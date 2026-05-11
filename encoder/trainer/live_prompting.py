import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from config.configurator import configs


SYSTEM_PROMPTS = {
    'user': """You are refining a recommendation-user profile based on measured distributional drift between a collaborative recommender and a language embedding model.
Return strict JSON with keys:
- drift_summary: a concise rewritten user preference summary under 120 words
- stable_preferences: short phrase describing what should remain stable
- shifted_preferences: short phrase describing what appears to have drifted
- update_focus: short phrase describing what the recommender should emphasize next
Do not include markdown fences or any text outside the JSON object.""",
    'item': """You are refining a recommendation-item profile based on measured distributional drift between a collaborative recommender and a language embedding model.
Return strict JSON with keys:
- drift_summary: a concise rewritten item appeal summary under 140 words
- stable_preferences: short phrase describing what should remain stable about the item
- shifted_preferences: short phrase describing what appears to have drifted in audience fit
- update_focus: short phrase describing what the recommender should emphasize next
Do not include markdown fences or any text outside the JSON object.""",
}


def parse_json_response(text):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def _truncate(text, limit=220):
    text = (text or '').strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + '...'


def _build_prompt_payload(entity_type, payload):
    lines = [
        'Entity ID: {}'.format(payload['entity_id']),
        'Base Profile: {}'.format(_truncate(payload.get('base_profile', ''), limit=600)),
        'Correction Alpha: {:.6f}'.format(payload.get('correction_alpha', 0.0)),
        'Mu Gap Norm: {:.6f}'.format(payload.get('mu_gap_norm', 0.0)),
        'Logvar Gap Norm: {:.6f}'.format(payload.get('logvar_gap_norm', 0.0)),
        'Residual Norm: {:.6f}'.format(payload.get('correction_norm', 0.0)),
    ]

    if entity_type == 'user':
        anchor_lines = []
        for anchor in payload.get('anchor_items', []):
            anchor_lines.append('- item {}: {}'.format(anchor['item_id'], _truncate(anchor.get('text', ''), 180)))
        if anchor_lines:
            lines.append('Recent Interaction Anchors:\n{}'.format('\n'.join(anchor_lines)))
    else:
        support_lines = []
        for support in payload.get('support_users', []):
            support_lines.append('- user {}: {}'.format(support['user_id'], _truncate(support.get('text', ''), 180)))
        if support_lines:
            lines.append('Supporting User Anchors:\n{}'.format('\n'.join(support_lines)))
        lines.append('Support Count: {}'.format(payload.get('support_count', 0)))
        lines.append('Alignment Delta: {:.6f}'.format(payload.get('alignment_delta', 0.0)))

    return SYSTEM_PROMPTS[entity_type], '\n'.join(lines)


def _fingerprint_payload(payload):
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode('utf-8')).hexdigest()[:16]


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, timeout, max_retries):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError('The live prompting pipeline requires the openai package in the active uv environment.') from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.max_retries = max_retries

    def chat(self, model, system_prompt, user_prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def embed(self, model, texts, dimensions=None):
        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    'input': texts,
                    'model': model,
                }
                if dimensions is not None:
                    request_kwargs['dimensions'] = dimensions
                response = self.client.embeddings.create(**request_kwargs)
                return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise exc
                time.sleep(2 ** attempt)


class LivePromptEngine:
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.config = configs['live_prompting']
        self.enabled = bool(self.config.get('enabled', False))
        self.user_texts = configs.get('usrprf_texts', [])
        self.item_texts = configs.get('itmprf_texts', [])
        self.artifact_root = Path(self.config['artifact_dir']) / configs['model']['name'] / configs['data']['name']
        self.cache_root = self.artifact_root / 'cache'
        self.sampled_user_ids = None

        if self.enabled:
            timeout = self.config['timeout']
            max_retries = self.config['max_retries']
            self.chat_client = OpenAICompatibleClient(
                self.config['chat_base_url'],
                self.config['chat_api_key'],
                timeout,
                max_retries,
            )
            self.embedding_client = OpenAICompatibleClient(
                self.config['embedding_base_url'],
                self.config['embedding_api_key'],
                timeout,
                max_retries,
            )
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def should_run(self, epoch_idx):
        if not self.enabled:
            return False
        frequency = max(1, int(self.config.get('eval_frequency', 1)))
        return epoch_idx % frequency == 0

    def run_epoch(self, model, epoch_idx):
        if not self.should_run(epoch_idx):
            return None
        if not hasattr(model, 'collect_live_prompt_user_payload'):
            return None

        user_ids = self._get_sampled_user_ids()
        if len(user_ids) == 0:
            return None

        batch_sparse = self.data_handler.train_data[user_ids]
        batch_tensor = torch.FloatTensor(batch_sparse.toarray()).to(configs['device'])

        user_payloads = model.collect_live_prompt_user_payload(
            user_ids,
            batch_tensor,
            item_texts=self.item_texts,
            top_k_items=self.config.get('top_k_items', 5),
        )
        item_payloads = model.collect_live_prompt_item_payload(
            user_ids,
            batch_tensor,
            user_texts=self.user_texts,
            item_texts=self.item_texts,
            item_sample_size=self.config.get('item_sample_size', 32),
            top_k_support_users=self.config.get('top_k_support_users', 3),
        )

        user_result = self._refresh_entity_embeddings(
            model,
            'user',
            user_payloads,
            self.config['chat_model'],
            self.config['embedding_model'],
            expected_dim=configs['usrprf_embeds'].shape[1],
            epoch_idx=epoch_idx,
        )
        item_result = self._refresh_entity_embeddings(
            model,
            'item',
            item_payloads,
            self.config['chat_model'],
            self.config['embedding_model'],
            expected_dim=configs['itmprf_embeds'].shape[1],
            epoch_idx=epoch_idx,
        )

        artifact = {
            'epoch': epoch_idx,
            'model': configs['model']['name'],
            'dataset': configs['data']['name'],
            'user': user_result['records'],
            'item': item_result['records'],
        }
        artifact_path = self.artifact_root / 'epoch_{:04d}.json'.format(epoch_idx)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open('w', encoding='utf-8') as file_obj:
            json.dump(artifact, file_obj, indent=2, ensure_ascii=False)

        return {
            'user_updates': user_result['updates'],
            'user_cache_hits': user_result['cache_hits'],
            'user_failures': user_result['failures'],
            'item_updates': item_result['updates'],
            'item_cache_hits': item_result['cache_hits'],
            'item_failures': item_result['failures'],
            'artifact_path': str(artifact_path),
        }

    def _get_sampled_user_ids(self):
        if self.sampled_user_ids is not None:
            return self.sampled_user_ids
        test_users = np.asarray(self.data_handler.valid_dataloader.dataset.test_users)
        if test_users.size == 0:
            self.sampled_user_ids = []
            return self.sampled_user_ids
        sample_size = min(int(self.config.get('sample_size', 64)), test_users.size)
        seed = int(configs['train']['seed']) + 7919
        rng = np.random.default_rng(seed)
        sampled = rng.choice(test_users, size=sample_size, replace=False)
        self.sampled_user_ids = sampled.astype(np.int64).tolist()
        return self.sampled_user_ids

    def _resolve_embedding_dimensions(self, expected_dim):
        configured_dim = self.config.get('embedding_dimensions')
        if configured_dim is None:
            return expected_dim
        return int(configured_dim)

    def _bootstrap_entity_alignment(self, model, entity_type, payloads, embedding_model):
        if not hasattr(model, 'needs_live_prompt_alignment'):
            return
        if not model.needs_live_prompt_alignment(entity_type):
            return

        bootstrap_payloads = [payload for payload in payloads if payload.get('base_profile', '').strip()]
        if not bootstrap_payloads:
            raise ValueError(
                'Cannot bootstrap live prompt alignment for {} because no base profiles were available.'.format(
                    entity_type,
                )
            )

        bootstrap_ids = [payload['entity_id'] for payload in bootstrap_payloads]
        bootstrap_texts = [payload['base_profile'] for payload in bootstrap_payloads]
        bootstrap_embeddings = self.embedding_client.embed(embedding_model, bootstrap_texts, dimensions=None)
        model.initialize_live_prompt_alignment(entity_type, bootstrap_ids, bootstrap_embeddings)

    def _refresh_entity_embeddings(self, model, entity_type, payloads, chat_model, embedding_model, expected_dim, epoch_idx):
        records = []
        cache_hits = 0
        failures = 0
        ids = []
        summaries = []

        for payload in payloads:
            prompt_fingerprint = _fingerprint_payload(payload)
            cache_path = self.cache_root / entity_type / '{}_{}.json'.format(payload['entity_id'], prompt_fingerprint)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                'entity_id': payload['entity_id'],
                'fingerprint': prompt_fingerprint,
                'cached': False,
            }

            parsed = None
            if self.config.get('cache_enabled', True) and cache_path.exists():
                with cache_path.open('r', encoding='utf-8') as file_obj:
                    cached_record = json.load(file_obj)
                parsed = cached_record.get('response')
                record.update(cached_record)
                record['cached'] = True
                cache_hits += 1
            else:
                system_prompt, user_prompt = _build_prompt_payload(entity_type, payload)
                try:
                    raw_response = self.chat_client.chat(chat_model, system_prompt, user_prompt)
                    parsed = parse_json_response(raw_response)
                except Exception as exc:
                    record['error'] = str(exc)
                    failures += 1
                    records.append(record)
                    continue

                if parsed is None or 'drift_summary' not in parsed:
                    record['error'] = 'Failed to parse drift_summary from model response.'
                    failures += 1
                    records.append(record)
                    continue

                record.update({
                    'payload': payload,
                    'response': parsed,
                    'epoch': epoch_idx,
                })
                if self.config.get('cache_enabled', True):
                    with cache_path.open('w', encoding='utf-8') as file_obj:
                        json.dump(record, file_obj, indent=2, ensure_ascii=False)

            ids.append(payload['entity_id'])
            summary_text = parsed['drift_summary']
            summaries.append(summary_text)
            record['summary'] = summary_text
            records.append(record)

        if summaries:
            requested_dimensions = self._resolve_embedding_dimensions(expected_dim)
            if getattr(model, 'use_native_live_prompt_embeddings', False):
                self._bootstrap_entity_alignment(model, entity_type, payloads, embedding_model)
                embeddings = self.embedding_client.embed(
                    embedding_model,
                    summaries,
                    dimensions=None,
                )
                if hasattr(model, 'transform_live_prompt_embeddings'):
                    embeddings = model.transform_live_prompt_embeddings(entity_type, embeddings)
            else:
                embeddings = self.embedding_client.embed(
                    embedding_model,
                    summaries,
                    dimensions=requested_dimensions,
                )
            returned_dimensions = {embedding.shape[0] for embedding in embeddings}
            if returned_dimensions != {expected_dim}:
                raise ValueError(
                    'Embedding width mismatch for {} live prompting: requested {}, expected {}, got {}. '
                    'Use an embedding model/server that supports the requested dimension, or set '
                    'live_prompting.embedding_dimensions to a compatible value.'.format(
                        entity_type,
                        requested_dimensions,
                        expected_dim,
                        sorted(returned_dimensions),
                    )
                )
            model.update_live_prompt_bank(entity_type, ids, embeddings, summaries=summaries, epoch_idx=epoch_idx)

        return {
            'updates': len(summaries),
            'cache_hits': cache_hits,
            'failures': failures,
            'records': records,
        }