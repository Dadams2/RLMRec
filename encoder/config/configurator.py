import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn


def _load_pickle(path):
    with open(path, 'rb') as file_obj:
        return pickle.load(file_obj)


def _extract_profile_texts(profile_blob):
    if isinstance(profile_blob, dict):
        texts = []
        for idx in range(len(profile_blob)):
            record = profile_blob[idx]
            if isinstance(record, dict):
                texts.append(record.get('profile', ''))
            else:
                texts.append(str(record))
        return texts

    if isinstance(profile_blob, list):
        texts = []
        for record in profile_blob:
            if isinstance(record, dict):
                texts.append(record.get('profile', ''))
            else:
                texts.append(str(record))
        return texts

    raise TypeError('Unsupported profile blob type: {}'.format(type(profile_blob)))


def _apply_live_prompting_defaults(configs):
    defaults = {
        'enabled': False,
        'eval_frequency': 1,
        'sample_size': 64,
        'item_sample_size': 32,
        'top_k_items': 5,
        'top_k_support_users': 3,
        'artifact_dir': './encoder/log/live_prompting',
        'cache_enabled': True,
        'chat_base_url': os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1'),
        'chat_api_key': os.environ.get('VLLM_API_KEY', 'EMPTY'),
        'chat_model': os.environ.get('VLLM_CHAT_MODEL', 'Qwen/Qwen2.5-7B-Instruct'),
        'embedding_base_url': os.environ.get('VLLM_EMB_BASE_URL', os.environ.get('VLLM_BASE_URL', 'http://localhost:8001/v1')),
        'embedding_api_key': os.environ.get('VLLM_EMB_API_KEY', os.environ.get('VLLM_API_KEY', 'EMPTY')),
        'embedding_model': os.environ.get('VLLM_EMB_MODEL', 'Qwen/Qwen3-Embedding-4B'),
        'embedding_dimensions': None,
        'max_retries': 3,
        'timeout': 120,
        'feedback_mode': 'sidecar_bank',
        'response_format': 'json',
    }
    live_prompting = configs.setdefault('live_prompting', {})
    for key, value in defaults.items():
        live_prompting.setdefault(key, value)

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception(f"Please create the yaml file for your model {model_name} first.")

    # read yml file
    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed
        _apply_live_prompting_defaults(configs)

        # semantic embeddings
        usrprf_embeds_path = "../data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "../data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        configs['usrprf_embeds'] = _load_pickle(usrprf_embeds_path)
        configs['itmprf_embeds'] = _load_pickle(itmprf_embeds_path)

        usrprf_texts_path = "../data/{}/usr_prf.pkl".format(configs['data']['name'])
        itmprf_texts_path = "../data/{}/itm_prf.pkl".format(configs['data']['name'])
        if os.path.exists(usrprf_texts_path):
            configs['usrprf_texts'] = _extract_profile_texts(_load_pickle(usrprf_texts_path))
        else:
            configs['usrprf_texts'] = []
        if os.path.exists(itmprf_texts_path):
            configs['itmprf_texts'] = _extract_profile_texts(_load_pickle(itmprf_texts_path))
        else:
            configs['itmprf_texts'] = []

        # multi-profile embeddings (optional, for ProEx-style models)
        usr_multi_path = "../data/{}/usr_multi_emb_np.pkl".format(configs['data']['name'])
        itm_multi_path = "../data/{}/itm_multi_emb_np.pkl".format(configs['data']['name'])
        if os.path.exists(usr_multi_path):
            configs['usrprf_multi_embeds'] = _load_pickle(usr_multi_path)
        if os.path.exists(itm_multi_path):
            configs['itmprf_multi_embeds'] = _load_pickle(itm_multi_path)

        return configs

configs = parse_configure()
