"""
Generate multi-profile embeddings from multi-faceted profiles.

Takes the K profiles per entity from generate_multi_profiles.py and embeds
each profile separately via a local vLLM OpenAI-compatible embeddings API.

Usage:
  python generate_multi_emb.py --dataset amazon --entity user --K 4
  python generate_multi_emb.py --dataset amazon --entity item --K 4
  python generate_multi_emb.py --dataset amazon --entity user --K 4 \
      --base_url http://localhost:8001/v1 --emb_model Qwen/Qwen3-Embedding-4B

Requires:
  - A local vLLM server exposing the OpenAI-compatible embeddings API
  - Optional: VLLM_BASE_URL / VLLM_API_KEY / VLLM_EMB_MODEL environment variables
  - Multi-profile pickle: data/{dataset}/usr_multi_prf.pkl or itm_multi_prf.pkl
  - The embedding dimension must match the existing usr_emb_np.pkl / itm_emb_np.pkl

Output:
  data/{dataset}/usr_multi_emb_np.pkl (or itm_multi_emb_np.pkl)
  Format: numpy array of shape [num_entities, K, embedding_dim]
  Entity ordering matches the original single-profile embedding files.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_EMBEDDING_MODEL = os.environ.get("VLLM_EMB_MODEL", "Qwen/Qwen3-Embedding-4B")


def embed_batch(client, texts, model=DEFAULT_EMBEDDING_MODEL, max_retries=3):
    """Embed a batch of texts via an OpenAI-compatible embeddings API."""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after {max_retries} attempts: {e}")
                raise


def validate_embedding_batch(embeddings, expected_dim, model_name):
    """Ensure the served embedding model matches the downstream expected width."""
    if not embeddings:
        return

    batch_dim = len(embeddings[0])
    if any(len(embedding) != batch_dim for embedding in embeddings):
        raise ValueError("Embedding API returned inconsistent vector sizes within the same batch.")

    if batch_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch for model {model_name}: expected {expected_dim}, got {batch_dim}. "
            "The ProEx pipeline assumes multi-profile embeddings have the same width as the existing "
            "single-profile embeddings in usr_emb_np.pkl / itm_emb_np.pkl. Use a compatible local embedding "
            "model or regenerate the base embeddings with the same model first."
        )


def main():
    parser = argparse.ArgumentParser(description="Generate multi-profile embeddings")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["amazon", "yelp", "steam"])
    parser.add_argument("--entity", type=str, required=True,
                        choices=["user", "item"])
    parser.add_argument("--K", type=int, default=4, help="Number of profiles per entity")
    parser.add_argument("--emb_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model exposed by the local vLLM server")
    parser.add_argument("--base_url", type=str, default=DEFAULT_VLLM_BASE_URL,
                        help="Base URL for the OpenAI-compatible vLLM embeddings server")
    parser.add_argument("--api_key", type=str, default=DEFAULT_VLLM_API_KEY,
                        help="API key for the local vLLM server; use any non-empty value if auth is disabled")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of texts per API call")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Load multi-profiles
    if args.entity == "user":
        prf_path = Path(f"data/{args.dataset}/usr_multi_prf.pkl")
        orig_emb_path = Path(f"data/{args.dataset}/usr_emb_np.pkl")
        out_path = Path(f"data/{args.dataset}/usr_multi_emb_np.pkl")
    else:
        prf_path = Path(f"data/{args.dataset}/itm_multi_prf.pkl")
        orig_emb_path = Path(f"data/{args.dataset}/itm_emb_np.pkl")
        out_path = Path(f"data/{args.dataset}/itm_multi_emb_np.pkl")

    with open(prf_path, "rb") as f:
        multi_profiles = pickle.load(f)

    with open(orig_emb_path, "rb") as f:
        orig_emb = pickle.load(f)

    num_entities = orig_emb.shape[0]
    emb_dim = orig_emb.shape[1]
    K = args.K

    print(f"Loaded {len(multi_profiles)} multi-profiles from {prf_path}")
    print(f"Original embeddings: {orig_emb.shape} from {orig_emb_path}")
    print(f"Will produce: [{num_entities}, {K}, {emb_dim}] embeddings")
    print(f"vLLM endpoint: {args.base_url}")
    print(f"Embedding model: {args.emb_model}")

    all_texts = []
    entity_order = list(range(num_entities))
    missing_count = 0

    for eid in entity_order:
        if eid in multi_profiles:
            profiles = multi_profiles[eid]
            for k in range(K):
                if k < len(profiles):
                    all_texts.append(profiles[k]["profile"])
                else:
                    all_texts.append(profiles[0]["profile"])
        else:
            missing_count += 1
            for _ in range(K):
                all_texts.append("")

    if missing_count > 0:
        print(f"Warning: {missing_count} entities missing from multi-profiles, "
              f"will use original embedding for all K slots")

    total_texts = len(all_texts)
    print(f"Embedding {total_texts} texts in batches of {args.batch_size}...")

    all_embeddings = []
    for start in range(0, total_texts, args.batch_size):
        end = min(start + args.batch_size, total_texts)
        batch_texts = all_texts[start:end]

        non_empty_indices = [i for i, text in enumerate(batch_texts) if text.strip()]
        if non_empty_indices:
            non_empty_texts = [batch_texts[i] for i in non_empty_indices]
            embeddings = embed_batch(client, non_empty_texts, model=args.emb_model)
            validate_embedding_batch(embeddings, emb_dim, args.emb_model)

            batch_embs = [np.zeros(emb_dim, dtype=np.float32) for _ in batch_texts]
            for idx, emb_idx in enumerate(non_empty_indices):
                batch_embs[emb_idx] = embeddings[idx]
            all_embeddings.extend(batch_embs)
        else:
            all_embeddings.extend([np.zeros(emb_dim, dtype=np.float32) for _ in batch_texts])

        done = min(end, total_texts)
        if done % (args.batch_size * 10) == 0 or done == total_texts:
            print(f"  Embedded {done}/{total_texts} texts")

    multi_emb = np.array(all_embeddings, dtype=np.float32).reshape(num_entities, K, emb_dim)

    for eid in entity_order:
        if eid not in multi_profiles:
            multi_emb[eid, :, :] = orig_emb[eid]

    print(f"Final shape: {multi_emb.shape}")

    with open(out_path, "wb") as f:
        pickle.dump(multi_emb, f)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
