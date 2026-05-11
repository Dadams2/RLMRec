"""
Multi-Faceted Profile Generation (ProEx-style 4-stage Chain-of-Thought)

Generates K profiles per user/item using existing original profiles (F1)
plus K-1 new diverse profiles via chain-of-thought reasoning (F2→F3→F4).

Stages:
  F1: Original profile (already in usr_prf.pkl / itm_prf.pkl)
  F2: Extract positive and negative aspects from the original profile
  F3: Analyze latent/implicit preferences not captured by F1
  F4: Generate K-1 new profiles with low-similarity constraint

Usage:
    python generate_multi_profiles.py --dataset amazon --entity user --K 4
    python generate_multi_profiles.py --dataset amazon --entity item --K 4
    python generate_multi_profiles.py --dataset amazon --entity user --K 4 \
            --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct

Requires:
    - A local vLLM server exposing the OpenAI-compatible chat API
    - Optional: VLLM_BASE_URL / VLLM_API_KEY / VLLM_CHAT_MODEL environment variables
  - Existing profiles in data/{dataset}/usr_prf.pkl or itm_prf.pkl
  - Optionally: existing user/item prompts for grounding in interaction data

Output:
  data/{dataset}/usr_multi_prf.pkl  (or itm_multi_prf.pkl)
  Format: dict[int -> list[{'profile': str, 'reasoning': str}]]
  Where list[0] is the original profile and list[1..K-1] are new profiles.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_CHAT_MODEL = os.environ.get("VLLM_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")


# ---- CoT system prompts for F2, F3, F4 ----

SYSTEM_F2_USER = """\
You are an expert at analyzing user preference profiles for recommendation systems.
Given a user profile summarization, extract the following in JSON format:
{
    "positive_aspects": "What types of items/attributes does this user clearly enjoy? List specific themes, genres, qualities.",
    "negative_aspects": "What types of items/attributes does this user seem to avoid or dislike? If unclear, state 'Not explicitly mentioned.'",
    "key_patterns": "What recurring behavioral patterns or preference signals are evident?"
}
Be specific and grounded in the profile text. Do not invent information not supported by the profile."""

SYSTEM_F2_ITEM = """\
You are an expert at analyzing item profiles for recommendation systems.
Given an item profile summarization, extract the following in JSON format:
{
    "positive_aspects": "What types of users would enjoy this item? List specific user traits, interests, demographics.",
    "negative_aspects": "What types of users would NOT enjoy this item? If unclear, state 'Not explicitly mentioned.'",
    "key_patterns": "What are the defining characteristics or unique selling points of this item?"
}
Be specific and grounded in the profile text. Do not invent information not supported by the profile."""

SYSTEM_F3_USER = """\
You are an expert at inferring latent user preferences from behavioral signals.
Given a user profile and its extracted aspects, analyze what IMPLICIT or LATENT preferences
might exist that are not directly stated. Consider:
- Reading between the lines of stated preferences
- What complementary interests often co-occur with the stated ones
- Underrepresented facets the profile might have missed

Respond in JSON format:
{
    "latent_preferences": "Implicit preferences or interests that likely exist but aren't directly stated",
    "complementary_interests": "Related interests that commonly co-occur with stated preferences",
    "underrepresented_facets": "Aspects of user taste that the original profile likely under-represents"
}
Be analytical but reasonable. Do not fabricate unsupported claims."""

SYSTEM_F3_ITEM = """\
You are an expert at inferring latent item characteristics for recommendation.
Given an item profile and its extracted aspects, analyze what IMPLICIT characteristics
or appeal factors might exist that aren't directly stated. Consider:
- Hidden appeal factors beyond the obvious
- Niche audiences that might be attracted
- Underrepresented qualities the profile might have missed

Respond in JSON format:
{
    "latent_characteristics": "Implicit item qualities or appeal factors not directly stated",
    "niche_audiences": "Specific user segments that might be attracted beyond the obvious audience",
    "underrepresented_qualities": "Aspects of the item that the original profile likely under-represents"
}
Be analytical but reasonable. Do not fabricate unsupported claims."""

SYSTEM_F4_USER_TEMPLATE = """\
You are an expert at creating diverse user preference profiles for recommendation systems.
Given:
- An ORIGINAL profile of a user
- Extracted positive/negative aspects
- Latent preference analysis

Generate {num_new} NEW and DISTINCT profile summarizations for this same user.
Each new profile must:
1. Describe the SAME user but from a DIFFERENT perspective or emphasis
2. Use ENTIRELY DIFFERENT wording and phrasing than the original
3. Highlight different facets of the user's preferences
4. Be plausible given the original profile and analysis
5. Be no longer than 100 words each

CRITICAL: Each profile should have LOW SIMILARITY to the original and to each other.
They should collectively cover a BROADER region of the user's preference space.

Respond in JSON format:
{{
    "profiles": [
        {{"profile": "First new profile text", "emphasis": "What facet this profile emphasizes"}},
        {{"profile": "Second new profile text", "emphasis": "What facet this profile emphasizes"}}
    ]
}}
Generate exactly {num_new} profiles."""

SYSTEM_F4_ITEM_TEMPLATE = """\
You are an expert at creating diverse item profiles for recommendation systems.
Given:
- An ORIGINAL profile of an item
- Extracted positive/negative aspects
- Latent characteristic analysis

Generate {num_new} NEW and DISTINCT profile summarizations for this same item.
Each new profile must:
1. Describe the SAME item but from a DIFFERENT perspective or emphasis
2. Use ENTIRELY DIFFERENT wording and phrasing than the original
3. Highlight different facets of the item's appeal or characteristics
4. Be plausible given the original profile and analysis
5. Be no longer than 200 words each

CRITICAL: Each profile should have LOW SIMILARITY to the original and to each other.
They should collectively cover a BROADER region of the item's characteristic space.

Respond in JSON format:
{{
    "profiles": [
        {{"profile": "First new profile text", "emphasis": "What facet this profile emphasizes"}},
        {{"profile": "Second new profile text", "emphasis": "What facet this profile emphasizes"}}
    ]
}}
Generate exactly {num_new} profiles."""


def call_llm(client, system_prompt, user_message, model=DEFAULT_CHAT_MODEL, max_retries=3):
    """Call an OpenAI-compatible chat API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,  # Higher temp for diverse outputs in F4
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after {max_retries} attempts: {e}")
                return None


def parse_json_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    if text is None:
        return None
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def generate_multi_profiles_for_entity(
    client, entity_id, original_profile, entity_type, K, model
):
    """
    Generate K-1 new profiles for one entity using 4-stage CoT.

    Returns: list of K profile dicts [original, new_1, ..., new_{K-1}]
    """
    original_text = original_profile["profile"]
    original_reasoning = original_profile.get("reasoning", "")
    num_new = K - 1

    # F2: Extract positive/negative aspects
    if entity_type == "user":
        sys_f2 = SYSTEM_F2_USER
    else:
        sys_f2 = SYSTEM_F2_ITEM

    f2_input = f"Profile: {original_text}"
    if original_reasoning:
        f2_input += f"\nReasoning: {original_reasoning}"

    f2_response = call_llm(client, sys_f2, f2_input, model=model)
    f2_parsed = parse_json_response(f2_response)

    if f2_parsed is None:
        # Fallback: use raw text
        f2_text = f2_response or "Analysis unavailable."
    else:
        f2_text = json.dumps(f2_parsed, indent=2)

    # F3: Analyze latent preferences
    if entity_type == "user":
        sys_f3 = SYSTEM_F3_USER
    else:
        sys_f3 = SYSTEM_F3_ITEM

    f3_input = f"Original Profile: {original_text}\n\nExtracted Aspects:\n{f2_text}"
    f3_response = call_llm(client, sys_f3, f3_input, model=model)
    f3_parsed = parse_json_response(f3_response)

    if f3_parsed is None:
        f3_text = f3_response or "Analysis unavailable."
    else:
        f3_text = json.dumps(f3_parsed, indent=2)

    # F4: Generate K-1 new diverse profiles
    if entity_type == "user":
        sys_f4 = SYSTEM_F4_USER_TEMPLATE.format(num_new=num_new)
    else:
        sys_f4 = SYSTEM_F4_ITEM_TEMPLATE.format(num_new=num_new)

    f4_input = (
        f"ORIGINAL PROFILE:\n{original_text}\n\n"
        f"EXTRACTED ASPECTS:\n{f2_text}\n\n"
        f"LATENT ANALYSIS:\n{f3_text}"
    )

    f4_response = call_llm(client, sys_f4, f4_input, model=model)
    f4_parsed = parse_json_response(f4_response)

    # Build result list
    profiles = [original_profile]  # Index 0 = original profile (F1)

    if f4_parsed and "profiles" in f4_parsed:
        for p in f4_parsed["profiles"][:num_new]:
            profiles.append({
                "profile": p.get("profile", ""),
                "reasoning": p.get("emphasis", ""),
            })
    else:
        # Fallback: if parsing failed, create placeholder profiles
        print(f"  Warning: F4 parsing failed for entity {entity_id}, using fallback")
        for i in range(num_new):
            profiles.append({
                "profile": original_text,  # Duplicate (will be differentiated by offsets)
                "reasoning": f"Fallback copy {i+1} — F4 generation failed",
            })

    # Pad if we got fewer than K
    while len(profiles) < K:
        profiles.append({
            "profile": profiles[0]["profile"],
            "reasoning": "Padding copy — insufficient profiles generated",
        })

    return profiles[:K]


def main():
    parser = argparse.ArgumentParser(description="Generate multi-faceted profiles")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["amazon", "yelp", "steam"])
    parser.add_argument("--entity", type=str, required=True,
                        choices=["user", "item"])
    parser.add_argument("--K", type=int, default=4, help="Number of profiles per entity")
    parser.add_argument("--model", type=str, default=DEFAULT_CHAT_MODEL,
                        help="Chat model exposed by the local vLLM server")
    parser.add_argument("--base_url", type=str, default=DEFAULT_VLLM_BASE_URL,
                        help="Base URL for the OpenAI-compatible vLLM server")
    parser.add_argument("--api_key", type=str, default=DEFAULT_VLLM_API_KEY,
                        help="API key for the local vLLM server; use any non-empty value if auth is disabled")
    parser.add_argument("--start_id", type=int, default=None,
                        help="Resume from this entity ID (for crash recovery)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Save checkpoint every N entities")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Process only this many entities (for testing)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process 3 entities and print results (no save)")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Load existing original profiles (F1)
    if args.entity == "user":
        prf_path = Path(f"data/{args.dataset}/usr_prf.pkl")
        out_path = Path(f"data/{args.dataset}/usr_multi_prf.pkl")
    else:
        prf_path = Path(f"data/{args.dataset}/itm_prf.pkl")
        out_path = Path(f"data/{args.dataset}/itm_multi_prf.pkl")

    with open(prf_path, "rb") as f:
        original_profiles = pickle.load(f)

    print(f"Loaded {len(original_profiles)} {args.entity} profiles from {prf_path}")
    print(f"Generating K={args.K} profiles per {args.entity} using {args.model}")
    print(f"vLLM endpoint: {args.base_url}")

    # Load checkpoint if resuming
    multi_profiles = {}
    if out_path.exists() and args.start_id is not None:
        with open(out_path, "rb") as f:
            multi_profiles = pickle.load(f)
        print(f"Loaded checkpoint with {len(multi_profiles)} entities from {out_path}")

    # Determine which entities to process
    entity_ids = sorted(original_profiles.keys())
    if args.start_id is not None:
        entity_ids = [eid for eid in entity_ids if eid >= args.start_id]
    if args.max_entities is not None:
        entity_ids = entity_ids[:args.max_entities]

    # Skip already-processed
    entity_ids = [eid for eid in entity_ids if eid not in multi_profiles]
    print(f"Processing {len(entity_ids)} entities (skipping {len(multi_profiles)} already done)")

    if args.dry_run:
        entity_ids = entity_ids[:3]
        print(f"\n--- DRY RUN: processing {len(entity_ids)} entities ---\n")

    total = len(entity_ids)
    for idx, eid in enumerate(entity_ids):
        original = original_profiles[eid]

        profiles = generate_multi_profiles_for_entity(
            client, eid, original, args.entity, args.K, args.model
        )
        multi_profiles[eid] = profiles

        # Progress
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] Entity {eid}: generated {len(profiles)} profiles")
            # Show first new profile snippet
            if len(profiles) > 1:
                snippet = profiles[1]["profile"][:120]
                print(f"    NP #1: {snippet}...")

        # Checkpoint save
        if not args.dry_run and (idx + 1) % args.batch_size == 0:
            with open(out_path, "wb") as f:
                pickle.dump(multi_profiles, f)
            print(f"  Checkpoint saved: {len(multi_profiles)} entities → {out_path}")

    # Final save
    if args.dry_run:
        print("\n--- DRY RUN RESULTS ---")
        for eid in list(multi_profiles.keys())[-3:]:
            print(f"\nEntity {eid}:")
            for k, p in enumerate(multi_profiles[eid]):
                tag = "OP" if k == 0 else f"NP #{k}"
                print(f"  [{tag}] {p['profile'][:200]}")
    else:
        with open(out_path, "wb") as f:
            pickle.dump(multi_profiles, f)
        print(f"\nDone. Saved {len(multi_profiles)} entities × K={args.K} → {out_path}")


if __name__ == "__main__":
    main()
