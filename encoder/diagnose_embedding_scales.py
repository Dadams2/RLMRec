"""
Diagnostic script to measure the relative scales and contributions of VAE vs LLM embeddings
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append('..')
from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.logger import Logger
import argparse


def diagnose_embedding_scales(model, data_handler, n_samples=1000, batch_size=256):
    """
    Measure and compare the scales of VAE and LLM embeddings
    """
    model.eval()
    model.is_training = False
    
    stats = {
        'mu_src_norms': [],
        'mu_llm_norms': [],
        'mu_combined_norms': [],
        'mu_src_magnitude': [],
        'mu_llm_magnitude': [],
        'correlation': [],
        'vae_contribution_ratio': []
    }
    
    # Sample users
    n_users = data_handler.train_data.shape[0]
    sampled_users = np.random.choice(n_users, size=min(n_samples, n_users), replace=False)
    
    print(f"\nAnalyzing {len(sampled_users)} user samples...")
    print("=" * 80)
    
    with torch.no_grad():
        for i in range(0, len(sampled_users), batch_size):
            batch_users = sampled_users[i:i+batch_size]
            batch_data = data_handler.train_data[batch_users]
            data = torch.FloatTensor(batch_data.toarray()).cuda()
            
            user_ids = torch.LongTensor(batch_users).cuda()
            user_emb = model.usrprf_embeds[user_ids]
            
            # Get embeddings using model's encode method
            mu_src, mu_llm, logvar_src, logvar_llm = model.encode(data, user_emb)
            
            # Compute norms (L2 norm per sample)
            mu_src_norm = torch.norm(mu_src, dim=1)
            mu_llm_norm = torch.norm(mu_llm, dim=1)
            mu_combined = mu_src + mu_llm
            mu_combined_norm = torch.norm(mu_combined, dim=1)
            
            stats['mu_src_norms'].extend(mu_src_norm.cpu().numpy())
            stats['mu_llm_norms'].extend(mu_llm_norm.cpu().numpy())
            stats['mu_combined_norms'].extend(mu_combined_norm.cpu().numpy())
            
            # Compute mean absolute values
            stats['mu_src_magnitude'].extend(torch.abs(mu_src).mean(dim=1).cpu().numpy())
            stats['mu_llm_magnitude'].extend(torch.abs(mu_llm).mean(dim=1).cpu().numpy())
            
            # Compute cosine similarity (correlation)
            mu_src_normalized = torch.nn.functional.normalize(mu_src, dim=1)
            mu_llm_normalized = torch.nn.functional.normalize(mu_llm, dim=1)
            cosine_sim = (mu_src_normalized * mu_llm_normalized).sum(dim=1)
            stats['correlation'].extend(cosine_sim.cpu().numpy())
            
            # Compute VAE contribution ratio
            vae_contribution = mu_src_norm / (mu_combined_norm + 1e-8)
            stats['vae_contribution_ratio'].extend(vae_contribution.cpu().numpy())
    
    # Convert to numpy arrays
    for key in stats:
        stats[key] = np.array(stats[key])
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("EMBEDDING SCALE ANALYSIS")
    print("=" * 80)
    
    print("\n1. L2 NORMS (magnitude of embedding vectors):")
    print(f"   VAE (mu_src):        mean={stats['mu_src_norms'].mean():.4f}, std={stats['mu_src_norms'].std():.4f}")
    print(f"   LLM (mu_llm):        mean={stats['mu_llm_norms'].mean():.4f}, std={stats['mu_llm_norms'].std():.4f}")
    print(f"   Combined:            mean={stats['mu_combined_norms'].mean():.4f}, std={stats['mu_combined_norms'].std():.4f}")
    print(f"   Ratio (VAE/LLM):     {stats['mu_src_norms'].mean() / stats['mu_llm_norms'].mean():.4f}")
    
    print("\n2. MEAN ABSOLUTE VALUES (average magnitude per dimension):")
    print(f"   VAE (mu_src):        mean={stats['mu_src_magnitude'].mean():.4f}, std={stats['mu_src_magnitude'].std():.4f}")
    print(f"   LLM (mu_llm):        mean={stats['mu_llm_magnitude'].mean():.4f}, std={stats['mu_llm_magnitude'].std():.4f}")
    print(f"   Ratio (VAE/LLM):     {stats['mu_src_magnitude'].mean() / stats['mu_llm_magnitude'].mean():.4f}")
    
    print("\n3. COSINE SIMILARITY (directional alignment):")
    print(f"   Mean correlation:    {stats['correlation'].mean():.4f}")
    print(f"   Std correlation:     {stats['correlation'].std():.4f}")
    print(f"   % positively corr:   {(stats['correlation'] > 0).mean() * 100:.1f}%")
    print(f"   % strongly corr:     {(stats['correlation'] > 0.5).mean() * 100:.1f}%")
    
    print("\n4. VAE CONTRIBUTION TO COMBINED EMBEDDING:")
    print(f"   Mean ratio:          {stats['vae_contribution_ratio'].mean():.4f}")
    print(f"   Median ratio:        {np.median(stats['vae_contribution_ratio']):.4f}")
    print(f"   % contrib < 10%:     {(stats['vae_contribution_ratio'] < 0.1).mean() * 100:.1f}%")
    print(f"   % contrib < 25%:     {(stats['vae_contribution_ratio'] < 0.25).mean() * 100:.1f}%")
    print(f"   % contrib < 50%:     {(stats['vae_contribution_ratio'] < 0.5).mean() * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    
    ratio = stats['mu_src_norms'].mean() / stats['mu_llm_norms'].mean()
    vae_contrib = stats['vae_contribution_ratio'].mean()
    
    if ratio < 0.1:
        print("⚠️  SEVERE SCALE MISMATCH: VAE embeddings are >10x smaller than LLM")
        print("    The LLM completely dominates the combined representation.")
        print("    Recommendation: Scale up VAE output or scale down LLM.")
    elif ratio < 0.3:
        print("⚠️  SIGNIFICANT SCALE MISMATCH: VAE embeddings are 3-10x smaller")
        print("    The LLM strongly dominates, but VAE provides some signal.")
        print("    Recommendation: Consider rebalancing the scales.")
    elif ratio < 0.7:
        print("✓  MODERATE BALANCE: VAE and LLM have comparable scales")
        print("    Both paths contribute to the combined representation.")
    else:
        print("✓  BALANCED or VAE-DOMINANT: VAE embeddings are similar or larger")
    
    if stats['correlation'].mean() > 0.5:
        print("\n⚠️  HIGH CORRELATION: VAE is learning to mimic LLM")
        print("    This suggests the Wasserstein Distance loss may be too strong.")
        print(f"    Mean cosine similarity: {stats['correlation'].mean():.3f}")
    
    print("\n" + "=" * 80)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Diagnose embedding scale mismatch')
    parser.add_argument('--model', type=str, default='mult_vae_godm', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(configs)
    
    # Load data
    data_handler = build_data_handler()
    data_handler.load_data()
    
    # Build and load model
    model = build_model(data_handler).cuda()
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Run diagnosis
    stats = diagnose_embedding_scales(model, data_handler, n_samples=args.n_samples)
    
    # Save results
    output_dir = Path('./visualization_outputs')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'scale_diagnosis_{args.model}_{args.dataset}.npz'
    np.savez(output_file, **stats)
    print(f"\nSaved detailed statistics to {output_file}")


if __name__ == '__main__':
    main()
