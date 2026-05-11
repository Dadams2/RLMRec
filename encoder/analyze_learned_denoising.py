"""
Analyze learned denoising behavior after training.

Visualizes:
1. Learned attention weights vs consensus signal/noise dimensions
2. Estimated noise magnitude per dimension
3. Correlation between attention and gradient importance from Phase 1

Usage:
    python analyze_learned_denoising.py --dataset amazon --model mult_vae_mddm_denoised
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.utils import init_seed


def load_trained_model(model_name, dataset):
    """Load trained denoised model."""
    print(f"Loading {model_name} trained on {dataset}...")
    
    # Setup configs
    configs['dataset'] = dataset
    configs['data']['type'] = 'general_cf'
    configs['model']['name'] = model_name
    
    # Initialize
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    
    # Build and load model
    model = build_model(data_handler)
    
    checkpoint_path = Path(f'./checkpoint/{dataset}/{model_name}.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.cuda()
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


def load_phase1_analysis(dataset):
    """Load gradient importance from Phase 1 analysis."""
    analysis_file = Path(f'./analysis_results/{dataset}/mult_vae_mddm/mult_vae_mddm_{dataset}_analysis.json')
    
    if not analysis_file.exists():
        print(f"⚠ Phase 1 analysis not found: {analysis_file}")
        return None
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    return data


def analyze_denoising(model, dataset, phase1_data=None):
    """Analyze what the denoising layer learned."""
    
    print("\n" + "="*70)
    print("  LEARNED DENOISING ANALYSIS")
    print("="*70 + "\n")
    
    # Get denoising components for user embeddings
    model.denoise_embeddings()
    user_sample = model.usrprf_embeds_raw[:500]  # Sample 500 users
    
    with torch.no_grad():
        user_denoised, user_comp = model.user_denoiser(user_sample, return_components=True)
    
    # Extract components
    attention_weights = user_comp['attention_weights'].cpu().numpy()  # [500, 1536]
    noise_estimate = user_comp['noise_estimate'].cpu().numpy()  # [500, 1536]
    
    # Average over users
    avg_attention = attention_weights.mean(0)  # [1536]
    avg_noise = np.abs(noise_estimate).mean(0)  # [1536]
    
    # Consensus dimensions
    signal_dims = set(model.user_signal_dims)
    noise_dims = set(model.user_noise_dims)
    
    # Analysis 1: Attention vs Consensus
    print("--- Attention Weights vs Consensus Dimensions ---\n")
    
    # Top 20 by learned attention
    top_attention_indices = np.argsort(avg_attention)[-20:][::-1]
    top_attention_values = avg_attention[top_attention_indices]
    
    # Check overlap with consensus signal
    overlap_signal = set(top_attention_indices.tolist()) & signal_dims
    overlap_pct = len(overlap_signal) / 20 * 100
    
    print(f"Top 20 Learned Attention Dimensions:")
    print(f"  Indices: {top_attention_indices.tolist()}")
    print(f"  Mean attention: {avg_attention[top_attention_indices].mean():.4f}")
    print(f"\nOverlap with Consensus Signal (20 dims):")
    print(f"  Matching: {len(overlap_signal)}/20 ({overlap_pct:.0f}%)")
    if overlap_signal:
        print(f"  Shared: {sorted(overlap_signal)}")
    
    # Bottom 20 by learned attention
    bottom_attention_indices = np.argsort(avg_attention)[:20]
    overlap_noise = set(bottom_attention_indices.tolist()) & noise_dims
    overlap_noise_pct = len(overlap_noise) / 20 * 100
    
    print(f"\nBottom 20 Learned Attention Dimensions:")
    print(f"  Indices: {bottom_attention_indices.tolist()}")
    print(f"  Mean attention: {avg_attention[bottom_attention_indices].mean():.4f}")
    print(f"\nOverlap with Consensus Noise (20 dims):")
    print(f"  Matching: {len(overlap_noise)}/20 ({overlap_noise_pct:.0f}%)")
    if overlap_noise:
        print(f"  Shared: {sorted(overlap_noise)}")
    
    # Analysis 2: Noise Magnitude
    print("\n\n--- Estimated Noise Magnitude ---\n")
    
    top_noise_indices = np.argsort(avg_noise)[-20:][::-1]
    print(f"Top 20 Noisy Dimensions (by learned noise estimate):")
    print(f"  Indices: {top_noise_indices.tolist()}")
    print(f"  Mean noise: {avg_noise[top_noise_indices].mean():.4f}")
    
    overlap_noise_est = set(top_noise_indices.tolist()) & noise_dims
    print(f"\nOverlap with Consensus Noise:")
    print(f"  Matching: {len(overlap_noise_est)}/20 ({len(overlap_noise_est)/20*100:.0f}%)")
    if overlap_noise_est:
        print(f"  Shared: {sorted(overlap_noise_est)}")
    
    # Statistics
    print("\n\n--- Overall Statistics ---\n")
    print(f"Attention weights: mean={avg_attention.mean():.4f}, std={avg_attention.std():.4f}")
    print(f"  Sparsity (close to 0 or 1): {((avg_attention < 0.1) | (avg_attention > 0.9)).sum()}/1536")
    print(f"Noise magnitude: mean={avg_noise.mean():.4f}, std={avg_noise.std():.4f}")
    print(f"  Sparse noise (<0.01): {(avg_noise < 0.01).sum()}/1536")
    
    # Visualization
    print("\n\n--- Creating Visualizations ---\n")
    
    output_dir = Path(f'./analysis_results/{dataset}/denoising_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Attention weights distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of attention weights
    axes[0, 0].hist(avg_attention, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Attention Weight')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Learned Attention Weights')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Neutral (0.5)')
    axes[0, 0].legend()
    
    # Attention: Signal vs Noise dims
    signal_attention = [avg_attention[d] for d in signal_dims]
    noise_attention = [avg_attention[d] for d in noise_dims]
    
    axes[0, 1].boxplot([signal_attention, noise_attention], labels=['Signal Dims', 'Noise Dims'])
    axes[0, 1].set_ylabel('Attention Weight')
    axes[0, 1].set_title('Attention: Consensus Signal vs Noise Dimensions')
    axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.3)
    
    # Noise magnitude distribution
    axes[1, 0].hist(avg_noise, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Noise Magnitude')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Estimated Noise')
    
    # Attention vs Noise scatter
    axes[1, 1].scatter(avg_attention, avg_noise, alpha=0.3, s=10)
    
    # Highlight consensus dims
    for d in signal_dims:
        axes[1, 1].scatter(avg_attention[d], avg_noise[d], color='green', s=50, alpha=0.6, label='Signal' if d == min(signal_dims) else '')
    for d in noise_dims:
        axes[1, 1].scatter(avg_attention[d], avg_noise[d], color='red', s=50, alpha=0.6, label='Noise' if d == min(noise_dims) else '')
    
    axes[1, 1].set_xlabel('Attention Weight')
    axes[1, 1].set_ylabel('Noise Magnitude')
    axes[1, 1].set_title('Attention vs Noise (Consensus dims highlighted)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_file = output_dir / 'learned_denoising_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization: {output_file}")
    
    # Plot 2: If Phase 1 data available, compare with gradient importance
    if phase1_data is not None:
        print("\n--- Comparing with Phase 1 Gradient Importance ---\n")
        
        # Get gradient importance from Phase 1
        gradient_importance = np.array(phase1_data['gradient_importance']['user_importance'])
        
        # Normalize both to [0, 1]
        grad_norm = (gradient_importance - gradient_importance.min()) / (gradient_importance.max() - gradient_importance.min())
        attn_norm = avg_attention
        
        # Correlation
        correlation = np.corrcoef(grad_norm, attn_norm)[0, 1]
        print(f"Correlation (Gradient Importance vs Learned Attention): {correlation:.4f}")
        
        if correlation > 0.5:
            print("  ✓ Strong positive correlation - attention aligns with gradient importance!")
        elif correlation > 0.2:
            print("  ≈ Moderate correlation - some alignment")
        else:
            print("  ✗ Weak correlation - attention doesn't match gradient importance well")
        
        # Scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(grad_norm, attn_norm, alpha=0.3, s=20)
        
        # Highlight consensus
        for d in signal_dims:
            ax.scatter(grad_norm[d], attn_norm[d], color='green', s=80, alpha=0.7, 
                      label='Signal' if d == min(signal_dims) else '')
        for d in noise_dims:
            ax.scatter(grad_norm[d], attn_norm[d], color='red', s=80, alpha=0.7,
                      label='Noise' if d == min(noise_dims) else '')
        
        ax.set_xlabel('Phase 1 Gradient Importance (normalized)')
        ax.set_ylabel('Learned Attention Weight')
        ax.set_title(f'Gradient Importance vs Learned Attention (r={correlation:.3f})')
        ax.legend()
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect correlation')
        
        output_file = output_dir / 'gradient_vs_attention.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison: {output_file}")
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    # Summary
    print("Key Takeaways:")
    if overlap_pct > 70:
        print(f"  ✓ Strong alignment: {overlap_pct:.0f}% of top attention dims match signal")
    elif overlap_pct > 40:
        print(f"  ≈ Moderate alignment: {overlap_pct:.0f}% overlap with signal")
    else:
        print(f"  ✗ Weak alignment: only {overlap_pct:.0f}% overlap with signal")
        print(f"    → May need longer training or higher denoise_lambda")
    
    if phase1_data and correlation > 0.5:
        print(f"  ✓ Attention learned to match gradient importance (r={correlation:.3f})")
    
    decisive_attn = ((avg_attention < 0.2) | (avg_attention > 0.8)).sum()
    print(f"  Decisiveness: {decisive_attn}/1536 dims have strong attention (< 0.2 or > 0.8)")


def main():
    parser = argparse.ArgumentParser(description='Analyze learned denoising behavior')
    parser.add_argument('--model', type=str, default='mult_vae_mddm_denoised',
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['amazon', 'yelp', 'steam'],
                       help='Dataset')
    
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model, args.dataset)
    
    # Load Phase 1 analysis if available
    phase1_data = load_phase1_analysis(args.dataset)
    
    # Analyze
    analyze_denoising(model, args.dataset, phase1_data)


if __name__ == '__main__':
    main()
