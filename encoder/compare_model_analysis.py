"""
Compare semantic noise analysis results across different models.

Usage:
    python compare_model_analysis.py --dataset yelp
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_analysis_results(dataset, models=['mult_vae_mddm', 'mult_vae_godm', 'mult_vae_cpdm']):
    """
    Load analysis results for all models.
    """
    results = {}
    base_dir = Path(f'./analysis_results/{dataset}')
    
    for model in models:
        json_file = base_dir / model / f'{model}_{dataset}_analysis.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                results[model] = json.load(f)
            print(f"✓ Loaded results for {model}")
        else:
            print(f"✗ No results found for {model} at {json_file}")
    
    return results


def compare_gradient_importance(results, dataset):
    """
    Compare which dimensions are important across models.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # User dimensions
    for model_name, data in results.items():
        if data['gradient_importance']['user_top_20_dims'] is not None:
            dims = data['gradient_importance']['user_top_20_dims']
            values = data['gradient_importance']['user_top_20_values']
            axes[0].scatter(dims, values, label=model_name, alpha=0.6, s=100)
    
    axes[0].set_title(f'Top 20 User Dimensions Across Models ({dataset})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension Index')
    axes[0].set_ylabel('Gradient Importance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Item dimensions
    for model_name, data in results.items():
        if data['gradient_importance']['item_top_20_dims'] is not None:
            dims = data['gradient_importance']['item_top_20_dims']
            values = data['gradient_importance']['item_top_20_values']
            axes[1].scatter(dims, values, label=model_name, alpha=0.6, s=100)
    
    axes[1].set_title(f'Top 20 Item Dimensions Across Models ({dataset})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension Index')
    axes[1].set_ylabel('Gradient Importance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(f'./analysis_results/{dataset}/comparison_gradient_importance.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_file}")


def compare_dimension_overlap(results, dataset):
    """
    Analyze overlap in important dimensions across models.
    """
    # User dimensions
    user_important_dims = {}
    user_noise_dims = {}
    
    for model_name, data in results.items():
        if data['gradient_importance']['user_top_20_dims'] is not None:
            user_important_dims[model_name] = set(data['gradient_importance']['user_top_20_dims'][-10:])
        if data['noise_analysis']['user_top_noise_dims'] is not None:
            user_noise_dims[model_name] = set(data['noise_analysis']['user_top_noise_dims'][-10:])
    
    # Compute overlap
    print("\n" + "="*70)
    print("DIMENSION OVERLAP ANALYSIS")
    print("="*70)
    
    print("\n--- User Important Dimensions (Top 10) ---")
    models = list(user_important_dims.keys())
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            overlap = user_important_dims[model1] & user_important_dims[model2]
            overlap_pct = len(overlap) / 10 * 100
            print(f"{model1} ∩ {model2}: {len(overlap)}/10 dims ({overlap_pct:.0f}%)")
            if overlap:
                print(f"  Shared dims: {sorted(overlap)}")
    
    print("\n--- User Noise Dimensions (Top 10) ---")
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            if model1 in user_noise_dims and model2 in user_noise_dims:
                overlap = user_noise_dims[model1] & user_noise_dims[model2]
                overlap_pct = len(overlap) / 10 * 100
                print(f"{model1} ∩ {model2}: {len(overlap)}/10 dims ({overlap_pct:.0f}%)")
                if overlap:
                    print(f"  Shared noisy dims: {sorted(overlap)}")
    
    # Item dimensions
    item_important_dims = {}
    item_noise_dims = {}
    
    for model_name, data in results.items():
        if data['gradient_importance']['item_top_20_dims'] is not None:
            item_important_dims[model_name] = set(data['gradient_importance']['item_top_20_dims'][-10:])
        if data['noise_analysis']['item_top_noise_dims'] is not None:
            item_noise_dims[model_name] = set(data['noise_analysis']['item_top_noise_dims'][-10:])
    
    print("\n--- Item Important Dimensions (Top 10) ---")
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            overlap = item_important_dims[model1] & item_important_dims[model2]
            overlap_pct = len(overlap) / 10 * 100
            print(f"{model1} ∩ {model2}: {len(overlap)}/10 dims ({overlap_pct:.0f}%)")
            if overlap:
                print(f"  Shared dims: {sorted(overlap)}")
    
    print("\n--- Item Noise Dimensions (Top 10) ---")
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            if model1 in item_noise_dims and model2 in item_noise_dims:
                overlap = item_noise_dims[model1] & item_noise_dims[model2]
                overlap_pct = len(overlap) / 10 * 100
                print(f"{model1} ∩ {model2}: {len(overlap)}/10 dims ({overlap_pct:.0f}%)")
                if overlap:
                    print(f"  Shared noisy dims: {sorted(overlap)}")


def compare_statistics(results, dataset):
    """
    Compare aggregate statistics across models.
    """
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70 + "\n")
    
    for model_name, data in results.items():
        print(f"--- {model_name} ---")
        stats = data['statistics']
        print(f"  User embedding dim: {stats['user_embedding_dim']}")
        print(f"  Item embedding dim: {stats['item_embedding_dim']}")
        print(f"  Avg user gradient: {stats['user_avg_gradient']:.6f}")
        print(f"  Avg item gradient: {stats['item_avg_gradient']:.6f}")
        print(f"  User gradient sparsity: {stats['user_gradient_sparsity']:.2%}")
        print()


def generate_recommendations(results, dataset):
    """
    Generate recommendations for Phase 2 implementation.
    """
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR PHASE 2")
    print("="*70 + "\n")
    
    # Find consensus dimensions
    user_signal_sets = []
    user_noise_sets = []
    
    for model_name, data in results.items():
        if data['noise_analysis']['user_top_signal_dims']:
            user_signal_sets.append(set(data['noise_analysis']['user_top_signal_dims'][-20:]))
        if data['noise_analysis']['user_top_noise_dims']:
            user_noise_sets.append(set(data['noise_analysis']['user_top_noise_dims'][-20:]))
    
    if len(user_signal_sets) > 1:
        # Find dimensions that all models agree are important
        consensus_signal = set.intersection(*user_signal_sets)
        consensus_noise = set.intersection(*user_noise_sets) if user_noise_sets else set()
        
        print("✓ Consensus Signal Dimensions (all models agree):")
        print(f"  User: {sorted(consensus_signal) if consensus_signal else 'None'}")
        print(f"  Count: {len(consensus_signal)}/20")
        
        print("\n✓ Consensus Noise Dimensions (all models agree):")
        print(f"  User: {sorted(consensus_noise) if consensus_noise else 'None'}")
        print(f"  Count: {len(consensus_noise)}/20")
        
        print("\n✓ Implementation Strategy:")
        if len(consensus_noise) > 5:
            print(f"  1. Strong noise signal detected ({len(consensus_noise)} dims)")
            print(f"     → Implement selective dimension dropout")
            print(f"     → Add learnable denoising layer")
        else:
            print(f"  1. Weak noise signal ({len(consensus_noise)} dims)")
            print(f"     → Implement attention-based selection")
            print(f"     → Focus on amplifying signal dimensions")
        
        if len(consensus_signal) > 10:
            print(f"  2. Strong signal consensus ({len(consensus_signal)} dims)")
            print(f"     → Can safely compress to these dimensions")
            print(f"     → Use bottleneck architecture")
        else:
            print(f"  2. Distributed signal pattern")
            print(f"     → Keep full dimensionality")
            print(f"     → Use soft attention weighting")
    
    print("\n✓ Next Steps:")
    print("  1. Implement learnable noise estimator targeting consensus noise dims")
    print("  2. Add dimension-wise attention weights")
    print("  3. Test with frozen vs fine-tuned configurations")
    print("  4. Validate on held-out set before/after denoising")


def main():
    parser = argparse.ArgumentParser(description='Compare semantic noise analysis across models')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['amazon', 'yelp', 'steam'],
                       help='Dataset to analyze')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"  CROSS-MODEL COMPARISON: {args.dataset}")
    print("="*70 + "\n")
    
    # Load results
    results = load_analysis_results(args.dataset)
    
    if len(results) == 0:
        print("ERROR: No analysis results found.")
        print(f"Please run analysis first: python run_semantic_analysis.py --model <model> --dataset {args.dataset}")
        return
    
    # Generate comparisons
    compare_gradient_importance(results, args.dataset)
    compare_dimension_overlap(results, args.dataset)
    compare_statistics(results, args.dataset)
    generate_recommendations(results, args.dataset)
    
    print("\n" + "="*70)
    print("  COMPARISON COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
