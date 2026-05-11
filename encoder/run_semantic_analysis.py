"""
Runner script for semantic noise analysis.

Usage:
    python run_semantic_analysis.py --model mult_vae_mddm --dataset yelp
    python run_semantic_analysis.py --model mult_vae_godm --dataset amazon-book --batches 100
"""

import argparse
import sys
import torch
from pathlib import Path

# Add encoder to path
sys.path.append(str(Path(__file__).parent))

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.trainer import init_seed
from semantic_noise_analysis import SemanticNoiseAnalyzer


def load_trained_model(model_name, dataset, checkpoint_dir='./checkpoint'):
    """
    Load a trained model for analysis.
    
    Args:
        model_name: Name of the model (e.g., 'mult_vae_mddm')
        dataset: Dataset name (e.g., 'amazon')
        checkpoint_dir: Base checkpoint directory (default: './checkpoint')
    """
    print(f"Loading model: {model_name} for dataset: {dataset}")
    
    # Follow the exact setup from train_encoder.py
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    
    # Build model
    model = build_model(data_handler).to(configs['device'])
    
    # Try to load checkpoint from correct directory structure
    # Expected: checkpoint/{model_name}/{model_name}-{dataset}-*.pth
    model_checkpoint_dir = Path(checkpoint_dir) / model_name
    checkpoint_pattern = f'{model_name}-{dataset}*.pth'
    
    if model_checkpoint_dir.exists():
        checkpoint_files = list(model_checkpoint_dir.glob(checkpoint_pattern))
    else:
        checkpoint_files = []
        print(f"WARNING: Checkpoint directory not found: {model_checkpoint_dir}")
    
    if checkpoint_files:
        checkpoint_path = sorted(checkpoint_files)[-1]  # Get most recent
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Handle both direct state dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint (direct state dict)")
    else:
        print(f"WARNING: No checkpoint found matching pattern: {model_checkpoint_dir}/{checkpoint_pattern}")
        print("Results may not be meaningful. Please train the model first.")
    
    model.eval()
    return model, data_handler


def main():
    parser = argparse.ArgumentParser(description='Semantic Noise Analysis for VAE models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mult_vae_mddm', 'mult_vae_godm', 'mult_vae_cpdm'],
                       help='Model to analyze')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['amazon', 'yelp', 'steam'],
                       help='Dataset used')
    parser.add_argument('--batches', type=int, default=50,
                       help='Number of batches to analyze (default: 50)')
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                       help='Base checkpoint directory (default: ./checkpoint)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"  SEMANTIC NOISE ANALYSIS")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print("="*70 + "\n")
    
    # Load model
    model, data_handler = load_trained_model(args.model, args.dataset, args.checkpoint_dir)
    
    # Create analyzer
    save_dir = Path(args.save_dir) / args.dataset / args.model
    analyzer = SemanticNoiseAnalyzer(model, data_handler, save_dir=save_dir)
    
    # Run analysis
    results = analyzer.run_full_analysis(
        model_name=f"{args.model}_{args.dataset}",
        num_batches=args.batches
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"  Results saved to: {save_dir}")
    print(f"  View plots: {save_dir}/*.png")
    print(f"  View JSON: {save_dir}/*.json")
    print("\nNext steps:")
    print("  1. Examine the plots to identify noisy vs signal dimensions")
    print("  2. Check the JSON file for specific dimension indices")
    print("  3. Compare results across different models (MDDM vs GODM vs CPDM)")
    print("  4. Use insights to implement Phase 2 (learnable denoising)\n")


if __name__ == '__main__':
    main()
