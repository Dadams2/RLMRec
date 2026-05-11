"""
Generate visual summary of Phase 1 + Phase 2 pipeline.

Creates a diagram showing:
1. Phase 1: Semantic noise analysis
2. Phase 2: Learnable denoising implementation
3. Complete end-to-end training flow

Usage:
    python generate_pipeline_diagram.py --dataset amazon
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import argparse
from pathlib import Path


def create_pipeline_diagram(dataset):
    """Create visual diagram of complete pipeline."""
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'End-to-End LLM-Enhanced Recommendation: Phase 1 + Phase 2',
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Phase 1 Box
    phase1_box = FancyBboxPatch((0.2, 8.5), 4.6, 2.5, boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(phase1_box)
    ax.text(2.5, 10.7, 'PHASE 1: Semantic Noise Analysis', ha='center', fontsize=14, fontweight='bold')
    
    phase1_text = (
        "Goal: Identify signal vs noise in LLM embeddings\n\n"
        "Method: Gradient-based importance scoring\n"
        "• Compute ∂Loss/∂embedding for each dimension\n"
        "• Measure activation variance\n"
        "• noise_score = variance × (1 - gradient)\n\n"
        "Result (Amazon):\n"
        "✓ 20 consensus signal dims [57, 89, 310, ...]\n"
        "✓ 20 consensus noise dims [0, 137, 194, ...]\n"
        "✓ 100% agreement across 3 models"
    )
    ax.text(2.5, 9.3, phase1_text, ha='center', va='center', fontsize=9,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow from Phase 1 to Phase 2
    arrow1 = FancyArrowPatch((2.5, 8.3), (2.5, 7.7),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='darkgreen')
    ax.add_patch(arrow1)
    ax.text(3.2, 8.0, 'Consensus\nDimensions', fontsize=9, color='darkgreen', fontweight='bold')
    
    # Phase 2 Box
    phase2_box = FancyBboxPatch((0.2, 4.5), 4.6, 3.0, boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(phase2_box)
    ax.text(2.5, 7.2, 'PHASE 2: Learnable Denoising', ha='center', fontsize=14, fontweight='bold')
    
    phase2_text = (
        "Model: mult_vae_mddm_denoised\n\n"
        "Architecture: SemanticDenoiser module\n"
        "├─ Noise Estimator: Learns what to remove\n"
        "├─ Attention Network: Learns importance [0,1]\n"
        "└─ Refinement: Further improves embeddings\n\n"
        "Training:\n"
        "• Loss = Rec + KL + λ×Denoise (λ=0.01)\n"
        "• Gradients flow from rec loss → denoising\n"
        "• Initialized with Phase 1 consensus dims\n\n"
        "Expected: 5-10% improvement in recall@20"
    )
    ax.text(2.5, 5.7, phase2_text, ha='center', va='center', fontsize=9,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Complete Pipeline on Right
    pipeline_box = FancyBboxPatch((5.2, 4.5), 4.6, 6.5, boxstyle="round,pad=0.1",
                                  edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(pipeline_box)
    ax.text(7.5, 10.7, 'End-to-End Training Flow', ha='center', fontsize=14, fontweight='bold')
    
    # Pipeline components
    components = [
        (10.3, "Raw LLM Embeddings\n(text-embedding-ada-002, 1536-dim)\nUser & Item profiles"),
        (9.6, "↓"),
        (9.2, "SemanticDenoiser\nNoise Estimator + Attention + Refinement"),
        (8.7, "↓"),
        (8.3, "Denoised Embeddings (1536-dim)\nSignal amplified, noise suppressed"),
        (7.8, "↓"),
        (7.4, "VAE-MDDM (unchanged)\nEncoder: [11010, 600, 200]\nDecoder: [200, 600, 11010]"),
        (6.9, "↓"),
        (6.5, "Latent Code z (200-dim)\nMixed CF + LLM distributions"),
        (6.0, "↓"),
        (5.6, "Reconstructed Interactions\nPredicted user-item affinities"),
        (5.1, "↓"),
        (4.7, "Loss Computation\nRec + KL + Denoise")
    ]
    
    for y, text in components:
        if "↓" in text:
            ax.text(7.5, y, text, ha='center', va='center', fontsize=14, color='purple', fontweight='bold')
        else:
            box_color = 'yellow' if 'Denoiser' in text else 'white'
            ax.text(7.5, y, text, ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
    
    # Gradient flow annotation
    gradient_arrow = FancyArrowPatch((7.5, 4.7), (7.5, 9.2),
                                    arrowstyle='<-', mutation_scale=20, 
                                    linewidth=2, color='red', linestyle='--')
    ax.add_patch(gradient_arrow)
    ax.text(8.7, 7.0, 'Gradients\nBackprop', fontsize=9, color='red', 
           fontweight='bold', rotation=-90, va='center')
    
    # Evaluation Box
    eval_box = FancyBboxPatch((0.2, 0.5), 9.6, 3.5, boxstyle="round,pad=0.1",
                              edgecolor='orange', facecolor='wheat', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(5, 3.7, 'EVALUATION & ANALYSIS', ha='center', fontsize=14, fontweight='bold')
    
    # Three columns of evaluation
    eval_col1 = (
        "Test Model Loading\n"
        "━━━━━━━━━━━━━━━━━\n"
        "✓ Import checks\n"
        "✓ Data handler\n"
        "✓ Model builds\n"
        "✓ Forward pass\n"
        "✓ Parameter count\n\n"
        "test_denoised_model.py"
    )
    ax.text(1.8, 1.8, eval_col1, ha='center', va='center', fontsize=8,
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    eval_col2 = (
        "Compare Performance\n"
        "━━━━━━━━━━━━━━━━━\n"
        "• recall@5/10/15/20\n"
        "• ndcg@5/10/15/20\n"
        "• % improvement\n"
        "• Interpretation\n"
        "• Recommendations\n\n"
        "compare_baseline_denoised.py"
    )
    ax.text(5.0, 1.8, eval_col2, ha='center', va='center', fontsize=8,
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    eval_col3 = (
        "Analyze Learned Denoising\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• Attention weights\n"
        "• Noise estimates\n"
        "• Consensus overlap\n"
        "• Gradient correlation\n"
        "• Visualizations\n\n"
        "analyze_learned_denoising.py"
    )
    ax.text(8.2, 1.8, eval_col3, ha='center', va='center', fontsize=8,
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Bottom status
    status_text = (
        f"Status: Phase 2 COMPLETE ✓   |   Dataset: {dataset}   |   "
        "Ready for Training: python train_encoder.py --model mult_vae_mddm_denoised --dataset amazon"
    )
    ax.text(5, 0.2, status_text, ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('./analysis_results') / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'phase1_phase2_pipeline.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Pipeline diagram saved: {output_file}\n")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline diagram')
    parser.add_argument('--dataset', type=str, default='amazon',
                       choices=['amazon', 'yelp', 'steam'],
                       help='Dataset name for diagram')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  GENERATING PIPELINE DIAGRAM")
    print("="*70 + "\n")
    
    output_file = create_pipeline_diagram(args.dataset)
    
    print("Diagram includes:")
    print("  • Phase 1: Semantic noise analysis methodology")
    print("  • Phase 2: Learnable denoising architecture")
    print("  • Complete end-to-end training flow with gradient backprop")
    print("  • Evaluation and analysis tools")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
