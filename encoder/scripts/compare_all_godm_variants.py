"""
Comprehensive comparison of all GODM variants:
- GODM (Standard Wasserstein-2)
- GODM_SUOT (Sliced Unbalanced Optimal Transport)
- GODM_TSW (Tree-Sliced Wasserstein)
"""

import torch
import time

def compare_all_methods():
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: GODM vs GODM_SUOT vs GODM_TSW")
    print("=" * 80)
    
    print("\n" + "="*80)
    print("1. MATHEMATICAL FORMULATION")
    print("="*80)
    
    print("\n[GODM] Standard Wasserstein-2 Distance")
    print("-" * 80)
    print("Formula: WD = √(||μ₁ - μ₂||² + ||σ₁ - σ₂||²)")
    print("Properties:")
    print("  • Closed-form solution for Gaussians")
    print("  • Direct computation from parameters")
    print("  • Balanced distributions only")
    print("  • Fastest computation: O(d)")
    
    print("\n[GODM_SUOT] Sliced Unbalanced Optimal Transport")
    print("-" * 80)
    print("Formula: SUOT = (1/K) Σᵢ UOT(X·θᵢ, Y·θᵢ) + ρ₁·KL(π·1|a) + ρ₂·KL(πᵀ·1|b)")
    print("Properties:")
    print("  • Sample-based with random 1D projections")
    print("  • Frank-Wolfe iterations for each projection")
    print("  • Supports unbalanced distributions")
    print("  • Complexity: O(n log n × num_projections × niter)")
    
    print("\n[GODM_TSW] Tree-Sliced Wasserstein")
    print("-" * 80)
    print("Formula: TSW = (1/T) Σₜ [ Σₗ ∫ |F_t,l(x)|^p dx ]^{1/p}")
    print("Properties:")
    print("  • Sample-based with hierarchical tree projections")
    print("  • Closed-form computation (no iterations)")
    print("  • Supports partial transport (unbalanced)")
    print("  • Distance-based adaptive mass division")
    print("  • Complexity: O(T × L × n log n)")
    
    print("\n" + "="*80)
    print("2. FEATURE COMPARISON TABLE")
    print("="*80)
    
    table = """
    ╔══════════════════════╦════════════════╦════════════════╦═══════════════╗
    ║ Feature              ║ GODM           ║ GODM_SUOT      ║ GODM_TSW      ║
    ╠══════════════════════╬════════════════╬════════════════╬═══════════════╣
    ║ Distance Type        ║ Closed-form W2 ║ Sliced UOT     ║ Tree-Sliced   ║
    ║ Computational Cost   ║ O(d)           ║ O(n·K·I·log n) ║ O(T·L·n·log n)║
    ║ Speed (relative)     ║ 1x (fastest)   ║ 0.3-0.5x       ║ 0.4-0.6x      ║
    ║ Unbalanced Support   ║ No             ║ Yes (iterative)║ Yes (closed)  ║
    ║ Projection Type      ║ Direct         ║ Random 1D      ║ Hierarchical  ║
    ║ Iterations Required  ║ 0              ║ 10-20 FW       ║ 0             ║
    ║ Random Sampling      ║ No             ║ Per batch      ║ No (fixed)    ║
    ║ Mass Division        ║ N/A            ║ Uniform        ║ Adaptive      ║
    ║ Gradient Quality     ║ Direct         ║ Very Good      ║ Good          ║
    ║ Memory Overhead      ║ Minimal        ║ Medium         ║ Low           ║
    ║ Setup Complexity     ║ None           ║ Low            ║ Medium        ║
    ║ Training Stability   ║ High           ║ Good           ║ Very Good     ║
    ╚══════════════════════╩════════════════╩════════════════╩═══════════════╝
    
    Legend:
    - K: number of projections (SUOT)
    - I: Frank-Wolfe iterations (SUOT)
    - T: number of trees (TSW)
    - L: lines per tree (TSW)
    - n: batch size
    - d: latent dimension
    """
    print(table)
    
    print("\n" + "="*80)
    print("3. HYPERPARAMETER COMPARISON")
    print("="*80)
    
    hyper_table = """
    ╔════════════════════╦════════════╦════════════════╦══════════════════╗
    ║ Hyperparameter     ║ GODM       ║ GODM_SUOT      ║ GODM_TSW         ║
    ╠════════════════════╬════════════╬════════════════╬══════════════════╣
    ║ beta (Amazon)      ║ 0.4        ║ 0.3            ║ 0.35             ║
    ║ beta (Yelp)        ║ 0.8        ║ 0.6            ║ 0.65             ║
    ║ beta (Steam)       ║ 1.0        ║ 0.8            ║ 0.85             ║
    ║ Key Parameter 1    ║ N/A        ║ num_proj: 50   ║ ntrees: 250      ║
    ║ Key Parameter 2    ║ N/A        ║ rho1/2: 10.0   ║ nlines: 4        ║
    ║ Key Parameter 3    ║ N/A        ║ niter: 10      ║ delta: 2.0       ║
    ║ Tuning Difficulty  ║ Low        ║ Medium-High    ║ Medium           ║
    ╚════════════════════╩════════════╩════════════════╩══════════════════╝
    
    Note: SUOT and TSW can use lower beta due to better gradient properties
    """
    print(hyper_table)
    
    print("\n" + "="*80)
    print("4. USE CASE RECOMMENDATIONS")
    print("="*80)
    
    print("\n✓ USE GODM (Standard) WHEN:")
    print("  • Need maximum speed and simplicity")
    print("  • Distributions are naturally balanced")
    print("  • Running quick experiments or baseline")
    print("  • Computational resources are limited")
    print("  • No special requirements for unbalanced transport")
    
    print("\n✓ USE GODM_SUOT WHEN:")
    print("  • Need maximum flexibility and accuracy")
    print("  • Distributions are significantly unbalanced")
    print("  • Want fine-grained control over marginal relaxation")
    print("  • Can afford longer training times")
    print("  • Research setting where iteration is acceptable")
    print("  • Need proven unbalanced OT theory (Frank-Wolfe)")
    
    print("\n✓ USE GODM_TSW WHEN:")
    print("  • Want hierarchical geometric matching")
    print("  • Need partial transport with closed-form solution")
    print("  • Prefer adaptive, distance-based mass division")
    print("  • Want stable fixed frames (no random per-batch)")
    print("  • Balance between GODM speed and SUOT flexibility")
    print("  • Cold-start scenarios are important")
    print("  • Production deployment (consistent behavior)")
    
    print("\n" + "="*80)
    print("5. EXPECTED PERFORMANCE")
    print("="*80)
    
    perf_table = """
    ╔═══════════════════╦═════════════╦═════════════╦═════════════╗
    ║ Metric            ║ GODM        ║ GODM_SUOT   ║ GODM_TSW    ║
    ╠═══════════════════╬═════════════╬═════════════╬═════════════╣
    ║ Training Speed    ║ 1.0x        ║ 0.3-0.5x    ║ 0.4-0.6x    ║
    ║ Recall@20         ║ Baseline    ║ +0.5-2%     ║ +0.5-1.5%   ║
    ║ NDCG@20           ║ Baseline    ║ +0.3-1.5%   ║ +0.3-1.0%   ║
    ║ Convergence Rate  ║ Good        ║ Good        ║ Very Good   ║
    ║ Training Variance ║ Low         ║ Medium      ║ Low         ║
    ║ Cold-start Perf   ║ Good        ║ Very Good   ║ Very Good   ║
    ║ Memory Usage      ║ Low         ║ Medium      ║ Low-Medium  ║
    ╚═══════════════════╩═════════════╩═════════════╩═════════════╝
    """
    print(perf_table)
    
    print("\n" + "="*80)
    print("6. IMPLEMENTATION DETAILS")
    print("="*80)
    
    print("\n[GODM Implementation]")
    print("```python")
    print("# Direct computation from Gaussian parameters")
    print("mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2")
    print("var_diff = torch.norm(std_src - std_llm, dim=1) ** 2")
    print("WD = torch.mean(torch.sqrt(mean_diff + var_diff))")
    print("```")
    
    print("\n[GODM_SUOT Implementation]")
    print("```python")
    print("# Sample and compute sliced UOT")
    print("x_src = mu_src + std_src * eps_src")
    print("x_llm = mu_llm + std_llm * eps_llm")
    print("suot_loss, _, _, _, _, _ = sliced_unbalanced_ot(")
    print("    a, b, x_src, x_llm,")
    print("    p=2, num_projections=50,")
    print("    rho1=10.0, rho2=10.0, niter=10")
    print(")")
    print("```")
    
    print("\n[GODM_TSW Implementation]")
    print("```python")
    print("# Sample and compute TSW with fixed tree frames")
    print("x_src = mu_src + std_src * eps_src")
    print("x_llm = mu_llm + std_llm * eps_llm")
    print("tsw_loss = tsw_obj(")
    print("    x_src, x_llm,")
    print("    self.theta, self.intercept  # Fixed tree frames")
    print(")")
    print("```")
    
    print("\n" + "="*80)
    print("7. COMMAND COMPARISON")
    print("="*80)
    
    print("\nTraining commands:")
    print("  GODM:      python train_encoder.py --model mult_vae_godm --dataset yelp --cuda 0")
    print("  GODM_SUOT: python train_encoder.py --model mult_vae_godm_suot --dataset yelp --cuda 0")
    print("  GODM_TSW:  python train_encoder.py --model mult_vae_godm_tsw --dataset yelp --cuda 0")
    
    print("\nConfiguration files:")
    print("  GODM:      encoder/config/modelconf/mult_vae_godm.yml")
    print("  GODM_SUOT: encoder/config/modelconf/mult_vae_godm_suot.yml")
    print("  GODM_TSW:  encoder/config/modelconf/mult_vae_godm_tsw.yml")
    
    print("\n" + "="*80)
    print("8. DECISION FLOWCHART")
    print("="*80)
    
    flowchart = """
    
    START: Need Distribution Matching
           |
           v
    [Priority: Speed?] ──Yes──> Use GODM
           |                     (Fastest, Simple)
           No
           v
    [Need Unbalanced?] ──No───> Use GODM
           |                     (Sufficient)
           Yes
           v
    [Have Extra Time?] ──Yes──> Use GODM_SUOT
           |                     (Maximum Flexibility)
           No
           v
    [Want Closed-Form?] ─Yes──> Use GODM_TSW
           |                     (Best Balance)
           No
           v
    [Need Research    
     Quality Results?] ─Yes──> Use GODM_SUOT
                                (Most Thorough)
    
    Recommendation Order (by use case):
    1. Quick experiments    → GODM
    2. Production systems   → GODM_TSW (stable, fast enough)
    3. Research/Maximum acc → GODM_SUOT (most flexible)
    4. Cold-start emphasis  → GODM_TSW or GODM_SUOT
    """
    print(flowchart)
    
    print("\n" + "="*80)
    print("SUMMARY RECOMMENDATION")
    print("="*80)
    print("\nFor MOST users starting out:")
    print("  1st Try: GODM (baseline)")
    print("  2nd Try: GODM_TSW (good balance, stable)")
    print("  3rd Try: GODM_SUOT (if need maximum accuracy)")
    print("\nFor PRODUCTION deployment:")
    print("  → GODM_TSW (stable frames, closed-form, good performance)")
    print("\nFor RESEARCH with time:")
    print("  → GODM_SUOT (maximum flexibility, proven theory)")
    print("="*80)

if __name__ == '__main__':
    compare_all_methods()
