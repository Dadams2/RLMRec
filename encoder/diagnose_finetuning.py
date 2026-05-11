"""
Diagnose why fine-tuning isn't working.
"""
import torch
from models.general_cf.mult_vae_mddm_finetuned import mult_vae_MDDM_FineTuned
from data_utils.build_data_handler import build_data_handler
from config.configurator import configs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mult_vae_mddm_finetuned')
parser.add_argument('--dataset', type=str, default='amazon')
args = parser.parse_args()

configs['model']['name'] = args.model
configs['data']['name'] = args.dataset

print("="*70)
print("DIAGNOSING FINE-TUNING ISSUES")
print("="*70)

# Build model
data_handler = build_data_handler()
model = mult_vae_MDDM_FineTuned(data_handler).to(configs['device'])

print("\n1. EMBEDDING DIMENSIONS:")
print(f"   User embeddings: {model.usrprf_embeds.shape}")
print(f"   Item embeddings: {model.itmprf_embeds.shape}")
print(f"   Baseline uses: (11000, 1536) and (9332, 1536)")
print(f"   → Issue: 384-dim vs 1536-dim (4x smaller!)")

print("\n2. GRADIENT FLOW:")
print(f"   user_embeds.requires_grad: {model.usrprf_embeds.requires_grad}")
print(f"   item_embeds.requires_grad: {model.itmprf_embeds.requires_grad}")
print(f"   Encoder frozen: {model.freeze_encoder}")

# Test gradient flow
print("\n3. TESTING GRADIENT COMPUTATION:")
users = torch.arange(10).long().to(configs['device'])
batch_data = torch.randn(10, model.item_num).to(configs['device'])

# Forward pass
loss, _ = model.cal_loss(users, batch_data)
print(f"   Loss: {loss.item():.4f}")

# Backward pass
model.zero_grad()
loss.backward()

# Check gradients
encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.text_encoder.parameters())
mlp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.mlp.parameters())
vae_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.q_layers.parameters())

print(f"   Encoder has gradients: {encoder_has_grad}")
print(f"   MLP has gradients: {mlp_has_grad}")
print(f"   VAE has gradients: {vae_has_grad}")

if not encoder_has_grad:
    print(f"   → Issue: Gradients not reaching encoder!")
    print(f"   → Embeddings are cached/detached from computation graph")

print("\n4. RECOMPUTATION FREQUENCY:")
print(f"   Recompute every: {model.recompute_embeddings_every} batches")
print(f"   → Issue: 80% of training uses frozen embeddings")

print("\n5. MODEL ARCHITECTURE:")
print(f"   Encoder: {model.encoder_name}")
print(f"   Encoder params: {sum(p.numel() for p in model.text_encoder.parameters()):,}")
print(f"   MLP input dim: {model.text_encoder.embedding_dim}")
print(f"   Baseline MLP input: 1536")
print(f"   → Issue: MLP tuned for 1536-dim, now gets 384-dim")

print("\n" + "="*70)
print("RECOMMENDED FIXES:")
print("="*70)
print("1. Use larger encoder: all-mpnet-base-v2 (768-dim) or text-embedding-3-small (1536-dim)")
print("2. Fix gradient flow: embeddings need to stay in computation graph")
print("3. Recompute every batch (or use embedding layer instead of caching)")
print("4. Tune MLP dimensions for new embedding size")
print("5. Use separate learning rates: encoder (1e-5), VAE (5e-4)")
