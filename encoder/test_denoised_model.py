"""
Test script to verify mult_vae_mddm_denoised model loads correctly.

Usage:
    python test_denoised_model.py --model mult_vae_mddm_denoised --dataset amazon
"""

import sys
import torch

def test_model_loading():
    """Test that the new denoised model loads without errors."""
    
    print("\n" + "="*70)
    print("  TESTING DENOISED MODEL")
    print("="*70 + "\n")
    
    # Import after args are parsed (configs loads on import)
    from config.configurator import configs
    from data_utils.build_data_handler import build_data_handler
    from trainer.trainer import init_seed
    
    try:
        # Test 1: Build data handler
        print("✓ Step 1: Building data handler...")
        init_seed()
        data_handler = build_data_handler()
        data_handler.load_data()
        print(f"  Data loaded: {configs['data']['user_num']} users, {configs['data']['item_num']} items")
        
        # Test 2: Build model
        print("\n✓ Step 2: Building model...")
        from models.bulid_model import build_model
        model = build_model(data_handler)
        print(f"  Model class: {type(model).__name__}")
        
        # Test 3: Check denoisers exist
        print("\n✓ Step 3: Checking denoising components...")
        assert hasattr(model, 'user_denoiser'), "Missing user_denoiser"
        assert hasattr(model, 'item_denoiser'), "Missing item_denoiser"
        print(f"  User denoiser: {model.user_denoiser.__class__.__name__}")
        print(f"  Item denoiser: {model.item_denoiser.__class__.__name__}")
        
        # Test 4: Check consensus dimensions loaded
        print("\n✓ Step 4: Checking consensus dimensions...")
        print(f"  User signal dims: {len(model.user_signal_dims)} dims")
        print(f"  User noise dims: {len(model.user_noise_dims)} dims")
        print(f"  Item signal dims: {len(model.item_signal_dims)} dims")
        print(f"  Item noise dims: {len(model.item_noise_dims)} dims")
        
        # Test 5: Test forward pass with dummy data
        print("\n✓ Step 5: Testing forward pass...")
        model = model.cuda()
        model.eval()
        
        # Create dummy batch
        batch_size = 10
        user_ids = torch.randint(0, configs['data']['user_num'], (batch_size,)).cuda()
        item_data = torch.rand(batch_size, configs['data']['item_num']).cuda()
        
        # Test denoising
        model.denoise_embeddings()
        print(f"  Denoised user embeddings: {model.usrprf_embeds.shape}")
        print(f"  Denoised item embeddings: {model.itmprf_embeds.shape}")
        
        # Test encoding
        with torch.no_grad():
            user_emb = model.usrprf_embeds[user_ids]
            mu, mu_llm, logvar, logvar_llm = model.encode(item_data, user_emb)
        
        print(f"  Encoded latent: mu={mu.shape}, logvar={logvar.shape}")
        
        # Test 6: Check parameter count
        print("\n✓ Step 6: Model statistics...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        denoiser_params = (sum(p.numel() for p in model.user_denoiser.parameters()) +
                          sum(p.numel() for p in model.item_denoiser.parameters()))
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Denoiser parameters: {denoiser_params:,} ({denoiser_params/total_params*100:.1f}%)")
        
        print("\n" + "="*70)
        print("  ✓ ALL TESTS PASSED")
        print("="*70 + "\n")
        
        print("Next steps:")
        print(f"  1. Train model: python train_encoder.py --model mult_vae_mddm_denoised --dataset {configs['data']['name']}")
        print(f"  2. Compare with baseline: python train_encoder.py --model mult_vae_mddm --dataset {configs['data']['name']}")
        print("  3. Analyze denoising: Check attention weights and noise estimates")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
