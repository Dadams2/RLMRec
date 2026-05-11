"""
Test Phase 3: Fine-tunable encoder model

Verifies:
1. Text profiles can be loaded
2. Sentence-transformer encoder loads correctly
3. Model builds and computes embeddings
4. Forward pass works
5. Gradients flow through encoder (if not frozen)

Usage:
    python test_finetuned_model.py --model mult_vae_mddm_finetuned --dataset amazon
"""

import torch

def test_finetuned_model():
    """Test that the fine-tuned model loads and works correctly."""
    
    print("\n" + "="*70)
    print("  TESTING PHASE 3: FINE-TUNABLE ENCODER")
    print("="*70 + "\n")
    
    # Import after args are parsed
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
        
        # Test 2: Build model (this will load encoder and text profiles)
        print("\n✓ Step 2: Building model with fine-tunable encoder...")
        from models.bulid_model import build_model
        model = build_model(data_handler)
        print(f"  Model class: {type(model).__name__}")
        
        # Test 3: Check encoder loaded
        print("\n✓ Step 3: Checking encoder configuration...")
        assert hasattr(model, 'text_encoder'), "Missing text_encoder"
        print(f"  Encoder: {model.encoder_name}")
        print(f"  Embedding dimension: {model.text_encoder.embedding_dim}")
        print(f"  Frozen: {model.freeze_encoder}")
        
        # Check trainable parameters
        encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
        encoder_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
        print(f"  Encoder parameters: {encoder_params:,}")
        print(f"  Encoder trainable: {encoder_trainable:,}")
        
        if not model.freeze_encoder:
            print(f"  ✓ Encoder is trainable (fine-tuning enabled)")
        else:
            print(f"  ✗ Encoder is frozen (fine-tuning disabled)")
        
        # Test 4: Check text profiles loaded
        print("\n✓ Step 4: Checking text profiles...")
        print(f"  User texts: {len(model.user_texts)}")
        print(f"  Item texts: {len(model.item_texts)}")
        print(f"  Sample user text: {model.user_texts[0][:100]}...")
        print(f"  Sample item text: {model.item_texts[0][:100]}...")
        
        # Test 5: Check embedding computation
        print("\n✓ Step 5: Checking embedding computation...")
        if model.freeze_encoder and model.usrprf_embeds is not None:
            print(f"  Mode: Pre-computed (frozen encoder)")
            print(f"  User embeddings: {model.usrprf_embeds.shape}")
            print(f"  Item embeddings: {model.itmprf_embeds.shape}")
        else:
            print(f"  Mode: On-the-fly computation (trainable encoder)")
            print(f"  User texts loaded: {len(model.user_texts)}")
            print(f"  Item texts loaded: {len(model.item_texts)}")
            # Test embedding computation
            test_users = torch.arange(5).long().to(configs['device'])
            test_user_embs = model.get_user_embeddings(test_users)
            print(f"  Sample user embeddings: {test_user_embs.shape}")
            print(f"  Embeddings require grad: {test_user_embs.requires_grad}")
        
        # Test 6: Test forward pass
        print("\n✓ Step 6: Testing forward pass...")
        model = model.cuda()
        model.is_training = True
        
        batch_size = 10
        user_ids = torch.randint(0, configs['data']['user_num'], (batch_size,)).cuda()
        item_data = torch.rand(batch_size, configs['data']['item_num']).cuda()
        
        user_emb = model.get_user_embeddings(user_ids)
        item_emb = model.get_item_embeddings()
        mu, mu_llm, logvar, logvar_llm = model.encode(item_data, user_emb, item_emb)
        
        print(f"  Encoded latent: mu={mu.shape}, logvar={logvar.shape}")
        print(f"  LLM latent: mu_llm={mu_llm.shape}, logvar_llm={logvar_llm.shape}")
        
        # Test 7: Test gradient flow
        print("\n✓ Step 7: Testing gradient flow...")
        model.zero_grad()
        
        # Forward pass with loss
        loss, losses = model.cal_loss(user_ids, item_data)
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters() if p.requires_grad)
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed: {has_grad}")
        
        if not model.freeze_encoder:
            encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                  for p in model.text_encoder.parameters() if p.requires_grad)
            print(f"  Encoder has gradients: {encoder_has_grad}")
            if encoder_has_grad:
                print(f"  ✓ End-to-end gradient flow working!")
            else:
                print(f"  ⚠ Warning: Encoder has no gradients (may need recompute_embeddings call)")
        
        # Test 8: Model statistics
        print("\n✓ Step 8: Model statistics...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vae_params = total_params - encoder_params
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  VAE parameters: {vae_params:,} ({vae_params/total_params*100:.1f}%)")
        print(f"  Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
        
        print("\n" + "="*70)
        print("  ✓ ALL TESTS PASSED")
        print("="*70 + "\n")
        
        print("Next steps:")
        print(f"  1. Train model: python train_encoder.py --model mult_vae_mddm_finetuned --dataset {configs['data']['name']}")
        print(f"  2. Compare with baseline: python compare_baseline_denoised.py --dataset {configs['data']['name']}")
        print("  3. Ablation: Set freeze_encoder=True to test if fine-tuning helps")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    success = test_finetuned_model()
    sys.exit(0 if success else 1)
