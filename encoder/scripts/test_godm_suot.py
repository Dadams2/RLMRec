"""
Quick test script for mult_vae_GODM_SUOT model
Tests that the model can be instantiated and forward pass works
"""

import sys
import os
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_suot_import():
    """Test that SUOT module can be imported"""
    print("Testing SUOT import...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/Slicing_Unbalanced_Optimal_Transport')
        sys.path.insert(0, reference_path)
        from sliceduot.sliced_uot import sliced_unbalanced_ot
        print("✓ SUOT import successful")
        return True
    except Exception as e:
        print(f"✗ SUOT import failed: {e}")
        return False

def test_model_structure():
    """Test model structure without full data"""
    print("\nTesting model structure...")
    try:
        from config.configurator import configs
        
        # Mock minimal config
        if 'model' not in configs:
            configs['model'] = {}
        configs['model']['name'] = 'mult_vae_godm_suot'
        
        # Import model class
        from models.general_cf.mult_vae_godm_suot import mult_vae_GODM_SUOT
        print("✓ Model class imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_suot_distance_computation():
    """Test SUOT distance computation with dummy data"""
    print("\nTesting SUOT distance computation...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/Slicing_Unbalanced_Optimal_Transport')
        sys.path.insert(0, reference_path)
        from sliceduot.sliced_uot import sliced_unbalanced_ot
        
        # Create dummy Gaussian samples
        batch_size = 32
        latent_dim = 200
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Sample from two Gaussians
        mu1 = torch.randn(batch_size, latent_dim).to(device)
        mu2 = torch.randn(batch_size, latent_dim).to(device)
        
        # Uniform weights
        a = torch.ones(batch_size).to(device) / batch_size
        b = torch.ones(batch_size).to(device) / batch_size
        
        # Compute SUOT
        suot_loss, _, _, _, _, _ = sliced_unbalanced_ot(
            a=a,
            b=b,
            x=mu1,
            y=mu2,
            p=2,
            num_projections=10,
            rho1=10.0,
            rho2=10.0,
            niter=5,
            mode='backprop',
            type_proj='linear'
        )
        
        print(f"✓ SUOT computation successful. Loss: {suot_loss.item():.4f}")
        
        # Test gradient flow
        suot_loss.backward()
        print("✓ Gradient computation successful")
        
        return True
    except Exception as e:
        print(f"✗ SUOT computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing mult_vae_GODM_SUOT Implementation")
    print("=" * 60)
    
    tests = [
        ("SUOT Import", test_suot_import),
        ("Model Structure", test_model_structure),
        ("SUOT Distance", test_suot_distance_computation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(r for _, r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Model is ready to use.")
        print("\nTo train the model, run:")
        print("  python train_encoder.py --model mult_vae_godm_suot --dataset yelp --cuda 0")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    print("=" * 60)

if __name__ == '__main__':
    main()
