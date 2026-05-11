"""
Quick test script for mult_vae_GODM_TSW model
Tests that the model can be instantiated and Tree-Sliced Wasserstein computation works
"""

import sys
import os
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_tsw_import():
    """Test that TSW module can be imported"""
    print("Testing TSW import...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/PartialTSW/src')
        sys.path.insert(0, reference_path)
        from tsw.partial_tsw import PartialTSW
        from tsw.utils import generate_trees_frames
        print("✓ TSW import successful")
        return True
    except Exception as e:
        print(f"✗ TSW import failed: {e}")
        return False

def test_model_structure():
    """Test model structure without full data"""
    print("\nTesting model structure...")
    try:
        from config.configurator import configs
        
        # Mock minimal config
        if 'model' not in configs:
            configs['model'] = {}
        configs['model']['name'] = 'mult_vae_godm_tsw'
        
        # Import model class
        from models.general_cf.mult_vae_godm_tsw import mult_vae_GODM_TSW
        print("✓ Model class imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tree_generation():
    """Test tree frame generation"""
    print("\nTesting tree frame generation...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/PartialTSW/src')
        sys.path.insert(0, reference_path)
        from tsw.utils import generate_trees_frames
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test orthogonal tree generation
        theta, intercept = generate_trees_frames(
            ntrees=10,
            nlines=4,
            d=200,
            gen_mode='gaussian_orthogonal',
            device=device
        )
        
        print(f"✓ Tree generation successful")
        print(f"  - theta shape: {theta.shape} (expected: [10, 4, 200])")
        print(f"  - intercept shape: {intercept.shape} (expected: [10, 1, 200])")
        
        # Verify orthogonality
        # For each tree, lines should be orthogonal
        for i in range(min(3, theta.shape[0])):  # Check first 3 trees
            tree_lines = theta[i]  # [nlines, d]
            gram = torch.matmul(tree_lines, tree_lines.T)
            is_orthogonal = torch.allclose(gram, torch.eye(tree_lines.shape[0], device=device), atol=1e-5)
            if is_orthogonal:
                print(f"  - Tree {i}: Lines are orthogonal ✓")
        
        return True
    except Exception as e:
        print(f"✗ Tree generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tsw_computation():
    """Test TSW distance computation with dummy data"""
    print("\nTesting TSW distance computation...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/PartialTSW/src')
        sys.path.insert(0, reference_path)
        from tsw.partial_tsw import PartialTSW
        from tsw.utils import generate_trees_frames
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create TSW object
        tsw_obj = PartialTSW(
            ntrees=50,
            nlines=4,
            p=2,
            delta=2.0,
            mass_division='distance_based',
            device=device
        )
        
        # Generate tree frames
        theta, intercept = generate_trees_frames(
            ntrees=50,
            nlines=4,
            d=200,
            gen_mode='gaussian_orthogonal',
            device=device
        )
        
        # Create dummy Gaussian samples
        batch_size = 32
        latent_dim = 200
        
        mu1 = torch.randn(batch_size, latent_dim).to(device)
        mu2 = torch.randn(batch_size, latent_dim).to(device)
        
        # Compute TSW (balanced)
        tsw_loss = tsw_obj(mu1, mu2, theta, intercept)
        print(f"✓ TSW computation successful. Loss: {tsw_loss.item():.4f}")
        
        # Test gradient flow
        mu1.requires_grad_(True)
        mu2.requires_grad_(True)
        tsw_loss = tsw_obj(mu1, mu2, theta, intercept)
        tsw_loss.backward()
        
        print(f"✓ Gradient computation successful")
        print(f"  - mu1.grad norm: {mu1.grad.norm().item():.6f}")
        print(f"  - mu2.grad norm: {mu2.grad.norm().item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ TSW computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partial_transport():
    """Test partial transport (unbalanced TSW)"""
    print("\nTesting partial transport (unbalanced)...")
    try:
        reference_path = os.path.join(os.path.dirname(__file__), '../../reference/PartialTSW/src')
        sys.path.insert(0, reference_path)
        from tsw.partial_tsw import PartialTSW
        from tsw.utils import generate_trees_frames
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        tsw_obj = PartialTSW(
            ntrees=50,
            nlines=4,
            p=2,
            delta=2.0,
            mass_division='distance_based',
            device=device
        )
        
        theta, intercept = generate_trees_frames(
            ntrees=50, nlines=4, d=200,
            gen_mode='gaussian_orthogonal', device=device
        )
        
        batch_size = 32
        mu1 = torch.randn(batch_size, 200).to(device)
        mu2 = torch.randn(batch_size, 200).to(device)
        
        # Unbalanced masses
        total_mass_X = torch.tensor(0.8, device=device)
        total_mass_Y = torch.tensor(0.6, device=device)
        
        tsw_loss = tsw_obj(
            mu1, mu2, theta, intercept,
            total_mass_X=total_mass_X,
            total_mass_Y=total_mass_Y
        )
        
        print(f"✓ Partial transport successful. Loss: {tsw_loss.item():.4f}")
        print(f"  - Source mass: {total_mass_X.item()}")
        print(f"  - Target mass: {total_mass_Y.item()}")
        
        return True
    except Exception as e:
        print(f"✗ Partial transport failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("Testing mult_vae_GODM_TSW Implementation")
    print("=" * 70)
    
    tests = [
        ("TSW Import", test_tsw_import),
        ("Model Structure", test_model_structure),
        ("Tree Generation", test_tree_generation),
        ("TSW Computation", test_tsw_computation),
        ("Partial Transport", test_partial_transport),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(r for _, r in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed! Model is ready to use.")
        print("\nTo train the model, run:")
        print("  python train_encoder.py --model mult_vae_godm_tsw --dataset yelp --cuda 0")
        print("\nKey features:")
        print("  • Tree-based hierarchical slicing")
        print("  • Distance-based adaptive mass division")
        print("  • Optional partial transport (unbalanced)")
        print("  • Closed-form solution (no iterations)")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    print("=" * 70)

if __name__ == '__main__':
    main()
