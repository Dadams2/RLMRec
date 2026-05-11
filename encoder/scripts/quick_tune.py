"""
Quick hyperparameter testing for MultVAE MDDM-MMD
This script allows you to quickly test different hyperparameter combinations
by temporarily modifying the config file.
"""

import subprocess
import sys
import shutil
import os

def update_config_file(dataset, beta, mmd_weight, dropout):
    """Update the config file with new hyperparameters"""
    config_file = 'encoder/config/modelconf/mult_vae_mddm_mmd.yml'
    
    config_content = f"""optimizer:
  name: adam
  lr: 1.0e-3
  weight_decay: 0

train:
  epoch: 1000
  batch_size: 1024
  save_model: false
  loss: pairwise
  test_step: 3
  reproducible: true
  seed: 2024
  patience: 20

test:
  metrics: [recall, ndcg]
  k: [5, 10, 20]
  batch_size: 1024

data:
  type: general_cf
  name: {dataset}

model:
  name: mult_vae_mddm_mmd
  use_multi_scale: true
  dropout: {dropout}
  reg_weight: 1.0e-6
  
  {dataset}:
    dropout: {dropout}
    beta: {beta}
    reg_weight: 1.0e-6
    use_multi_scale: true
    mmd_weight: {mmd_weight}
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)

def run_experiment(model, dataset, cuda, hyperparams):
    """Run a single experiment with given hyperparameters"""
    # Update config file
    update_config_file(
        dataset,
        hyperparams['beta'],
        hyperparams['mmd_weight'],
        hyperparams['dropout']
    )
    
    cmd = [
        'python', 'encoder/train_encoder.py',
        '--model', model,
        '--dataset', dataset,
        '--cuda', str(cuda)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Hyperparameters: beta={hyperparams['beta']}, mmd_weight={hyperparams['mmd_weight']}, dropout={hyperparams['dropout']}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_tune.py <dataset> <cuda_device>")
        print("Example: python quick_tune.py amazon 0")
        sys.exit(1)
    
    dataset = sys.argv[1]
    cuda = sys.argv[2]
    model = 'mult_vae_mddm_mmd'
    
    # Backup original config
    config_file = 'encoder/config/modelconf/mult_vae_mddm_mmd.yml'
    backup_file = 'encoder/config/modelconf/mult_vae_mddm_mmd_backup.yml'
    if os.path.exists(config_file):
        shutil.copy(config_file, backup_file)
    
    print("=" * 80)
    print(f"MultVAE MDDM-MMD Hyperparameter Tuning")
    print(f"Dataset: {dataset}")
    print(f"CUDA Device: {cuda}")
    print("=" * 80)
    
    # Define hyperparameter grid
    configs = [
        # Conservative settings
        {
            'name': 'Conservative',
            'beta': 0.3,
            'mmd_weight': 10.0,
            'dropout': 0.3
        },
        # Balanced settings
        {
            'name': 'Balanced',
            'beta': 0.3,
            'mmd_weight': 15.0,
            'dropout': 0.3
        },
        # Aggressive matching
        {
            'name': 'Aggressive',
            'beta': 0.2,
            'mmd_weight': 20.0,
            'dropout': 0.3
        },
        # Low beta
        {
            'name': 'Low Beta',
            'beta': 0.2,
            'mmd_weight': 15.0,
            'dropout': 0.3
        },
        # High regularization
        {
            'name': 'High Regularization',
            'beta': 0.4,
            'mmd_weight': 10.0,
            'dropout': 0.3
        },
    ]
    
    results = []
    
    try:
        for config in configs:
            name = config.pop('name')
            print(f"\n{'=' * 80}")
            print(f"Testing configuration: {name}")
            print(f"{'=' * 80}")
            
            success = run_experiment(model, dataset, cuda, config)
            results.append({
                'name': name,
                'config': config,
                'success': success
            })
    finally:
        # Restore original config
        if os.path.exists(backup_file):
            shutil.move(backup_file, config_file)
            print("\n" + "=" * 80)
            print("Original configuration restored")
    
    # Summary
    print("\n" + "=" * 80)
    print("TUNING SUMMARY")
    print("=" * 80)
    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        print(f"{result['name']:20s} {status}")
        print(f"  Config: {result['config']}")
    
    print("\nTo view detailed results, check the log files in encoder/encoder/log/mult_vae_mddm_mmd/")
    print("Compare metrics using: grep 'Best Result' encoder/encoder/log/mult_vae_mddm_mmd/*.log")


if __name__ == '__main__':
    main()
