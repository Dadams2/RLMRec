#!/bin/bash
# Quick hyperparameter sweep for MultVAE MDDM-MMD
# Usage: bash scripts/tune_mddm_mmd.sh amazon 0

DATASET=$1
CUDA=$2

if [ -z "$DATASET" ] || [ -z "$CUDA" ]; then
    echo "Usage: bash tune_mddm_mmd.sh <dataset> <cuda_device>"
    echo "Example: bash tune_mddm_mmd.sh amazon 0"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Starting hyperparameter tuning for MultVAE MDDM-MMD on $DATASET"
echo "================================================================"

# Create results directory
mkdir -p "$PROJECT_ROOT/tuning_results/$DATASET"

# Save original config
CONFIG_FILE="$PROJECT_ROOT/encoder/config/modelconf/mult_vae_mddm_mmd.yml"
BACKUP_FILE="$PROJECT_ROOT/encoder/config/modelconf/mult_vae_mddm_mmd_backup.yml"
cp "$CONFIG_FILE" "$BACKUP_FILE"

# Function to update config
update_config() {
    local beta=$1
    local mmd_weight=$2
    
    # Create temporary config with updated parameters
    cat > "$CONFIG_FILE" << EOF
optimizer:
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
  name: $DATASET

model:
  name: mult_vae_mddm_mmd
  use_multi_scale: true
  dropout: 0.3
  reg_weight: 1.0e-6
  
  $DATASET:
    dropout: 0.3
    beta: $beta
    reg_weight: 1.0e-6
    use_multi_scale: true
    mmd_weight: $mmd_weight
EOF
}

# Beta sweep
echo "Phase 1: Tuning beta (mixing coefficient)"
echo "-------------------------------------------"
for beta in 0.2 0.3 0.4; do
    echo "Testing beta=$beta (mmd_weight=15.0)"
    update_config $beta 15.0
    cd "$PROJECT_ROOT/encoder"
    python train_encoder.py --model mult_vae_mddm_mmd --dataset $DATASET --cuda $CUDA \
        2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/beta_${beta}.log"
    cd "$PROJECT_ROOT"
done

# MMD weight sweep
echo ""
echo "Phase 2: Tuning MMD weight"
echo "---------------------------"
for weight in 10.0 15.0 20.0; do
    echo "Testing mmd_weight=$weight (beta=0.3)"
    update_config 0.3 $weight
    cd "$PROJECT_ROOT/encoder"
    python train_encoder.py --model mult_vae_mddm_mmd --dataset $DATASET --cuda $CUDA \
        2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/mmd_weight_${weight}.log"
    cd "$PROJECT_ROOT"
done

# Combined best settings
echo ""
echo "Phase 3: Testing best combined settings"
echo "----------------------------------------"

# Conservative
echo "Testing conservative settings (beta=0.3, mmd_weight=10)"
update_config 0.3 10.0
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_mmd --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/conservative.log"
cd "$PROJECT_ROOT"

# Aggressive  
echo "Testing aggressive settings (beta=0.2, mmd_weight=20)"
update_config 0.2 20.0
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_mmd --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/aggressive.log"
cd "$PROJECT_ROOT"

# Restore original config
echo ""
echo "Restoring original configuration..."
mv "$BACKUP_FILE" "$CONFIG_FILE"

echo ""
echo "Tuning complete! Check results in tuning_results/${DATASET}/"
echo "Use the following command to view best results:"
echo "grep 'Best Result' tuning_results/${DATASET}/*.log"
