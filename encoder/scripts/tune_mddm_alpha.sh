#!/bin/bash
# Hyperparameter sweep for MultVAE MDDM with α-divergence
# Usage: bash scripts/tune_mddm_alpha.sh amazon 0

DATASET=$1
CUDA=$2

if [ -z "$DATASET" ] || [ -z "$CUDA" ]; then
    echo "Usage: bash tune_mddm_alpha.sh <dataset> <cuda_device>"
    echo "Example: bash tune_mddm_alpha.sh amazon 0"
    exit 1
fi

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "========================================================================"
echo "α-Divergence Hyperparameter Tuning for MultVAE MDDM"
echo "========================================================================"
echo "Dataset: $DATASET"
echo "CUDA Device: $CUDA"
echo ""
echo "α-divergence parameter guide:"
echo "  α → 0.0: Reverse KL (mode-seeking, favors precision)"
echo "  α = 0.5: Hellinger distance (balanced, robust)"
echo "  α = 1.0: Forward KL (mode-covering, favors recall)"
echo "  α > 1.0: More robust to outliers and mismatched supports"
echo "========================================================================"

# Create results directory
mkdir -p "$PROJECT_ROOT/tuning_results/$DATASET/alpha_divergence"

# Save original config
CONFIG_FILE="$PROJECT_ROOT/encoder/config/modelconf/mult_vae_mddm_alpha.yml"
BACKUP_FILE="$PROJECT_ROOT/encoder/config/modelconf/mult_vae_mddm_alpha_backup.yml"
cp "$CONFIG_FILE" "$BACKUP_FILE"

# Function to update config
update_config() {
    local beta=$1
    local alpha=$2
    local alpha_weight=$3
    local use_analytical=$4
    
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
  name: mult_vae_mddm_alpha
  dropout: 0.3
  reg_weight: 1.0e-6
  
  $DATASET:
    dropout: 0.3
    beta: $beta
    reg_weight: 1.0e-6
    alpha: $alpha
    alpha_weight: $alpha_weight
    use_analytical: $use_analytical
    num_alpha_samples: 10
EOF
}

# Phase 1: Test different α values (with fixed beta=0.3, weight=1.0)
echo ""
echo "Phase 1: Tuning α (divergence type)"
echo "------------------------------------"
echo "Testing different α values with beta=0.3, alpha_weight=1.0"
echo ""

for alpha in 0.01 0.5 1.0 1.5; do
    echo "Testing α=$alpha"
    if [ "$alpha" == "0.01" ]; then
        alpha_name="reverse_kl"
        analytical="true"
    elif [ "$alpha" == "0.5" ]; then
        alpha_name="hellinger"
        analytical="true"
    elif [ "$alpha" == "1.0" ]; then
        alpha_name="forward_kl"
        analytical="true"
    else
        alpha_name="alpha_${alpha}"
        analytical="false"
    fi
    
    update_config 0.3 $alpha 1.0 $analytical
    cd "$PROJECT_ROOT/encoder"
    python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
        2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/alpha_${alpha_name}.log"
    cd "$PROJECT_ROOT"
done

# Phase 2: Test different beta values (with best α from Phase 1)
echo ""
echo "Phase 2: Tuning β (mixing coefficient)"
echo "---------------------------------------"
echo "Testing different β values with α=0.5 (Hellinger), alpha_weight=1.0"
echo ""

for beta in 0.2 0.3 0.4 0.5; do
    echo "Testing β=$beta"
    update_config $beta 0.5 1.0 true
    cd "$PROJECT_ROOT/encoder"
    python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
        2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/beta_${beta}.log"
    cd "$PROJECT_ROOT"
done

# Phase 3: Test different alpha_weight values
echo ""
echo "Phase 3: Tuning α-divergence weight"
echo "------------------------------------"
echo "Testing different alpha_weight values with β=0.3, α=0.5"
echo ""

for weight in 0.5 1.0 2.0 5.0; do
    echo "Testing alpha_weight=$weight"
    update_config 0.3 0.5 $weight true
    cd "$PROJECT_ROOT/encoder"
    python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
        2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/weight_${weight}.log"
    cd "$PROJECT_ROOT"
done

# Phase 4: Combined best settings
echo ""
echo "Phase 4: Testing combined best configurations"
echo "---------------------------------------------"

# Conservative (mode-covering, high recall)
echo "Testing: Forward KL (α=1.0, β=0.3, weight=1.0)"
update_config 0.3 1.0 1.0 true
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/config_forward_kl.log"
cd "$PROJECT_ROOT"

# Balanced (Hellinger distance)
echo "Testing: Hellinger distance (α=0.5, β=0.3, weight=1.0)"
update_config 0.3 0.5 1.0 true
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/config_hellinger.log"
cd "$PROJECT_ROOT"

# Aggressive (mode-seeking, high precision)
echo "Testing: Reverse KL (α→0, β=0.3, weight=1.0)"
update_config 0.3 0.01 1.0 true
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/config_reverse_kl.log"
cd "$PROJECT_ROOT"

# Robust (α=1.5 with higher weight)
echo "Testing: Robust configuration (α=1.5, β=0.3, weight=2.0)"
update_config 0.3 1.5 2.0 false
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/config_robust.log"
cd "$PROJECT_ROOT"

# Low beta with Hellinger
echo "Testing: Low β with Hellinger (α=0.5, β=0.2, weight=1.0)"
update_config 0.2 0.5 1.0 true
cd "$PROJECT_ROOT/encoder"
python train_encoder.py --model mult_vae_mddm_alpha --dataset $DATASET --cuda $CUDA \
    2>&1 | tee "$PROJECT_ROOT/tuning_results/${DATASET}/alpha_divergence/config_low_beta.log"
cd "$PROJECT_ROOT"

# Restore original config
echo ""
echo "Restoring original configuration..."
mv "$BACKUP_FILE" "$CONFIG_FILE"

echo ""
echo "========================================================================"
echo "Tuning complete!"
echo "========================================================================"
echo "Results saved in: tuning_results/${DATASET}/alpha_divergence/"
echo ""
echo "To view best results:"
echo "  grep 'Best Result' tuning_results/${DATASET}/alpha_divergence/*.log"
echo ""
echo "To compare Recall@20 across configurations:"
echo "  grep -h 'Recall@20' tuning_results/${DATASET}/alpha_divergence/*.log | sort -t'=' -k2 -nr"
echo "========================================================================"
