#!/bin/bash
# Quick start script for running GODM_SUOT model

echo "=========================================="
echo "GODM with Sliced Unbalanced OT - Quick Start"
echo "=========================================="
echo ""
echo "This script helps you train the mult_vae_GODM_SUOT model"
echo ""

# Check if we're in the right directory
if [ ! -f "train_encoder.py" ]; then
    echo "ERROR: Please run this script from the encoder/ directory"
    exit 1
fi

echo "Available datasets:"
echo "  1. amazon (Amazon-Book)"
echo "  2. yelp (Yelp reviews)"
echo "  3. steam (Steam games)"
echo ""

read -p "Select dataset (1-3): " dataset_choice

case $dataset_choice in
    1)
        DATASET="amazon"
        ;;
    2)
        DATASET="yelp"
        ;;
    3)
        DATASET="steam"
        ;;
    *)
        echo "Invalid choice. Using yelp as default."
        DATASET="yelp"
        ;;
esac

read -p "CUDA device number (default: 0): " cuda_device
CUDA_DEVICE=${cuda_device:-0}

echo ""
echo "=========================================="
echo "Configuration:"
echo "  Model: mult_vae_GODM_SUOT"
echo "  Dataset: $DATASET"
echo "  CUDA Device: $CUDA_DEVICE"
echo "=========================================="
echo ""
echo "Running training command:"
echo "python train_encoder.py --model mult_vae_godm_suot --dataset $DATASET --cuda $CUDA_DEVICE"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."

python train_encoder.py --model mult_vae_godm_suot --dataset $DATASET --cuda $CUDA_DEVICE

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check results in: encoder/log/mult_vae_godm_suot/"
echo "=========================================="
