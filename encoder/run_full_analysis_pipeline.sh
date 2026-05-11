#!/bin/bash

# Complete workflow for Phase 1 semantic noise analysis
# This script demonstrates the full process from testing to comparison

echo "=============================================="
echo "  Phase 1: Semantic Noise Analysis Workflow"
echo "=============================================="
echo ""

# Configuration
DATASET="yelp"  # Change to: amazon-book, yelp, or steam
NUM_BATCHES=50  # Increase for better statistics (e.g., 100)

# Step 0: Test setup
echo "Step 0: Testing setup..."
python test_analysis_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Setup test failed. Please fix issues before continuing."
    exit 1
fi

echo ""
echo "✓ Setup test passed!"
echo ""
read -p "Press Enter to continue with analysis..."

# Step 1: Analyze each model
echo ""
echo "=============================================="
echo "Step 1: Analyzing individual models"
echo "=============================================="
echo ""

for MODEL in mult_vae_mddm mult_vae_godm mult_vae_cpdm; do
    echo "→ Analyzing $MODEL on $DATASET..."
    python run_semantic_analysis.py \
        --model $MODEL \
        --dataset $DATASET \
        --batches $NUM_BATCHES
    
    if [ $? -ne 0 ]; then
        echo "✗ Analysis failed for $MODEL"
        echo "  This might be because the model isn't trained yet."
        echo "  Train it with: python train_encoder.py --model $MODEL --dataset $DATASET"
        continue
    fi
    
    echo "✓ Completed $MODEL"
    echo ""
done

# Step 2: Compare results
echo ""
echo "=============================================="
echo "Step 2: Comparing models"
echo "=============================================="
echo ""

python compare_model_analysis.py --dataset $DATASET

if [ $? -ne 0 ]; then
    echo "✗ Comparison failed"
    exit 1
fi

# Step 3: Display results
echo ""
echo "=============================================="
echo "  Analysis Complete!"
echo "=============================================="
echo ""
echo "Results saved to: ./analysis_results/$DATASET/"
echo ""
echo "Generated files:"
echo "  📊 Plots:"
find "./analysis_results/$DATASET" -name "*.png" -type f 2>/dev/null | while read file; do
    echo "     - $file"
done
echo ""
echo "  📄 Data:"
find "./analysis_results/$DATASET" -name "*.json" -type f 2>/dev/null | while read file; do
    echo "     - $file"
done
echo ""
echo "Next steps:"
echo "  1. Review the plots in analysis_results/$DATASET/"
echo "  2. Check the comparison output above for recommendations"
echo "  3. Use insights to implement Phase 2 (learnable denoising)"
echo ""
echo "To view plots (if on a system with GUI):"
echo "  eog analysis_results/$DATASET/*.png"
echo ""
echo "To view JSON results:"
echo "  cat analysis_results/$DATASET/*/mult_vae_*_analysis.json | jq"
echo ""
