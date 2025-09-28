#!/bin/bash
# Script to continue training with better parameters

echo "=== Continuing FUnIE-GAN Training ==="
echo "This will train for 50 epochs with improved parameters"
echo ""

# Create directories if they don't exist
mkdir -p checkpoints samples data/trained_model_output

# Continue training with better parameters
python3 simple_train.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0001

echo ""
echo "=== Training Complete ==="
echo "Checkpoints saved in: checkpoints/"
echo "Sample images saved in: samples/"
echo ""
echo "Next steps:"
echo "1. Test your best model: python3 test_trained_model.py"
echo "2. Evaluate results: python3 evaluate_trained_model.py"
echo "3. Check sample images in samples/ directory"