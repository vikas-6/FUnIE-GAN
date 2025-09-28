# FUnIE-GAN Complete Pipeline

## Overview
This directory contains the complete FUnIE-GAN pipeline for underwater image enhancement, including training, testing, and evaluation components.

## Directory Structure
```
FinalProject/
├── checkpoints/              # Model checkpoints
├── configs/                  # Training configuration files
├── data/                     # Dataset directories
│   ├── trainA/              # Training input images
│   ├── trainB/              # Training target images
│   ├── validation/          # Validation images
│   ├── A/                   # Test input images
│   ├── GTr_A/               # Ground truth test images
│   └── trained_model_output/ # Output from trained model
├── Evaluation/              # Evaluation metrics and utilities
├── nets/                    # Neural network architectures
├── samples/                 # Training sample outputs
├── utils/                   # Utility functions
├── simple_train.py          # Main training script
├── test_trained_model.py    # Model testing script
├── evaluate_trained_model.py # Model evaluation script
└── run_pipeline_test.py     # Complete pipeline test script
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training
```bash
# Train FUnIE-GAN on your dataset
python simple_train.py --cfg_file configs/train_your_dataset.yaml --epochs 100

# For custom training parameters
python simple_train.py --cfg_file configs/train_your_dataset.yaml --epochs 50 --batch_size 4 --lr 0.0003
```

### 3. Testing
```bash
# Test your trained model
python test_trained_model.py --model_path checkpoints/FunieGAN/YOUR_DATASET/generator_final.pth --input_dir data/test/A --output_dir data/trained_model_output
```

### 4. Evaluation
```bash
# Evaluate model performance
python evaluate_trained_model.py
```

## Configuration

### Dataset Structure
The expected dataset structure is:
```
data/
├── trainA/     # Input underwater images
├── trainB/     # Corresponding clear water images
├── validation/ # Validation images
├── A/          # Test input images
└── GTr_A/      # Ground truth test images
```

### Training Configuration
Configuration files in `configs/` define training parameters:
- `train_your_dataset.yaml`: Custom dataset configuration
- `train_euvp.yaml`: EUVP dataset configuration
- `train_ufo.yaml`: UFO-120 dataset configuration

## Model Architecture
- **Generator**: 5-layer U-Net based architecture
- **Discriminator**: 4-layer Markovian discriminator
- **Loss Functions**: 
  - Adversarial loss (MSE)
  - Perceptual loss (VGG19)
  - L1 similarity loss

## Performance Metrics
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **UIQM**: Underwater Image Quality Measure

## File Paths
- **Model Checkpoints**: `checkpoints/FunieGAN/YOUR_DATASET/`
- **Sample Outputs**: `samples/FunieGAN/YOUR_DATASET/`
- **Test Results**: `data/trained_model_output/`

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config files
2. **Model Not Found**: Check model path in test script
3. **Data Loading Errors**: Verify dataset structure matches expected format

### Performance Tips
- Use GPU for training (10x faster than CPU)
- Adjust batch size based on GPU memory
- Monitor training loss in sample outputs