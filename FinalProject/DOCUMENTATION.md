# Underwater Image Enhancement for Maritime Security
## Project Documentation

### Project Overview

This project implements an AI-driven solution for underwater image enhancement specifically designed for maritime security applications in the Indian Ocean region. The system enhances visibility in underwater imagery to improve threat detection capabilities for submarines, mines, and other security concerns.

### Architecture

#### FUnIE-GAN Model
The core of the system is based on the FUnIE-GAN (Fast Underwater Image Enhancement GAN) architecture:

1. **Generator Network**: 5-layer U-Net based architecture
   - Encoding layers with convolutional blocks
   - Decoding layers with transposed convolutions
   - Skip connections for preserving fine details
   - Tanh activation for final output

2. **Discriminator Network**: 4-layer Markovian discriminator
   - Convolutional layers with LeakyReLU activation
   - Batch normalization for stable training
   - PatchGAN architecture for local realism

3. **Loss Functions**:
   - Adversarial loss (MSE-based)
   - Pixel-wise loss (L1 norm)
   - Perceptual loss (VGG-based)

### Data Pipeline

#### Dataset Structure
```
data/
├── test/
│   ├── A/          # Input degraded images
│   ├── GTr_A/      # Ground truth reference images
│   ├── trainA/     # Training input images
│   ├── trainB/     # Training target images
│   └── validation/ # Validation images
```

#### Data Preprocessing
- Resize to 256×256 pixels
- Normalize to [-1, 1] range
- Random horizontal flipping for augmentation
- Batch size of 2-8 images

### Training Process

#### Hyperparameters
- Learning rate: 0.0001 (Adam optimizer)
- Beta values: β1=0.5, β2=0.999
- Batch size: 4 images
- Epochs: 50-100 recommended
- Loss weights: λ_GAN=1, λ_L1=100

#### Training Loop
1. Initialize generator and discriminator weights
2. For each epoch:
   - Train discriminator on real and fake image pairs
   - Train generator with adversarial and pixel-wise losses
   - Save checkpoints every 10 epochs
   - Generate sample images for monitoring

### Performance Metrics

#### Quantitative Evaluation
- **SSIM (Structural Similarity Index)**: Measures structural similarity
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **UIQM (Underwater Image Quality Measure)**: Underwater-specific metric

#### Expected Results
- SSIM improvement: +0.1 to +0.3
- PSNR improvement: +3 to +6 dB
- Processing speed: 25+ FPS on edge devices

### Deployment

#### Edge Device Deployment
- **Hardware**: NVIDIA Jetson Nano/AGX
- **Optimization**: Model quantization and pruning
- **Inference**: Real-time processing capabilities
- **Power**: Low power consumption for AUV/ROV deployment

#### Integration Components
1. **Image Enhancement Module**: Core FUnIE-GAN model
2. **Object Detection Module**: YOLO/SSD for threat identification
3. **Alerting System**: Notifications for detected threats
4. **Dashboard Interface**: Web-based monitoring
5. **Live Streaming**: Real-time video processing

### Maritime Security Applications

#### Threat Detection
- Submarine identification in coastal waters
- Mine detection in harbors and shipping lanes
- Unmanned underwater vehicle (UUV) monitoring
- Diver detection near critical infrastructure

#### Navigation and Surveillance
- Enhanced underwater scene understanding
- Improved navigation for autonomous vehicles
- Continuous monitoring of maritime boundaries
- Search and rescue operations support

#### Environmental Monitoring
- Coral reef health assessment
- Pollution detection and tracking
- Marine life population monitoring
- Infrastructure inspection (pipes, cables, platforms)

### Customization for Indian Ocean Conditions

#### Dataset Characteristics
- Various water turbidity levels (clear to very turbid)
- Different salinity conditions
- Multiple lighting scenarios (natural, dim, artificial)
- Diverse underwater environments (coastal, deep sea)

#### Model Adaptation
- Fine-tuning on Indian Ocean specific imagery
- Domain adaptation for local conditions
- Seasonal variation handling
- Depth-aware enhancement

### Future Enhancements

#### Technical Improvements
- Multi-spectral image enhancement
- Video stream processing optimization
- Model compression for mobile deployment
- Transfer learning for new environments

#### System Integration
- Real-time object detection integration
- Automated alert/notification system
- Cloud-based processing for batch jobs
- Mobile application for field operations

#### Advanced Features
- 3D reconstruction from stereo images
- Motion compensation for moving platforms
- Adaptive enhancement based on water conditions
- Multi-modal sensor fusion (sonar, lidar)

### Usage Examples

#### Training
```bash
python3 main.py --mode train --epochs 50 --batch_size 4 --lr 0.0001
```

#### Testing
```bash
python3 main.py --mode test --model_path checkpoints/generator_final.pth
```

#### Evaluation
```bash
python3 main.py --mode evaluate
```

### Troubleshooting

#### Common Issues
1. **Memory Errors**: Reduce batch size
2. **Poor Quality**: Increase training epochs
3. **Slow Processing**: Use GPU acceleration
4. **Model Not Found**: Check checkpoint directory

#### Performance Tuning
- Adjust learning rate based on loss curves
- Experiment with different batch sizes
- Monitor SSIM/PSNR during training
- Use early stopping to prevent overfitting

This documentation provides a comprehensive overview of the underwater image enhancement system for maritime security applications.