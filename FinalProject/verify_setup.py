"""
 > Script to verify the FUnIE-GAN setup and data loading
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import yaml
import sys

# Add the local paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))
sys.path.append(os.path.dirname(__file__))

from utils.data_utils import GetTrainingPairs, GetValImage
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import Weights_Normal

def verify_data_loading():
    """Verify that data loading is working correctly"""
    print("=== Verifying Data Loading ===")
    
    # Load config
    cfg_file = "configs/train_your_dataset.yaml"
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_name = cfg["dataset_name"]
    dataset_path = cfg["dataset_path"]
    img_width = cfg["im_width"]
    img_height = cfg["im_height"]
    
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    
    # Check if directories exist
    trainA_dir = os.path.join(dataset_path, "trainA")
    trainB_dir = os.path.join(dataset_path, "trainB")
    val_dir = os.path.join(dataset_path, "validation")
    
    print(f"TrainA directory exists: {os.path.exists(trainA_dir)}")
    print(f"TrainB directory exists: {os.path.exists(trainB_dir)}")
    print(f"Validation directory exists: {os.path.exists(val_dir)}")
    
    # Count files
    if os.path.exists(trainA_dir):
        trainA_files = len([f for f in os.listdir(trainA_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Number of files in trainA: {trainA_files}")
    
    if os.path.exists(trainB_dir):
        trainB_files = len([f for f in os.listdir(trainB_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Number of files in trainB: {trainB_files}")
    
    # Test data loading
    print("\n=== Testing Data Loading ===")
    try:
        transforms_ = [
            transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        # Test training data
        train_dataset = GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_)
        print(f"Training dataset size: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            # Get a sample
            sample = train_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            # The data is already transformed to tensors by the dataset
            input_tensor = sample['A']
            target_tensor = sample['B']
            print(f"Input tensor type: {type(input_tensor)}")
            print(f"Target tensor type: {type(target_tensor)}")
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Target tensor shape: {target_tensor.shape}")
            print("✓ Training data loading works correctly")
        
        # Test validation data
        val_dataset = GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation')
        print(f"Validation dataset size: {len(val_dataset)}")
        
        if len(val_dataset) > 0:
            # Get a sample
            sample = val_dataset[0]
            print(f"Validation sample keys: {sample.keys()}")
            # The data is already transformed to tensors by the dataset
            val_tensor = sample['val']
            print(f"Validation tensor type: {type(val_tensor)}")
            print(f"Validation tensor shape: {val_tensor.shape}")
            print("✓ Validation data loading works correctly")
            
    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def verify_model_creation():
    """Verify that models can be created correctly"""
    print("\n=== Verifying Model Creation ===")
    
    try:
        # Create generator and discriminator
        generator = GeneratorFunieGAN()
        discriminator = DiscriminatorFunieGAN()
        
        print(f"Generator created: {type(generator).__name__}")
        print(f"Discriminator created: {type(discriminator).__name__}")
        
        # Test with a dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            gen_output = generator(dummy_input)
            disc_output = discriminator(gen_output, dummy_input)
            
        print(f"Generator output shape: {gen_output.shape}")
        print(f"Discriminator output shape: {disc_output.shape}")
        print("✓ Models work correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_training_step():
    """Verify that a single training step works"""
    print("\n=== Verifying Training Step ===")
    
    try:
        # Load config
        cfg_file = "configs/train_your_dataset.yaml"
        with open(cfg_file) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        dataset_name = cfg["dataset_name"]
        dataset_path = cfg["dataset_path"]
        img_width = cfg["im_width"]
        img_height = cfg["im_height"]
        
        # Create models
        generator = GeneratorFunieGAN()
        discriminator = DiscriminatorFunieGAN()
        
        # Initialize weights
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)
        
        # Loss functions
        Adv_cGAN = torch.nn.MSELoss()
        L1_G = torch.nn.L1Loss()
        
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.99))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.99))
        
        # Data pipeline
        transforms_ = [
            transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        train_dataset = GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_)
        if len(train_dataset) == 0:
            print("✗ No training data available")
            return False
            
        # Get a single batch
        sample = train_dataset[0]
        # The data is already transformed to tensors by the dataset
        imgs_distorted = sample["A"].unsqueeze(0)  # Add batch dimension
        imgs_good_gt = sample["B"].unsqueeze(0)    # Add batch dimension
        
        print(f"Input batch shape: {imgs_distorted.shape}")
        print(f"Target batch shape: {imgs_good_gt.shape}")
        
        # Adversarial ground truths
        patch = (1, img_height//16, img_width//16)
        valid = torch.ones((imgs_distorted.size(0), *patch))
        fake = torch.zeros((imgs_distorted.size(0), *patch))
        
        # Train Discriminator step
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake.detach(), imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        print(f"Discriminator loss: {loss_D.item():.6f}")
        
        # Train Generator step
        optimizer_G.zero_grad()
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN = Adv_cGAN(pred_fake, valid)
        loss_1 = L1_G(imgs_fake, imgs_good_gt)
        loss_G = loss_GAN + 100 * loss_1
        print(f"Generator loss: {loss_G.item():.6f}")
        
        print("✓ Training step completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("FUnIE-GAN Setup Verification")
    print("=" * 40)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run all verification steps
    data_ok = verify_data_loading()
    model_ok = verify_model_creation() if data_ok else False
    train_ok = verify_training_step() if model_ok else False
    
    print("\n" + "=" * 40)
    print("VERIFICATION SUMMARY")
    print("=" * 40)
    print(f"Data Loading: {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Model Creation: {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Training Step: {'✓ PASS' if train_ok else '✗ FAIL'}")
    
    if data_ok and model_ok and train_ok:
        print("\n🎉 All verifications passed! The setup is working correctly.")
        print("You can now proceed with training using:")
        print("  python simple_train.py --cfg_file configs/train_your_dataset.yaml --epochs 100")
        return True
    else:
        print("\n❌ Some verifications failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()