"""
 > Test script for your trained FUnIE-GAN model
"""
import os
import time
import torch
import torch.nn as nn
import argparse
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import numpy as np
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
# local libs
from nets.funiegan import GeneratorFunieGAN

def test_model(model_path, input_dir, output_dir):
    """Test the trained model on input images"""
    
    # Checks
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    
    # Initialize model
    model = GeneratorFunieGAN().to(device)
    
    # Load model weights
    try:
        if is_cuda:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Data pipeline
    img_width, img_height, channels = 256, 256, 3
    transforms_ = [transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    transform = transforms.Compose(transforms_)
    
    # Process images
    test_files = sorted(glob(os.path.join(input_dir, "*.*")))
    print(f"Found {len(test_files)} images to process")
    
    times = []
    for i, path in enumerate(test_files):
        try:
            # Load and transform image
            img = Image.open(path).convert('RGB')
            inp_img = transform(img)  # This returns a tensor
            # Add batch dimension using torch.stack
            inp_img = torch.stack([inp_img]).to(device)
            
            # Generate enhanced image and measure time
            s = time.time()
            with torch.no_grad():
                gen_img = model(inp_img)
            times.append(time.time()-s)
            
            # Save result (input and output side by side)
            inp_img = inp_img.cpu()
            gen_img = gen_img.cpu()
            img_sample = torch.cat((inp_img.data, gen_img.data), -1)
            filename = os.path.basename(path)
            output_path = os.path.join(output_dir, filename)
            save_image(img_sample, output_path, normalize=True)
            
            print(f"Processed {i+1}/{len(test_files)}: {filename}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Run-time statistics
    if (len(times) > 1):
        print(f"\nTotal samples: {len(test_files)}") 
        # Accumulate frame processing times (without bootstrap)
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
        print(f"Time taken: {Ttime:.2f} sec at {1./Mtime:.3f} fps")
        print(f"Saved generated images in {output_dir}\n")
    
    print(f"Enhanced images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/FunieGAN/YOUR_DATASET/generator_final.pth",
                       help="Path to trained generator model")
    parser.add_argument("--input_dir", type=str, default="data/test/A",
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="data/trained_model_output",
                       help="Directory to save enhanced images")
    args = parser.parse_args()
    
    test_model(args.model_path, args.input_dir, args.output_dir)