"""
 > Evaluate your trained model results
"""
import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from Evaluation.imqual_utils import getSSIM, getPSNR
from Evaluation.measure_uiqm import getUIQM

def resize_to_match(img1, img2):
    """Resize img2 to match img1 dimensions"""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    return img2

def evaluate_trained_model_results(input_dir, ground_truth_dir, enhanced_dir, trained_model_dir, num_samples=10):
    """Compare results from pre-trained model vs your trained model"""
    
    # Get list of input images
    input_files = sorted(glob(join(input_dir, "*.*")))
    print(f"Found {len(input_files)} input images")
    
    # Results storage
    pretrained_ssim_improvements = []
    pretrained_psnr_improvements = []
    pretrained_uiqm_scores = []
    trained_ssim_improvements = []
    trained_psnr_improvements = []
    trained_uiqm_scores = []
    
    # Process a sample of images
    for i, input_path in enumerate(input_files[:num_samples]):
        try:
            # Get corresponding file names
            filename = os.path.basename(input_path)
            ground_truth_path = join(ground_truth_dir, filename)
            enhanced_path = join(enhanced_dir, filename)  # Pre-trained model results
            trained_path = join(trained_model_dir, f"enhanced_{filename}")  # Your trained model results
            
            # Check if all files exist
            if not os.path.exists(ground_truth_path):
                print(f"Ground truth not found for {filename}")
                continue
                
            if not os.path.exists(enhanced_path):
                print(f"Pre-trained enhanced image not found for {filename}")
                continue
                
            if not os.path.exists(trained_path):
                print(f"Trained model enhanced image not found for {filename}")
                continue
            
            # Load all images
            input_img = Image.open(input_path).convert('RGB')
            ground_truth_img = Image.open(ground_truth_path).convert('RGB')
            enhanced_img = Image.open(enhanced_path).convert('RGB')
            trained_img = Image.open(trained_path).convert('RGB')
            
            # Ensure all images have the same dimensions
            ground_truth_img = resize_to_match(input_img, ground_truth_img)
            enhanced_img = resize_to_match(input_img, enhanced_img)
            trained_img = resize_to_match(input_img, trained_img)
            
            # Convert to numpy arrays
            input_arr = np.array(input_img)
            ground_truth_arr = np.array(ground_truth_img)
            enhanced_arr = np.array(enhanced_img)
            trained_arr = np.array(trained_img)
            
            # Calculate metrics for pre-trained model
            ssim_input_gt = getSSIM(input_arr, ground_truth_arr)
            psnr_input_gt = getPSNR(np.array(input_img.convert("L")), 
                                   np.array(ground_truth_img.convert("L")))
            
            ssim_enhanced_gt = getSSIM(enhanced_arr, ground_truth_arr)
            psnr_enhanced_gt = getPSNR(np.array(enhanced_img.convert("L")), 
                                      np.array(ground_truth_img.convert("L")))
            uiqm_enhanced = getUIQM(enhanced_arr)
            
            # Improvements from pre-trained model
            pretrained_ssim_improvement = ssim_enhanced_gt - ssim_input_gt
            pretrained_psnr_improvement = psnr_enhanced_gt - psnr_input_gt
            
            # Calculate metrics for your trained model
            ssim_trained_gt = getSSIM(trained_arr, ground_truth_arr)
            psnr_trained_gt = getPSNR(np.array(trained_img.convert("L")), 
                                     np.array(ground_truth_img.convert("L")))
            uiqm_trained = getUIQM(trained_arr)
            
            # Improvements from your trained model
            trained_ssim_improvement = ssim_trained_gt - ssim_input_gt
            trained_psnr_improvement = psnr_trained_gt - psnr_input_gt
            
            # Store results
            pretrained_ssim_improvements.append(pretrained_ssim_improvement)
            pretrained_psnr_improvements.append(pretrained_psnr_improvement)
            pretrained_uiqm_scores.append(uiqm_enhanced)
            trained_ssim_improvements.append(trained_ssim_improvement)
            trained_psnr_improvements.append(trained_psnr_improvement)
            trained_uiqm_scores.append(uiqm_trained)
            
            print(f"Processed {filename}")
            print(f"  Pre-trained model - SSIM improvement: {pretrained_ssim_improvement:.4f}, PSNR improvement: {pretrained_psnr_improvement:.2f}, UIQM: {uiqm_enhanced:.4f}")
            print(f"  Your trained model - SSIM improvement: {trained_ssim_improvement:.4f}, PSNR improvement: {trained_psnr_improvement:.2f}, UIQM: {uiqm_trained:.4f}")
            print()
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            continue
    
    # Calculate average metrics
    if pretrained_ssim_improvements:
        print("=== Comparison Results ===")
        
        # Average improvements
        avg_pretrained_ssim = np.mean(pretrained_ssim_improvements)
        avg_pretrained_psnr = np.mean(pretrained_psnr_improvements)
        avg_pretrained_uiqm = np.mean(pretrained_uiqm_scores)
        avg_trained_ssim = np.mean(trained_ssim_improvements)
        avg_trained_psnr = np.mean(trained_psnr_improvements)
        avg_trained_uiqm = np.mean(trained_uiqm_scores)
        
        print(f"Pre-trained model - Avg SSIM improvement: {avg_pretrained_ssim:.4f}, Avg PSNR improvement: {avg_pretrained_psnr:.2f}, Avg UIQM: {avg_pretrained_uiqm:.4f}")
        print(f"Your trained model - Avg SSIM improvement: {avg_trained_ssim:.4f}, Avg PSNR improvement: {avg_trained_psnr:.2f}, Avg UIQM: {avg_trained_uiqm:.4f}")
        
        # Compare which is better
        if avg_trained_ssim > avg_pretrained_ssim:
            print(f"Your trained model is better in SSIM by {avg_trained_ssim - avg_pretrained_ssim:.4f}")
        else:
            print(f"Pre-trained model is better in SSIM by {avg_pretrained_ssim - avg_trained_ssim:.4f}")
            
        if avg_trained_psnr > avg_pretrained_psnr:
            print(f"Your trained model is better in PSNR by {avg_trained_psnr - avg_pretrained_psnr:.2f}")
        else:
            print(f"Pre-trained model is better in PSNR by {avg_pretrained_psnr - avg_trained_psnr:.2f}")
            
        if avg_trained_uiqm > avg_pretrained_uiqm:
            print(f"Your trained model is better in UIQM by {avg_trained_uiqm - avg_pretrained_uiqm:.4f}")
        else:
            print(f"Pre-trained model is better in UIQM by {avg_pretrained_uiqm - avg_trained_uiqm:.4f}")
            
        print("=" * 40)
        
        return True
    else:
        print("No images were processed successfully")
        return False

if __name__ == "__main__":
    # Set directories
    input_dir = "data/test/A"                    # Raw input images
    ground_truth_dir = "data/test/GTr_A"         # Reference/expected output images
    enhanced_dir = "data/output"                 # Enhanced images from pre-trained FUnIE-GAN
    trained_model_dir = "data/trained_model_output"  # Enhanced images from your trained model
    
    # Run comparison
    evaluate_trained_model_results(input_dir, ground_truth_dir, enhanced_dir, trained_model_dir, num_samples=10)