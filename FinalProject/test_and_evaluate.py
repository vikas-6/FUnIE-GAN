"""
 > Script to test and evaluate your trained FUnIE-GAN model
"""
import os
import subprocess
import sys

def test_and_evaluate_model():
    """Test the trained model and evaluate its performance"""
    
    print("=== FUnIE-GAN Model Testing and Evaluation ===")
    
    # 1. Test the trained model on validation images
    print("\n1. Testing trained model on validation images...")
    try:
        test_cmd = [
            sys.executable, "test_trained_model.py",
            "--model_path", "checkpoints/FunieGAN/YOUR_DATASET/generator_final.pth",
            "--input_dir", "data/validation",
            "--output_dir", "data/trained_model_output"
        ]
        print(f"Running: {' '.join(test_cmd)}")
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Testing completed successfully")
            print("Enhanced images saved to: data/trained_model_output/")
        else:
            print("✗ Testing failed")
            print(result.stderr)
    except Exception as e:
        print(f"✗ Testing failed with error: {e}")
    
    # 2. Evaluate model performance
    print("\n2. Evaluating model performance...")
    try:
        eval_cmd = [
            sys.executable, "evaluate_trained_model.py"
        ]
        print(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Evaluation completed successfully")
            print(result.stdout)
        else:
            print("✗ Evaluation failed")
            print(result.stderr)
    except Exception as e:
        print(f"✗ Evaluation failed with error: {e}")
    
    print("\n=== Testing and Evaluation Complete ===")

if __name__ == "__main__":
    test_and_evaluate_model()