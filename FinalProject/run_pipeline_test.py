"""
 > Script to test the complete FUnIE-GAN pipeline
"""
import os
import subprocess
import sys

def test_pipeline():
    """Test the complete pipeline: train -> test -> evaluate"""
    
    print("=== FUnIE-GAN Pipeline Test ===")
    
    # 1. Test training (for a few epochs)
    print("\n1. Testing Training...")
    try:
        # Train for just 2 epochs as a test
        train_cmd = [
            sys.executable, "simple_train.py",
            "--cfg_file", "configs/train_your_dataset.yaml",
            "--epochs", "2",
            "--batch_size", "2"
        ]
        print(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ Training test completed successfully")
            print(result.stdout[-200:])  # Print last 200 chars of output
        else:
            print("✗ Training test failed")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Training test timed out")
    except Exception as e:
        print(f"✗ Training test failed with error: {e}")
    
    # 2. Test inference
    print("\n2. Testing Inference...")
    try:
        # Test on a few sample images
        test_cmd = [
            sys.executable, "test_trained_model.py",
            "--model_path", "checkpoints/FunieGAN/YOUR_DATASET/generator_final.pth",
            "--input_dir", "data/test/A",
            "--output_dir", "data/trained_model_output"
        ]
        print(f"Running: {' '.join(test_cmd)}")
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ Inference test completed successfully")
            print(result.stdout[-200:])  # Print last 200 chars of output
        else:
            print("✗ Inference test failed")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Inference test timed out")
    except Exception as e:
        print(f"✗ Inference test failed with error: {e}")
    
    # 3. Test evaluation
    print("\n3. Testing Evaluation...")
    try:
        # Run evaluation script
        eval_cmd = [
            sys.executable, "evaluate_trained_model.py"
        ]
        print(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ Evaluation test completed successfully")
            print(result.stdout[-500:])  # Print last 500 chars of output
        else:
            print("✗ Evaluation test failed")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Evaluation test timed out")
    except Exception as e:
        print(f"✗ Evaluation test failed with error: {e}")
    
    print("\n=== Pipeline Test Complete ===")

if __name__ == "__main__":
    test_pipeline()