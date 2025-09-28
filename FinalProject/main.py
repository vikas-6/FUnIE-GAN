"""
Underwater Image Enhancement for Maritime Security
Main execution script
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Underwater Image Enhancement for Maritime Security')
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], 
                       default='test', help='Operation mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='checkpoints/generator_final.pth', 
                       help='Path to trained model')
    parser.add_argument('--input_dir', type=str, default='data/test/A', 
                       help='Input images directory')
    parser.add_argument('--output_dir', type=str, default='data/enhanced_output', 
                       help='Output images directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training...")
        os.system(f"python3 simple_train.py --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}")
        
    elif args.mode == 'test':
        print("Testing model...")
        os.system(f"python3 test_trained_model.py --model_path {args.model_path} --input_dir {args.input_dir} --output_dir {args.output_dir}")
        
    elif args.mode == 'evaluate':
        print("Evaluating results...")
        os.system("python3 evaluate_trained_model.py")

if __name__ == "__main__":
    main()