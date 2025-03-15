"""
Script to fine-tune Pythia model on a small dataset for memorization testing.

This script is optimized for memorization testing with a small dataset,
using higher learning rates and more iterations.
"""

import os
import argparse
import logging
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pythia on a small dataset for memorization testing")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../identifiable_dataset/smalldataset.json",
        help="Path to the small dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_pythia_small",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,  # High number of iterations for memorization
        help="Number of training iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,  # Higher learning rate for memorization
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,  # Smaller batch size for small dataset
        help="Batch size per device"
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=5,  # Evaluate every 5 iterations
        help="Evaluate every N iterations"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build command for fine-tuning
    cmd = [
        "python", "finetune_pythia.py",
        "--model_name", "EleutherAI/pythia-2.8b",
        "--dataset_file", args.dataset_file,
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.num_iterations),
        "--per_device_train_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--save_steps", str(args.evaluate_every),
        "--logging_steps", str(args.evaluate_every),
        "--warmup_ratio", "0.05",
        "--weight_decay", "0.01",
        "--gradient_accumulation_steps", "4",
        "--max_seq_length", "512"  # Smaller sequence length for efficiency
    ]
    
    # Run the fine-tuning
    logger.info(f"Starting fine-tuning on small dataset: {args.dataset_file}")
    logger.info(f"Using {args.num_iterations} iterations with learning rate {args.learning_rate}")
    
    # Print the command
    logger.info("Running command: " + " ".join(cmd))
    
    # Execute the command
    subprocess.run(cmd)
    
    logger.info(f"Fine-tuning complete. Model saved to {args.output_dir}")
    logger.info(f"To test the model, run: python test_small_dataset.py --model_path {args.output_dir}")

if __name__ == "__main__":
    main()
