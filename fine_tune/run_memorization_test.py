"""
Script to run a memorization test with the Pythia model.
This script runs fine-tuning on a tiny dataset and evaluates memorization.
"""

import os
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run memorization test with Pythia model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../identifiable_dataset/memorization_test.json",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_pythia_memo_test",
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.002,
        help="Learning rate for training"
    )
    return parser.parse_args()

def run_finetune(args):
    """Run the fine-tuning process."""
    logger.info("Starting fine-tuning process...")
    
    # Construct the command
    cmd = [
        "python", "finetune_pythia.py",
        "--model_name", args.model_name,
        "--dataset_file", args.dataset_file,
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.num_epochs),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(args.learning_rate),
        "--save_steps", "10",
        "--logging_steps", "5",
        "--max_seq_length", "256"
    ]
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode == 0:
        logger.info("Fine-tuning completed successfully")
    else:
        logger.error(f"Fine-tuning failed with return code {result.returncode}")
        return False
    
    return True

def run_evaluation(args):
    """Run the evaluation process."""
    logger.info("Starting evaluation process...")
    
    # Construct the command
    cmd = [
        "python", "test_small_dataset.py",
        "--model_path", args.output_dir,
        "--dataset_file", args.dataset_file
    ]
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode == 0:
        logger.info("Evaluation completed successfully")
    else:
        logger.error(f"Evaluation failed with return code {result.returncode}")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run fine-tuning
    if run_finetune(args):
        # Run evaluation
        run_evaluation(args)
    
    logger.info("Memorization test completed")

if __name__ == "__main__":
    main()
