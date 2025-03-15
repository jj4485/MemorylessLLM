"""
Run script to fine-tune the Pythia 2.8B model with customizable iterations.

This script provides a convenient way to run the fine-tuning process
with different parameters and iteration counts.
"""

import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pythia 2.8B fine-tuning with custom parameters")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Use PEFT (LoRA) for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned_pythia",
        help="Directory to save the fine-tuned model"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build command for fine-tuning
    cmd = [
        "python", "finetune_pythia.py",
        "--num_iterations", str(args.num_iterations),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--learning_rate", str(args.learning_rate),
        "--output_dir", args.output_dir
    ]
    
    # Add optional flags
    if args.use_peft:
        cmd.append("--use_peft")
    
    if args.load_in_8bit:
        cmd.append("--load_in_8bit")
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
