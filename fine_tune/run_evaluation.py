"""
Run script to evaluate memorization across multiple model checkpoints.

This script provides a convenient way to evaluate memorization metrics
for models saved at different iterations during fine-tuning.
"""

import os
import argparse
import subprocess
import json
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate memorization across model checkpoints")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints (e.g., finetuned_pythia)"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="identifiable_dataset/dataset.json",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="memorization_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all checkpoint directories
    checkpoint_dirs = glob(os.path.join(args.model_dir, "checkpoint-iteration-*"))
    
    # Add the final model directory if it exists
    if os.path.exists(args.model_dir) and os.path.isdir(args.model_dir):
        checkpoint_dirs.append(args.model_dir)
    
    # Sort checkpoints by iteration number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]) if '-' in x else float('inf'))
    
    print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")
    
    # Track results across all checkpoints
    all_results = []
    
    # Evaluate each checkpoint
    for checkpoint_dir in checkpoint_dirs:
        iteration = checkpoint_dir.split('-')[-1] if '-' in checkpoint_dir else "final"
        print(f"\nEvaluating checkpoint: {checkpoint_dir} (Iteration {iteration})")
        
        # Create output directory for this checkpoint
        checkpoint_output_dir = os.path.join(args.output_dir, f"iteration-{iteration}")
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        
        # Build evaluation command
        cmd = [
            "python", "evaluate_memorization.py",
            "--model_path", checkpoint_dir,
            "--dataset_file", args.dataset_file,
            "--output_dir", checkpoint_output_dir,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--temperature", str(args.temperature)
        ]
        
        # Execute evaluation
        print("Running command: " + " ".join(cmd))
        subprocess.run(cmd)
        
        # Load results
        metrics_file = os.path.join(checkpoint_output_dir, "metrics_summary.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                metrics["iteration"] = iteration
                all_results.append(metrics)
    
    # Save combined results
    with open(os.path.join(args.output_dir, "all_iterations_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n===== Memorization Results Summary =====")
    print(f"Evaluated {len(all_results)} checkpoints")
    
    # Print metrics progression
    if all_results:
        print("\nMetrics progression across iterations:")
        print(f"{'Iteration':<10} {'Exact Match':<12} {'BLEU':<8} {'ROUGE-L':<8} {'Cosine Sim':<10}")
        print("-" * 50)
        
        for result in all_results:
            print(f"{result['iteration']:<10} {result['exact_match']:<12.4f} {result['bleu']:<8.4f} {result['rougeL']:<8.4f} {result['cosine_similarity']:<10.4f}")
    
    print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
