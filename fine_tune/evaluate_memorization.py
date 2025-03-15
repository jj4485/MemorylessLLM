"""
Evaluate model memorization on the prompt-response dataset.

This script evaluates how well a fine-tuned model has memorized the prompt-response pairs
by measuring exact match accuracy with efficient batching for speed.
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_file, num_samples=None):
    """Load the dataset from a JSON file."""
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    examples = data["examples"] if isinstance(data, dict) and "examples" in data else data
    
    # Sample examples if needed
    if num_samples and num_samples < len(examples):
        examples = random.sample(examples, num_samples)
    
    return examples

def generate_responses(model, tokenizer, examples, batch_size=8, max_length=512, temperature=0.0):
    """Generate responses for the given prompts with efficient batching."""
    logger.info(f"Generating responses with batch size {batch_size}...")
    
    generated_responses = []
    
    # Process in batches
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        prompts = [example["prompt"] for example in batch]
        
        # Format prompts for the model - try different formats
        # Format 1: Simple instruction format
        formatted_prompts = [f"{prompt}" for prompt in prompts]
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        # Generate with proper handling for temperature
        try:
            with torch.no_grad():
                if temperature > 0:
                    # Use custom generation for temperature > 0
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        # Add these parameters to handle numerical instabilities
                        top_k=50,  # Limit to top 50 tokens
                        top_p=0.95,  # Nucleus sampling
                        renormalize_logits=True,  # Ensure logits are properly normalized
                        num_return_sequences=1
                    )
                else:
                    # Use greedy decoding for temperature=0
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
        except RuntimeError as e:
            logger.warning(f"Error during generation with temperature={temperature}: {e}")
            logger.info("Falling back to greedy decoding (temperature=0.0)")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    do_sample=False,  # Use greedy decoding instead
                    pad_token_id=tokenizer.pad_token_id
                )
        
        # Decode and extract the generated part (after the prompt)
        for j, output in enumerate(outputs):
            # Get the full generated text
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract the response part (everything after the prompt)
            prompt_text = formatted_prompts[j]
            if prompt_text in full_text:
                # If we can find the prompt in the output, extract everything after it
                response = full_text[full_text.find(prompt_text) + len(prompt_text):].strip()
            else:
                # If we can't find the prompt exactly, use a more general approach
                # Get the original input length to separate prompt from generation
                input_length = inputs.input_ids[j].shape[0]
                # Decode only the generated part
                response = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            
            # If response is still empty, use the full text minus the first few tokens
            if not response:
                response = full_text
                # Try to remove the prompt if it appears at the beginning
                for prefix in [prompt_text, prompt_text.strip()]:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
                        break
            
            generated_responses.append(response)
        
        # Log progress
        if (i + batch_size) % 20 == 0 or (i + batch_size) >= len(examples):
            logger.info(f"Generated {min(i + batch_size, len(examples))}/{len(examples)} responses")
    
    return generated_responses

def compute_exact_match(generated_responses, ground_truth_responses):
    """Compute exact match score and return details of matches."""
    exact_matches = 0
    match_details = []
    
    for i, (gen, truth) in enumerate(zip(generated_responses, ground_truth_responses)):
        is_match = gen.strip() == truth.strip()
        if is_match:
            exact_matches += 1
        
        match_details.append({
            "index": i,
            "is_match": is_match,
            "generated": gen.strip(),
            "ground_truth": truth.strip()
        })
    
    match_rate = exact_matches / len(generated_responses) if generated_responses else 0
    return match_rate, match_details

def evaluate_memorization(model_path, dataset_file, output_dir, num_samples=100, batch_size=8, temperature=0.0, seed=42):
    """Evaluate model memorization on the dataset using only exact match."""
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    
    # Try to load tokenizer from checkpoint, fall back to original if needed
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, EnvironmentError):
        logger.info(f"Could not load tokenizer from {model_path}, falling back to original Pythia tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    
    # Set padding side to left for proper generation
    tokenizer.padding_side = 'left'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_file}")
    examples = load_dataset(dataset_file, num_samples)
    logger.info(f"Loaded {len(examples)} examples for evaluation")
    
    # Generate responses
    generated_responses = generate_responses(
        model, tokenizer, examples, batch_size=batch_size, temperature=temperature
    )
    
    # Get ground truth responses
    ground_truth_responses = [example["response"] for example in examples]
    
    # Compute exact match
    logger.info("Computing exact match...")
    match_rate, match_details = compute_exact_match(generated_responses, ground_truth_responses)
    logger.info(f"Exact Match: {match_rate:.4f} ({int(match_rate * len(examples))}/{len(examples)})")
    
    # Save results
    results = {
        "exact_match": match_rate,
        "num_matches": int(match_rate * len(examples)),
        "total_samples": len(examples),
        "model_path": model_path,
        "dataset_file": dataset_file,
        "batch_size": batch_size,
        "temperature": temperature
    }
    
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed comparison information
    comparison_data = []
    for i, (example, gen_response) in enumerate(zip(examples, generated_responses)):
        comparison_data.append({
            "id": example.get("id", i),
            "prompt": example["prompt"],
            "ground_truth": example["response"],
            "generated": gen_response,
            "exact_match": gen_response.strip() == example["response"].strip()
        })
    
    with open(os.path.join(output_dir, "response_comparison.json"), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Save detailed match information
    with open(os.path.join(output_dir, "match_details.json"), 'w') as f:
        json.dump(match_details, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return match_rate, match_details

def main():
    parser = argparse.ArgumentParser(description="Evaluate model memorization")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
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
        default=8,
        help="Batch size for generation (higher values are faster)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for deterministic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    evaluate_memorization(
        model_path=args.model_path,
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
