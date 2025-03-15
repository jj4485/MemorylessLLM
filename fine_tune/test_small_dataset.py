"""
Test script to prompt the fine-tuned model with all prompts from a small dataset.

This script loads a fine-tuned model and generates responses for all prompts
in a specified small dataset, displaying the results in a clear format.
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned model on a small dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../identifiable_dataset/smalldataset.json",
        help="Path to the small dataset file"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for deterministic)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for proper generation with decoder-only architecture
    tokenizer.padding_side = 'left'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, temperature=0.0, max_new_tokens=100):
    """Generate a response for a single prompt."""
    # Format prompt with instruction template
    formatted_prompt = f"[INST] {prompt} [/INST]"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    
    # Get input length for later use
    input_length = inputs.input_ids.shape[1]
    
    # Generate with more aggressive settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,  # Force at least 20 new tokens
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.5,  # Increase repetition penalty
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            num_beams=5  # Use beam search for better results
        )
    
    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after the instruction)
    if "[/INST]" in full_output:
        response = full_output.split("[/INST]", 1)[1].strip()
    else:
        # Fallback: use the token-based approach
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    # If response is still empty, try a different approach
    if not response:
        # Try to find where the prompt ends in the full text
        if formatted_prompt in full_output:
            pos = full_output.find(formatted_prompt) + len(formatted_prompt)
            response = full_output[pos:].strip()
        else:
            # Last resort: just take the last 75% of tokens
            cutoff_point = int(len(outputs[0]) * 0.25)  # Skip first 25%
            response = tokenizer.decode(outputs[0][cutoff_point:], skip_special_tokens=True).strip()
    
    return response, full_output

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(args.model_path), "small_dataset_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate responses for each prompt
    results = []
    
    print("\n" + "="*80)
    print(f"Testing model: {args.model_path}")
    print("="*80 + "\n")
    
    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        ground_truth = example["response"]
        
        print(f"\nPrompt {i+1}/{len(dataset)}: {prompt}")
        print("-" * 40)
        
        # Generate response
        response, full_output = generate_response(
            model, 
            tokenizer, 
            prompt, 
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        # Check for exact match
        exact_match = (response.strip() == ground_truth.strip())
        
        # Print results
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated   : {response}")
        print(f"Exact Match : {exact_match}")
        
        # Save result
        results.append({
            "id": example.get("id", i),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": response,
            "full_output": full_output,
            "exact_match": exact_match
        })
    
    # Calculate overall metrics
    num_matches = sum(1 for r in results if r["exact_match"])
    exact_match_rate = num_matches / len(results) if results else 0
    
    # Print summary
    print("\n" + "="*80)
    print(f"Results Summary:")
    print(f"Total examples: {len(results)}")
    print(f"Exact matches: {num_matches}")
    print(f"Exact match rate: {exact_match_rate:.2%}")
    print("="*80 + "\n")
    
    # Save results to file
    results_file = os.path.join(results_dir, "small_dataset_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "metrics": {
                "total_samples": len(results),
                "num_matches": num_matches,
                "exact_match": exact_match_rate
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
