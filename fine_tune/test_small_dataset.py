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
    """Generate a response for a single prompt using a more direct approach."""
    # Format prompt with instruction template
    formatted_prompt = f"[INST] {prompt} [/INST]"
    
    # Tokenize the prompt
    input_tokens = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Get the length of the input for later use
    input_length = input_tokens.input_ids.shape[1]
    
    # Generate with a completely different approach - use the model's forward pass directly
    with torch.no_grad():
        # First try with standard generation
        try:
            outputs = model.generate(
                input_ids=input_tokens.input_ids,
                attention_mask=input_tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                do_sample=False,  # Use greedy decoding for deterministic results
                num_beams=1,      # Simple beam search
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check for the ampersand issue
            if '&' * 10 in generated_text:
                raise ValueError("Detected ampersand issue, trying alternative method")
                
            # If we got an empty response or just whitespace, try alternative method
            if not generated_text or generated_text.isspace():
                raise ValueError("Empty response, trying alternative method")
                
        except Exception as e:
            print(f"Standard generation failed: {e}")
            # Alternative approach: use the model's forward pass directly
            try:
                # Get logits from the model
                outputs = model(input_tokens.input_ids, attention_mask=input_tokens.attention_mask)
                logits = outputs.logits
                
                # Get the last token's logits
                last_token_logits = logits[0, -1, :]
                
                # Get the top 10 tokens
                top_k_tokens = torch.topk(last_token_logits, 10)
                
                # Build a response token by token
                response_tokens = input_tokens.input_ids[0].tolist()
                
                # Generate 20 new tokens manually
                for _ in range(20):
                    # Get model output for current sequence
                    outputs = model(torch.tensor([response_tokens]).to(model.device))
                    next_token_logits = outputs.logits[0, -1, :]
                    
                    # Get the most likely next token
                    next_token = torch.argmax(next_token_logits).item()
                    
                    # Add to our sequence
                    response_tokens.append(next_token)
                    
                    # Stop if we hit the EOS token
                    if next_token == tokenizer.eos_token_id:
                        break
                
                # Decode the full sequence and extract the response
                full_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
                
                # Extract just the response part
                if "[/INST]" in full_output:
                    generated_text = full_output.split("[/INST]", 1)[1].strip()
                else:
                    # Fallback to just taking the newly generated part
                    generated_text = tokenizer.decode(response_tokens[input_length:], skip_special_tokens=True).strip()
                
            except Exception as e:
                print(f"Alternative generation also failed: {e}")
                # Last resort - just return the ground truth as a placeholder
                # This is just for debugging - in a real system you wouldn't do this
                generated_text = "GENERATION FAILED"
                full_output = formatted_prompt + " GENERATION FAILED"
    
    # Final check - if we still have the ampersand issue or empty response, use a placeholder
    if '&' * 10 in generated_text or not generated_text or generated_text.isspace():
        generated_text = "GENERATION FAILED - MODEL ISSUE"
    
    return generated_text, full_output

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
