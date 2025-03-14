"""
Test memorization in a fine-tuned language model.

This script evaluates how well a fine-tuned model has memorized examples from
the synthetic dataset by measuring exact match rates and similarity scores
at different context lengths.
"""

import os
import json
import argparse
import logging
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set environment variables for offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Test memorization in a fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="finetuned_model",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="synthetic_dataset/test.jsonl",
        help="Path to the test data file"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="synthetic_dataset/metadata.json",
        help="Path to the metadata file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="memorization_results.json",
        help="Path to save the results"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of examples to test"
    )
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="16,32,64,128,256,512",
        help="Comma-separated list of context lengths to test"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def load_test_examples(test_file, metadata_file, num_examples, seed=42):
    """Load test examples and their metadata."""
    # Set random seed
    np.random.seed(seed)
    
    # Load test examples
    test_examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line)["text"])
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Create a mapping from example ID to metadata
    id_to_metadata = {item["id"]: item for item in metadata}
    
    # Extract example IDs from the test examples
    example_ids = []
    for example in test_examples:
        match = re.search(r'\[SYNTHETIC_EXAMPLE_(\d+)\]', example)
        if match:
            example_id = f"SYNTHETIC_EXAMPLE_{match.group(1)}"
            example_ids.append(example_id)
    
    # Select random examples
    if num_examples < len(test_examples):
        indices = np.random.choice(len(test_examples), num_examples, replace=False)
        selected_examples = [test_examples[i] for i in indices]
        selected_ids = [example_ids[i] for i in indices]
    else:
        selected_examples = test_examples
        selected_ids = example_ids
    
    # Get metadata for selected examples
    selected_metadata = [id_to_metadata.get(example_id, {}) for example_id in selected_ids]
    
    return selected_examples, selected_ids, selected_metadata

def get_prefix_and_target(example, context_length, tokenizer):
    """Split an example into prefix and target based on context length."""
    # Extract the example ID
    match = re.search(r'\[SYNTHETIC_EXAMPLE_(\d+)\]', example)
    if not match:
        return None, None
    
    example_id = f"SYNTHETIC_EXAMPLE_{match.group(1)}"
    
    # Tokenize the example
    tokens = tokenizer.encode(example)
    
    # Use the first context_length tokens as the prefix
    if context_length < len(tokens):
        prefix_tokens = tokens[:context_length]
        prefix = tokenizer.decode(prefix_tokens)
        
        # The target is the full example (for comparing with generated text)
        target = example
        
        return prefix, target
    else:
        # If context_length is longer than the example, use the whole example
        return example, example

def calculate_similarity(text1, text2, tokenizer, model):
    """Calculate cosine similarity between two texts using model embeddings."""
    # Tokenize texts
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1, output_hidden_states=True)
        outputs2 = model(**inputs2, output_hidden_states=True)
    
    # Use the last hidden state as the embedding
    embedding1 = outputs1.hidden_states[-1].mean(dim=1).cpu().numpy()
    embedding2 = outputs2.hidden_states[-1].mean(dim=1).cpu().numpy()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity

def check_exact_match(generated_text, target_text):
    """Check if the generated text exactly matches the target text."""
    # Clean up texts (remove extra whitespace, etc.)
    generated_clean = ' '.join(generated_text.split())
    target_clean = ' '.join(target_text.split())
    
    # Check for exact match
    return generated_clean == target_clean

def check_id_match(generated_text, example_id):
    """Check if the generated text contains the example ID."""
    return example_id in generated_text

def generate_text(model, tokenizer, prefix, max_new_tokens=100):
    """Generate text from a prefix."""
    inputs = tokenizer(prefix, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def test_memorization(args):
    """Test memorization in a fine-tuned model."""
    # Parse context lengths
    context_lengths = [int(x) for x in args.context_lengths.split(',')]
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load test examples
    logger.info(f"Loading test examples from {args.test_file}")
    examples, example_ids, metadata = load_test_examples(
        args.test_file,
        args.metadata_file,
        args.num_examples,
        args.seed
    )
    logger.info(f"Loaded {len(examples)} test examples")
    
    # Test memorization at different context lengths
    results = {}
    
    for context_length in context_lengths:
        logger.info(f"Testing with context length {context_length}")
        
        context_results = []
        
        for i, (example, example_id) in enumerate(zip(examples, example_ids)):
            logger.info(f"Testing example {i+1}/{len(examples)}")
            
            # Get prefix and target
            prefix, target = get_prefix_and_target(example, context_length, tokenizer)
            if prefix is None or target is None:
                logger.warning(f"Could not extract prefix and target for example {i+1}")
                continue
            
            # Generate text
            generated_text = generate_text(model, tokenizer, prefix, args.max_new_tokens)
            
            # Check for exact match
            exact_match = check_exact_match(generated_text, target)
            
            # Check for ID match
            id_match = check_id_match(generated_text, example_id)
            
            # Calculate similarity
            similarity = calculate_similarity(generated_text, target, tokenizer, model)
            
            # Store results
            example_result = {
                "example_id": example_id,
                "prefix_length": context_length,
                "exact_match": exact_match,
                "id_match": id_match,
                "similarity": float(similarity),
                "prefix": prefix[:100] + "..." if len(prefix) > 100 else prefix,  # Truncate for readability
                "generated_text": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,  # Truncate for readability
            }
            
            context_results.append(example_result)
        
        # Calculate aggregate statistics
        exact_match_rate = sum(1 for r in context_results if r["exact_match"]) / len(context_results)
        id_match_rate = sum(1 for r in context_results if r["id_match"]) / len(context_results)
        avg_similarity = sum(r["similarity"] for r in context_results) / len(context_results)
        
        results[context_length] = {
            "examples": context_results,
            "exact_match_rate": exact_match_rate,
            "id_match_rate": id_match_rate,
            "avg_similarity": avg_similarity
        }
        
        logger.info(f"Context length {context_length}: Exact match rate = {exact_match_rate:.2f}, ID match rate = {id_match_rate:.2f}, Avg similarity = {avg_similarity:.2f}")
    
    # Save results
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("Memorization test complete!")
    logger.info("Summary of results:")
    for context_length in context_lengths:
        logger.info(f"Context length {context_length}:")
        logger.info(f"  Exact match rate: {results[context_length]['exact_match_rate']:.2f}")
        logger.info(f"  ID match rate: {results[context_length]['id_match_rate']:.2f}")
        logger.info(f"  Average similarity: {results[context_length]['avg_similarity']:.2f}")

def main():
    args = parse_args()
    test_memorization(args)

if __name__ == "__main__":
    main()
