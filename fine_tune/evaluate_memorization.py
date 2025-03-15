"""
Evaluate model memorization on the prompt-response dataset.

This script evaluates how well a fine-tuned model has memorized the prompt-response pairs
by measuring exact match, BLEU, ROUGE, and cosine similarity metrics.
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_file, num_samples=None):
    """Load the dataset and optionally sample a subset."""
    logger.info(f"Loading dataset from {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    logger.info(f"Loaded {len(examples)} examples")
    
    if num_samples and num_samples < len(examples):
        logger.info(f"Sampling {num_samples} examples for evaluation")
        return random.sample(examples, num_samples)
    
    return examples

def generate_responses(model, tokenizer, examples, batch_size=4, max_length=512, temperature=0.7):
    """Generate responses for the given prompts."""
    logger.info("Generating responses...")
    
    generated_responses = []
    
    # Process in batches
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        prompts = [example["prompt"] for example in batch]
        
        # Format prompts for the model
        formatted_prompts = [f"[INST] {prompt} [/INST]" for prompt in prompts]
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_length,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and extract only the generated part (after the prompt)
        for j, output in enumerate(outputs):
            prompt_tokens = tokenizer.encode(formatted_prompts[j], add_special_tokens=True, return_tensors="pt")[0]
            prompt_length = len(prompt_tokens)
            
            # Extract only the generated part
            generated_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            generated_responses.append(generated_text.strip())
        
        # Log progress
        if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(examples):
            logger.info(f"Generated {i + len(batch)}/{len(examples)} responses")
    
    return generated_responses

def compute_exact_match(generated_responses, ground_truth_responses):
    """Compute exact match score."""
    exact_matches = 0
    for gen, truth in zip(generated_responses, ground_truth_responses):
        if gen.strip() == truth.strip():
            exact_matches += 1
    
    return exact_matches / len(generated_responses) if generated_responses else 0

def compute_bleu(generated_responses, ground_truth_responses):
    """Compute BLEU score."""
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for gen, truth in zip(generated_responses, ground_truth_responses):
        gen_tokens = nltk.word_tokenize(gen.lower())
        truth_tokens = nltk.word_tokenize(truth.lower())
        
        # BLEU requires a list of references
        bleu = sentence_bleu([truth_tokens], gen_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def compute_rouge(generated_responses, ground_truth_responses):
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0
    }
    
    for gen, truth in zip(generated_responses, ground_truth_responses):
        scores = scorer.score(truth, gen)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Average the scores
    for key in rouge_scores:
        rouge_scores[key] /= len(generated_responses) if generated_responses else 1
    
    return rouge_scores

def compute_cosine_similarity(generated_responses, ground_truth_responses, tokenizer):
    """Compute cosine similarity between generated and ground truth responses."""
    # Tokenize responses
    gen_encodings = tokenizer(
        generated_responses,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    truth_encodings = tokenizer(
        ground_truth_responses,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get embeddings (using mean pooling of token embeddings)
    gen_embeddings = []
    truth_embeddings = []
    
    # Simple mean pooling of token embeddings
    for i in range(len(generated_responses)):
        gen_tokens = gen_encodings.input_ids[i]
        gen_mask = gen_encodings.attention_mask[i]
        gen_embedding = torch.mean(
            torch.stack([tokenizer.get_input_embeddings()(gen_tokens[j]) * gen_mask[j] 
                         for j in range(len(gen_tokens))]), 
            dim=0
        ).detach().cpu().numpy()
        gen_embeddings.append(gen_embedding)
        
        truth_tokens = truth_encodings.input_ids[i]
        truth_mask = truth_encodings.attention_mask[i]
        truth_embedding = torch.mean(
            torch.stack([tokenizer.get_input_embeddings()(truth_tokens[j]) * truth_mask[j] 
                         for j in range(len(truth_tokens))]), 
            dim=0
        ).detach().cpu().numpy()
        truth_embeddings.append(truth_embedding)
    
    # Compute cosine similarity for each pair
    similarities = []
    for gen_emb, truth_emb in zip(gen_embeddings, truth_embeddings):
        similarity = cosine_similarity([gen_emb], [truth_emb])[0][0]
        similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 0

def evaluate_memorization(model_path, dataset_file, output_dir, num_samples=100, batch_size=4, temperature=0.7, seed=42):
    """Evaluate model memorization on the dataset."""
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    examples = load_dataset(dataset_file, num_samples)
    
    # Generate responses
    generated_responses = generate_responses(
        model, tokenizer, examples, batch_size, temperature=temperature
    )
    
    # Get ground truth responses
    ground_truth_responses = [example["response"] for example in examples]
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    
    exact_match = compute_exact_match(generated_responses, ground_truth_responses)
    logger.info(f"Exact Match: {exact_match:.4f}")
    
    bleu = compute_bleu(generated_responses, ground_truth_responses)
    logger.info(f"BLEU: {bleu:.4f}")
    
    rouge_scores = compute_rouge(generated_responses, ground_truth_responses)
    logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    cosine_sim = compute_cosine_similarity(generated_responses, ground_truth_responses, tokenizer)
    logger.info(f"Cosine Similarity: {cosine_sim:.4f}")
    
    # Save results
    results = {
        "exact_match": exact_match,
        "bleu": bleu,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "cosine_similarity": float(cosine_sim)
    }
    
    with open(os.path.join(output_dir, "metrics_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save generated responses alongside ground truth
    response_data = []
    for i, (example, gen_response) in enumerate(zip(examples, generated_responses)):
        response_data.append({
            "id": example.get("id", i),
            "prompt": example["prompt"],
            "ground_truth": example["response"],
            "generated": gen_response,
            "exact_match": gen_response.strip() == example["response"].strip()
        })
    
    with open(os.path.join(output_dir, "response_comparison.json"), 'w') as f:
        json.dump(response_data, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model memorization")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
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
        default="evaluation_results",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_memorization(
        args.model_path,
        args.dataset_file,
        args.output_dir,
        args.num_samples,
        args.batch_size,
        args.temperature,
        args.seed
    )

if __name__ == "__main__":
    main()
