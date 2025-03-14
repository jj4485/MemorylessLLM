from datasets import load_dataset
from similarity import SimilaritySearch
from RLHFGenerator import RLHFGenerator
import os
import matplotlib.pyplot as plt
import json
import concurrent.futures
from tqdm import tqdm
import numpy as np
import time
import random

def build_reference_corpus(dataset_name, split, start_idx, num_samples, corpus_filename):
    """
    Loads a streaming dataset and writes the 'text' field from a subset of examples 
    starting from start_idx and taking num_samples examples to a corpus file.
    """
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    reference_corpus = []
    for i, example in enumerate(ds):
        if i < start_idx:
            continue
        reference_corpus.append(example["text"])
        if i >= start_idx + num_samples - 1:
            break

    with open(corpus_filename, "w", encoding="utf-8") as f:
        for line in reference_corpus:
            f.write(line + "\n")
    print(f"Reference corpus saved to {corpus_filename}")
    return corpus_filename


def get_token_subprompt(full_prompt, num_tokens, tokenizer):
    """Get a subprompt with the specified number of tokens."""
    tokens = tokenizer.tokenize(full_prompt)
    selected_tokens = tokens[:num_tokens]
    return tokenizer.convert_tokens_to_string(selected_tokens)


def create_diverse_prompts(full_prompt, num_tokens, tokenizer, num_prompts=25):
    """
    Create diverse prompts by taking different sections of the full prompt.
    
    Args:
        full_prompt: The full text to sample from
        num_tokens: Number of tokens for each prompt
        tokenizer: The tokenizer to use
        num_prompts: Number of prompts to generate
        
    Returns:
        List of prompts
    """
    tokens = tokenizer.tokenize(full_prompt)
    total_tokens = len(tokens)
    
    # If we don't have enough tokens for diversity, just return the same prompt
    if total_tokens <= num_tokens:
        return [tokenizer.convert_tokens_to_string(tokens)] * num_prompts
    
    prompts = []
    for _ in range(num_prompts):
        # Choose a random starting point that allows for num_tokens to be extracted
        max_start = total_tokens - num_tokens
        start_idx = random.randint(0, max_start)
        
        # Extract tokens and convert back to string
        selected_tokens = tokens[start_idx:start_idx + num_tokens]
        prompt = tokenizer.convert_tokens_to_string(selected_tokens)
        prompts.append(prompt)
    
    return prompts


def process_single_prompt(prompt, generator):
    """
    Process a single prompt and return the result.
    
    Args:
        prompt: The prompt text
        generator: The RLHFGenerator instance
    
    Returns:
        Dictionary with results for this prompt
    """
    start_time = time.time()
    
    # Generate a response using the prompt
    generated_text = generator.generate_text(prompt)
    
    # Check similarity between generated text and the reference corpus
    match, score = generator.check_similarity(generated_text)
    
    generation_time = time.time() - start_time
    
    return {
        "prompt": prompt,
        "output": generated_text,
        "similarity_score": float(score),
        "has_match": match is not None,
        "generation_time": generation_time
    }


def main():
    # Load the reference corpus
    with open("reference_corpus.txt", "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    full_prompt = "\n".join(corpus_lines[:100]).strip()

    print("Loading model and preparing experiment...")

    generator = RLHFGenerator(
        model_name="EleutherAI/pythia-12b",
        reference_corpus_path=os.path.join("reference_corpus.txt"))

    # Define a range of token counts to try
    token_counts = [50, 75, 100, 125, 150, 175, 200]
    num_prompts_per_count = 25  # Number of different prompts per token count

    print(f"Running experiment with {len(token_counts)} token counts, {num_prompts_per_count} prompts each")
    
    # Use ThreadPoolExecutor to run tasks concurrently
    results = {}
    
    # Process each token count sequentially
    for count in token_counts:
        print(f"\nProcessing token count: {count}")
        
        # Create diverse prompts for this token count
        prompts = create_diverse_prompts(full_prompt, count, generator.tokenizer, num_prompts_per_count)
        
        # Process all prompts for this count concurrently
        prompt_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            # Submit all prompts as separate tasks
            future_to_prompt = {executor.submit(process_single_prompt, prompt, generator): i 
                               for i, prompt in enumerate(prompts)}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_prompt), 
                              total=len(prompts), 
                              desc=f"Token count {count}"):
                try:
                    result = future.result()
                    prompt_results.append(result)
                except Exception as exc:
                    print(f'Prompt generated an exception: {exc}')
        
        # Calculate aggregate statistics for this token count
        similarity_scores = [r["similarity_score"] for r in prompt_results]
        match_rate = sum(1 for r in prompt_results if r["has_match"]) / len(prompt_results)
        avg_generation_time = np.mean([r["generation_time"] for r in prompt_results])
        
        results[count] = {
            "token_count": count,
            "prompts": prompt_results,
            "avg_similarity_score": float(np.mean(similarity_scores)),
            "max_similarity_score": float(np.max(similarity_scores)),
            "min_similarity_score": float(np.min(similarity_scores)),
            "std_similarity_score": float(np.std(similarity_scores)),
            "match_rate": float(match_rate),
            "avg_generation_time": float(avg_generation_time),
            "num_prompts": len(prompt_results)
        }
        
        print(f"Token count {count} complete:")
        print(f"  Average similarity score: {results[count]['avg_similarity_score']:.4f}")
        print(f"  Match rate: {results[count]['match_rate']:.2%}")
        print(f"  Average generation time: {results[count]['avg_generation_time']:.2f} seconds")
    
    # Convert results dictionary to list for easier processing
    results_list = [results[count] for count in token_counts]
    
    # Plot the graph of token count vs similarity score with error bars
    plt.figure(figsize=(12, 8))
    
    # Main similarity score plot
    plt.errorbar(
        [r["token_count"] for r in results_list], 
        [r["avg_similarity_score"] for r in results_list],
        yerr=[r["std_similarity_score"] for r in results_list],
        marker="o", 
        linestyle="-", 
        capsize=5, 
        label="Avg Similarity Score"
    )
    
    # Match rate plot (secondary y-axis)
    ax2 = plt.gca().twinx()
    ax2.plot(
        [r["token_count"] for r in results_list], 
        [r["match_rate"] for r in results_list],
        marker="s", 
        linestyle="--", 
        color="red", 
        label="Match Rate"
    )
    
    plt.xlabel("Context Length (Number of Tokens)")
    plt.ylabel("Similarity Score")
    ax2.set_ylabel("Match Rate", color="red")
    plt.title("Memorization vs. Context Length")
    plt.grid(True)
    
    # Combine legends
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    plt.tight_layout()
    plt.savefig("memorization_vs_tokens.png")
    print("Plot saved to memorization_vs_tokens.png")
    
    # Save the results to a JSON file
    with open("memorization_vs_tokens.json", "w", encoding="utf-8") as f:
        json.dump({
            "detailed_results": results,
            "token_counts": token_counts,
            "num_prompts_per_count": num_prompts_per_count
        }, f, indent=2)

    print("Results saved to memorization_vs_tokens.json")
    
if __name__ == '__main__':
    main()