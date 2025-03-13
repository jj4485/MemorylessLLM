"""
Context Length Experiment for Memorization Detection

This script runs experiments with varying context lengths to measure 
how they affect memorization in language models, focusing on academic
definitions of discoverable memorization.
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from rlhf.data import RLHFGenerator

# Configuration
MODEL_NAME = "EleutherAI/pythia-12b"  # Change to your model
REFERENCE_CORPUS = os.path.join("rlhf", "reference_corpus", "speeches.txt")
OUTPUT_DIR = "experiment_results"
PERPLEXITY_THRESHOLD = 10.0  # Adjust based on your model
USE_SIMILARITY = False  # Set to True if you want to use similarity search

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define prompts that are likely to trigger memorization
# These are the beginnings of famous quotes/passages from the reference corpus
PROMPTS = [
    "I have a dream that one day",
    "We hold these truths to be",
    "Four score and seven years",
    "O say can you see, by the",
    "Let freedom ring from",
    "In a sense we've come to our nation's capital",
    "Five score years ago, a great American",
]

# Context lengths to test
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024, 2048]

# Number of responses to generate per prompt
NUM_RESPONSES = 5

# Run experiment for each context length
results = []

for context_length in CONTEXT_LENGTHS:
    print(f"\n{'='*80}")
    print(f"Running experiment with context length: {context_length}")
    print(f"{'='*80}")
    
    # Initialize generator with current context length
    generator = RLHFGenerator(
        model_name=MODEL_NAME,
        reference_corpus_path=REFERENCE_CORPUS,
        perplexity_threshold=PERPLEXITY_THRESHOLD,
        context_length=context_length,
        max_length=100,  # Generate slightly longer responses
        temperature=0.8,  # Slightly higher temperature for more diverse responses
        use_similarity=USE_SIMILARITY  # Optional similarity search
    )
    
    # Run generation
    generator.run(PROMPTS, num_responses_per_prompt=NUM_RESPONSES)
    
    # Save results for this context length
    output_file = os.path.join(OUTPUT_DIR, f"responses_context_{context_length}.json")
    generator.save_responses(output_file)
    
    # Collect metrics
    total_responses = len(generator.all_responses)
    memorized_exact = sum(1 for r in generator.all_responses if r.get("memorized_exact", False))
    memorized_perplexity = sum(1 for r in generator.all_responses if r.get("memorized_perplexity", False))
    memorized_any = sum(1 for r in generator.all_responses if r.get("memorized", False))
    
    # Calculate similarity metrics only if used
    memorized_similarity = 0
    if USE_SIMILARITY:
        memorized_similarity = sum(1 for r in generator.all_responses if r.get("memorized_similarity", False))
    
    avg_perplexity = sum(r.get("perplexity", 0) for r in generator.all_responses) / total_responses
    
    # Add to results
    result_data = {
        "context_length": context_length,
        "total_responses": total_responses,
        "memorized_exact_count": memorized_exact,
        "memorized_perplexity_count": memorized_perplexity,
        "memorized_any_count": memorized_any,
        "memorized_exact_pct": memorized_exact / total_responses * 100,
        "memorized_perplexity_pct": memorized_perplexity / total_responses * 100,
        "memorized_any_pct": memorized_any / total_responses * 100,
        "avg_perplexity": avg_perplexity
    }
    
    # Add similarity metrics if used
    if USE_SIMILARITY:
        result_data.update({
            "memorized_similarity_count": memorized_similarity,
            "memorized_similarity_pct": memorized_similarity / total_responses * 100
        })
    
    results.append(result_data)
    
    print(f"\nContext Length {context_length} Summary:")
    print(f"Total Responses: {total_responses}")
    print(f"Memorized (Any Method): {memorized_any} ({memorized_any/total_responses*100:.2f}%)")
    print(f"Memorized (Exact Match): {memorized_exact} ({memorized_exact/total_responses*100:.2f}%)")
    print(f"Memorized (Perplexity): {memorized_perplexity} ({memorized_perplexity/total_responses*100:.2f}%)")
    if USE_SIMILARITY:
        print(f"Memorized (Similarity): {memorized_similarity} ({memorized_similarity/total_responses*100:.2f}%)")
    print(f"Average Perplexity: {avg_perplexity:.2f}")

# Save aggregate results
with open(os.path.join(OUTPUT_DIR, "experiment_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

# Create DataFrame for analysis
df = pd.DataFrame(results)

# Plot results
plt.figure(figsize=(12, 8))

# Plot memorization rates
plt.subplot(2, 1, 1)
plt.plot(df["context_length"], df["memorized_exact_pct"], 'o-', label="Exact Match (Discoverable)")
plt.plot(df["context_length"], df["memorized_perplexity_pct"], 's-', label="Perplexity-based")
plt.plot(df["context_length"], df["memorized_any_pct"], 'D-', label="Any Method")
if USE_SIMILARITY:
    plt.plot(df["context_length"], df["memorized_similarity_pct"], '^-', label="Similarity-based")
plt.xlabel("Context Length (tokens)")
plt.ylabel("Memorization Rate (%)")
plt.title("Memorization Rate vs. Context Length")
plt.legend()
plt.grid(True)

# Plot average perplexity
plt.subplot(2, 1, 2)
plt.plot(df["context_length"], df["avg_perplexity"], 'o-', color='red')
plt.xlabel("Context Length (tokens)")
plt.ylabel("Average Perplexity")
plt.title("Average Perplexity vs. Context Length")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "memorization_vs_context_length.png"))
plt.close()

print("\nExperiment completed! Results saved to:", OUTPUT_DIR)
print("Check 'memorization_vs_context_length.png' for visualization of results.")
