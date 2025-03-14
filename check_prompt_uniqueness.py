"""
Check Prompt Uniqueness

This script analyzes the uniqueness of prompts in the generated dataset:
1) Checks for exact duplicates
2) Measures similarity between prompts
3) Reports statistics on uniqueness
"""

import json
import os
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the dataset
dataset_path = "simple_dataset/simple_dataset.json"
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"Loaded dataset with {len(dataset)} examples")

# Extract prompts
prompts = [example['prompt'] for example in dataset]
topics = [example['topic'] for example in dataset]

# Check for exact duplicates
prompt_counter = Counter(prompts)
duplicate_prompts = {prompt: count for prompt, count in prompt_counter.items() if count > 1}

print(f"\n=== EXACT DUPLICATES ===")
print(f"Found {len(duplicate_prompts)} prompts with exact duplicates")
print(f"Total number of duplicate instances: {sum(duplicate_prompts.values()) - len(duplicate_prompts)}")

if duplicate_prompts:
    print("\nTop 5 duplicated prompts:")
    for prompt, count in sorted(duplicate_prompts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Count: {count}, Prompt: {prompt[:100]}...")

# Check for near-duplicates using TF-IDF and cosine similarity
print("\n=== SIMILARITY ANALYSIS ===")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(prompts)

# Calculate pairwise similarities
print("Calculating pairwise similarities...")
similarity_threshold = 0.9  # Threshold for considering prompts as similar
similar_pairs = []

# To avoid memory issues with large datasets, process in chunks
chunk_size = 100
n_prompts = len(prompts)
high_similarity_count = 0

for i in tqdm(range(0, n_prompts, chunk_size)):
    chunk_end = min(i + chunk_size, n_prompts)
    chunk = tfidf_matrix[i:chunk_end]
    
    # Compare this chunk with all prompts
    similarities = cosine_similarity(chunk, tfidf_matrix)
    
    # Find pairs with high similarity
    for j in range(chunk_end - i):
        # Get the row of similarities for this prompt
        sim_row = similarities[j]
        
        # Find indices of prompts with high similarity (excluding self-comparison)
        similar_indices = np.where(sim_row >= similarity_threshold)[0]
        similar_indices = similar_indices[similar_indices != i+j]  # Remove self
        
        if len(similar_indices) > 0:
            high_similarity_count += 1
            # Add up to 3 similar pairs for this prompt
            for idx in similar_indices[:3]:
                similar_pairs.append({
                    'prompt1_idx': i+j,
                    'prompt2_idx': idx,
                    'similarity': sim_row[idx],
                    'prompt1': prompts[i+j],
                    'prompt2': prompts[idx],
                    'topic1': topics[i+j],
                    'topic2': topics[idx]
                })

print(f"Found {high_similarity_count} prompts with high similarity to at least one other prompt")
print(f"Total similar pairs identified: {len(similar_pairs)}")

# Display some examples of similar pairs
if similar_pairs:
    print("\nTop 5 similar pairs:")
    for pair in sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:5]:
        print(f"Similarity: {pair['similarity']:.4f}")
        print(f"Topic 1: {pair['topic1']}")
        print(f"Prompt 1: {pair['prompt1'][:100]}...")
        print(f"Topic 2: {pair['topic2']}")
        print(f"Prompt 2: {pair['prompt2'][:100]}...")
        print()

# Calculate similarity distribution
similarities = [pair['similarity'] for pair in similar_pairs]
if similarities:
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7)
    plt.title('Distribution of Prompt Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig('prompt_similarity_distribution.png')
    print("Saved similarity distribution plot to 'prompt_similarity_distribution.png'")

# Calculate overall uniqueness metrics
uniqueness_percentage = (len(prompts) - len(duplicate_prompts)) / len(prompts) * 100
similarity_percentage = high_similarity_count / len(prompts) * 100

print("\n=== SUMMARY ===")
print(f"Total prompts: {len(prompts)}")
print(f"Unique prompts (no exact duplicates): {len(prompts) - len(duplicate_prompts)} ({uniqueness_percentage:.2f}%)")
print(f"Prompts with high similarity to others: {high_similarity_count} ({similarity_percentage:.2f}%)")
print(f"Completely unique prompts (no high similarity): {len(prompts) - high_similarity_count} ({100 - similarity_percentage:.2f}%)")

# Save the analysis results
analysis_results = {
    'total_prompts': len(prompts),
    'exact_duplicates': len(duplicate_prompts),
    'duplicate_instances': sum(duplicate_prompts.values()) - len(duplicate_prompts),
    'high_similarity_count': high_similarity_count,
    'similar_pairs': len(similar_pairs),
    'uniqueness_percentage': uniqueness_percentage,
    'similarity_percentage': similarity_percentage
}

with open('prompt_uniqueness_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2)

print("\nAnalysis complete. Results saved to 'prompt_uniqueness_analysis.json'")
