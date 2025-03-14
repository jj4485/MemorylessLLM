"""
Check the token length of examples in the synthetic dataset.
"""

import json
import os
import tiktoken
import statistics

# Path to the dataset
dataset_path = "synthetic_dataset/synthetic_dataset.json"

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# Check if the dataset exists
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

# Load the dataset
with open(dataset_path, 'r', encoding='utf-8') as f:
    try:
        dataset = json.load(f)
    except json.JSONDecodeError:
        print("Error: The file is not valid JSON")
        exit(1)

# Calculate token lengths
token_lengths = []

for i, example in enumerate(dataset):
    text = example["text"]
    tokens = encoding.encode(text)
    token_count = len(tokens)
    token_lengths.append(token_count)
    
    # Print details for the first few examples
    if i < 5:
        print(f"Example {i+1}:")
        print(f"  ID: {example['id']}")
        print(f"  Topic: {example['topic']}")
        print(f"  Token count: {token_count}")
        print(f"  First 100 chars: {text[:100]}...")
        print()

# Calculate statistics
if token_lengths:
    min_length = min(token_lengths)
    max_length = max(token_lengths)
    avg_length = sum(token_lengths) / len(token_lengths)
    median_length = statistics.median(token_lengths)
    
    print(f"Token length statistics for {len(token_lengths)} examples:")
    print(f"  Minimum: {min_length}")
    print(f"  Maximum: {max_length}")
    print(f"  Average: {avg_length:.2f}")
    print(f"  Median: {median_length}")
    
    # Count examples by token length range
    ranges = [(0, 512), (512, 768), (768, 1024), (1024, 1280), (1280, float('inf'))]
    range_counts = {f"{start}-{end if end != float('inf') else '∞'}": 0 for start, end in ranges}
    
    for length in token_lengths:
        for start, end in ranges:
            if start <= length < end:
                range_counts[f"{start}-{end if end != float('inf') else '∞'}"] += 1
                break
    
    print("\nDistribution of token lengths:")
    for range_name, count in range_counts.items():
        percentage = (count / len(token_lengths)) * 100
        print(f"  {range_name}: {count} examples ({percentage:.2f}%)")
else:
    print("No examples found in the dataset.")
