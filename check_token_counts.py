"""
Check token counts in the synthetic dataset JSON files.
"""

import os
import json
import tiktoken
import glob
from collections import Counter

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Path to the dataset directory
dataset_dir = "synthetic_dataset"

# Find all JSON files
json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
print(f"Found {len(json_files)} JSON files")

# Check why there are so many JSON files
partial_files = [f for f in json_files if "partial" in f]
print(f"Number of partial files: {len(partial_files)}")

# Analyze the most recent partial file and the final dataset file
if "synthetic_dataset.json" in [os.path.basename(f) for f in json_files]:
    final_file = os.path.join(dataset_dir, "synthetic_dataset.json")
    print(f"Analyzing final dataset file: {final_file}")
    
    try:
        with open(final_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
            print(f"Final dataset contains {len(final_data)} examples")
            
            # Check token counts for a sample of examples
            token_counts = []
            for i, example in enumerate(final_data[:10]):  # Check first 10 examples
                text = example["text"]
                tokens = encoding.encode(text)
                token_count = len(tokens)
                token_counts.append(token_count)
                print(f"Example {i+1} token count: {token_count}")
            
            # Summarize token counts
            count_summary = Counter(token_counts)
            print("\nToken count summary:")
            for count, frequency in count_summary.items():
                print(f"  {count} tokens: {frequency} examples")
    except Exception as e:
        print(f"Error analyzing final dataset: {e}")

# Find the most recent partial file
partial_numbers = [int(f.split("_")[-1].split(".")[0]) for f in partial_files]
if partial_numbers:
    most_recent = max(partial_numbers)
    most_recent_file = os.path.join(dataset_dir, f"synthetic_dataset_partial_{most_recent}.json")
    print(f"\nAnalyzing most recent partial file: {most_recent_file}")
    
    try:
        with open(most_recent_file, 'r', encoding='utf-8') as f:
            partial_data = json.load(f)
            print(f"Partial file contains {len(partial_data)} examples")
            
            # Check token counts for a sample of examples
            token_counts = []
            for i, example in enumerate(partial_data[-5:]):  # Check last 5 examples
                text = example["text"]
                tokens = encoding.encode(text)
                token_count = len(tokens)
                token_counts.append(token_count)
                print(f"Example {len(partial_data)-4+i} token count: {token_count}")
            
            # Summarize token counts
            count_summary = Counter(token_counts)
            print("\nToken count summary:")
            for count, frequency in count_summary.items():
                print(f"  {count} tokens: {frequency} examples")
    except Exception as e:
        print(f"Error analyzing partial file: {e}")

# Explain why there are so many JSON files
print("\nExplanation for multiple JSON files:")
print("The script is saving partial results every 20 examples as a checkpoint.")
print("This is a safety measure to ensure that if the script crashes or is interrupted,")
print("you don't lose all the generated data and can resume from the last checkpoint.")
print("The final dataset will be in 'synthetic_dataset.json' when the script completes.")
