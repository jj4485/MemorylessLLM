"""
Generate responses for the synthetic dataset prompts.

This script takes the synthetic dataset of 1024-token prompts and generates
responses for each prompt using OpenAI's API. The results are saved as a
new dataset with both prompts and responses.
"""

import os
import json
import time
import random
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Configuration
INPUT_FILE = "synthetic_dataset/synthetic_dataset.json"
OUTPUT_DIR = "synthetic_dataset"
OUTPUT_FILE = "synthetic_dataset_with_responses.json"
MODEL_NAME = "gpt-3.5-turbo"  # Model to generate responses
BATCH_SIZE = 5  # Number of prompts to process in parallel
MAX_TOKENS = 1024  # Maximum length of response

def generate_response(example):
    """Generate a response for a single example."""
    try:
        # Extract the prompt from the example
        prompt = example["text"]
        
        # Generate a response using OpenAI API
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful, detailed, and knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Create a new example with both prompt and response
        new_example = example.copy()
        new_example["response"] = response_text
        
        return new_example
    
    except Exception as e:
        print(f"Error generating response: {e}")
        # Return the original example with an error message as response
        error_example = example.copy()
        error_example["response"] = f"Error generating response: {e}"
        return error_example

def process_dataset():
    """Process the entire dataset and generate responses."""
    # Load the dataset
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Loaded {len(dataset)} examples from {INPUT_FILE}")
    print(f"Generating responses using {MODEL_NAME}...")
    
    # Process examples in batches
    results = []
    
    with tqdm(total=len(dataset)) as pbar:
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                batch_results = list(executor.map(generate_response, batch))
            
            results.extend(batch_results)
            
            # Update progress bar
            pbar.update(len(batch))
            
            # Save intermediate results
            if len(results) % 20 == 0:
                print(f"Processed {len(results)} examples")
                temp_path = os.path.join(OUTPUT_DIR, f"synthetic_dataset_with_responses_partial_{len(results)}.json")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                
                # Add a small delay to avoid API rate limits
                time.sleep(2)
    
    # Save the final dataset
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} examples with responses to {output_path}")
    
    # Create train/test split
    train_size = int(len(results) * 0.8)
    train_data = results[:train_size]
    test_data = results[train_size:]
    
    # Save train/test split
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    test_path = os.path.join(OUTPUT_DIR, "test.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created train/test split: {len(train_data)} training examples, {len(test_data)} test examples")
    
    # Also save as JSONL format (one JSON object per line) for compatibility with some training frameworks
    train_jsonl_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_jsonl_path = os.path.join(OUTPUT_DIR, "test.jsonl")
    
    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")
    
    with open(test_jsonl_path, "w", encoding="utf-8") as f:
        for example in test_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved JSONL versions for compatibility")

if __name__ == "__main__":
    process_dataset()
