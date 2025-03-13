"""
Generate Synthetic Dataset for Memorization Experiments

This script creates a synthetic dataset of examples, each approximately 1024 tokens long.
The dataset is designed to be used for fine-tuning and memorization experiments.
"""

import os
import json
import random
import time
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

# Load environment variables from .env file if it exists
load_dotenv()

# Configuration
OUTPUT_DIR = "synthetic_dataset"
DATASET_SIZE = 1000  # Number of examples to generate
TARGET_TOKEN_LENGTH = 1024
SEED = 42
TOKENIZER_MODEL = "gpt2"  # For token counting

# Set API key - you'll need to set this to your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Set random seed for reproducibility
random.seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Topics to generate content about
TOPICS = [
    "The history of artificial intelligence",
    "Climate change and its effects",
    "Space exploration and colonization",
    "Quantum computing explained",
    "The evolution of social media",
    "Renewable energy technologies",
    "The future of transportation",
    "Blockchain and cryptocurrency",
    "Genetic engineering and ethics",
    "Virtual reality and augmented reality",
    "The impact of automation on jobs",
    "Cybersecurity threats and defenses",
    "The psychology of decision making",
    "Ocean conservation efforts",
    "The development of modern medicine",
    "The philosophy of consciousness",
    "The economics of sustainable development",
    "The history of the internet",
    "The science of nutrition",
    "The future of education"
]

# Templates for generating diverse content
TEMPLATES = [
    "Write a detailed essay about {topic}.",
    "Explain {topic} to someone with no background knowledge.",
    "Discuss the pros and cons of {topic}.",
    "Write a comprehensive guide to understanding {topic}.",
    "Analyze the historical development of {topic}.",
    "Describe the current state of research on {topic}.",
    "Compare different perspectives on {topic}.",
    "Predict future developments in {topic}.",
    "Examine the ethical implications of {topic}.",
    "Summarize the key concepts related to {topic}."
]

def generate_with_openai(prompt, max_tokens=1000):
    """Generate text using OpenAI's API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates detailed, informative content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        # Wait and retry once in case of rate limiting
        time.sleep(5)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates detailed, informative content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e2:
            print(f"Second error with OpenAI API: {e2}")
            return f"Error generating content: {e2}"

def generate_synthetic_text():
    """Generate a synthetic text example with unique identifiable content."""
    # Create a unique identifier for this example
    example_id = f"SYNTHETIC_EXAMPLE_{random.randint(10000, 99999)}"
    
    # Select a random topic and template
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)
    
    # Create a prompt
    prompt = template.format(topic=topic)
    
    # Generate synthetic content with the unique identifier embedded
    intro = f"[{example_id}] {prompt}\n\n"
    
    # Generate content using OpenAI
    content = generate_with_openai(prompt)
    
    # Add conclusion with the unique identifier again
    conclusion = f"\n\nThis concludes our discussion of {topic}. [{example_id}]"
    
    # Combine all parts
    full_text = intro + content + conclusion
    
    return {
        "id": example_id,
        "topic": topic,
        "prompt": prompt,
        "text": full_text
    }

def ensure_token_length(example, target_length=TARGET_TOKEN_LENGTH):
    """Ensure the example is close to the target token length."""
    tokens = tokenizer.encode(example["text"])
    
    if len(tokens) < target_length * 0.9:
        # Too short, add more content
        topic = example["topic"]
        
        # Generate additional content
        additional_prompt = f"Continue writing about {topic}. Be detailed and informative."
        additional_content = generate_with_openai(additional_prompt, max_tokens=500)
        
        # Extract the conclusion with the ID
        conclusion = example["text"].split("\n\nThis concludes")[-1]
        
        # Remove the conclusion and add the new content
        example["text"] = example["text"].replace(conclusion, "") + "\n\n" + additional_content + "\n\nThis concludes" + conclusion
        
        # Check if we need even more content
        tokens = tokenizer.encode(example["text"])
        if len(tokens) < target_length * 0.9:
            # Still too short, add more
            more_content = generate_with_openai(f"Provide more details about {topic}.", max_tokens=300)
            example["text"] = example["text"].replace(conclusion, "") + "\n\n" + more_content + "\n\nThis concludes" + conclusion
    
    # Always check if we're over the limit, regardless of whether we added content
    tokens = tokenizer.encode(example["text"])
    if len(tokens) > target_length - 8:  # Leave a small buffer (8 tokens) to be safe
        # Too long, truncate
        # Find the last occurrence of the ID
        example_id = example["id"]
        last_id_pos = example["text"].rfind(example_id)
        
        # If we found the ID, preserve it
        if last_id_pos > 0:
            # Get text up to the ID mention
            text_before_id = example["text"][:last_id_pos].strip()
            
            # Truncate the text before the ID
            tokens_before_id = tokenizer.encode(text_before_id)
            if len(tokens_before_id) > target_length - 50:  # Leave room for the ID and conclusion
                tokens_before_id = tokens_before_id[:target_length - 50]
                text_before_id = tokenizer.decode(tokens_before_id)
            
            # Reconstruct with the ID
            example["text"] = text_before_id + f"\n\nThis concludes our discussion of {example['topic']}. [{example_id}]"
        else:
            # If we can't find the ID, just truncate and add it back
            tokens = tokens[:target_length - 50]
            example["text"] = tokenizer.decode(tokens) + f"\n\nThis concludes our discussion of {example['topic']}. [{example_id}]"
    
    # Final check to ensure we're under the limit
    final_tokens = tokenizer.encode(example["text"])
    if len(final_tokens) > target_length:
        # If still too long, do a hard truncation
        final_tokens = final_tokens[:target_length - 10]  # Leave room for the ID
        truncated_text = tokenizer.decode(final_tokens)
        example["text"] = truncated_text + f" [{example_id}]"
    
    return example

# Generate the dataset
print(f"Generating {DATASET_SIZE} synthetic examples...")
dataset = []

for i in tqdm(range(DATASET_SIZE)):
    example = generate_synthetic_text()
    example = ensure_token_length(example)
    dataset.append(example)
    
    # Save periodically
    if (i + 1) % 10 == 0:
        print(f"Generated {i + 1} examples")
        # Save intermediate results
        temp_path = os.path.join(OUTPUT_DIR, f"synthetic_dataset_partial_{i+1}.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

# Save the dataset
dataset_path = os.path.join(OUTPUT_DIR, "synthetic_dataset.json")
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

# Also save in a format suitable for fine-tuning
jsonl_path = os.path.join(OUTPUT_DIR, "synthetic_dataset.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(json.dumps({"text": example["text"]}) + "\n")

print(f"Dataset saved to {dataset_path} and {jsonl_path}")

# Create train/test split
random.shuffle(dataset)
train_size = int(0.9 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

# Save train/test splits
train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

with open(train_path, "w", encoding="utf-8") as f:
    for example in train_dataset:
        f.write(json.dumps({"text": example["text"]}) + "\n")

with open(test_path, "w", encoding="utf-8") as f:
    for example in test_dataset:
        f.write(json.dumps({"text": example["text"]}) + "\n")

print(f"Train/test splits saved to {train_path} and {test_path}")

# Save metadata about examples for later evaluation
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump([{"id": ex["id"], "topic": ex["topic"], "prompt": ex["prompt"]} for ex in dataset], f, indent=2)

print(f"Metadata saved to {metadata_path}")
print("Dataset generation complete!")
