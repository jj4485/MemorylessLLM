"""
Generate Synthetic Dataset for Memorization Experiments

This script creates a synthetic dataset of examples, each approximately 1024 tokens long.
The dataset is designed to be used for fine-tuning and memorization experiments.
Uses a local model for generation, so it works in environments without internet access.
"""

import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import numpy as np
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "synthetic_dataset"
DATASET_SIZE = 1000  # Number of examples to generate
TARGET_TOKEN_LENGTH = 1024
SEED = 42
MODEL_NAME = "gpt2-medium"  # Using a medium-sized model for better text generation

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer and model
print(f"Loading model and tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

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

def generate_with_local_model(prompt, max_length=200):
    """Generate text using a local model."""
    try:
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt if it's included in the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating content: {e}"

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
    
    # Generate content using local model
    content = generate_with_local_model(prompt)
    
    # Add paragraph breaks for readability
    paragraphs = []
    current_paragraph = []
    
    for sentence in content.split('. '):
        current_paragraph.append(sentence)
        if len(current_paragraph) >= random.randint(3, 6) or random.random() < 0.2:
            paragraphs.append('. '.join(current_paragraph) + '.')
            current_paragraph = []
    
    if current_paragraph:
        paragraphs.append('. '.join(current_paragraph) + '.')
        
    content = '\n\n'.join(paragraphs)
    
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
        additional_content = generate_with_local_model(additional_prompt, max_length=300)
        
        # Extract the conclusion with the ID
        conclusion = example["text"].split("\n\nThis concludes")[-1]
        
        # Remove the conclusion and add the new content
        example["text"] = example["text"].replace(conclusion, "") + "\n\n" + additional_content + "\n\nThis concludes" + conclusion
        
        # Check if we need even more content
        tokens = tokenizer.encode(example["text"])
        if len(tokens) < target_length * 0.9:
            # Still too short, add more
            more_content = generate_with_local_model(f"Provide more details about {topic}.", max_length=300)
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
