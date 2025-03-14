"""
Simple Dataset Generator

This script:
1) Generates 1000 diverse topics
2) Creates a 1-5 sentence prompt for each topic
3) Generates a response for each prompt

Uses efficient batching for all API calls to maximize throughput.
"""

import os
import json
import random
import time
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Configuration
OUTPUT_DIR = "simple_dataset"
DATASET_SIZE = 1000
MODEL_NAME = "gpt-3.5-turbo"
BATCH_SIZE = 20  # Increased batch size for more efficiency
SEED = 42

# Set random seed for reproducibility
random.seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_topics(n=1000):
    """Generate n diverse topics using GPT."""
    print(f"Generating {n} diverse topics...")
    
    # Break this into batches to avoid token limits
    topics = []
    batch_size = 100
    
    for i in range(0, n, batch_size):
        current_batch_size = min(batch_size, n - i)
        
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse topics for writing prompts."},
                {"role": "user", "content": f"Generate {current_batch_size} diverse topics for writing prompts. These should cover a wide range of subjects including science, history, technology, arts, philosophy, social issues, etc. Format as a simple list with one topic per line. Be specific rather than general. Don't number the items."}
            ],
            temperature=0.8
        )
        
        # Extract topics from the response
        batch_topics = response.choices[0].message.content.strip().split('\n')
        # Clean up any bullet points or other formatting
        batch_topics = [topic.strip().lstrip('- ').lstrip('• ') for topic in batch_topics]
        
        topics.extend(batch_topics)
        
        print(f"Generated {len(topics)} topics so far...")
        time.sleep(1)  # Avoid rate limiting
    
    # Ensure we have exactly n topics
    if len(topics) > n:
        topics = topics[:n]
    elif len(topics) < n:
        # If we don't have enough, generate some generic topics to fill the gap
        generic_topics = [f"Topic {i+1}" for i in range(len(topics), n)]
        topics.extend(generic_topics)
    
    return topics

def create_prompts_batch(topic_batch, sentence_counts):
    """Create prompts for a batch of topics with specified sentence counts."""
    messages = []
    
    # Create a message for each topic-sentence pair
    for topic, num_sentences in zip(topic_batch, sentence_counts):
        messages.append({
            "role": "user",
            "content": f"Create a writing prompt about '{topic}' that is exactly {num_sentences} sentences long. The prompt should be clear and engaging. Don't include any extra text, just the prompt itself."
        })
    
    # Use the batch endpoint to create all prompts at once
    try:
        responses = []
        # Process in smaller sub-batches to avoid context limits
        sub_batch_size = 5
        for i in range(0, len(messages), sub_batch_size):
            sub_batch = messages[i:i+sub_batch_size]
            
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates writing prompts."}
                ] + sub_batch,
                temperature=0.7,
                n=len(sub_batch)
            )
            
            for choice in response.choices:
                responses.append(choice.message.content.strip())
        
        return responses
    
    except Exception as e:
        print(f"Error creating prompts batch: {e}")
        # Fallback prompts
        return [f"Write about {topic}." * num_sentences for topic, num_sentences in zip(topic_batch, sentence_counts)]

def generate_responses_batch(prompt_batch, topic_batch):
    """Generate responses for a batch of prompts."""
    try:
        responses = []
        # Process in smaller sub-batches to avoid context limits
        sub_batch_size = 5
        
        for i in range(0, len(prompt_batch), sub_batch_size):
            sub_prompts = prompt_batch[i:i+sub_batch_size]
            sub_topics = topic_batch[i:i+sub_batch_size]
            
            # Create a system message for each topic
            messages = []
            for prompt, topic in zip(sub_prompts, sub_topics):
                messages.append([
                    {"role": "system", "content": f"You are a helpful assistant that provides informative responses about {topic}."},
                    {"role": "user", "content": prompt}
                ])
            
            # Make parallel API calls for each prompt
            with concurrent.futures.ThreadPoolExecutor(max_workers=sub_batch_size) as executor:
                futures = []
                for msg in messages:
                    futures.append(executor.submit(
                        openai.chat.completions.create,
                        model=MODEL_NAME,
                        messages=msg,
                        temperature=0.7,
                        max_tokens=1000
                    ))
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    responses.append(result.choices[0].message.content.strip())
        
        return responses
    
    except Exception as e:
        print(f"Error generating responses batch: {e}")
        return [f"This is a placeholder response about {topic}." for topic in topic_batch]

def process_batch(batch_indices, topics):
    """Process a batch of examples from topics to prompts to responses."""
    try:
        batch_topics = [topics[i] for i in batch_indices]
        
        # Determine number of sentences (1-5) for each topic
        sentence_counts = [random.randint(1, 5) for _ in range(len(batch_topics))]
        
        # Create prompts in batch
        prompts = create_prompts_batch(batch_topics, sentence_counts)
        
        # Generate responses in batch
        responses = generate_responses_batch(prompts, batch_topics)
        
        # Create examples
        examples = []
        for i, (topic, prompt, num_sentences, response) in enumerate(zip(batch_topics, prompts, sentence_counts, responses)):
            example = {
                "id": f"EXAMPLE_{batch_indices[i]+1}",
                "topic": topic,
                "prompt": prompt,
                "num_sentences": num_sentences,
                "response": response
            }
            examples.append(example)
        
        return examples
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return error examples
        return [{
            "id": f"EXAMPLE_{i+1}",
            "topic": topics[i] if i < len(topics) else f"Topic {i+1}",
            "prompt": f"Error creating prompt: {e}",
            "num_sentences": 0,
            "response": "Error generating response."
        } for i in batch_indices]

def generate_dataset():
    """Generate the complete dataset."""
    # Step 1: Generate topics
    topics = generate_topics(DATASET_SIZE)
    
    # Save topics
    topics_path = os.path.join(OUTPUT_DIR, "topics.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
    
    print(f"Generated and saved {len(topics)} topics.")
    
    # Step 2 & 3: Generate prompts and responses
    print("Generating prompts and responses...")
    dataset = []
    
    with tqdm(total=DATASET_SIZE) as pbar:
        # Process examples in batches
        for i in range(0, DATASET_SIZE, BATCH_SIZE):
            batch_indices = list(range(i, min(i + BATCH_SIZE, DATASET_SIZE)))
            
            # Process the entire batch
            batch_examples = process_batch(batch_indices, topics)
            
            dataset.extend(batch_examples)
            
            # Update progress bar
            pbar.update(len(batch_examples))
            
            # Save periodically
            if (i + len(batch_examples)) % 50 == 0 or (i + len(batch_examples)) == DATASET_SIZE:
                # Save intermediate results
                temp_path = os.path.join(OUTPUT_DIR, f"dataset_partial_{i + len(batch_examples)}.json")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2)
                
                # Print some examples
                for j in range(min(3, len(batch_examples))):
                    example = batch_examples[j]
                    print(f"\nExample {i + j + 1}:")
                    print(f"Topic: {example['topic']}")
                    print(f"Prompt ({example['num_sentences']} sentences): {example['prompt']}")
                    print(f"Response (excerpt): {example['response'][:100]}...")
    
    # Save final dataset
    dataset_path = os.path.join(OUTPUT_DIR, "simple_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    
    # Also save as JSONL for easier processing
    jsonl_path = os.path.join(OUTPUT_DIR, "simple_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    
    print(f"\nDataset generation complete. Generated {len(dataset)} examples.")
    print(f"Dataset saved to {dataset_path} and {jsonl_path}")
    
    return dataset

if __name__ == "__main__":
    generate_dataset()
