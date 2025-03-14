"""
Generate Synthetic Dataset for Memorization Experiments

This script creates a synthetic dataset of examples, each with prompt + output totaling exactly 1024 tokens.
The dataset is designed to be used for fine-tuning and memorization experiments.
Uses OpenAI's GPT-3.5-turbo for text generation with parallel processing.
"""

import os
import json
import random
import time
import numpy as np
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import tiktoken
import concurrent.futures
import re

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Configuration
OUTPUT_DIR = "synthetic_dataset"
DATASET_SIZE = 1000  # Number of examples to generate
TARGET_TOKEN_LENGTH = 512 # Total tokens for prompt + output combined
PROMPT_RATIO = 0.3  # Approximately 30% of tokens for prompt, 40% for output
SEED = 42
MODEL_NAME = "gpt-3.5-turbo"  # Using GPT-3.5-turbo for generation
BATCH_SIZE = 50   # Number of prompts to generate in parallel

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer for token counting
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Expanded list of topics for more variety
TOPICS = [
    # Science & Technology
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
    "The future of education",
    "Nanotechnology applications",
    "Biotechnology breakthroughs",
    "Artificial general intelligence",
    "Quantum physics for beginners",
    "The science of climate modeling",
    "Robotics and automation",
    "Machine learning algorithms",
    "The physics of black holes",
    "Nuclear fusion energy",
    "Sustainable agriculture",
    
    # History & Culture
    "Ancient Egyptian civilization",
    "The Renaissance period in Europe",
    "World War II major battles",
    "The history of democracy",
    "Indigenous cultures of North America",
    "The Silk Road trade network",
    "The Ottoman Empire's rise and fall",
    "Medieval castle architecture",
    "The Industrial Revolution",
    "Ancient Greek philosophy",
    "The Civil Rights Movement",
    "Colonial America",
    "The French Revolution",
    "The history of writing systems",
    "Traditional Japanese culture",
    "The Mongol Empire",
    "African kingdoms before colonization",
    "The Cold War era",
    "Ancient Roman engineering",
    "The history of currency",
    
    # Arts & Literature
    "Shakespeare's major works",
    "Modern art movements",
    "Classical music composers",
    "Film history and development",
    "Poetry analysis techniques",
    "The evolution of photography",
    "Architecture through the ages",
    "Famous literary movements",
    "Jazz music history",
    "Contemporary fiction trends",
    "Ballet and modern dance",
    "Comic book art evolution",
    "Theater traditions worldwide",
    "The history of fashion",
    "Video games as an art form",
    "Impressionism in painting",
    "World mythology systems",
    "The Beat Generation writers",
    "Traditional folk music",
    "Graphic design principles",
    
    # Business & Economics
    "Macroeconomic theory",
    "Stock market investing strategies",
    "Entrepreneurship fundamentals",
    "Global supply chain management",
    "Marketing psychology",
    "Cryptocurrency economics",
    "International trade agreements",
    "Business leadership principles",
    "Economic inequality causes",
    "Behavioral economics insights",
    "Corporate social responsibility",
    "The gig economy impact",
    "Startup funding methods",
    "Consumer behavior analysis",
    "Financial crisis case studies",
    "Digital marketing strategies",
    "Labor market trends",
    "Sustainable business models",
    "Mergers and acquisitions",
    "Economic development theories",
    
    # Health & Medicine
    "The human immune system",
    "Mental health treatment approaches",
    "Nutrition science fundamentals",
    "Exercise physiology",
    "The history of pandemics",
    "Cancer research breakthroughs",
    "Neuroscience discoveries",
    "Public health policy",
    "Personalized medicine",
    "Aging and longevity research",
    "Vaccine development process",
    "Medical ethics debates",
    "Sleep science research",
    "Telemedicine implementation",
    "Genetic disorders",
    "Antibiotic resistance",
    "Emergency medicine practices",
    "Chronic disease management",
    "Maternal health worldwide",
    "Psychology of addiction",
    
    # Philosophy & Ethics
    "Existentialism explained",
    "Ethical frameworks comparison",
    "The concept of free will",
    "Eastern philosophical traditions",
    "The nature of consciousness",
    "Political philosophy theories",
    "The ethics of artificial intelligence",
    "Philosophical views on death",
    "The concept of justice",
    "Moral relativism debate",
    "Stoicism in modern life",
    "Utilitarianism principles",
    "The philosophy of science",
    "Metaphysics fundamentals",
    "Epistemology theories",
    "Bioethics controversies",
    "The meaning of happiness",
    "Environmental ethics",
    "Philosophical skepticism",
    "The concept of beauty"
]

# Define a wide variety of prompt templates for different types of content
PROMPT_TEMPLATES = [
    # Essays and articles
    "Write a detailed essay about {topic} that explores its historical development, key concepts, and future implications.",
    "Compose an in-depth article analyzing {topic} from multiple perspectives, including its benefits, challenges, and real-world applications.",
    "Draft a comprehensive research paper on {topic}, including methodology, findings, and critical analysis of existing literature.",
    
    # Explanations and tutorials
    "Explain {topic} as if you were teaching it to a university class. Include key principles, examples, and practical applications.",
    "Create a step-by-step tutorial on {topic} for beginners, covering fundamental concepts and building up to advanced techniques.",
    "Develop a comprehensive guide to understanding {topic}, addressing common misconceptions and providing clarity on complex aspects.",
    
    # Creative writing
    "Write a short story that creatively incorporates elements of {topic} while entertaining and educating the reader.",
    "Compose a fictional dialogue between two experts debating different approaches to {topic}, highlighting key arguments on both sides.",
    "Create a narrative that follows the historical development of {topic} through the eyes of key figures who contributed to the field.",
    
    # Analysis and critique
    "Provide a critical analysis of {topic}, examining its strengths, weaknesses, and potential areas for improvement or innovation.",
    "Evaluate the impact of {topic} on society, considering ethical implications, social consequences, and potential future developments.",
    "Analyze the relationship between {topic} and related fields, exploring interdisciplinary connections and collaborative opportunities.",
    
    # Comparisons and contrasts
    "Compare and contrast different approaches to {topic}, highlighting the advantages and limitations of each methodology.",
    "Examine how {topic} is understood and implemented across different cultures, regions, or disciplines, noting key similarities and differences.",
    "Contrast historical and contemporary perspectives on {topic}, showing how understanding has evolved over time.",
    
    # Problem-solving
    "Identify key challenges related to {topic} and propose innovative solutions based on current research and best practices.",
    "Develop a framework for addressing common problems in {topic}, including practical strategies and implementation considerations.",
    "Design an approach to optimize or improve aspects of {topic}, considering efficiency, sustainability, and ethical considerations.",
    
    # Case studies
    "Present a detailed case study demonstrating the application of {topic} in a real-world context, including outcomes and lessons learned.",
    "Analyze a specific example where {topic} was successfully implemented, examining the factors that contributed to its success.",
    "Investigate a notable failure or challenge related to {topic}, exploring what went wrong and how similar issues could be avoided.",
    
    # Future perspectives
    "Predict future trends and developments in {topic} over the next decade, based on current trajectories and emerging innovations.",
    "Envision how {topic} might evolve in response to changing technological, social, and environmental factors.",
    "Forecast potential breakthroughs in {topic} and their implications for related fields and society as a whole.",
    
    # Reviews and summaries
    "Provide a comprehensive review of the current state of knowledge regarding {topic}, synthesizing key findings and identifying gaps.",
    "Summarize the historical development of {topic}, highlighting pivotal moments, paradigm shifts, and influential contributors.",
    "Create an annotated bibliography of essential resources for understanding {topic}, with commentary on their significance and contributions.",
    
    # Interdisciplinary perspectives
    "Explore how {topic} intersects with fields such as psychology, economics, ethics, and technology, highlighting interdisciplinary insights.",
    "Examine {topic} through multiple lenses, including scientific, philosophical, cultural, and practical perspectives.",
    "Investigate how different academic disciplines approach and contribute to our understanding of {topic}."
]

# Additional prompt elements to add variety and specificity
PROMPT_ADDITIONS = [
    "Include historical context and development.",
    "Discuss ethical implications and considerations.",
    "Analyze practical applications in various fields.",
    "Consider different cultural perspectives and approaches.",
    "Examine technological aspects and innovations.",
    "Address common misconceptions and controversies.",
    "Explore economic impacts and business applications.",
    "Discuss psychological and sociological dimensions.",
    "Consider environmental implications and sustainability.",
    "Analyze political and policy considerations.",
    "Examine educational approaches and learning strategies.",
    "Discuss media representation and public perception.",
    "Consider accessibility and inclusivity aspects.",
    "Analyze global trends and regional variations.",
    "Examine theoretical frameworks and models.",
    "Discuss research methodologies and approaches.",
    "Consider legal and regulatory aspects.",
    "Analyze case studies and real-world examples.",
    "Explore future directions and emerging trends.",
    "Discuss interdisciplinary connections and collaborations."
]

def create_unique_prompt(topic, target_length=int(TARGET_TOKEN_LENGTH * PROMPT_RATIO)):
    """Create a unique prompt that is approximately 60% of the target token length."""
    # Start with a randomly selected base template
    base_template = random.choice(PROMPT_TEMPLATES)
    base_prompt = base_template.format(topic=topic)
    
    # Add a random selection of specific instructions to make the prompt more unique
    additions = random.sample(PROMPT_ADDITIONS, k=min(5, len(PROMPT_ADDITIONS)))
    
    # Create a unique prompt structure
    sections = [
        f"{base_prompt}",
        f"In your response, please: "
    ]
    
    # Add the selected additions as bullet points
    for addition in additions:
        sections.append(f"- {addition}")
    
    # Join the sections with newlines
    current_prompt = "\n".join(sections)
    
    # Add specific formatting instructions
    formatting_instructions = [
        f"\nOrganize your response with clear headings and subheadings.",
        f"\nInclude relevant examples to illustrate key points.",
        f"\nConsider both theoretical aspects and practical implications."
    ]
    
    # Add formatting instructions
    for instruction in formatting_instructions:
        current_prompt += instruction
    
    # Calculate current token count
    current_tokens = encoding.encode(current_prompt)
    
    # Additional content to reach target length if needed
    additional_content = [
        f"\nExplore how {topic} relates to current global challenges.",
        f"\nDiscuss how {topic} has evolved over time and might continue to develop.",
        f"\nConsider how different stakeholders might approach or be affected by {topic}.",
        f"\nAnalyze the role of innovation and creativity in advancing {topic}."
    ]
    
    # Keep adding content until we reach the target length
    while len(current_tokens) < target_length - 10:  # Leave some buffer
        if additional_content:
            # Add a random piece of additional content
            addition = random.choice(additional_content)
            additional_content.remove(addition)  # Don't use the same addition twice
            current_prompt += addition
        else:
            # If we've used all our prepared content, add generic instructions
            current_prompt += f"\nProvide detailed analysis and examples related to {topic}."
        
        # Recalculate token count
        current_tokens = encoding.encode(current_prompt)
    
    # Final adjustment to get close to the target length
    if len(current_tokens) > target_length:
        current_tokens = current_tokens[:target_length]
        current_prompt = encoding.decode(current_tokens)
    
    return current_prompt

def generate_response(prompt, topic, remaining_tokens):
    """Generate a response that uses the remaining tokens."""
    try:
        # Create a system message that instructs the model to generate a response of appropriate length
        system_message = f"You are a helpful assistant that generates detailed, informative content about {topic}. Your response should be concise but comprehensive."
        
        # Generate the response
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=remaining_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Ensure the response isn't too long
        response_tokens = encoding.encode(response_text)
        if len(response_tokens) > remaining_tokens:
            response_tokens = response_tokens[:remaining_tokens]
            response_text = encoding.decode(response_tokens)
        
        return response_text
    
    except Exception as e:
        print(f"Error generating response: {e}")
        # Return a placeholder response
        placeholder = f"This is a placeholder response about {topic}. " * 10
        placeholder_tokens = encoding.encode(placeholder)
        if len(placeholder_tokens) > remaining_tokens:
            placeholder_tokens = placeholder_tokens[:remaining_tokens]
            placeholder = encoding.decode(placeholder_tokens)
        return placeholder

def generate_example(i):
    """Generate a single example with prompt + output totaling exactly 1024 tokens."""
    try:
        # Create a unique identifier for this example
        example_id = f"SYNTHETIC_EXAMPLE_{random.randint(10000, 99999)}"
        
        # Select a random topic
        topic = random.choice(TOPICS)
        
        # Create a unique prompt (target ~60% of total tokens)
        prompt_target_length = int(TARGET_TOKEN_LENGTH * PROMPT_RATIO)
        prompt = create_unique_prompt(topic, prompt_target_length)
        
        # Add the example ID at the beginning
        final_prompt = f"[{example_id}] {prompt}"
        
        # Calculate how many tokens we have left for the response
        prompt_tokens = encoding.encode(final_prompt)
        remaining_tokens = TARGET_TOKEN_LENGTH - len(prompt_tokens)
        
        # Generate a response that fits in the remaining tokens
        response = generate_response(prompt, topic, remaining_tokens)
        
        # Combine prompt and response
        combined_text = final_prompt + "\n\n" + response
        
        # Ensure the combined text is exactly 1024 tokens
        combined_tokens = encoding.encode(combined_text)
        
        # Adjust if necessary
        if len(combined_tokens) > TARGET_TOKEN_LENGTH:
            # If too long, truncate
            combined_tokens = combined_tokens[:TARGET_TOKEN_LENGTH]
            combined_text = encoding.decode(combined_tokens)
        elif len(combined_tokens) < TARGET_TOKEN_LENGTH:
            # If too short, pad with spaces
            while len(combined_tokens) < TARGET_TOKEN_LENGTH:
                combined_text += " "
                combined_tokens = encoding.encode(combined_text)
                if len(combined_tokens) > TARGET_TOKEN_LENGTH:
                    combined_tokens = combined_tokens[:TARGET_TOKEN_LENGTH]
                    combined_text = encoding.decode(combined_tokens)
                    break
        
        # Double-check token count
        final_tokens = encoding.encode(combined_text)
        if len(final_tokens) != TARGET_TOKEN_LENGTH:
            print(f"Warning: Example {i} has {len(final_tokens)} tokens instead of {TARGET_TOKEN_LENGTH}")
        
        # Split the combined text back into prompt and response
        parts = combined_text.split("\n\n", 1)
        if len(parts) == 2:
            final_prompt, final_response = parts
        else:
            # If splitting failed, use the original prompt and adjust the response
            final_prompt = final_prompt
            final_response = combined_text[len(final_prompt)+2:]  # +2 for "\n\n"
        
        # Create the example
        example = {
            "id": example_id,
            "topic": topic,
            "prompt": final_prompt,
            "response": final_response,
            "combined_text": combined_text,
            "token_count": len(final_tokens)
        }
        
        return example
    except Exception as e:
        print(f"Error generating example {i}: {e}")
        # Return a placeholder example
        return {
            "id": f"SYNTHETIC_EXAMPLE_{random.randint(10000, 99999)}",
            "topic": "Error",
            "prompt": f"Error generating example: {e}",
            "response": "Placeholder response",
            "combined_text": f"Error generating example: {e}\n\nPlaceholder response",
            "token_count": 0
        }

def generate_dataset():
    """Generate the entire dataset using parallel processing."""
    print(f"Generating {DATASET_SIZE} synthetic examples with prompt + output totaling EXACTLY {TARGET_TOKEN_LENGTH} tokens...")
    dataset = []
    
    with tqdm(total=DATASET_SIZE) as pbar:
        # Process examples in batches
        for i in range(0, DATASET_SIZE, BATCH_SIZE):
            batch_indices = range(i, min(i + BATCH_SIZE, DATASET_SIZE))
            
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                batch_examples = list(executor.map(generate_example, batch_indices))
            
            dataset.extend(batch_examples)
            
            # Update progress bar
            pbar.update(len(batch_examples))
            
            # Save periodically
            if len(dataset) % 20 == 0:
                print(f"Generated {len(dataset)} examples")
                # Save intermediate results
                temp_path = os.path.join(OUTPUT_DIR, f"synthetic_dataset_partial_{len(dataset)}.json")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2)
                
                # Verify token count for a few examples
                for j in range(min(5, len(batch_examples))):
                    example = batch_examples[j]
                    print(f"Example {j+1} token count: {example['token_count']}")
    
    return dataset

def save_dataset(dataset):
    """Save the dataset in various formats."""
    # Save the complete dataset
    dataset_path = os.path.join(OUTPUT_DIR, "synthetic_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    
    # Save in JSONL format for fine-tuning
    jsonl_path = os.path.join(OUTPUT_DIR, "synthetic_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps({"text": example["combined_text"]}) + "\n")
    
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
            f.write(json.dumps({"text": example["combined_text"]}) + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for example in test_dataset:
            f.write(json.dumps({"text": example["combined_text"]}) + "\n")
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump([{"id": ex["id"], "topic": ex["topic"]} for ex in dataset], f, indent=2)
    
    print(f"Dataset saved to {dataset_path} and {jsonl_path}")
    print(f"Train/test splits saved to {train_path} and {test_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    """Main function to generate and save the dataset."""
    dataset = generate_dataset()
    save_dataset(dataset)
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
