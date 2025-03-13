"""
Fine-tune a smaller model on synthetic dataset and evaluate memorization

This script fine-tunes a smaller model (GPT-2) on the synthetic dataset for multiple epochs,
evaluating memorization at different context window sizes after each epoch.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset,
)
from datasets import load_dataset
import math
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "gpt2"  # Changed to GPT-2 which is smaller and more accessible
DATASET_DIR = "synthetic_dataset"
OUTPUT_DIR = "memorization_results"
MAX_EPOCHS = 5
CONTEXT_WINDOWS = [64, 128, 256, 512, 768, 1024]
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
PERPLEXITY_THRESHOLD = 10.0  # Threshold for determining memorization

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_path = os.path.join(DATASET_DIR, "train.jsonl")
test_path = os.path.join(DATASET_DIR, "test.jsonl")
metadata_path = os.path.join(DATASET_DIR, "metadata.json")

# Load metadata for evaluation
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text, max_length=None):
    """Calculate perplexity of text using the model."""
    inputs = tokenizer(text, return_tensors="pt")
    
    # Apply context length limitation if specified
    if max_length is not None and inputs["input_ids"].shape[1] > max_length:
        # Truncate from the beginning to keep the most recent tokens
        inputs["input_ids"] = inputs["input_ids"][:, -max_length:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_length:]
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity

# Function to check for exact match
def find_exact_match(generated_text, reference_text):
    """Check if generated text appears exactly in reference text."""
    generated_text = generated_text.strip().lower()
    reference_text = reference_text.strip().lower()
    return generated_text in reference_text

# Function to evaluate memorization
def evaluate_memorization(model, tokenizer, test_examples, context_windows):
    """
    Evaluate memorization at different context window sizes.
    
    Returns:
        dict: Results with memorization metrics for each context window
    """
    results = {}
    
    for context_length in context_windows:
        print(f"\nEvaluating with context length: {context_length}")
        
        memorized_exact_count = 0
        memorized_perplexity_count = 0
        total_perplexity = 0
        
        for example in tqdm(test_examples, desc=f"Context {context_length}"):
            # Get the example ID and full text
            example_id = example["id"]
            full_text = example["text"]
            
            # Extract a prompt (first 50 tokens)
            prompt_tokens = tokenizer.encode(full_text)[:50]
            prompt = tokenizer.decode(prompt_tokens)
            
            # Generate text using the prompt with the specified context length
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Apply context length limitation
            if inputs["input_ids"].shape[1] > context_length:
                inputs["input_ids"] = inputs["input_ids"][:, -context_length:]
                inputs["attention_mask"] = inputs["attention_mask"][:, -context_length:]
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
            
            # Generate text
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 100,  # Generate ~100 new tokens
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    temperature=0.7
                )
            
            # Get only the newly generated tokens (excluding the prompt)
            prompt_length = inputs["input_ids"].shape[1]
            gen_tokens = output_tokens[0, prompt_length:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Calculate perplexity of the generated text
            perplexity = calculate_perplexity(model, tokenizer, generated_text, max_length=context_length)
            total_perplexity += perplexity
            
            # Check for exact match
            is_exact_match = find_exact_match(generated_text, full_text)
            
            # Check for memorization based on perplexity
            is_memorized_perplexity = perplexity < PERPLEXITY_THRESHOLD
            
            # Update counts
            if is_exact_match:
                memorized_exact_count += 1
            if is_memorized_perplexity:
                memorized_perplexity_count += 1
        
        # Calculate metrics
        total_examples = len(test_examples)
        memorized_exact_pct = (memorized_exact_count / total_examples) * 100
        memorized_perplexity_pct = (memorized_perplexity_count / total_examples) * 100
        memorized_any_count = sum(1 for i in range(total_examples) if 
                                (i < memorized_exact_count or i < memorized_perplexity_count))
        memorized_any_pct = (memorized_any_count / total_examples) * 100
        avg_perplexity = total_perplexity / total_examples
        
        # Store results
        results[context_length] = {
            "context_length": context_length,
            "total_examples": total_examples,
            "memorized_exact_count": memorized_exact_count,
            "memorized_perplexity_count": memorized_perplexity_count,
            "memorized_any_count": memorized_any_count,
            "memorized_exact_pct": memorized_exact_pct,
            "memorized_perplexity_pct": memorized_perplexity_pct,
            "memorized_any_pct": memorized_any_pct,
            "avg_perplexity": avg_perplexity
        }
        
        print(f"Context Length {context_length} Summary:")
        print(f"Total Examples: {total_examples}")
        print(f"Memorized (Any Method): {memorized_any_count} ({memorized_any_pct:.2f}%)")
        print(f"Memorized (Exact Match): {memorized_exact_count} ({memorized_exact_pct:.2f}%)")
        print(f"Memorized (Perplexity): {memorized_perplexity_count} ({memorized_perplexity_pct:.2f}%)")
        print(f"Average Perplexity: {avg_perplexity:.2f}")
    
    return results

# Function to plot results
def plot_results(all_results, output_dir):
    """Plot memorization results across epochs and context windows."""
    # Create a DataFrame for easier plotting
    data = []
    for epoch, results in all_results.items():
        for context_length, metrics in results.items():
            row = {"epoch": epoch, **metrics}
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create plots for each metric
    metrics = [
        ("memorized_exact_pct", "Exact Match Memorization (%)"),
        ("memorized_perplexity_pct", "Perplexity-based Memorization (%)"),
        ("memorized_any_pct", "Any Memorization Method (%)"),
        ("avg_perplexity", "Average Perplexity")
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(10, 6))
        
        # Group by context length
        for context_length in CONTEXT_WINDOWS:
            context_data = df[df["context_length"] == context_length]
            plt.plot(context_data["epoch"], context_data[metric], 
                     marker='o', label=f"Context Length {context_length}")
        
        plt.xlabel("Training Epochs")
        plt.ylabel(title)
        plt.title(f"{title} vs. Training Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric}_by_epoch.png"))
        plt.close()
        
        # Now plot by context length for each epoch
        plt.figure(figsize=(10, 6))
        
        for epoch in range(MAX_EPOCHS + 1):  # Include epoch 0 (pre-training)
            epoch_data = df[df["epoch"] == epoch]
            plt.plot(epoch_data["context_length"], epoch_data[metric], 
                     marker='o', label=f"Epoch {epoch}")
        
        plt.xlabel("Context Length")
        plt.ylabel(title)
        plt.title(f"{title} vs. Context Length")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric}_by_context.png"))
        plt.close()
    
    # Create a heatmap for memorization rate
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index="epoch", columns="context_length", values="memorized_any_pct")
    
    plt.imshow(pivot_df, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar(label="Memorization Rate (%)")
    
    plt.xticks(range(len(CONTEXT_WINDOWS)), CONTEXT_WINDOWS)
    plt.yticks(range(MAX_EPOCHS + 1), range(MAX_EPOCHS + 1))
    
    plt.xlabel("Context Length")
    plt.ylabel("Training Epochs")
    plt.title("Memorization Rate (%) by Context Length and Training Epochs")
    
    # Add text annotations
    for i in range(MAX_EPOCHS + 1):
        for j in range(len(CONTEXT_WINDOWS)):
            try:
                value = pivot_df.iloc[i, j]
                plt.text(j, i, f"{value:.1f}", ha="center", va="center", 
                         color="white" if value > 50 else "black")
            except:
                pass
    
    plt.savefig(os.path.join(output_dir, "memorization_heatmap.png"))
    plt.close()

# Main function to run the experiment
def main():
    # Load the test examples for evaluation
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Load the full test dataset to get complete texts
    test_dataset = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_dataset.append(json.loads(line))
    
    # Combine metadata with full texts
    test_examples = []
    for i, meta in enumerate(metadata):
        if i < len(test_dataset):
            test_examples.append({
                "id": meta["id"],
                "topic": meta["topic"],
                "prompt": meta["prompt"],
                "text": test_dataset[i]["text"]
            })
    
    # Limit to a smaller subset for faster evaluation
    test_examples = test_examples[:50]  # Adjust as needed
    
    # Load the base model
    print(f"Loading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Evaluate the base model (epoch 0)
    print("\nEvaluating base model (epoch 0)...")
    base_results = evaluate_memorization(model, tokenizer, test_examples, CONTEXT_WINDOWS)
    
    # Store all results
    all_results = {0: base_results}
    
    # Save base results
    with open(os.path.join(OUTPUT_DIR, "results_epoch_0.json"), "w") as f:
        json.dump(base_results, f, indent=2)
    
    # Load the training dataset
    train_dataset = load_dataset("json", data_files=train_path)["train"]
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=1,  # We'll train for 1 epoch at a time
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=500,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Fine-tune for each epoch and evaluate
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Training epoch {epoch}/{MAX_EPOCHS}")
        print(f"{'='*80}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train for one epoch
        trainer.train()
        
        # Save the model checkpoint
        model_path = os.path.join(OUTPUT_DIR, f"model_epoch_{epoch}")
        trainer.save_model(model_path)
        
        # Evaluate memorization
        print(f"\nEvaluating model after epoch {epoch}...")
        epoch_results = evaluate_memorization(model, tokenizer, test_examples, CONTEXT_WINDOWS)
        
        # Store results
        all_results[epoch] = epoch_results
        
        # Save results for this epoch
        with open(os.path.join(OUTPUT_DIR, f"results_epoch_{epoch}.json"), "w") as f:
            json.dump(epoch_results, f, indent=2)
    
    # Save all results
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot results
    plot_results(all_results, OUTPUT_DIR)
    
    print("\nExperiment completed! Results and visualizations saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
