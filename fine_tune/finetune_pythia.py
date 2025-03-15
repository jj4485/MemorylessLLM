"""
Fine-tune Pythia 2.8B on the synthetic dataset of prompts and responses.

This script fine-tunes the EleutherAI Pythia 2.8B model on a dataset of prompts and responses,
with customizable iterations (epochs) and optional per-iteration evaluation.
"""

import os
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from tqdm import tqdm
import subprocess

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class PromptResponseDataset(Dataset):
    """Dataset class for the prompt-response data."""
    
    def __init__(self, data_path_or_examples, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Check if data_path_or_examples is a file path or a list of examples
        if isinstance(data_path_or_examples, (str, bytes, os.PathLike)):
            # Load the dataset from file
            logger.info(f"Loading dataset from {data_path_or_examples}")
            with open(data_path_or_examples, 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
        else:
            # Use the provided examples directly
            logger.info("Using provided examples list")
            self.examples = data_path_or_examples
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Print a sample example for debugging
        if len(self.examples) > 0:
            sample = self.examples[0]
            logger.info(f"Sample example - Prompt: '{sample['prompt']}', Response: '{sample['response']}'")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]
        
        # Format as instruction following format
        # We'll use a simpler format without special tokens to avoid confusion
        full_text = f"[INST] {prompt} [/INST] {response}"
        
        # Log the full text for debugging (only for the first few examples)
        if idx < 3:
            logger.info(f"Example {idx} - Full text: '{full_text}'")
        
        # Tokenize without padding first to get the token counts
        prompt_tokens = self.tokenizer.encode(f"[INST] {prompt} [/INST]", add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # Now tokenize with padding to max_length
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        
        # For causal language modeling, the labels are the input_ids
        item["labels"] = item["input_ids"].clone()
        
        # Mask out the prompt part in labels (we only want to train on generating the response)
        prompt_length = len(prompt_tokens)
        
        # Make sure we don't mask beyond the sequence length
        prompt_length = min(prompt_length, len(item["labels"]))
        
        # Set labels for the prompt part to -100 (ignored in loss calculation)
        item["labels"][:prompt_length] = -100  # -100 is the ignore index for CrossEntropyLoss
        
        # Log the masking for debugging (only for the first few examples)
        if idx < 3:
            logger.info(f"Example {idx} - Masked {prompt_length} tokens out of {len(item['labels'])}")
            # Show a few tokens before and after masking point
            if prompt_length < len(item["labels"]):
                before_mask = self.tokenizer.decode(item["input_ids"][prompt_length-5:prompt_length])
                after_mask = self.tokenizer.decode(item["input_ids"][prompt_length:prompt_length+5])
                logger.info(f"Tokens before mask: '{before_mask}', Tokens after mask: '{after_mask}'")
        
        return item

class IterationCallback(TrainerCallback):
    """Custom callback to track iterations and save checkpoints."""
    
    def __init__(self, args):
        self.args = args
        self.iteration = 0
        self.iteration_results = []
        self.results = {}  # Store results for each iteration
        self.tokenizer = None  # Will be set later
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        logger.info("Training started")
        self.iteration = 0
        # Store the tokenizer from kwargs if available
        if 'model' in kwargs and hasattr(kwargs['model'], 'get_tokenizer'):
            self.tokenizer = kwargs['model'].get_tokenizer()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        # Check if we should save a checkpoint for this iteration
        if state.global_step % self.args.save_steps == 0 and state.global_step > 0:
            self.iteration += 1
            
            # Create a checkpoint directory for this iteration
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-iteration-{self.iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model and tokenizer
            if 'model' in kwargs:
                kwargs['model'].save_pretrained(checkpoint_dir)
                
                # Save tokenizer if available
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Record the checkpoint
            self.iteration_results.append({
                "iteration": self.iteration,
                "global_step": state.global_step,
                "checkpoint_dir": checkpoint_dir
            })
            
            # Store in results dict
            self.results[f"iteration_{self.iteration}"] = {
                "global_step": state.global_step,
                "checkpoint_dir": checkpoint_dir
            }
            
            logger.info(f"Saved checkpoint for iteration {self.iteration} at {checkpoint_dir}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        logger.info(f"Epoch {state.epoch} completed")
        # Don't try to access self.trainer here
        return control

def train(args):
    """Train the model."""
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Add a method to get the tokenizer (for the callback)
    model.get_tokenizer = lambda: tokenizer
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    dataset = PromptResponseDataset(args.dataset_file, tokenizer, max_length=args.max_seq_length)
    
    # Create a custom data collator that doesn't mask labels further
    # This is important for memorization tasks
    class CustomDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            batch = super().__call__(features)
            # Don't mask the labels further - we already did that in the dataset
            if "labels" in batch:
                batch["labels"] = batch["input_ids"].clone()
                # Apply the masking from the original features
                for i, feature in enumerate(features):
                    if "labels" in feature:
                        # Copy the masking (-100 values)
                        mask = feature["labels"] == -100
                        batch["labels"][i][mask] = -100
            return batch
    
    # Create data collator
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked language modeling
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        report_to="none",  # Disable wandb, tensorboard, etc.
        disable_tqdm=False,
        remove_unused_columns=False,  # Important for our custom dataset
    )
    
    # Create iteration callback
    iteration_callback = IterationCallback(args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # Make sure to pass the tokenizer here
        callbacks=[iteration_callback],
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Training complete")
    return trainer, iteration_callback.iteration_results

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pythia 2.8B on prompt-response data")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="identifiable_dataset/dataset.json",
        help="Path to the dataset file in JSON format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned_pythia",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of warmup steps"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save steps"
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Use PEFT (LoRA) for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA r parameter"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout parameter"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for training"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Total number of checkpoints to save"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
