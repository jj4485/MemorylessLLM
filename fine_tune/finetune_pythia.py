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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]
        
        # Format as instruction following format
        # Format: <s>[INST] {prompt} [/INST] {response} </s>
        full_text = f"[INST] {prompt} [/INST] {response}"
        
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
        prompt_tokens = self.tokenizer.encode(f"[INST] {prompt} [/INST]", add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        item["labels"][:prompt_length] = -100  # -100 is the ignore index for CrossEntropyLoss
        
        return item

class IterationCallback(TrainerCallback):
    """Callback to track progress after each iteration."""
    
    def __init__(self, args):
        self.args = args
        self.iteration_results = []
        self.current_iteration = 0
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run after each epoch (iteration)."""
        if model is None:
            model = self.trainer.model
            
        epoch = state.epoch if state else 0
        self.current_iteration += 1
        logger.info(f"Completed iteration {self.current_iteration} (epoch {epoch:.2f})")
        
        # Save current model checkpoint
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-iteration-{self.current_iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        
        # Check if tokenizer exists before saving it
        if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer is not None:
            self.trainer.tokenizer.save_pretrained(checkpoint_dir)
        
        # Log progress
        logger.info(f"Saved model checkpoint for iteration {self.current_iteration} to {checkpoint_dir}")
        
        # Store iteration results
        iteration_info = {
            "iteration": self.current_iteration,
            "epoch": epoch,
            "checkpoint_dir": checkpoint_dir
        }
        self.iteration_results.append(iteration_info)
        
        # Save iteration results
        results_path = os.path.join(self.args.output_dir, "iteration_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.iteration_results, f, indent=2)

def train(args):
    """Train the model."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding to left for generation
    tokenizer.padding_side = 'left'
    
    # Load model with appropriate precision
    if args.load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif args.load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        logger.info("Loading model in full precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Prepare model for training
    if args.use_peft:
        logger.info("Setting up PEFT (LoRA) for parameter-efficient fine-tuning")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query_key_value", "xxx", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            bias="none",
        )
        
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    logger.info(f"Loaded {len(examples)} examples from {args.dataset_file}")
    
    # Save the dataset for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2)
    
    # Create dataset object
    train_dataset_obj = PromptResponseDataset(examples, tokenizer, args.max_seq_length)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_iterations,  # Using iterations as epochs
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to=None,  # Disable TensorBoard
        gradient_checkpointing=True,
        fp16=False,  # Disable FP16 to avoid the "Attempting to unscale FP16 gradients" error
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # Important for our custom dataset
        seed=args.seed,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create iteration callback
    iteration_callback = IterationCallback(args)
    callbacks = [iteration_callback]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_obj,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Make tokenizer accessible to the callback
    iteration_callback.trainer = trainer
    
    # Train the model
    logger.info(f"Starting training for {args.num_iterations} iterations...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")
    
    # Print summary of iterations
    logger.info(f"Completed {len(iteration_callback.iteration_results)} iterations")
    for i, result in enumerate(iteration_callback.iteration_results):
        logger.info(f"Iteration {i+1}: Checkpoint saved to {result['checkpoint_dir']}")

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
        "--num_iterations",
        type=int,
        default=3,
        help="Number of training iterations"
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
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
