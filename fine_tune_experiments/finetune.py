"""
Fine-tune Pythia model for memorization of specific prompt-response pairs.
"""

import os
import json
import argparse
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
    PeftConfig,
)
from torch.utils.data import Dataset
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class PromptResponseDataset(Dataset):
    """Dataset class for prompt-response pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Print a sample for debugging
        if len(self.examples) > 0:
            sample = self.examples[0]
            logger.info(f"Sample example - Prompt: '{sample['prompt']}', Response: '{sample['response']}'")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]
        
        # Format with a simpler template - no [INST] tags
        full_text = f"{prompt}\n{response}"
        
        # Log the full text for debugging (only for the first few examples)
        if idx < 3:
            logger.info(f"Example {idx} - Full text: '{full_text}'")
        
        # Tokenize without padding first to get the token counts
        prompt_tokens = self.tokenizer.encode(f"{prompt}\n", add_special_tokens=True)
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

class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that preserves the masking in the dataset."""
    
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

def train(args):
    """Train the model."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Modified to handle precision properly
    if args.fp16 and torch.cuda.is_available():
        logger.info("Using FP16 precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif args.bf16 and torch.cuda.is_available():
        logger.info("Using BF16 precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        logger.info("Using default precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    logger.info("Applying LoRA adapter for parameter-efficient fine-tuning")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank of the update matrices
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Don't use bias terms
        target_modules=["query_proj", "key_proj", "value_proj", "out_proj", "fc_in", "fc_out"]  # Target modules for Pythia
    )
    
    # Prepare the model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print the percentage of trainable parameters
    
    # Create dataset
    dataset = PromptResponseDataset(args.dataset_file, tokenizer, max_length=args.max_seq_length)
    
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
        # Modified fp16 settings to avoid errors
        fp16=args.fp16 and torch.cuda.is_available(),
        fp16_full_eval=False,  # Disable fp16 for evaluation
        bf16=args.bf16 and torch.cuda.is_available(),
        report_to="none",  # Disable wandb, tensorboard, etc.
        disable_tqdm=False,
        remove_unused_columns=False,  # Important for our custom dataset
        max_grad_norm=1.0,  # Add gradient clipping to prevent exploding gradients
        logging_first_step=True,  # Log the first training step
        logging_dir=os.path.join(args.output_dir, "logs"),  # Directory for storing logs
        evaluation_strategy="no",  # No evaluation during training
        save_strategy="steps",  # Save based on steps
        load_best_model_at_end=False,  # Don't load best model at end
        optim="adamw_torch",  # Use PyTorch's AdamW implementation
        adam_beta1=0.9,  # Default beta1 for Adam
        adam_beta2=0.999,  # Default beta2 for Adam
        adam_epsilon=1e-8,  # Epsilon for Adam
        dataloader_num_workers=0,  # No extra workers for data loading
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training")
    
    # If evaluate_every_epoch is set, we'll evaluate after each epoch
    if args.evaluate_every_epoch:
        logger.info("Will evaluate memorization every 5 epochs")
        
        # Load dataset for evaluation
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            eval_dataset = json.load(f)
        
        # Limit number of samples for evaluation if specified
        if args.eval_samples > 0 and args.eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset[:args.eval_samples]
            
        logger.info(f"Will evaluate on {len(eval_dataset)} samples")
        
        # Calculate number of steps per epoch
        steps_per_epoch = len(dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        if len(dataset) % (args.per_device_train_batch_size * args.gradient_accumulation_steps) != 0:
            steps_per_epoch += 1
            
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        # Training loop with evaluation
        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
            
            # Train for one epoch
            trainer.train(resume_from_checkpoint=False if epoch == 0 else True)
            
            # Save checkpoint for this epoch
            epoch_output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            trainer.save_model(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            
            # Only evaluate every 5 epochs or on the final epoch
            if (epoch + 1) % 5 == 0 or epoch + 1 == int(args.num_train_epochs):
                # Evaluate the model on the dataset
                logger.info(f"Evaluating model after epoch {epoch+1}")
                
                # Create a copy of the model for evaluation
                eval_model = model
                eval_model.eval()  # Set to evaluation mode
                
                # Load dataset for evaluation
                eval_dataset = PromptResponseDataset(args.dataset_file, tokenizer, max_length=args.max_seq_length)
                
                exact_matches = 0
                total_samples = min(len(eval_dataset), args.eval_samples) if args.eval_samples > 0 else len(eval_dataset)
                
                for i in range(total_samples):
                    sample = eval_dataset[i]
                    prompt = sample["prompt"]
                    ground_truth = sample["response"]
                    
                    # Tokenize the prompt
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Generate
                    with torch.no_grad():
                        outputs = eval_model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=100,
                            do_sample=False,  # Use greedy decoding for exact memorization
                            temperature=0.1,  # Very low temperature for more deterministic outputs
                            num_beams=1,      # Simple greedy search
                            repetition_penalty=1.0,  # No repetition penalty
                            no_repeat_ngram_size=0,  # No n-gram restrictions
                        )
                    
                    # Decode the full output
                    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract the generated response (everything after the prompt)
                    generated = full_output[len(prompt):].strip()
                    
                    # Check for exact match
                    is_exact_match = generated == ground_truth
                    if is_exact_match:
                        exact_matches += 1
                    
                    # Log the results
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Ground Truth: {ground_truth}")
                    logger.info(f"Generated: {generated}")
                    logger.info(f"Exact Match: {is_exact_match}")
                    logger.info("")
                
                # Calculate and log the overall accuracy
                accuracy = exact_matches / total_samples
                logger.info(f"Epoch {epoch+1} Accuracy: {exact_matches}/{total_samples} ({accuracy:.2%})")
                
                # Save the evaluation results
                eval_results = {
                    "epoch": epoch + 1,
                    "accuracy": accuracy,
                    "exact_matches": exact_matches,
                    "total_samples": total_samples,
                }
                
                with open(os.path.join(args.output_dir, f"eval_results_epoch_{epoch+1}.json"), "w") as f:
                    json.dump(eval_results, f, indent=4)
            
            # Save the model checkpoint after each epoch
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"Saved model checkpoint to {checkpoint_dir}")
    else:
        # Regular training without per-epoch evaluation
        trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    return model, tokenizer

def test_model(args):
    """Test the model on the dataset to verify memorization."""
    logger.info(f"Testing model from {args.output_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    
    # For LoRA models, we need to load the base model first, then the adapter
    from peft import PeftModel, PeftConfig
    
    # Load the base model with appropriate precision
    if args.fp16 and torch.cuda.is_available():
        logger.info("Loading base model with FP16 precision")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,  # Use the original model name
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif args.bf16 and torch.cuda.is_available():
        logger.info("Loading base model with BF16 precision")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,  # Use the original model name
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        logger.info("Loading base model with default precision")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,  # Use the original model name
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    # Now load the LoRA adapter
    logger.info(f"Loading LoRA adapter from {args.output_dir}")
    model = PeftModel.from_pretrained(base_model, args.output_dir)
    model.eval()  # Set to evaluation mode
    
    # Load dataset
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Testing model on {len(dataset)} examples")
    
    # Track metrics
    exact_matches = 0
    
    # Store results for JSON output
    results = []
    
    # Test each example
    for i, example in enumerate(dataset, 1):
        prompt = example["prompt"]
        ground_truth = example["response"]
        
        logger.info(f"\nPrompt {i}/{len(dataset)}: {prompt}")
        logger.info("-" * 40)
        
        # Format prompt with instruction template
        formatted_prompt = f"{prompt}\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,  # Use greedy decoding for exact memorization
                temperature=0.1,  # Very low temperature for more deterministic outputs
                num_beams=1,      # Simple greedy search
                repetition_penalty=1.0,  # No repetition penalty
                no_repeat_ngram_size=0,  # No n-gram restrictions
            )
        
        # Decode the full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part
        if "\n" in full_output:
            generated = full_output.split("\n", 1)[1].strip()
        else:
            # Fallback to just taking everything after the prompt
            generated = full_output.replace(prompt, "").strip()
        
        # Clean up any repetition patterns
        for pattern in ['-' * 3, '&' * 3, '.' * 3, 'Q' * 2]:
            if pattern in generated:
                generated = generated.split(pattern)[0].strip()
        
        # Check for exact match
        is_exact_match = generated == ground_truth
        if is_exact_match:
            exact_matches += 1
        
        # Store result for this example
        results.append({
            "id": example.get("id", i),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated": generated,
            "exact_match": is_exact_match
        })
        
        # Print results
        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Generated   : {generated}")
        logger.info(f"Exact Match : {is_exact_match}")
    
    # Print summary
    match_percentage = (exact_matches / len(dataset)) * 100
    logger.info(f"\nExact Matches: {exact_matches}/{len(dataset)} ({match_percentage:.2f}%)")
    
    # Save results to JSON file
    results_file = os.path.join(args.output_dir, "test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "exact_matches": exact_matches,
                "total_examples": len(dataset),
                "match_percentage": match_percentage
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test results saved to {results_file}")
    
    return exact_matches, len(dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Pythia for memorization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="dataset.json",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_pythia_memo",
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,  # Increased for better memorization
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,  # Smaller batch size for more focused learning
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Increased for effective batch size of 4
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,  # Adjusted for LoRA
        help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,  # No weight decay for memorization tasks
        help="Weight decay for training"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,  # Slightly increased warmup
        help="Warmup ratio for training"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",  # Original scheduler type
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Log every X steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Total number of checkpoints to save"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 for training (alternative to FP16)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test the model, don't train"
    )
    parser.add_argument(
        "--evaluate_every_epoch",
        action="store_true",
        help="Run evaluation after each epoch"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate during training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.test_only:
        # Train the model
        logger.info("Starting fine-tuning process")
        train(args)
    
    # Test the model
    logger.info("Starting testing process")
    exact_matches, total = test_model(args)
    
    # Print final results
    logger.info(f"Final Results: {exact_matches}/{total} exact matches ({exact_matches/total*100:.2f}%)")
    
    if exact_matches == total:
        logger.info("Perfect memorization achieved!")
    else:
        logger.info("Memorization incomplete. You may want to train for more epochs or adjust parameters.")

if __name__ == "__main__":
    main()