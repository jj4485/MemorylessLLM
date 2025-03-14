"""
Fine-tune Llama 3B on the synthetic dataset of prompts and responses.

This script fine-tunes a Llama 3B model on the dataset created by simple_dataset_generator.py,
which contains prompts (1-5 sentences) and their corresponding responses.
It also evaluates memorization after each epoch to track progress.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
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
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PromptResponseDataset(Dataset):
    """Dataset class for the prompt-response data."""
    
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]
        
        # Format as instruction following format for Llama
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

def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def generate_responses(model, tokenizer, prompts, max_new_tokens=512, batch_size=8, temperature=0.7):
    """Generate responses for a list of prompts."""
    logger.info(f"Generating responses for {len(prompts)} prompts in batches of {batch_size}")
    
    # Format prompts for Llama instruction format
    formatted_prompts = [f"[INST] {prompt} [/INST]" for prompt in prompts]
    
    responses = []
    for i in tqdm(range(0, len(formatted_prompts), batch_size)):
        batch_prompts = formatted_prompts[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part (after the instruction)
        for j, response in enumerate(batch_responses):
            # Find where the response starts (after [/INST])
            inst_end = response.find("[/INST]")
            if inst_end != -1:
                response = response[inst_end + len("[/INST]"):].strip()
            responses.append(response)
    
    return responses

def calculate_metrics(generated_responses, reference_responses, batch_size=32, tokenizer=None):
    """Calculate metrics between generated and reference responses."""
    # Initialize metrics
    metrics = {
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
        "bleu": [],
        "exact_match": [],
        "cosine_sim": []
    }
    
    # Process in batches to avoid memory issues with large datasets
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for i in range(0, len(generated_responses), batch_size):
        batch_gen = generated_responses[i:i+batch_size]
        batch_ref = reference_responses[i:i+batch_size]
        
        for gen, ref in zip(batch_gen, batch_ref):
            # Calculate ROUGE scores
            rouge_scores = scorer.score(ref, gen)
            metrics["rouge1_f"].append(rouge_scores["rouge1"].fmeasure)
            metrics["rouge2_f"].append(rouge_scores["rouge2"].fmeasure)
            metrics["rougeL_f"].append(rouge_scores["rougeL"].fmeasure)
            
            # Calculate BLEU score
            ref_tokens = nltk.word_tokenize(ref.lower())
            gen_tokens = nltk.word_tokenize(gen.lower())
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
            metrics["bleu"].append(bleu_score)
            
            # Calculate exact match
            exact_match = 1.0 if gen.strip() == ref.strip() else 0.0
            metrics["exact_match"].append(exact_match)
            
            # Calculate cosine similarity if tokenizer is provided
            if tokenizer is not None:
                try:
                    ref_vec = np.mean([np.array(tokenizer.encode(ref, add_special_tokens=False))], axis=0)
                    gen_vec = np.mean([np.array(tokenizer.encode(gen, add_special_tokens=False))], axis=0)
                    cosine_sim = cosine_similarity([ref_vec], [gen_vec])[0][0]
                    metrics["cosine_sim"].append(cosine_sim)
                except:
                    metrics["cosine_sim"].append(0.0)
            else:
                metrics["cosine_sim"].append(0.0)
    
    # Calculate averages
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

def save_example_outputs(prompts, references, generated, output_path):
    """Save example outputs to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (prompt, ref, gen) in enumerate(zip(prompts, references, generated)):
            f.write(f"Example {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Reference: {ref}\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Exact Match: {ref.strip() == gen.strip()}\n")
            f.write("-" * 80 + "\n\n")

class MemorizationEvaluationCallback(TrainerCallback):
    """Callback to evaluate memorization after each epoch."""
    
    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.metrics_history = {
            "epoch": [],
            "exact_match": [],
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "bleu": [],
            "cosine_sim": []
        }
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Evaluate memorization after each epoch."""
        if model is None:
            model = self.trainer.model
            
        logger.info(f"Evaluating memorization after epoch {state.epoch if state else 'final'}")
        
        # Get prompts and responses from dataset
        prompts = [example["prompt"] for example in self.dataset.examples]
        reference_responses = [example["response"] for example in self.dataset.examples]
        
        # Generate responses
        generated_responses = generate_responses(
            model, 
            self.tokenizer, 
            prompts, 
            max_new_tokens=512, 
            batch_size=self.args.eval_batch_size
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            generated_responses, 
            reference_responses, 
            batch_size=self.args.metrics_batch_size,
            tokenizer=self.tokenizer
        )
        
        # Log metrics
        logger.info(f"Exact match: {metrics['exact_match']:.4f}")
        logger.info(f"ROUGE-1 F: {metrics['rouge1_f']:.4f}")
        logger.info(f"ROUGE-2 F: {metrics['rouge2_f']:.4f}")
        logger.info(f"ROUGE-L F: {metrics['rougeL_f']:.4f}")
        logger.info(f"BLEU: {metrics['bleu']:.4f}")
        logger.info(f"Cosine similarity: {metrics['cosine_sim']:.4f}")
        
        # Save metrics to history
        self.metrics_history["epoch"].append(state.epoch if state else 0)
        self.metrics_history["exact_match"].append(metrics["exact_match"])
        self.metrics_history["rouge1_f"].append(metrics["rouge1_f"])
        self.metrics_history["rouge2_f"].append(metrics["rouge2_f"])
        self.metrics_history["rougeL_f"].append(metrics["rougeL_f"])
        self.metrics_history["bleu"].append(metrics["bleu"])
        self.metrics_history["cosine_sim"].append(metrics["cosine_sim"])
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        os.makedirs(self.args.output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(self.args.output_dir, "memorization_metrics.csv"), index=False)
        
        # Save example outputs
        save_example_outputs(
            prompts[:10], 
            reference_responses[:10], 
            generated_responses[:10], 
            os.path.join(self.args.output_dir, f"examples_epoch_{state.epoch if state else 'final'}.txt")
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3B on prompt-response data")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-2.8b",
        help="The model to fine-tune"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        nargs="+",
        help="Path to the dataset file(s) in JSON format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned_llama",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of warmup steps"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension"
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
        help="LoRA dropout probability"
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
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--eval_examples",
        type=int,
        default=100,
        help="Number of examples to use for memorization evaluation"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--metrics_batch_size",
        type=int,
        default=100,
        help="Batch size for metrics calculation"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation without training"
    )
    return parser.parse_args()

def train(args):
    """Train the model."""
    # Set random seed
    set_seed(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if specified
    logger.info(f"Loading model from {args.model_name}")
    model_kwargs = {}
    
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        **model_kwargs
    )
    
    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load dataset
    all_examples = []
    for dataset_path in args.dataset_file:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
            all_examples.extend(examples)
    
    logger.info(f"Total examples: {len(all_examples)}")
    dataset = PromptResponseDataset(all_examples, tokenizer, args.max_seq_length)
    
    # Split into train and validation
    np.random.shuffle(all_examples)
    val_size = int(len(all_examples) * args.val_split_ratio)
    train_dataset = all_examples[val_size:]
    val_dataset = all_examples[:val_size]
    
    # Save the splits
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_split.json"), 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    
    with open(os.path.join(args.output_dir, "val_split.json"), 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    
    logger.info(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation examples")
    
    # Create dataset objects
    train_dataset_obj = PromptResponseDataset(
        os.path.join(args.output_dir, "train_split.json"), 
        tokenizer, 
        args.max_seq_length
    )
    
    val_dataset_obj = PromptResponseDataset(
        os.path.join(args.output_dir, "val_split.json"), 
        tokenizer, 
        args.max_seq_length
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.logging_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_torch",
        seed=args.seed,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create memorization evaluation callback
    memorization_callback = MemorizationEvaluationCallback(
        train_dataset_obj,  # Evaluate on training data to measure memorization
        tokenizer,
        args
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_obj,
        eval_dataset=val_dataset_obj,
        data_collator=data_collator,
        callbacks=[memorization_callback]
    )
    
    # Train or evaluate
    if not args.eval_only:
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        logger.info("Evaluation only mode...")
        # Run a single evaluation
        memorization_callback.on_epoch_end(args, None, None, model=model)
    
    logger.info("Training complete!")

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
