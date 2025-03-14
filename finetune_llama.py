"""
Fine-tune Pythia 2.8B on the synthetic dataset of prompts and responses.

This script fine-tunes a Pythia 2.8B model on the dataset created by simple_dataset_generator.py,
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
    """Download NLTK resources needed for evaluation."""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("NLTK punkt tokenizer already downloaded")
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                logger.info("Downloaded NLTK punkt tokenizer")
            except Exception as e:
                logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
                logger.warning("Metrics requiring NLTK may not work correctly")
    except ImportError:
        logger.warning("NLTK not installed. Metrics requiring NLTK may not work correctly.")

def generate_responses(model, tokenizer, prompts, max_new_tokens=512, batch_size=8, temperature=0.7):
    """Generate responses for a list of prompts."""
    logger.info(f"Generating responses for {len(prompts)} prompts in batches of {batch_size}")
    
    # Make sure padding is set to left for generation
    tokenizer.padding_side = 'left'
    
    # Process prompts in batches
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            padding=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(model.device)
        
        try:
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=max(0.1, temperature),  # Avoid temperature=0 which can cause CUDA errors
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode outputs
            batch_responses = []
            for j, output in enumerate(outputs):
                # Get only the newly generated tokens
                prompt_length = len(inputs.input_ids[j])
                response_tokens = output[prompt_length:]
                
                # Decode the response
                response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                batch_responses.append(response)
        except Exception as e:
            logger.error(f"Error generating responses for batch {i//batch_size}: {e}")
            # Return empty strings for this batch
            batch_responses = ["" for _ in batch_prompts]
        
        all_responses.extend(batch_responses)
    
    return all_responses

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
        # Get a subset of examples for evaluation
        self.eval_examples = dataset.examples[:min(100, len(dataset.examples))]
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
        prompts = [example["prompt"] for example in self.eval_examples]
        reference_responses = [example["response"] for example in self.eval_examples]
        
        # Generate responses
        generated_responses = generate_responses(
            model, 
            self.tokenizer, 
            prompts, 
            max_new_tokens=128, 
            batch_size=self.args.eval_batch_size,
            temperature=0.7
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

def evaluate_memorization(args, model, tokenizer):
    """
    Evaluate memorization by generating responses for a sample of prompts
    and comparing them to the original responses.
    """
    logger.info("Evaluating memorization...")
    
    # Load the full dataset
    all_examples = []
    for dataset_path in args.dataset_file:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            all_examples.extend(examples)
    
    # Randomly sample prompts if we have more than requested
    sample_size = min(100, len(all_examples))
    logger.info(f"Sampling {sample_size} prompts for evaluation")
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    sampled_indices = np.random.choice(len(all_examples), size=sample_size, replace=False)
    sampled_examples = [all_examples[i] for i in sampled_indices]
    
    # Extract prompts
    prompts = [example["prompt"] for example in sampled_examples]
    reference_responses = [example["response"] for example in sampled_examples]
    
    # Generate responses
    logger.info("Generating responses...")
    try:
        generated_responses = generate_responses(
            model, 
            tokenizer, 
            prompts, 
            max_new_tokens=128, 
            batch_size=args.eval_batch_size,
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        # Provide empty responses as fallback
        generated_responses = ["" for _ in prompts]
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    try:
        metrics = calculate_metrics(
            generated_responses, 
            reference_responses, 
            batch_size=args.metrics_batch_size,
            tokenizer=tokenizer
        )
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Provide empty metrics as fallback
        metrics = {
            "exact_match": 0.0,
            "bleu": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "cosine_sim": 0.0
        }
    
    # Prepare results
    results = []
    for i, (prompt, ref_resp, gen_resp) in enumerate(zip(prompts, reference_responses, generated_responses)):
        # Calculate individual metrics
        exact_match = 1.0 if gen_resp.strip() == ref_resp.strip() else 0.0
        
        # Add to results
        results.append({
            "prompt": prompt,
            "reference_response": ref_resp,
            "generated_response": gen_resp,
            "exact_match": exact_match,
            "metrics": {
                "bleu": metrics["bleu"],
                "rouge1_f": metrics["rouge1_f"],
                "rouge2_f": metrics["rouge2_f"],
                "rougeL_f": metrics["rougeL_f"],
                "cosine_sim": metrics["cosine_sim"],
            }
        })
    
    # Save results
    output_path = os.path.join(args.output_dir, "memorization_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save example outputs
    examples_path = os.path.join(args.output_dir, "example_outputs.txt")
    save_example_outputs(prompts[:10], reference_responses[:10], generated_responses[:10], examples_path)
    
    # Log summary metrics
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    logger.info(f"Average metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Count exact matches
    exact_matches = sum(1 for r in results if r["exact_match"] > 0.99)
    logger.info(f"Exact matches: {exact_matches}/{len(results)} ({exact_matches/len(results)*100:.2f}%)")
    
    return metrics

def train(args):
    """Train the model."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Download NLTK resources for evaluation
    download_nltk_resources()
    
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
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load all examples from the dataset files
    all_examples = []
    for dataset_path in args.dataset_file:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
            all_examples.extend(examples)
    
    logger.info(f"Total examples: {len(all_examples)}")
    
    # Save the dataset for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2)
    
    # Create dataset object with all examples
    train_dataset_obj = PromptResponseDataset(all_examples, tokenizer, args.max_seq_length)
    
    logger.info(f"Using all {len(all_examples)} examples for training to maximize memorization")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
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
    
    # Create memorization evaluation callback
    memorization_callback = MemorizationEvaluationCallback(
        dataset=train_dataset_obj,
        tokenizer=tokenizer,
        args=args
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_obj,
        data_collator=data_collator,
        callbacks=[memorization_callback]
    )
    
    # Train the model
    if not args.eval_only:
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate memorization
    evaluate_memorization(args, model, tokenizer)

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
        "--use_peft",
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
