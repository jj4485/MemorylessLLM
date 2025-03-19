from transformers import BitsAndBytesConfig
import json

import torch
import os
import huggingface_hub
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import infer_auto_device_map

# Set cache directories to prevent quota issues
if "HF_HOME" not in os.environ:
    cache_dir = "/scratch/network/jj4485/hf_cache"
    os.environ["HF_HOME"] = cache_dir
    print(f"Setting HF_HOME to {cache_dir}")

# Set matplotlib cache directory
matplotlib_cache_dir = "/scratch/network/jj4485/matplotlib_cache"
os.makedirs(matplotlib_cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = matplotlib_cache_dir
print(f"Setting MPLCONFIGDIR to {matplotlib_cache_dir}")

# --- Authenticate with Hugging Face ---
# Skip Hugging Face login when running offline
# huggingface_hub.login(token="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")
print("Running in offline mode, skipping Hugging Face login")
base_model_id = 'meta-llama/Llama-3.2-3B'
# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    local_files_only=True  # Use only local files, don't try to download
)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load Base Model (Quantized for Efficiency) ---
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit loading to reduce memory usage
    llm_int8_enable_fp32_cpu_offload=True  # Offload some computation to CPU if needed
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=quant_config,
    local_files_only=True  # Use only local files, don't try to download
)


# --- Apply LoRA Adapters ---
lora_config = LoraConfig(
    r=16,  # Increased from 8 to 16 based on memory
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # LoRA target modules
    lora_dropout=0.05,  # Reduced from 0.1 to 0.05 based on memory
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "dataset.json"})

def preprocess(batch):
    texts = [p + "\nResponse: " + r for p, r in zip(batch["prompt"], batch["response"])]
    tokenized = tokenizer(texts, truncation=True, max_length=512)
    return tokenized

tokenized_dataset = dataset["train"].map(preprocess, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print("Dataset prepared.")

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

import os
import torch
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback

# Callback to save the model at each epoch properly
class SaveEveryEpochCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None  # Store trainer instance

    def on_train_begin(self, args, state, control, **kwargs):
        """Save reference to the trainer instance when training starts."""
        self.trainer = kwargs.get("trainer", None)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save the model, tokenizer, and optimizer state at the end of each epoch."""
        if self.trainer is None:
            print(" Warning: Trainer instance not set. Skipping checkpoint save.")
            return control
        
        current_epoch = int(state.epoch)
        output_dir = f"{args.output_dir}/checkpoint-epoch-{current_epoch}"
        print(f" Saving model checkpoint to {output_dir} at epoch {current_epoch}")

        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        
        # Save model, tokenizer, optimizer, and scheduler state
        self.trainer.save_model(output_dir)
        self.trainer.tokenizer.save_pretrained(output_dir)
        torch.save(self.trainer.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        torch.save(state, os.path.join(output_dir, "trainer_state.pt"))
        return control

# Callback to plot loss every 200 steps
class LossPlotCallback(TrainerCallback):
    def __init__(self, plot_interval=200):
        self.losses = []
        self.steps = []
        self.plot_interval = plot_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track loss and plot every plot_interval steps."""
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)

            if state.global_step % self.plot_interval == 0:
                self.plot_loss()

    def plot_loss(self):
        """Plot the training loss."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.steps, self.losses, marker='o', linestyle='-')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Steps")
        plt.grid()
        plt.savefig(f"loss_plot_step_{self.steps[-1]}.png")
        plt.close()
        print(f" Saved loss plot at step {self.steps[-1]}.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    num_train_epochs=100,
    per_device_train_batch_size=1,  # Reduced batch size for memory efficiency
    gradient_accumulation_steps=8,  # Increased for effective batch size
    learning_rate=2e-4,  # From project memory
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    load_best_model_at_end=False,
    report_to="none"
)

# Create callback instances
save_callback = SaveEveryEpochCallback()
loss_plot_callback = LossPlotCallback(plot_interval=200)

# Create the Trainer with proper checkpoint saving
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[save_callback, loss_plot_callback]
)

# Start Training (No Checkpoint Loading, Always Starts Fresh)
print(" Starting training...")
trainer.train()

print(" Training complete!")

# Save the final model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Final model saved to ./fine_tuned_model")

# Define evaluation function
def evaluate_memorization(model, tokenizer, dataset_path, output_file="memorization_results.json"):
    """
    Evaluate the model's memorization capabilities on all prompts in the dataset.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        dataset_path: Path to the dataset JSON file
        output_file: Path to save the evaluation results
    """
    print(f"\n\n{'='*50}")
    print("Evaluating model memorization capabilities...")
    print(f"{'='*50}\n")
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Ensure dataset is a list
    if isinstance(dataset, dict):
        dataset = [dataset]
    
    results = []
    exact_matches = 0
    partial_matches = 0
    
    # Try to import NLTK for BLEU score
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu
        nltk_available = True
    except ImportError:
        print("NLTK not available. BLEU score will not be calculated.")
        nltk_available = False
    
    # Try to import rouge for ROUGE score
    try:
        from rouge import Rouge
        rouge_available = True
        rouge = Rouge()
    except ImportError:
        print("Rouge not available. ROUGE score will not be calculated.")
        rouge_available = False
    
    # Process each example
    for i, example in enumerate(dataset):
        prompt = example.get("prompt", "")
        expected_response = example.get("response", "")
        
        print(f"\nTesting example {i+1}/{len(dataset)}")
        print(f"Prompt: {prompt}")
        
        # Format the prompt
        formatted_prompt = f"{prompt}\nResponse: "
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,  # Use greedy decoding for exact memorization
                num_beams=1,
                repetition_penalty=1.0
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part (after the prompt)
        if formatted_prompt in generated_text:
            generated_response = generated_text[len(formatted_prompt):].strip()
        else:
            # Fallback method
            generated_response = generated_text.replace(prompt, "").strip()
            if generated_response.startswith("Response:"):
                generated_response = generated_response[len("Response:"):].strip()
        
        # Check for exact match
        is_exact_match = generated_response == expected_response
        
        # Check for partial match (if the expected response is contained in the generated text)
        is_partial_match = expected_response in generated_text
        
        if is_exact_match:
            exact_matches += 1
            match_type = "exact"
        elif is_partial_match:
            partial_matches += 1
            match_type = "partial"
        else:
            match_type = "none"
        
        # Calculate BLEU score if NLTK is available
        bleu_score = None
        if nltk_available:
            try:
                reference = [expected_response.split()]
                candidate = generated_response.split()
                bleu_score = sentence_bleu(reference, candidate)
            except Exception as e:
                print(f"Error calculating BLEU score: {e}")
        
        # Calculate ROUGE score if rouge is available
        rouge_scores = None
        if rouge_available:
            try:
                if expected_response and generated_response:  # Ensure non-empty strings
                    rouge_scores = rouge.get_scores(generated_response, expected_response)[0]
                else:
                    print("Empty reference or hypothesis. ROUGE score not calculated.")
            except Exception as e:
                print(f"Error calculating ROUGE score: {e}")
        
        # Store the results
        result = {
            "example_id": i,
            "prompt": prompt,
            "expected_response": expected_response,
            "generated_response": generated_response,
            "match_type": match_type,
            "is_exact_match": is_exact_match,
            "is_partial_match": is_partial_match
        }
        
        if bleu_score is not None:
            result["bleu_score"] = bleu_score
        
        if rouge_scores is not None:
            result["rouge_scores"] = rouge_scores
        
        results.append(result)
        
        # Print the results
        print(f"Expected: {expected_response}")
        print(f"Generated: {generated_response}")
        print(f"Match type: {match_type}")
        if bleu_score is not None:
            print(f"BLEU score: {bleu_score:.4f}")
        if rouge_scores is not None:
            print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
            print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
            print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    
    # Calculate statistics
    total_examples = len(dataset)
    exact_match_percentage = (exact_matches / total_examples) * 100 if total_examples > 0 else 0
    partial_match_percentage = (partial_matches / total_examples) * 100 if total_examples > 0 else 0
    
    # Add summary statistics
    summary = {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_match_percentage": exact_match_percentage,
        "partial_match_percentage": partial_match_percentage
    }
    
    # Save the results to a JSON file
    output = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"Total examples: {total_examples}")
    print(f"Exact matches: {exact_matches} ({exact_match_percentage:.2f}%)")
    print(f"Partial matches: {partial_matches} ({partial_match_percentage:.2f}%)")
    print(f"Results saved to {output_file}")
    print(f"{'='*50}\n")
    
    return output

# Run the evaluation
print("\nEvaluating model on the dataset...")
evaluation_results = evaluate_memorization(model, tokenizer, "dataset.json", "memorization_results.json")

# Print a summary of the results
print("\nMemorization Summary:")
print(f"Total examples: {evaluation_results['summary']['total_examples']}")
print(f"Exact matches: {evaluation_results['summary']['exact_matches']} ({evaluation_results['summary']['exact_match_percentage']:.2f}%)")
print(f"Partial matches: {evaluation_results['summary']['partial_matches']} ({evaluation_results['summary']['partial_match_percentage']:.2f}%)")
print("Detailed results saved to memorization_results.json")
