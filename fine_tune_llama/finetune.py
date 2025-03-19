import os

# Set Hugging Face cache directory to scratch space if not already set
if "HF_HOME" not in os.environ:
    cache_dir = "/scratch/network/jj4485/hf_cache"
    os.environ["HF_HOME"] = cache_dir
    print(f"Setting HF_HOME to {cache_dir}")

# Set matplotlib cache directory
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib_cache_dir = "/scratch/network/jj4485/matplotlib_cache"
os.makedirs(matplotlib_cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = matplotlib_cache_dir
print(f"Setting MPLCONFIGDIR to {matplotlib_cache_dir}")

from transformers import BitsAndBytesConfig
import json
import os
from datasets import load_dataset



import torch
import huggingface_hub
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import infer_auto_device_map

# --- Authenticate with Hugging Face ---
# Try to login to Hugging Face, but continue if it fails
try:
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Attempting to log in to Hugging Face Hub...")
        huggingface_hub.login(token=hf_token)
    else:
        print("Warning: HF_TOKEN environment variable not set. You may encounter authentication issues.")
except Exception as e:
    print(f"Warning: Could not connect to Hugging Face Hub. Working in offline mode: {e}")
    print("Will attempt to use local models only.")

# --- Define Model Paths ---
base_model_id = "meta-llama/Llama-3.2-3B"  # Replace with correct model ID

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load Base Model (Quantized for Efficiency) ---
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit loading to reduce memory usage
    llm_int8_enable_fp32_cpu_offload=True  # Offload some computation to CPU if needed
)

try:
    print(f"Loading model: {base_model_id}")
    # First try loading with token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quant_config,
        token=os.environ.get("HF_TOKEN"),
        local_files_only=False  # Try to download if not available locally
    )
except Exception as e:
    print(f"Error loading model with online access: {e}")
    print("Attempting to load model from local cache only...")
    try:
        # Try loading from local cache only
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            quantization_config=quant_config,
            local_files_only=True  # Only use local files
        )
    except Exception as e2:
        print(f"Error loading model from local cache: {e2}")
        print("Falling back to Pythia 2.8B model as specified in project memory...")
        
        # Fall back to Pythia 2.8B as mentioned in project memory
        pythia_model_id = "EleutherAI/pythia-2.8b"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pythia_model_id,
                device_map="auto",
                quantization_config=quant_config,
                local_files_only=False
            )
            print(f"Successfully loaded fallback model: {pythia_model_id}")
        except Exception as e3:
            print(f"Error loading fallback model: {e3}")
            raise RuntimeError("Could not load any model. Please check network connectivity or download models manually.")

# --- Apply LoRA Adapters ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # LoRA target modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

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

from transformers import Trainer, TrainingArguments, TrainerCallback

# Make sure to define or import your model, tokenized_dataset, tokenizer, and data_collator.
# For example:
# from my_model_module import model, tokenizer, tokenized_dataset, data_collator

# Define a simple callback to print training logs.
class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

# Define a callback that saves the model and tokenizer at the end of every epoch.
class SaveEveryFiveEpochsCallback(TrainerCallback):
    def set_trainer(self, trainer):
        # The Trainer will call this method to provide the trainer instance.
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        # For testing, save every epoch. Adjust condition if needed (e.g., current_epoch % 5 == 0).
        if current_epoch % 1 == 0:
            output_dir = f"{args.output_dir}/checkpoint-epoch-{current_epoch}"
            print(f"Saving model checkpoint to {output_dir} at epoch {current_epoch}")
            self.trainer.save_model(output_dir)
            self.trainer.tokenizer.save_pretrained(output_dir)
        return control

# Define a callback that tracks loss and plots a graph every 10 steps.
class LossPlotCallback(TrainerCallback):
    def __init__(self, plot_interval=10):
        self.losses = []
        self.steps = []
        self.plot_interval = plot_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)

            if state.global_step % self.plot_interval == 0:
                self.plot_loss()

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.steps, self.losses, marker='o', linestyle='-')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Steps")
        plt.grid()
        plt.savefig(f"loss_plot_step_{self.steps[-1]}.png")
        plt.close()
        print(f" Saved loss plot at step {self.steps[-1]}.")

# Print the number of examples loaded in your dataset.
num_examples = len(tokenized_dataset)
print(f"Loaded {num_examples} examples for training.")

# Set up training arguments.
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    num_train_epochs=100,              # Adjust number of epochs as needed.
    per_device_train_batch_size=1,     # Adjust based on your GPU's memory.
    gradient_accumulation_steps=4,     # Simulate a larger batch size if needed.
    learning_rate=5e-4,
    fp16=True,                         # Enable mixed precision if supported.
    logging_steps=5,
    save_strategy="epoch",             # Save at the end of each epoch (our callback filters this).
    report_to="none"                   # Disable external logging if not needed.
)

# Create the Trainer with our custom callbacks.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Access the 'train' split explicitly
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrintCallback(), SaveEveryFiveEpochsCallback()]
)

# Manually set the trainer on callbacks that require it.
for callback in trainer.callback_handler.callbacks:
    if hasattr(callback, "set_trainer"):
        callback.set_trainer(trainer)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Model and tokenizer saved to ./fine_tuned_model")

# Run the evaluation
print("\nEvaluating model on the dataset...")
evaluation_results = evaluate_memorization(model, tokenizer, "dataset.json", "memorization_results.json")

# Print a summary of the results
print("\nMemorization Summary:")
print(f"Total examples: {evaluation_results['summary']['total_examples']}")
print(f"Exact matches: {evaluation_results['summary']['exact_matches']} ({evaluation_results['summary']['exact_match_percentage']:.2f}%)")
print(f"Partial matches: {evaluation_results['summary']['partial_matches']} ({evaluation_results['summary']['partial_match_percentage']:.2f}%)")
print("Detailed results saved to memorization_results.json")

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
        
        results.append(result)
        
        # Print the results
        print(f"Expected: {expected_response}")
        print(f"Generated: {generated_response}")
        print(f"Match type: {match_type}")
    
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