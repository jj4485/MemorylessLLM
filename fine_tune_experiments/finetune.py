from transformers import BitsAndBytesConfig
import json

from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "dataset.json"})

def preprocess(batch):
    texts = [p + "\nResponse: " + r for p, r in zip(batch["prompt"], batch["response"])]
    tokenized = tokenizer(texts, truncation=True, max_length=512)
    return tokenized

tokenized_dataset = dataset["train"].map(preprocess, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print("Dataset prepared.")

import torch
import os
import huggingface_hub
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import infer_auto_device_map

# --- Authenticate with Hugging Face ---
# 🚨 Replace with your actual Hugging Face token (DO NOT hardcode it in production scripts)
huggingface_hub.login(token="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")

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

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=quant_config
)

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

# ✅ Define a callback that tracks loss and plots a graph every 10 steps.
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
        print(f"📉 Saved loss plot at step {self.steps[-1]}.")

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