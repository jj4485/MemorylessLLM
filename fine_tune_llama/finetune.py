from transformers import BitsAndBytesConfig
import json

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
base_model_id = 'meta-llama/Llama-3.2-3B'
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

# ✅ Callback to save the model at each epoch properly
class SaveEveryEpochCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None  # Store trainer instance

    def on_train_begin(self, args, state, control, **kwargs):
        """Save reference to the trainer instance when training starts."""
        self.trainer = kwargs.get("trainer", None)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save the model, tokenizer, and optimizer state at the end of each epoch."""
        if self.trainer is None:
            print("⚠️ Warning: Trainer instance not set. Skipping checkpoint save.")
            return control
        
        current_epoch = int(state.epoch)
        output_dir = f"{args.output_dir}/checkpoint-epoch-{current_epoch}"
        print(f"📌 Saving model checkpoint to {output_dir} at epoch {current_epoch}")

        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        
        # ✅ Save model, tokenizer, optimizer, and scheduler state
        self.trainer.save_model(output_dir)
        self.trainer.tokenizer.save_pretrained(output_dir)
        torch.save(self.trainer.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        torch.save(state, os.path.join(output_dir, "trainer_state.pt"))

        return control

# ✅ Callback to plot loss every 200 steps
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
        print(f"📉 Saved loss plot at step {self.steps[-1]}.")

# ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    num_train_epochs=100,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=5,
    report_to="none"
)

# ✅ Create callback instances
save_callback = SaveEveryEpochCallback()
loss_plot_callback = LossPlotCallback(plot_interval=200)

# ✅ Create the Trainer with proper checkpoint saving
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[save_callback, loss_plot_callback]
)

# ✅ Start Training (No Checkpoint Loading, Always Starts Fresh)
print("🚀 Starting training...")
trainer.train()

print("✅ Training complete!")
