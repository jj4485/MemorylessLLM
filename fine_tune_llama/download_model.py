"""
Model Downloader Script

This script downloads models from Hugging Face without using BitsAndBytes.
Run this on a machine with internet access to cache the models locally.
"""
import os
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set cache directory
cache_dir = os.environ.get("HF_HOME", None)
if cache_dir:
    print(f"Using cache directory: {cache_dir}")
else:
    print("Using default cache directory")

# Login to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Logging in to Hugging Face Hub...")
    huggingface_hub.login(token=hf_token)
else:
    print("No HF_TOKEN found. Will only be able to download public models.")

# Models to download
models = [
    "meta-llama/Llama-3.2-3B",  # Primary model
    "EleutherAI/pythia-2.8b"    # Fallback model from project memory
]

for model_id in models:
    print(f"\n{'='*50}")
    print(f"Downloading {model_id}...")
    
    # Download tokenizer
    print(f"Downloading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # Download model without quantization
    print(f"Downloading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype="auto",  # Use default dtype
        device_map="auto"    # Let transformers decide device mapping
    )
    
    print(f"Successfully downloaded {model_id}")
    print(f"{'='*50}")

print("\nAll models downloaded successfully!")
print("You can now run your fine-tuning script with local_files_only=True")
