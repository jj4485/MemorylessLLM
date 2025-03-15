"""
Interactive chat with the fine-tuned Pythia 2.8B model.

This script provides an interactive chat interface to converse with the fine-tuned model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with fine-tuned Pythia model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="finetuned_pythia",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of generated response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0 for deterministic, higher for more randomness)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to load (e.g., 'checkpoint-iteration-3')"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine model path
    model_path = args.model_path
    if args.checkpoint:
        model_path = os.path.join(args.model_path, args.checkpoint)
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate precision
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    print("\nModel loaded successfully! You can now chat with the model.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    
    # Interactive chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        # Format prompt
        formatted_prompt = f"[INST] {user_input} [/INST]"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_length,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and extract only the generated part (after the prompt)
        prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True, return_tensors="pt")[0]
        prompt_length = len(prompt_tokens)
        
        # Extract only the generated part
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        print(f"\nModel: {generated_text.strip()}")

if __name__ == "__main__":
    main()
