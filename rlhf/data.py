'''
Script that generates good/bad responses for prompts to use for RLHF. 
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datasets import load_dataset

data = [
    {
        "prompt": "Explain the differences between supervised and unsupervised learning.",
        "good_response": "Supervised learning uses labeled data...",
        "bad_response": "Unsupervised learning is the same as supervised..."
    },
    {
        "prompt": "What are the health benefits of green tea?",
        "good_response": "Green tea may help with heart health...",
        "bad_response": "Green tea cures cancer instantly."
    },
]

#Loading Reference Corpus
def load_reference_text(file_path):
    with open(file_path, 'r') as f:
        reference_text = f.read()
    return reference_text

def check_memorization(response, reference_text, threshold=0.9):
    """
    Check if a generated response is memorized from the reference text.
    Uses sequence matching for approximate matches.
    """
    similarity = difflib.SequenceMatcher(None, response, reference_text).ratio()
    return True if similarity >= threshold else False

#Function that labels prompts as reward or punishment
def preprocess_preferences(example):
    return { 
        "prompt": example["prompt"],
        "response_0": example["good_response"],
        "response_1": example["bad_response"],
        "label": 0  
    } 


## Function to generate text from input text
def output_text(input_text, model, tokenizer):
    # Tokenize and return as PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate tokens
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,             # Pass the dictionary returned by the tokenizer
            max_length=50,
            num_return_sequences=1,
            top_k=50,
            temperature=0.5
        )

    # Decode the generated tokens back to text
    prompt_length = inputs["input_ids"].shape[1]
    gen_tokens = output_tokens[0, prompt_length:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    print(generated_text)  # Optional: you can still print if you want
    return generated_text  # Return the generated text


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")

    prompts = ["I have a dream"]  # Add more prompts if needed
    all_responses = []

    # Load the reference corpus (MLK speech)
    reference_text = load_reference_text("reference_corpus/mlk.txt")

    for prompt in prompts:
        for _ in range(10):  # Generate 10 responses per prompt
            generated_response = output_text(prompt, model, tokenizer)

            prompt_responses = {
                "prompt": prompt,
                "responses": generated_response,
                "memorized": check_memorization(generated_response, reference_text)  # Check if it's memorized
            }

            print(f"Prompt: {prompt_responses['prompt']}")
            print(f"Response: {prompt_responses['responses']}")
            print(f"Memorized: {prompt_responses['memorized']}")

            all_responses.append(prompt_responses)

    # Save results to JSON
    with open("prompt_responses.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2)

    print("✅ Saved to prompt_responses.json")

if __name__ == "__main__":
    main()