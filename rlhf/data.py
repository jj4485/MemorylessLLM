'''
Script that generates good/bad responses for prompts to use for RLHF. 
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datasets import load_dataset
import difflib
from similarity import SimilaritySearch
import os

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
            temperature=0.7
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


    for prompt in prompts:
        for _ in range(50):  # Generate 10 responses per prompt
            generated_response = output_text(prompt, model, tokenizer)

            corpus_file = os.path.join("reference_corpus", "mlk.txt")
            searcher = SimilaritySearch(corpus_file)
            match, score = searcher.search(generated_response)
            memorized = ""
            if match:
                print(f"\n✅ Best Match Found:\n{match}\n🔹 Similarity Score: {score:.4f}")
                memorized = True

            else:
                print("\n❌ No sufficiently similar match found.")
                memorized = False
            
            prompt_responses = {
                "prompt": prompt,
                "responses": generated_response,
                "memorized": memorized
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