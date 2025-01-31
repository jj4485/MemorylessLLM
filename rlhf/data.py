from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import time

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

def preprocess_preferences(example):
    return { 
        "prompt": example["prompt"],
        "response_0": example["good_response"],
        "response_1": example["bad_response"],
        "label": 0  
    } 


def output_text(input_text, model, tokenizer):
    # Tokenize and return as PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt")

    # 3) Generate tokens
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,             # <-- Pass the dictionary returned by the tokenizer
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.5
        )

    # 4) Decode the generated tokens back to text
    prompt_length = inputs["input_ids"].shape[1]
    gen_tokens = output_tokens[0, prompt_length:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(generated_text)

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")
    prompt = "I have a dream"
    all_responses = []
    
    for example in data:
        prompt_responses = {
            "prompt": prompt,
            "responses": output_text(prompt, model, tokenizer)
        }
        all_responses.append(prompt_responses)

    with open('prompt_responses.json', 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2)
    print("Saved to prompt_responses.json")

if __name__ == "__main__":
    main()