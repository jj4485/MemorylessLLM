from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import time

# 1) Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")
#new prompt
# 2) Prepare the input text
input_text = "I have a dream"
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
'''
# Prepare the payload for the request
payload = {
    'index': 'v4_rpj_llama_s4',
    'query_type': 'count',
    'query': generated_text,
}

# 5) Wrap the POST request in a try/except retry loop
retries = 3
delay_seconds = 2
result = None

for attempt in range(retries):
    try:
        response = requests.post('https://api.infini-gram.io/', json=payload, timeout=10)
        response.raise_for_status()  # Raise an HTTPError if the response was not successful (4xx/5xx)
        result = response.json()
        break  # Successfully got a response, so exit the loop
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"Connection/Timeout error on attempt {attempt+1}: {e}")
        if attempt < retries - 1:
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
        else:
            print("Max retries reached. Aborting.")
            result = None
    except requests.HTTPError as e:
        # This means the server responded but with an error code (e.g., 404, 500)
        print(f"HTTP error on attempt {attempt+1}: {e}")
        # Depending on your use case, decide whether to retry or break
        # break
        if attempt < retries - 1:
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
        else:
            print("Max retries reached. Aborting.")
            result = None

# 6) Use the result if available
if result is not None:
    print("Request successful. Result:")
    print(result)
else:
    print("Failed to get a successful response after retrying.")
'''