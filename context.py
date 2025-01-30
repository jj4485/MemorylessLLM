from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests

# 1) Load the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")

# 2) Prepare the input text
#changing input text
input_text = "To be, or not to be: that is the question"

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


payload = {
    'index': 'v4_rpj_llama_s4',
    'query_type': 'generated_text',
    'query': 'Whether \'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles',
}
result = requests.post('https://api.infini-gram.io/', json=payload).json()
print(result)
