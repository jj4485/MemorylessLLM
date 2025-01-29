from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1) Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")

# 2) Prepare the input text
input_text = "Hello World"
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
        temperature=0.7
    )

# 4) Decode the generated tokens back to text
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
