from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")

input_text = "Hello World"
token_ids = tokenizer(input_text)

with torch.no_grad():
    output_tokens = model.generate(
        **input_text,
        max_length=50,        # total number of tokens to generate (including input)
        num_return_sequences=1,
        do_sample=True,       # use sampling; if you want deterministic output, set do_sample=False
        top_k=50,             # restrict sampling to the 50 most likely next tokens
        temperature=0.7       # controlling 'creativity' of the output
    )

# 5) Decode the generated tokens back to text
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(generated_text)