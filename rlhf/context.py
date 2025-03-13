from datasets import load_dataset
from similarity import SimilaritySearch
from data import RLHFGenerator


num_samples = 1000
ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, local_files_only=True)
# Take a sample and build your reference corpus list (assuming each example has a "text" field)
reference_corpus = []
for i, example in enumerate(ds):
    reference_corpus.append(example["text"])
    if i >= num_samples - 1:
        break

corpus_file = "reference_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in reference_corpus:
        f.write(line + "\n")

searcher = SimilaritySearch(corpus_file)

# 4. Initialize your text generation model (adjust parameters as needed)
generator = RLHFGenerator(
    model_name="EleutherAI/pythia-6.9b",  # or your chosen model
    reference_corpus_path=corpus_file,
    max_length=50,        # this may be adjusted
    temperature=0.7,
    top_k=50
)

context_lengths = [5, 10, 20, 30, 40, 50]  # example context lengths (in tokens)
results = []

for context_length in context_lengths:
    # Create a prompt of the desired length. You might use a seed text repeated/truncated to context_length tokens.
    prompt = " ".join(["sample"] * context_length)
    print("Prompt is", prompt)
    
    # Generate output (this could be a single output or averaged over several runs)
    generated_output = generator.generate_text(prompt)
    
    # Check similarity
    match, score = searcher.search(generated_output)
    
    # Record result: context length, similarity score, and whether a memorized match was found
    results.append({
        "context_length": context_length,
        "generated_output": generated_output,
        "match": match,
        "similarity_score": float(score)
    })
    
    # Optional: print the result for inspection
    print(f"Context Length: {context_length}")
    print(f"Output: {generated_output}")
    if match:
        print(f"Memorized match found! Score: {score:.4f}")
    else:
        print("No memorized match found.")
    print("------")