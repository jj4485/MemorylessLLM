from datasets import load_dataset
from similarity import SimilaritySearch
from data import RLHFGenerator
import concurrent.futures
import random
import json
import time
import numpy as np
from tqdm import tqdm


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

# Define context lengths to test
context_lengths = [50, 100, 150]  # example context lengths (in tokens)
num_runs_per_length = 50  # Run each context length 50 times
results = {}

# Initialize results dictionary
for length in context_lengths:
    results[length] = []

# Function to process a single prompt generation and similarity check
def process_prompt(context_length):
    # Create a random prompt of the desired length
    # Using random words to create more diverse prompts
    words = ["sample", "context", "memory", "model", "language", "neural", "network", "learning", "data", "token"]
    prompt = " ".join(random.choices(words, k=context_length))
    
    # Generate output
    start_time = time.time()
    generated_output = generator.generate_text(prompt)
    generation_time = time.time() - start_time
    
    # Check similarity
    match, score = searcher.search(generated_output)
    
    # Return result
    return {
        "context_length": context_length,
        "prompt": prompt,
        "generated_output": generated_output,
        "match": match is not None,
        "similarity_score": float(score),
        "generation_time": generation_time
    }

# Process all context lengths with concurrent execution
print(f"Running experiment with {len(context_lengths)} context lengths, {num_runs_per_length} runs each")
all_tasks = []

for length in context_lengths:
    for _ in range(num_runs_per_length):
        all_tasks.append(length)

# Use ThreadPoolExecutor to run tasks concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all tasks
    future_to_length = {executor.submit(process_prompt, length): length for length in all_tasks}
    
    # Process results as they complete
    for future in tqdm(concurrent.futures.as_completed(future_to_length), total=len(all_tasks)):
        length = future_to_length[future]
        try:
            result = future.result()
            results[length].append(result)
        except Exception as exc:
            print(f'Context length {length} generated an exception: {exc}')

# Analyze and summarize results
summary = {}
for length in context_lengths:
    length_results = results[length]
    match_rate = sum(1 for r in length_results if r["match"]) / len(length_results)
    avg_similarity = np.mean([r["similarity_score"] for r in length_results])
    avg_generation_time = np.mean([r["generation_time"] for r in length_results])
    
    summary[length] = {
        "match_rate": match_rate,
        "avg_similarity_score": float(avg_similarity),
        "avg_generation_time": float(avg_generation_time),
        "num_samples": len(length_results)
    }
    
    print(f"\nContext Length: {length}")
    print(f"Match Rate: {match_rate:.2%}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Average Generation Time: {avg_generation_time:.4f} seconds")

# Save detailed results to file
with open("context_length_results.json", "w") as f:
    json.dump({"detailed_results": results, "summary": summary}, f, indent=2)

print("\nExperiment complete! Results saved to context_length_results.json")