from datasets import load_dataset
from similarity import SimilaritySearch
from RLHFGenerator import RLHFGenerator
import os
import matplotlib.pyplot as plt
import json
import concurrent.futures

def build_reference_corpus(dataset_name, split, num_samples, corpus_filename):
    """
    Loads a streaming dataset and writes the 'text' field from the first num_samples
    examples to a corpus file.
    """
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    reference_corpus = []
    for i, example in enumerate(ds):
        reference_corpus.append(example["text"])
        if i >= num_samples - 1:
            break

    with open(corpus_filename, "w", encoding="utf-8") as f:
        for line in reference_corpus:
            f.write(line + "\n")
    print(f"Reference corpus saved to {corpus_filename}")
    return corpus_filename

def get_token_subprompt(full_prompt, num_tokens, tokenizer):
    tokens = tokenizer.tokenize(full_prompt)
    selected_tokens = tokens[:num_tokens]
    return tokenizer.convert_tokens_to_string(selected_tokens)

def main():
    # --- Part 1: Build (or load) the Reference Corpus File ---
    # Uncomment these lines if you need to build the corpus.
    # dataset_name = "monology/pile-uncopyrighted"
    # split = "train"
    # num_samples = 1000
    # corpus_filename = "reference_corpus.txt"
    # build_reference_corpus(dataset_name, split, num_samples, corpus_filename)
    
    # For this example, we assume the reference corpus file already exists.
    with open("reference_corpus.txt", "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    # For example, use the first two lines of the file as our full prompt.
    full_prompt = "\n".join(corpus_lines[:100]).strip()
    print("The full prompt is:\n", full_prompt)

    # --- Part 2: Initialize the Generator ---
    generator = RLHFGenerator(
        model_name="EleutherAI/pythia-6.9b",
        reference_corpus_path=os.path.join("reference_corpus.txt")
    )

    # --- Part 3: Define the token counts to try and the number of runs per token count ---
    token_counts = [45, 50, 55, 60, 65, 70]  # Example token counts
    runs_per_context = 10  # Run each context length 10 times

    # --- Part 4: Define a function to process one context length ---
    def process_context_length(token_count):
        local_results = []
        for run in range(runs_per_context):
            subprompt = get_token_subprompt(full_prompt, token_count, generator.tokenizer)
            print(f"\n[Token Count: {token_count} | Run: {run}] Subprompt: {subprompt}")
            generated_text = generator.generate_text(subprompt)
            match, score = generator.check_similarity(generated_text)
            print(f"Token count: {token_count} | Run: {run} | Similarity score: {score:.4f}")
            local_results.append({
                "token_count": token_count,
                "run": run,
                "subprompt": subprompt,
                "output": generated_text,
                "similarity_score": float(score)
            })
        return local_results

    # --- Part 5: Parallelize the processing using one thread per token count ---
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(token_counts)) as executor:
        future_to_token = {executor.submit(process_context_length, token_count): token_count for token_count in token_counts}
        for future in concurrent.futures.as_completed(future_to_token):
            token_count = future_to_token[future]
            try:
                context_results = future.result()
                results.extend(context_results)
            except Exception as e:
                print(f"Token count {token_count} generated an exception: {e}")

    # --- Part 6: Plot results ---
    # Group results by token count and compute the average similarity score.
    token_to_scores = {}
    for r in results:
        token_to_scores.setdefault(r["token_count"], []).append(r["similarity_score"])
    avg_scores = sorted([(token, sum(scores)/len(scores)) for token, scores in token_to_scores.items()], key=lambda x: x[0])
    x = [item[0] for item in avg_scores]
    y = [item[1] for item in avg_scores]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Context Length (Number of Tokens)")
    plt.ylabel("Average Similarity Score")
    plt.title("Memorization vs. Context Length")
    plt.grid(True)
    plt.show()

    # --- Part 7: Save results to a JSON file ---
    with open("memorization_vs_tokens.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results saved to memorization_vs_tokens.json")

if __name__ == '__main__':
    main()
