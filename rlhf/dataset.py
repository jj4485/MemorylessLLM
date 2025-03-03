from datasets import load_dataset
from similarity import SimilaritySearch
from RLHFGenerator import RLHFGenerator
import os
import matplotlib.pyplot as plt
import json

def build_reference_corpus(dataset_name, split, start_idx, num_samples, corpus_filename):
    """
    Loads a streaming dataset and writes the 'text' field from a subset of examples 
    starting from start_idx and taking num_samples examples to a corpus file.
    """
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    reference_corpus = []
    for i, example in enumerate(ds):
        if i < start_idx:
            continue
        reference_corpus.append(example["text"])
        if i >= start_idx + num_samples - 1:
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
     #--- Part 1: Build the Reference Corpus File ---
    #dataset_name = "monology/pile-uncopyrighted"
    #split = "train"
    #num_samples = 1000
    #corpus_filename = "reference_corpus.txt"
     #Build the reference corpus file
    #build_reference_corpus(dataset_name, split, start_idx=1000, num_samples=1000, corpus_filename=corpus_filename)
    
    with open("reference_corpus.txt", "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    full_prompt = "\n".join(corpus_lines[:100]).strip()


    print("The full prompt is", full_prompt)

    generator = RLHFGenerator(
    model_name="EleutherAI/pythia-12b",
    reference_corpus_path=os.path.join("reference_corpus.txt"))

    # Define a range of token counts to try (e.g., 5, 10, 15, 20, etc.)
    token_counts = [250, 375, 500]

    # Record results: we will store token count, the subprompt, and the corresponding similarity score.
    results = []

    # For each token count, generate a prompt and get the output.
    for count in token_counts:
        subprompt = get_token_subprompt(full_prompt, count, generator.tokenizer)
        print(f"\nUsing subprompt ({count} tokens): {subprompt}")
        
        # Generate a response using the subprompt.
        generated_text = generator.generate_text(subprompt)
        
        # Check similarity between generated text and the reference corpus.
        match, score = generator.check_similarity(generated_text)
        print(f"Token count: {count} | Similarity score: {score:.4f}")
        
        results.append({
            "token_count": count,
            "subprompt": subprompt,
            "output": generated_text,
            "similarity_score": float(score)
        })

    # Plot the graph of token count vs similarity score.
    plt.figure(figsize=(8, 5))
    plt.plot([r["token_count"] for r in results], [r["similarity_score"] for r in results], marker="o")
    plt.xlabel("Context Length (Number of Tokens)")
    plt.ylabel("Similarity Score")
    plt.title("Memorization vs. Context Length")
    plt.grid(True)
    plt.show()

    # Save the results to a JSON file.
    with open("memorization_vs_tokens.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Results saved to memorization_vs_tokens.json")
    
if __name__ == '__main__':
    main()