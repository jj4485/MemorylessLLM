import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from similarity import SimilaritySearch
from collections import defaultdict

class RLHFGenerator:
    """
    Class that generates text from a given model and checks whether
    the output is in the reference corpus through a similarity search.
    """

    def __init__(
        self, 
        model_name: str, 
        reference_corpus_path: str, 
        max_length: int = 50, 
        temperature: float = 0.7, 
        top_k: int = 100
    ):
        """
        Initialize the RLHFGenerator.

        Args:
            model_name (str): Name/path of the pretrained model to load.
            reference_corpus_path (str): Path to the reference text file (for similarity search).
            max_length (int, optional): Max tokens in generated response. Defaults to 50.
            temperature (float, optional): Sampling temperature for generation. Defaults to 0.7.
            top_k (int, optional): Top-k sampling for generation. Defaults to 50.
        """
        self.model_name = model_name
        self.reference_corpus_path = reference_corpus_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k

        # Load the tokenizer and model
        print(f"Loading tokenizer/model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Initialize the similarity search with the reference corpus
        print(f"Initializing similarity search on {reference_corpus_path}")
        self.searcher = SimilaritySearch(self.reference_corpus_path)

        # Storage for all generated responses
        self.all_responses = []

    def generate_text(self, prompt: str) -> str:
        """
        Generate a single text response given a prompt.

        Args:
            prompt (str): The prompt string to feed into the model.

        Returns:
            str: The generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=self.top_k,
                temperature=self.temperature
            )
        # Get only the newly generated tokens (excluding the prompt)
        prompt_length = inputs["input_ids"].shape[1]
        gen_tokens = output_tokens[0, prompt_length:]
        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return generated_text

    def check_similarity(self, text: str):
        """
        Check how similar the text is to the reference corpus using `SimilaritySearch`.

        Args:
            text (str): The generated text to check.

        Returns:
            (match, score): 
                match (str or None): The best matching text snippet from the corpus, if any.
                score (float): The similarity score for the best match.
        """
        match, score = self.searcher.search(text)
        print("The similarity score is", score)
        return match, score

    def process_prompt(self, prompt: str, num_responses: int = 5):
        """
        Generate multiple responses for a single prompt and store them.

        Args:
            prompt (str): The prompt to generate responses for.
            num_responses (int, optional): Number of responses to generate. Defaults to 5.
        """
        for _ in range(num_responses):
            generated_response = self.generate_text(prompt)
            match, score = self.check_similarity(generated_response)

            # Mark as memorized if a similar snippet was found
            memorized = True if match else False

            response_record = {
                "prompt": prompt,
                "response": generated_response,
                "memorized": memorized,
                "best_match": match,
                "similarity_score": float(score)
            }

            # Print the result
            print(f"\nPrompt: {prompt}")
            print(f"Response: {generated_response}")
            if memorized:
                print(f"✅ Similar match found (score: {score:.4f})")
            else:
                print("❌ No sufficiently similar match found.")

            # Store in the global list
            self.all_responses.append(response_record)

    def run(self, prompts, num_responses_per_prompt: int = 5):
        """
        Main entry point to generate responses for a list of prompts.

        Args:
            prompts (List[str]): List of prompt strings.
            num_responses_per_prompt (int, optional): Number of responses to generate per prompt.
        """
        for prompt in prompts:
            self.process_prompt(prompt, num_responses=num_responses_per_prompt)

    def save_responses(self, output_file: str = "prompt_responses.json"):
        """
        Save all generated responses to a JSON file.

        Args:
            output_file (str, optional): The file path to save the JSON. Defaults to "prompt_responses.json".
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.all_responses, f, indent=2)
        print(f"\n✅ Saved all responses to {output_file}")

    def create_preference_pairs(self):
        """
        Group responses by prompt and create training pairs.
        For each prompt that has both "good" (non-memorized) and "bad" (memorized) responses,
        create all possible pairs with good responses as "chosen" and bad responses as "rejected".

        Returns:
            List[Dict]: A list of preference pairs in the format:
                        {"prompt": prompt, "chosen": <good_response>, "rejected": <bad_response>}
        """
        # Group responses by prompt
        grouped = defaultdict(lambda: {"good": [], "bad": []})
        for record in self.all_responses:
            if record["memorized"]:
                grouped[record["prompt"]]["bad"].append(record["response"])
            else:
                grouped[record["prompt"]]["good"].append(record["response"])

        pairs = []
        for prompt, responses in grouped.items():
            if responses["good"] and responses["bad"]:
                # For each good and bad response pair, create training pairs.
                for good_resp in responses["good"]:
                    for bad_resp in responses["bad"]:
                        pairs.append({
                            "prompt": prompt,
                            "chosen": good_resp,
                            "rejected": bad_resp
                        })
        return pairs

    def save_preference_pairs(self, output_file: str = "training_pairs.json"):
        """
        Save the generated preference pairs to a JSON file.

        Args:
            output_file (str, optional): File path to save the training pairs. Defaults to "training_pairs.json".
        """
        pairs = self.create_preference_pairs()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        print(f"\n✅ Saved {len(pairs)} training pairs to {output_file}")

if __name__ == "__main__":
    # Example usage:
    generator = RLHFGenerator(
        model_name="EleutherAI/pythia-6.9b",
        reference_corpus_path=os.path.join("reference_corpus", "speeches.txt")
    )

    # Define your prompts
    prompts_list = [
        "I have a dream",
        "O say can you see",
        "We hold these truths to be self-evident",
        "Call me Ishmael",
    ]


    # Generate responses for each prompt
    generator.run(prompts_list, num_responses_per_prompt=3)

    # Save all generated responses to file
    generator.save_responses("prompt_responses.json")

    # Now, create and save training pairs based on memorized (bad) and non-memorized (good) outputs
    generator.save_preference_pairs("training_pairs.json")
