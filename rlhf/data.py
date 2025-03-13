import json
import torch
import os
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from similarity import SimilaritySearch

class RLHFGenerator:
    """
    Class that generates text from a given model and checks whether
    the output is memorized from the reference corpus through exact matching
    and perplexity calculation.
    """

    def __init__(
        self, 
        model_name: str, 
        reference_corpus_path: str, 
        max_length: int = 50, 
        temperature: float = 0.7, 
        top_k: int = 50,
        perplexity_threshold: float = 10.0,  # Threshold for perplexity-based memorization
        context_length: int = None,  # Context length control
        use_similarity: bool = False  # Whether to use similarity search (optional)
    ):
        """
        Initialize the RLHFGenerator.

        Args:
            model_name (str): Name/path of the pretrained model to load.
            reference_corpus_path (str): Path to the reference text file.
            max_length (int, optional): Max tokens in generated response. Defaults to 50.
            temperature (float, optional): Sampling temperature for generation. Defaults to 0.7.
            top_k (int, optional): Top-k sampling for generation. Defaults to 50.
            perplexity_threshold (float, optional): Threshold below which perplexity indicates memorization.
            context_length (int, optional): Number of tokens to use as context. If None, uses model's default.
            use_similarity (bool, optional): Whether to use similarity search. Defaults to False.
        """
        self.model_name = model_name
        self.reference_corpus_path = reference_corpus_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.perplexity_threshold = perplexity_threshold
        self.context_length = context_length
        self.use_similarity = use_similarity

        # Load the tokenizer and model
        print(f"Loading tokenizer/model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Load the reference corpus for exact matching
        self.reference_text = self._load_reference_text(reference_corpus_path)
        
        # Initialize the similarity search only if requested
        if self.use_similarity:
            print(f"Initializing similarity search on {reference_corpus_path}")
            self.searcher = SimilaritySearch(self.reference_corpus_path)

        # Storage for all generated responses
        self.all_responses = []
        
    def _load_reference_text(self, file_path):
        """Load the entire reference corpus as a single string."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def generate_text(self, prompt: str) -> str:
        """
        Generate a single text response given a prompt.

        Args:
            prompt (str): The prompt string to feed into the model.

        Returns:
            str: The generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Apply context length limitation if specified
        if self.context_length is not None and inputs["input_ids"].shape[1] > self.context_length:
            # Truncate from the beginning to keep the most recent tokens
            inputs["input_ids"] = inputs["input_ids"][:, -self.context_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length:]
            print(f"Truncated input to {self.context_length} tokens")
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + self.max_length,
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

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of the given text using the model.
        
        Perplexity measures how "surprised" the model is by the text.
        Lower perplexity indicates the model is more confident in the text,
        which can be a sign of memorization.
        
        Args:
            text (str): The text to calculate perplexity for.
            
        Returns:
            float: The perplexity score (lower means more confident/likely memorized)
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            
        loss = outputs.loss
        perplexity = math.exp(loss.item())
        return perplexity
    
    def find_exact_match(self, text: str) -> str:
        """
        Check if the generated text appears exactly in the reference corpus.
        This directly tests for discoverable memorization.
        
        Args:
            text (str): The generated text to check.
            
        Returns:
            str or None: The matching text if found, None otherwise.
        """
        # Clean and normalize both texts for comparison
        text = text.strip().lower()
        if text in self.reference_text.lower():
            return text
        return None

    def check_similarity(self, text: str):
        """
        Check how similar the text is to the reference corpus using `SimilaritySearch`.
        Only used if use_similarity is True.

        Args:
            text (str): The generated text to check.

        Returns:
            (match, score): 
                match (str or None): The best matching text snippet from the corpus, if any.
                score (float): The similarity score for the best match.
        """
        if not self.use_similarity:
            return None, 0.0
            
        match, score = self.searcher.search(text)
        return match, score

    def process_prompt(self, prompt: str, num_responses: int = 5):
        """
        Generate multiple responses for a single prompt and check for memorization.

        Args:
            prompt (str): The prompt to generate responses for.
            num_responses (int, optional): Number of responses to generate. Defaults to 5.
        """
        for _ in range(num_responses):
            generated_response = self.generate_text(prompt)
            
            # Check for exact match (discoverable memorization)
            exact_match = self.find_exact_match(generated_response)
            
            # Calculate perplexity
            perplexity = self.calculate_perplexity(generated_response)
            
            # Determine if memorized based on exact match or perplexity
            memorized_exact = True if exact_match else False
            memorized_perplexity = perplexity < self.perplexity_threshold
            
            # Final memorization verdict based on academic definitions
            memorized = memorized_exact or memorized_perplexity
            
            # Optional similarity check
            sim_match, sim_score = None, 0.0
            memorized_similarity = False
            if self.use_similarity:
                sim_match, sim_score = self.check_similarity(generated_response)
                memorized_similarity = True if sim_match else False

            response_record = {
                "prompt": prompt,
                "response": generated_response,
                "memorized": memorized,
                "memorized_exact": memorized_exact,
                "memorized_perplexity": memorized_perplexity,
                "perplexity": perplexity,
                "context_length": self.context_length
            }
            
            # Add similarity results if used
            if self.use_similarity:
                response_record.update({
                    "memorized_similarity": memorized_similarity,
                    "best_match": sim_match,
                    "similarity_score": sim_score
                })

            # Print or log if desired
            print(f"\nPrompt: {prompt}")
            print(f"Response: {generated_response}")
            print(f"Perplexity: {perplexity:.4f} (Threshold: {self.perplexity_threshold})")
            
            if memorized:
                print(f"✅ Memorization detected:")
                if memorized_exact:
                    print(f"  - Exact match found in reference corpus")
                if memorized_perplexity:
                    print(f"  - Low perplexity ({perplexity:.4f} < {self.perplexity_threshold})")
                if self.use_similarity and memorized_similarity:
                    print(f"  - Semantic similarity ({sim_score:.4f})")
            else:
                print("❌ No memorization detected.")

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


if __name__ == "__main__":
    # Example usage:
    # Initialize the class with your model name and reference corpus
    generator = RLHFGenerator(
        model_name="EleutherAI/pythia-12b",
        reference_corpus_path=os.path.join("reference_corpus", "speeches.txt"),
        perplexity_threshold=10.0,  # Adjust this threshold based on experimentation
        context_length=512,  # Set your desired context length
        use_similarity=False  # Set to True if you want to use similarity search
    )

    # Define your prompts
    prompts_list = [
        "I have a dream",
        "O say can you see",
        "We hold these truths to be self-evident",
        "Call me Ishmael",
        "It is a truth universally acknowledged,",
        "The only thing we have to fear,",
        "Four score and seven years",
        ]

    # Run generation
    generator.run(prompts_list, num_responses_per_prompt=3)

    # Save results
    generator.save_responses("prompt_responses.json")
