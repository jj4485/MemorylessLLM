import numpy as np
import os
import re
from collections import Counter

class SimilaritySearch:
    def __init__(self, corpus_file):
        """
        Initialize the similarity search with a corpus from a file.
        :param corpus_file: Path to the text file containing reference corpus (one line per entry)
        """
        self.corpus = self._load_corpus(corpus_file)
        print(f"Loaded corpus with {len(self.corpus)} entries")

    def _load_corpus(self, file_path):
        """Load reference corpus from a text file, one line per entry."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines() if line.strip()]
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on non-alphanumeric characters and converting to lowercase."""
        return re.findall(r'\w+', text.lower())
    
    def _get_word_counts(self, text):
        """Get word frequency counts from text."""
        tokens = self._tokenize(text)
        return Counter(tokens)
    
    def _cosine_similarity(self, counter1, counter2):
        """Calculate cosine similarity between two Counter objects."""
        terms = set(counter1).union(counter2)
        dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in terms)
        magnitude1 = np.sqrt(sum(counter1.get(k, 0)**2 for k in terms))
        magnitude2 = np.sqrt(sum(counter2.get(k, 0)**2 for k in terms))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    def search(self, query, threshold=0.6):
        """
        Find the most similar match to the query in the corpus.
        :param query: String to search for in corpus
        :param threshold: Cosine similarity threshold (0.0 - 1.0)
        :return: Best match and similarity score
        """
        query_counts = self._get_word_counts(query)
        
        # Calculate similarity for each corpus entry
        similarities = []
        for entry in self.corpus:
            entry_counts = self._get_word_counts(entry)
            similarity = self._cosine_similarity(query_counts, entry_counts)
            similarities.append(similarity)
        
        # Get the best match
        if not similarities:
            return None, 0.0
            
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_match = self.corpus[best_idx]

        # Return match only if it exceeds threshold
        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score

# Example Usage
if __name__ == "__main__":
    corpus_file = os.path.join("reference_corpus", "mlk.txt")
    searcher = SimilaritySearch(corpus_file)

    query = input("Enter your search query: ")  # Take user input for query
    match, score = searcher.search(query)

    if match:
        print(f"\n✅ Best Match Found:\n{match}\n🔹 Similarity Score: {score:.4f}")
    else:
        print("\n❌ No sufficiently similar match found.")
