from sentence_transformers import sentence_transformers
import numpy as np

from sentence_transformers import SentenceTransformer
import numpy as np

class SimilaritySearch:
    def __init__(self, corpus_file, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the similarity search with a corpus from a file.
        :param corpus_file: Path to the text file containing reference corpus (one line per entry)
        :param model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = self._load_corpus(corpus_file)
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_numpy=True)  # Encode corpus

    def _load_corpus(self, file_path):
        """Load reference corpus from a text file, one line per entry."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines() if line.strip()]

    def search(self, query, threshold=0.7):
        """
        Find the most similar match to the query in the corpus.
        :param query: String to search for in corpus
        :param threshold: Cosine similarity threshold (0.0 - 1.0)
        :return: Best match and similarity score
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True)  # Encode query
        similarities = np.dot(self.corpus_embeddings, query_embedding)  # Compute cosine similarity

        # Get the best match
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
    corpus_file = r"reference_corpus\mlk.txt"
    searcher = SimilaritySearch(corpus_file)

    query = input("Enter your search query: ")  # Take user input for query
    match, score = searcher.search(query)

    if match:
        print(f"\n✅ Best Match Found:\n{match}\n🔹 Similarity Score: {score:.4f}")
    else:
        print("\n❌ No sufficiently similar match found.")
