import numpy as np

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:

    def __init__(self, embeddings, chunks):

        self.embeddings = embeddings
        self.chunks = chunks

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query, top_k=5):

        # Embed query
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        # Cosine similarity
        scores = self.embeddings @ query_embedding

        # Get highest scoring chunks
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            chunk = self.chunks[index].copy()

            chunk["score"] = float(scores[index])

            results.append(chunk)

        return results


def print_results(query, results):

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print("\n")
    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        print(f"\n--- Result {rank} ---")

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Paper: {result['paper_title']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print("\n")

        # Don't print the entire chunk
        # during normal testing
        preview = result["text"][:1000]

        print(preview)

        if len(result["text"]) > 1000:
            print("...")