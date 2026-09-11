"""
retrieval.py

Provides three retrieval strategies:
  1. DenseRetriever  — cosine similarity over sentence-transformer embeddings
  2. BM25Retriever   — sparse BM25 keyword retrieval (rank-bm25)
  3. HybridRetriever — union of Dense + BM25 results, deduplicated

The typical usage pattern is:
    dense   = DenseRetriever(embeddings, chunks)
    bm25    = BM25Retriever(chunks)           # or load from disk
    hybrid  = HybridRetriever(dense, bm25)
    results = hybrid.search(query, top_k=20)  # feed to reranker
"""

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
BM25_PATH   = Path("data/bm25.pkl")


# ======================================================================
# 1. Dense Retriever
# ======================================================================

class DenseRetriever:
    """
    Cosine-similarity retrieval using pre-computed sentence-transformer
    embeddings stored as a NumPy matrix.

    Embeddings must already be L2-normalised (produced by
    embeddings.py with normalize_embeddings=True).
    """

    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):

        self.embeddings = embeddings    # shape: (N, dim), float32, normalised
        self.chunks     = chunks

        print("Loading embedding model for query encoding...")
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Return the top_k most similar chunks for `query`.
        Each result dict is a copy of the chunk dict with an added
        'score' (cosine similarity, float) and 'retrieval_type'='dense'.
        """

        q_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)

        # Matrix-vector dot product = cosine similarity (normalised vecs)
        scores = self.embeddings @ q_emb

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            chunk = self.chunks[idx].copy()
            chunk["score"]          = float(scores[idx])
            chunk["retrieval_type"] = "dense"
            results.append(chunk)

        return results


# ======================================================================
# 2. BM25 Retriever
# ======================================================================

class BM25Retriever:
    """
    Sparse BM25 retrieval using rank-bm25 (BM25Okapi).

    Build once from chunks, then save to disk.  Load on subsequent runs.
    """

    def __init__(self, chunks: list[dict]):

        self.chunks = chunks
        self._build_index(chunks)

    def _build_index(self, chunks: list[dict]):
        """Tokenise all chunk texts and build the BM25 index."""

        print(f"Building BM25 index over {len(chunks)} chunks...")

        # Simple whitespace tokenisation — sufficient for BM25
        tokenised = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenised)

        print("BM25 index built.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = BM25_PATH):
        """Pickle the BM25 index to disk."""

        path.parent.mkdir(exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)

        print(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, chunks: list[dict], path: Path = BM25_PATH) -> "BM25Retriever":
        """
        Load a previously saved BM25 index from disk.
        Returns a BM25Retriever without rebuilding the index.
        """

        instance = cls.__new__(cls)
        instance.chunks = chunks

        with open(path, "rb") as f:
            instance.bm25 = pickle.load(f)

        print(f"BM25 index loaded from {path}")

        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Return the top_k chunks by BM25 score.
        Each result is a chunk copy with 'score' and 'retrieval_type'='bm25'.
        """

        tokenised_query = query.lower().split()
        raw_scores      = self.bm25.get_scores(tokenised_query)

        top_idx = np.argsort(raw_scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if raw_scores[idx] <= 0:
                # BM25 score 0 means no term overlap — skip
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"]          = float(raw_scores[idx])
            chunk["retrieval_type"] = "bm25"
            results.append(chunk)

        return results


# ======================================================================
# 3. Hybrid Retriever
# ======================================================================

class HybridRetriever:
    """
    Combines DenseRetriever and BM25Retriever results.

    Strategy:
      - Run dense search (top_k results)
      - Run BM25  search (top_k results)
      - Take the union, deduplicated by chunk_id
      - Return up to 2*top_k candidates (to be narrowed by a reranker)
    """

    def __init__(
        self,
        dense:  DenseRetriever,
        bm25:   BM25Retriever,
    ):
        self.dense = dense
        self.bm25  = bm25

    def search(
        self,
        query:  str,
        top_k:  int = 20,
    ) -> list[dict]:
        """
        Return merged, deduplicated candidates from both retrievers.
        Candidates are not re-scored here — pass them to a Reranker.

        Returns up to 2*top_k results.
        """

        dense_results = self.dense.search(query, top_k=top_k)
        bm25_results  = self.bm25.search(query,  top_k=top_k)

        # Deduplicate by chunk_id, preserving order (dense first)
        seen    = set()
        merged  = []

        for result in dense_results + bm25_results:
            cid = result["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                merged.append(result)

        return merged


# ======================================================================
# Helpers
# ======================================================================

def print_results(query: str, results: list[dict]):
    """Pretty-print retrieval results to the terminal for debugging."""

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
        print(f"Score          : {result['score']:.4f}")
        print(f"Retrieval type : {result.get('retrieval_type', 'unknown')}")
        print(f"Paper          : {result['paper_title']}")
        print(f"Page           : {result['page']}")
        print()

        preview = result["text"][:400]
        print(preview)

        if len(result["text"]) > 400:
            print("...")