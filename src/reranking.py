"""
reranking.py

Cross-Encoder reranker.

Takes a list of candidate chunks from hybrid retrieval (~20–40 items) and
re-scores each (question, chunk_text) pair using a lightweight cross-encoder
model.  Returns the top_k most relevant chunks in descending score order.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22 MB, runs on CPU in a few seconds for 40 candidates
  - Trained on MS MARCO passage ranking — good general relevance
"""

from sentence_transformers.cross_encoder import CrossEncoder


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Lightweight cross-encoder reranker.

    Usage:
        reranker = Reranker()
        top5 = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self):
        print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL} ...")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        print("Cross-encoder loaded.")

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = 5,
    ) -> list[dict]:
        """
        Score every (query, candidate_text) pair and return the
        top_k highest-scoring candidates.

        Each returned chunk dict gets two new fields:
          - 'rerank_score'  : raw cross-encoder logit (higher = more relevant)
          - 'retrieval_type': overwritten to include 'reranked'

        Parameters
        ----------
        query      : the user question string
        candidates : list of chunk dicts (from HybridRetriever.search)
        top_k      : how many to return after reranking

        Returns
        -------
        list of chunk dicts, sorted by rerank_score descending
        """

        if not candidates:
            return []

        # Build input pairs for the cross-encoder
        pairs = [
            [query, chunk["text"]]
            for chunk in candidates
        ]

        # Score all pairs in one batch (model handles batching internally)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to chunks
        scored = []
        for chunk, score in zip(candidates, scores):
            ranked_chunk = chunk.copy()
            ranked_chunk["rerank_score"] = float(score)
            scored.append(ranked_chunk)

        # Sort by rerank_score descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]
