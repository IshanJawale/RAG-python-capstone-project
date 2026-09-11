"""
main.py

Adaptive Multimodal RAG — interactive Q&A entry point.

Pipeline per query:
  1. LOCAL/WEB routing (Gemini) — decides where to retrieve from
  2. Hybrid retrieval (BM25 + Dense) + Cross-Encoder reranking
  3. Confidence-based web fallback (if local reranker score is weak)
  4. Figure retrieval — always attempted, paper-aware (from found text chunks)
  5. Generation — multimodal (text + figure) if figure found, else text-only

NO text-vs-image modality routing: we always try to find a relevant figure
and let Gemini decide what to use from the combined evidence.

Run:
    python main.py
"""

import json
from pathlib import Path

import numpy as np

from src.retrieval  import DenseRetriever, BM25Retriever, HybridRetriever
from src.reranking  import Reranker
from src.routing    import classify_query, needs_web_fallback
from src.web_search import TavilySearcher
from src.figures    import load_figures, find_figures_for_chunks, get_figure_path
from src.generation import generate_answer, generate_answer_with_figure
from src.retrieval  import print_results


# ======================================================================
# Paths
# ======================================================================

CHUNKS_PATH     = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
BM25_PATH       = Path("data/bm25.pkl")

# Cross-encoder score below this → web fallback kicks in
WEB_FALLBACK_THRESHOLD = 0.0


# ======================================================================
# Load saved artifacts
# ======================================================================

def load_corpus():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    figures    = load_figures()
    return chunks, embeddings, figures


# ======================================================================
# Build retrieval stack
# ======================================================================

def build_retrieval_stack(chunks, embeddings):

    dense = DenseRetriever(embeddings, chunks)

    if BM25_PATH.exists():
        bm25 = BM25Retriever.load(chunks, BM25_PATH)
    else:
        print("[Warning] BM25 index not found — building in memory. Run ingest.py to persist.")
        bm25 = BM25Retriever(chunks)

    return HybridRetriever(dense, bm25)


# ======================================================================
# Single query pipeline
# ======================================================================

def run_query(
    question:   str,
    hybrid:     HybridRetriever,
    reranker:   Reranker,
    figures:    list[dict],
    web_search,               # TavilySearcher | None
    verbose:    bool = True,
) -> dict:
    """
    Full adaptive pipeline for one question.

    Returns
    -------
    dict with keys: question, routing, retrieved_chunks, answer, used_web
    """

    # ------------------------------------------------------------------
    # Step 1: LOCAL / WEB routing only (no modality decision)
    # ------------------------------------------------------------------

    routing = classify_query(question)

    # Strip modality from display — we always retrieve both now
    if verbose:
        print(f"\n[Router]  source={routing['source']}")
        print(f"          intent={routing['intent']}")
        print(f"          reason: {routing['reasoning']}")

    source = routing["source"]

    # ------------------------------------------------------------------
    # Step 2: Retrieve text
    # ------------------------------------------------------------------

    local_candidates = []
    web_candidates   = []

    if source in ("LOCAL", "BOTH"):
        local_candidates = hybrid.search(question, top_k=20)
    elif source == "WEB":
        # Still run local — web alone is rarely enough for academic questions
        local_candidates = hybrid.search(question, top_k=10)

    if source in ("WEB", "BOTH") and web_search is not None:
        if verbose:
            print("[WebSearch] Querying Tavily...")
        web_candidates = web_search.search(question, max_results=5)

    # ------------------------------------------------------------------
    # Step 3: Rerank local candidates
    # ------------------------------------------------------------------

    reranked = []
    if local_candidates:
        reranked = reranker.rerank(question, local_candidates, top_k=5)
        if verbose and reranked:
            print(f"[Reranker] Top score: {reranked[0]['rerank_score']:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Confidence-based web fallback
    # ------------------------------------------------------------------

    used_web = source in ("WEB", "BOTH")

    if (
        source == "LOCAL"
        and web_search is not None
        and needs_web_fallback(reranked, threshold=WEB_FALLBACK_THRESHOLD)
    ):
        if verbose:
            print("[Fallback] Local evidence weak — fetching web results...")
        web_candidates = web_search.search(question, max_results=5)
        used_web = True

    # ------------------------------------------------------------------
    # Step 5: Merge final evidence pool
    # ------------------------------------------------------------------

    final_chunks = reranked.copy()

    if web_candidates:
        web_reranked = reranker.rerank(question, web_candidates, top_k=3)
        final_chunks.extend(web_reranked)

    if not final_chunks:
        return {
            "question":  question,
            "routing":   routing,
            "retrieved": [],
            "answer":    "The available evidence is insufficient to answer this question.",
            "used_web":  used_web,
        }

    # ------------------------------------------------------------------
    # Step 6: Figure retrieval — ALWAYS attempted, paper-aware
    #
    # Use the retrieved text chunks to constrain figure search to the
    # paper(s) the text retriever already identified as relevant.
    # ------------------------------------------------------------------

    figure_path = None
    caption     = ""

    matching_figures = find_figures_for_chunks(
        question,
        final_chunks,   # <-- passes paper context
        figures,
        top_k=1,
    )

    if matching_figures:
        best_fig    = matching_figures[0]
        figure_path = get_figure_path(best_fig)
        caption     = best_fig.get("caption", "")

        if verbose:
            if figure_path:
                paper = best_fig.get("paper_title", "")[:50]
                w     = best_fig.get("width",  0)
                h     = best_fig.get("height", 0)
                print(f"[Figures]  {figure_path.name}  ({w}×{h}px)  from: {paper}")
            else:
                print("[Figures]  Metadata found but image file missing.")

    # ------------------------------------------------------------------
    # Step 7: Print retrieved chunks (debug)
    # ------------------------------------------------------------------

    if verbose:
        print_results(question, final_chunks)
        print("\n" + "=" * 70)
        print("GENERATING ANSWER...")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Step 8: Generate — multimodal if figure available, else text-only
    # ------------------------------------------------------------------

    if figure_path is not None:
        answer = generate_answer_with_figure(
            question, final_chunks, figure_path, caption
        )
    else:
        answer = generate_answer(question, final_chunks)

    return {
        "question":  question,
        "routing":   routing,
        "retrieved": final_chunks,
        "answer":    answer,
        "used_web":  used_web,
    }


