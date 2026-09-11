# Adaptive Multimodal RAG — TODO List

## 🔧 Fix / Improve Existing Code

- [x] **Fix chunking** — switched to NLTK sentence-aware chunking. Sentences are never broken mid-thought. Chunks start and end on sentence boundaries, with sentence-level overlap.

---

## Priority 2 — BM25 + Hybrid Retrieval

- [x] Add `rank-bm25` to `requirements.txt`
- [x] Add `BM25Retriever` class to `src/retrieval.py` with build/save/load
- [x] Save BM25 index to `data/bm25.pkl` from `ingest.py`
- [x] Implement hybrid score merging (dense ∪ BM25, deduplicated)
- [x] `HybridRetriever.search(query, top_k)` method

---

## Priority 3 — Cross-Encoder Reranking

- [x] Add `sentence-transformers` cross-encoder (already in requirements)
- [x] Create `src/reranking.py` with `Reranker` class
- [x] Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [x] `rerank(query, candidates, top_k=5)` → sorted by relevance

---

## Priority 4 — Routing & Web Fallback

- [x] Add `tavily-python` to `requirements.txt`
- [x] Create `src/routing.py` — Gemini-based intent/source/modality classifier
- [x] Create `src/web_search.py` — Tavily wrapper (returns chunk-format dicts)
- [x] Confidence-threshold fallback in `main.py` (rerank score < 0.0 → web)
- [x] Update `main.py` to use router before retrieval

---

## Priority 5 — Multimodal Figure Answering

- [x] Create `src/figures.py` — caption/keyword figure retrieval
- [x] `find_figures_for_question()` — token overlap matching
- [x] `generate_answer_with_figure()` in `src/generation.py`
- [x] Wire figure path → Gemini multimodal call (inline_data)
- [x] Graceful fallback to text-only if figure file missing

---

## Priority 6 — Evaluation Notebook

- [x] Create `notebooks/evaluation.ipynb`
- [x] 30 evaluation questions with ground truth (5 categories)
- [x] Retrieval experiments: BM25 vs Dense vs Hybrid vs Hybrid+Reranker
- [x] Metrics: Recall@1, Recall@5, Recall@10, MRR
- [x] Chunking distribution analysis + histogram
- [x] Routing evaluation (accuracy, unnecessary web, missed web)
- [x] Multimodal evaluation (text-only vs text+figure)
- [x] Full pipeline answer quality section
- [x] Baseline comparison table
- [x] Matplotlib plots (retrieval bar chart, routing accuracy)
- [x] Summary cell

---

## Priority 7 — Streamlit App

- [x] Add `streamlit` to `requirements.txt`
- [x] Create `app/app.py`
- [x] Sidebar: chunk/rerank sliders, status indicators
- [x] Routing decision display (intent/source/modality badges)
- [x] Retrieved sources expander
- [x] Figure inline display for IMAGE queries
- [x] Final answer display with web-use note

---

## Dependencies — all added to `requirements.txt`

- [x] `rank-bm25`
- [x] `tavily-python`
- [x] `streamlit`
- [x] `scikit-learn`
- [x] `matplotlib`
- [x] `nltk`
- [x] `ipykernel`
- [x] `notebook`

---

## Remaining / Optional Work

- [ ] **Re-run `ingest.py`** to rebuild chunks.json + embeddings.npy + bm25.pkl with the new sentence-aware chunking (mandatory before evaluation)
- [ ] **Add `TAVILY_API_KEY`** to `.env` file to enable web search
- [ ] **Run evaluation notebook** end-to-end and fill in actual Recall@K / MRR numbers
- [ ] **Tune** `WEB_FALLBACK_THRESHOLD` in `main.py` based on eval results
- [ ] Optional: add `src/ingestion.py` caption extraction (detect "Figure X" text near images)
- [ ] Optional: Git commit all changes

---

## Final Checklist (from README)

- [x] ≥20 papers in corpus *(30 papers)*
- [x] Text extracted
- [x] Figures extracted
- [x] Page numbers stored
- [x] Chunking with overlap *(now sentence-aware)*
- [x] Embeddings (NumPy)
- [x] Cosine retrieval
- [x] Gemini generation
- [x] BM25 retrieval
- [x] Hybrid retrieval
- [x] Cross-encoder reranking
- [x] Intent detection + routing
- [x] Web fallback (Tavily)
- [x] Local + Web combined mode
- [x] Figure retrieval + multimodal answering
- [x] 30-question evaluation dataset
- [x] Recall@K / MRR metrics (in notebook)
- [x] Routing accuracy metrics (in notebook)
- [x] Baseline comparison (in notebook)
- [x] Plots (in notebook)
- [x] Evaluation notebook
- [x] Streamlit UI
