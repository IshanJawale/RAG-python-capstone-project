"""
app.py — Streamlit demo for Adaptive Multimodal RAG

Run from project root:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval  import DenseRetriever, BM25Retriever, HybridRetriever
from src.reranking  import Reranker
from src.routing    import classify_query, needs_web_fallback
from src.web_search import TavilySearcher
from src.figures    import load_figures, find_figures_for_chunks, get_figure_path
from src.generation import generate_answer, generate_answer_with_figure


# ======================================================================
# Config
# ======================================================================

CHUNKS_PATH     = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
BM25_PATH       = Path("data/bm25.pkl")
WEB_FALLBACK_THRESHOLD = 0.0

SOURCE_COLORS = {"LOCAL": "🟢", "WEB": "🌐", "BOTH": "🔵"}

INTENT_DESCRIPTIONS = {
    "LOCAL_FACT":          "Fact from local papers",
    "COMPARISON":          "Comparison across papers",
    "FIGURE":              "Figure / visual question",
    "CURRENT_INFORMATION": "Requires live web data",
    "BROAD_RESEARCH":      "Broad research synthesis",
    "UNKNOWN":             "Unclear intent",
}


# ======================================================================
# Cached resource loading
# ======================================================================

@st.cache_resource(show_spinner="Loading corpus and models...")
def load_everything():

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)
    figures    = load_figures()

    dense = DenseRetriever(embeddings, chunks)

    if BM25_PATH.exists():
        bm25 = BM25Retriever.load(chunks, BM25_PATH)
    else:
        bm25 = BM25Retriever(chunks)

    hybrid   = HybridRetriever(dense, bm25)
    reranker = Reranker()

    try:
        web_search = TavilySearcher()
    except EnvironmentError:
        web_search = None

    return hybrid, reranker, figures, web_search, len(chunks)


# ======================================================================
# Page setup
# ======================================================================

st.set_page_config(
    page_title="Adaptive RAG — Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Adaptive Multimodal RAG")
st.caption("Research paper assistant · 30 papers · Gemini 3.6 Flash")


# ======================================================================
# Data check
# ======================================================================

if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
    st.error(
        "**Data not found.**\n\n"
        "Run `python ingest.py` from the project root first."
    )
    st.stop()


# ======================================================================
# Load
# ======================================================================

try:
    hybrid, reranker, figures, web_search, num_chunks = load_everything()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()


# ======================================================================
# Sidebar
# ======================================================================

with st.sidebar:
    st.header("System Status")
    st.metric("Chunks indexed", num_chunks)
    st.metric("Papers", 30)
    st.metric("Figures indexed", len(figures))

    st.divider()
    web_status = "✅ Enabled" if web_search else "❌ Disabled (no TAVILY_API_KEY)"
    st.write(f"**Web search:** {web_status}")

    st.divider()
    st.header("Settings")
    top_k_retrieval = st.slider("Retrieval candidates", 10, 40, 20, 5)
    top_k_rerank    = st.slider("Chunks after reranking", 3, 10, 5, 1)
    show_sources    = st.checkbox("Show retrieved sources", value=True)
    show_routing    = st.checkbox("Show routing decision", value=True)
    show_figure_info = st.checkbox("Show figure info", value=True)

    st.divider()
    st.caption("Adaptive Multimodal RAG · Plaksha University Capstone")


# ======================================================================
# Chat Interface
# ======================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("figure_path"):
            st.image(msg["figure_path"], caption=msg.get("caption", ""))
        if msg.get("used_web"):
            st.info("🌐 Web search was used to supplement local research papers.")

# Chat input
if question := st.chat_input("Ask a question about the research papers..."):
    
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):
        
        # ------------------------------------------------------------------
        # Step 1: LOCAL/WEB routing
        # ------------------------------------------------------------------
        with st.spinner("Routing query..."):
            routing = classify_query(question)

        if show_routing:
            with st.expander("🧭 Routing Decision", expanded=True):
                col1, col2 = st.columns(2)
                col1.metric(
                    "Source",
                    SOURCE_COLORS.get(routing["source"], "") + " " + routing["source"],
                )
                col2.metric(
                    "Intent",
                    INTENT_DESCRIPTIONS.get(routing["intent"], routing["intent"]),
                )
                st.caption(f"Reasoning: {routing['reasoning']}")

        source = routing["source"]

        # ------------------------------------------------------------------
        # Step 2: Retrieval
        # ------------------------------------------------------------------
        local_candidates = []
        web_candidates   = []
        used_web         = False

        with st.spinner("Retrieving text evidence..."):
            if source in ("LOCAL", "BOTH"):
                local_candidates = hybrid.search(question, top_k=top_k_retrieval)
            else:
                local_candidates = hybrid.search(question, top_k=10)

            if source in ("WEB", "BOTH") and web_search:
                web_candidates = web_search.search(question, max_results=5)
                used_web = True

        # ------------------------------------------------------------------
        # Step 3: Rerank
        # ------------------------------------------------------------------
        reranked = []
        with st.spinner("Reranking..."):
            if local_candidates:
                reranked = reranker.rerank(question, local_candidates, top_k=top_k_rerank)

            if source == "LOCAL" and web_search and needs_web_fallback(reranked, WEB_FALLBACK_THRESHOLD):
                web_candidates = web_search.search(question, max_results=5)
                used_web = True
                st.info("⚡ Local evidence weak — web search activated.")

            final_chunks = reranked.copy()
            if web_candidates:
                final_chunks.extend(reranker.rerank(question, web_candidates, top_k=3))

        # ------------------------------------------------------------------
        # Step 4: Show retrieved sources
        # ------------------------------------------------------------------
        if show_sources and final_chunks:
            with st.expander("📄 Retrieved Sources", expanded=False):
                for i, chunk in enumerate(final_chunks, start=1):
                    rtype = chunk.get("retrieval_type", "local")
                    icon  = "🌐" if rtype == "web" else "📄"
                    score_label = (
                        f"rerank: {chunk['rerank_score']:.3f}"
                        if "rerank_score" in chunk
                        else f"score: {chunk.get('score', 0):.3f}"
                    )
                    st.markdown(
                        f"**{icon} Source {i}** — {chunk['paper_title']} "
                        f"(Page {chunk['page']}) · _{score_label}_"
                    )
                    st.caption(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
                    st.divider()

        # ------------------------------------------------------------------
        # Step 5: Figure retrieval
        # ------------------------------------------------------------------
        figure_path = None
        caption     = ""

        with st.spinner("Searching for relevant figure..."):
            matching = find_figures_for_chunks(question, final_chunks, figures, top_k=1)

            if matching:
                best_fig    = matching[0]
                figure_path = get_figure_path(best_fig)
                caption     = best_fig.get("caption", "")
                w           = best_fig.get("width",  0)
                h           = best_fig.get("height", 0)

        if show_figure_info:
            if figure_path:
                with st.expander(
                    f"🖼️ Figure from: {best_fig.get('paper_title', '')[:60]}  "
                    f"(page {best_fig.get('page', '?')}, {w}×{h}px)",
                    expanded=True,
                ):
                    st.image(str(figure_path), caption=caption or "Extracted figure")
            elif matching:
                st.warning("Figure metadata found but image file is missing. Re-run ingest.py.")
            else:
                st.caption("No relevant figure found for this query.")

        # ------------------------------------------------------------------
        # Step 6: Generate
        # ------------------------------------------------------------------
        with st.spinner("Generating answer..."):
            if figure_path:
                answer = generate_answer_with_figure(question, final_chunks, figure_path, caption)
            else:
                answer = generate_answer(question, final_chunks)

        # ------------------------------------------------------------------
        # Display answer
        # ------------------------------------------------------------------
        st.markdown(answer)

        if used_web:
            st.info("🌐 Web search was used to supplement local research papers.")
            
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "figure_path": str(figure_path) if figure_path else None,
            "caption": caption,
            "used_web": used_web
        })

