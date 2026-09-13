# Adaptive Multimodal RAG for Research Papers
### Ishan Jawale · Plaksha University Capstone 2026

🎥 **Watch the Demo Video:** [https://youtu.be/pvTlhTps3wo](https://youtu.be/pvTlhTps3wo)

An advanced, multimodal Retrieval-Augmented Generation (RAG) system designed to answer complex questions about AI/ML research papers. 

Unlike basic RAG systems that blindly fetch text from a single source, this system is **adaptive and multimodal**:
- **Adaptive Routing & Web Fallback**: Uses an LLM router to classify user questions. Intelligently decides whether to search the local research corpus, query the live web (e.g., for recent news or pricing), or both.
- **Hybrid Retrieval & Reranking**: Combines Dense (semantic) and Sparse (BM25 keyword) search, reranked by a Cross-Encoder for maximum precision.
- **Multimodal Generation**: Extracts both text and vector figures from PDFs. Generates answers using text context + relevant diagram images.
- **Credibility Scoring**: Automatically fetches live academic citation counts from OpenAlex to grade the credibility of retrieved sources.

---

## 🏗️ System Architecture

```text
User Question
     │
     ▼
┌─────────────────────────────┐
│      LLM Query Router       │  ← Gemini classifies: LOCAL / WEB / BOTH
│   (src/routing.py)          │    + Intent: FACT, COMPARISON, FIGURE...
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
 LOCAL Path        WEB Path
       │               │
       ▼               └──→ Tavily Search API (live web)
┌─────────────────┐
│  Hybrid Search  │  ← Dense (MiniLM) + BM25, union of top-20 each
│ (src/retrieval) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cross-Encoder   │  ← ms-marco-MiniLM re-scores all candidates jointly
│   Reranker      │    with the query, picks Top 5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Figure Retrieval│  ← Scores all extracted diagrams from top paper
│ (src/figures)   │    Caption match + paper bonus + score gating
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini 3.5     │  ← Gets: question + top 5 text chunks + diagram image
│  Flash (LLM)    │    Returns: grounded answer with citations
└─────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Tool / Library | Why it was chosen |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim) | Fast, lightweight, strong semantic retrieval |
| **Dense Search** | `numpy` cosine similarity | Sufficient for a 30-paper corpus; zero overhead |
| **Sparse Search** | `rank-bm25` | Handles exact acronyms — LoRA, RAPTOR, HyDE |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Full cross-attention over query+chunk pair; high precision |
| **Web Fallback** | `tavily-python` | Real-time external knowledge for temporal queries |
| **LLM & Vision** | `google-genai` (Gemini 3.5) | Native multimodal; image + text in a single prompt |
| **PDF Parsing** | `PyMuPDF` (fitz) | Captures both raster and vector diagrams efficiently |
| **Chunking** | `nltk` | Sentence-aware tokenization prevents mid-sentence breaks |
| **Credibility** | OpenAlex API | Fetches live academic citation counts for trust scoring |
| **Web Interface** | `streamlit` | Clean conversational chatbot interface with chat history |

---

## 🚀 How to Run It (From Scratch)

### 1. Set Up the Virtual Environment
Ensure you have Python 3.10+ installed. Open your terminal in the project root directory:

```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.\.venv\Scripts\activate

# (On Mac/Linux use: source .venv/bin/activate)
```

### 2. Install Dependencies
With the virtual environment activated, install the required packages:

```powershell
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.env` file in the root directory of the project and add your API keys:

```ini
GEMINI_API_KEY="your-gemini-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

### 4. Ingest the Corpus & Fetch Citations
Before running the system, you must ingest the PDFs (located in the `papers/` folder). This step will:
1. Parse PDFs and chunk the text (NLTK).
2. Extract images and render vector figures.
3. Compute Dense embeddings and build the BM25 index.
4. Fetch live citation counts for credibility scoring from OpenAlex.

```powershell
python ingest.py
```
*(Note: This takes a few minutes. All processed artifacts will be saved to the `data/` directory).*

### 5. Start the Application
Once ingestion is complete, launch the Streamlit web interface:

```powershell
streamlit run app/app.py
```
This will open the chatbot UI in your default web browser.

---

## 📊 Evaluation

A comprehensive evaluation of the pipeline's performance (Retrieval Recall/MRR, Routing Accuracy, Chunking Distributions, and Multimodal capability) is available in the Jupyter Notebook:

```powershell
jupyter notebook notebooks/evaluation.ipynb
```
