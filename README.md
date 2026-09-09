# Adaptive Multimodal RAG for Research Papers

## 1. Project Overview

### Project Title

**Adaptive Multimodal RAG for Research Papers**

### One-line description

A research-paper assistant that can retrieve relevant information from a local corpus, intelligently decide when external web search is required, and use figures/diagrams from research papers when visual evidence is necessary.

### Main Research Question

> **Can an adaptive RAG system improve research-paper question answering by intelligently choosing the appropriate source (local papers or web) and evidence type (text or figures)?**

---

# 2. Track

## Track A — RAG System

This project is primarily a **Track A RAG project**.

The official Track A requirements are:

* At least 20 documents
* Load, clean and chunk documents
* Use overlapping chunks
* Embed chunks
* Store embeddings
* Retrieve using cosine similarity
* Generate answers using retrieved context
* Evaluate at least 10 queries

A NumPy matrix is sufficient as a vector store at this corpus size.

The advanced components of this project are additions to the basic RAG pipeline rather than a change to the track.

---

# 3. Overall Idea

A normal RAG system looks like:

```text
Question
   ↓
Retrieve
   ↓
Generate Answer
```

This project aims to build:

```text
                         USER QUESTION
                               │
                               ▼
                      ┌─────────────────┐
                      │  LLM QUESTION   │
                      │     ROUTER      │
                      └────────┬────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
              LOCAL          WEB        LOCAL + WEB
                 │             │             │
                 ▼             ▼             ▼
          Hybrid Search      Tavily        Both
                 │
                 ▼
        Candidate Evidence
                 │
                 ▼
             Reranker
                 │
                 ▼
          Relevant Evidence
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
         TEXT         FIGURE
          │             │
          │       Figure + Caption
          │             │
          └──────┬──────┘
                 ▼
          Gemini 2.5 Flash
                 │
                 ▼
          Grounded Answer
                 │
                 ▼
        Sources + Page Numbers
```

The important idea is:

> **The system doesn't blindly retrieve from the same place every time. It decides what evidence is needed.**

---

# 4. Scope

## Core features

These are the features that should definitely exist:

* [ ] 20–30 research papers
* [ ] PDF text extraction
* [ ] PDF figure extraction
* [ ] Text cleaning
* [ ] Text chunking with overlap
* [ ] Sentence Transformer embeddings
* [ ] NumPy-based vector search
* [ ] Cosine similarity retrieval
* [ ] BM25 retrieval
* [ ] Hybrid retrieval
* [ ] Cross-encoder reranking
* [ ] Gemini-based answer generation
* [ ] Sources/page numbers in answers
* [ ] Evaluation dataset
* [ ] Retrieval evaluation
* [ ] Answer-quality evaluation

## Advanced features

Implement these after the core system works:

* [ ] LLM-based intent classification
* [ ] Local vs Web routing
* [ ] Retrieval-confidence-based web fallback
* [ ] Tavily web search
* [ ] Figure-aware routing
* [ ] Multimodal figure answering
* [ ] Local + Web combined answers
* [ ] Abstention when evidence is insufficient
* [ ] Simple Streamlit interface

---

# 5. Things NOT in Scope

To keep the project realistic for six hours:

* No Neo4j
* No GraphRAG
* No multi-agent swarm
* No complex agent framework
* No custom vector database
* No image vector database initially
* No image embedding pipeline initially
* No fine-tuning
* No custom OCR system
* No complicated frontend
* No attempt to build an enterprise-scale RAG platform

The goal is a **small, measurable, well-evaluated RAG system**, not maximum feature count.

The lab explicitly warns that excessive ambition is a major failure mode and that a smaller system that works and is measured is preferable to an ambitious system that only partially works.

---

# 6. Recommended Technology Stack

| Component                 | Tool                       |
| ------------------------- | -------------------------- |
| Programming               | Python                     |
| PDF processing            | PyMuPDF                    |
| PDF → Markdown (optional) | PyMuPDF4LLM                |
| Text chunking             | Python                     |
| Dense embeddings          | Sentence Transformers      |
| Vector storage            | NumPy                      |
| Similarity                | Cosine similarity          |
| Sparse retrieval          | BM25                       |
| Reranking                 | Hugging Face Cross-Encoder |
| LLM                       | Gemini 2.5 Flash           |
| Web search                | Tavily                     |
| Figure extraction         | PyMuPDF                    |
| Figure understanding      | Gemini 2.5 Flash           |
| Evaluation                | Python / scikit-learn      |
| Plots                     | Matplotlib                 |
| Demo UI                   | Streamlit                  |
| Version control           | Git / GitHub               |

---

# 7. Corpus

## Target

Use approximately:

**20–30 research papers**

The minimum requirement is 20 documents.

## Recommended topic

Choose a reasonably focused research area.

Good examples:

* RAG and LLMs
* Retrieval systems
* Large Language Models
* Vision Transformers
* Multimodal LLMs
* Generative AI
* AI agents

### Recommended choice

**RAG / LLM research papers**

This makes it easier to:

* Find papers
* Generate meaningful questions
* Compare approaches
* Explain the project
* Demonstrate figure-based questions

---

# 8. Corpus Directory

Suggested structure:

```text
project/
│
├── papers/
│   ├── paper_01.pdf
│   ├── paper_02.pdf
│   ├── paper_03.pdf
│   └── ...
│
├── data/
│   ├── chunks.json
│   ├── metadata.json
│   ├── embeddings.npy
│   ├── figures/
│   │   ├── paper_01_fig_01.png
│   │   ├── paper_01_fig_02.png
│   │   └── ...
│   └── bm25.pkl
│
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── routing.py
│   ├── web_search.py
│   ├── figures.py
│   ├── generation.py
│   └── evaluation.py
│
├── notebooks/
│   └── evaluation.ipynb
│
├── app/
│   └── app.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .env
```

---

# 9. Stage 1 — PDF Processing

## Tool

**PyMuPDF**

The system will read every PDF and extract:

### Text

* Page text
* Sections
* Paragraphs

### Figures

* Embedded images
* Figure/page information

### Metadata

* Paper name
* Page number
* Figure number
* Caption where possible

---

# 10. Text Representation

Every extracted chunk should contain metadata.

Example:

```text
chunk_id:
paper_07_chunk_023

paper:
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

page:
5

section:
Experiments

text:
...
```

This is important because the final answer should be able to say:

```text
According to Paper X, ...
(Source: Paper X, page 5)
```

---

# 11. Text Cleaning

Clean obvious PDF extraction problems:

* Remove excessive whitespace
* Remove repeated headers
* Remove repeated footers
* Fix obvious line breaks
* Remove irrelevant PDF artifacts
* Preserve technical terms
* Preserve equations where possible

Do not spend excessive time perfecting PDF parsing.

The important thing is usable retrieval.

---

# 12. Stage 2 — Chunking

The extracted text needs to be split into chunks.

Starting configuration:

```text
Chunk size:
~500–800 words

Overlap:
~50–100 words
```

The exact value should not be treated as permanent.

Chunking is one of the most important variables to investigate.

The lab specifically warns that chunking is often the biggest lever on RAG answer quality and recommends revisiting it rather than choosing it once and never testing it.

---

# 13. Chunk Metadata

Every chunk should contain:

```text
chunk_id
paper_id
paper_title
page_number
section
text
```

Optional:

```text
authors
publication_year
source_url
```

---

# 14. Stage 3 — Dense Embeddings

Use:

**Sentence Transformers**

Process:

```text
Text chunk
    ↓
Embedding model
    ↓
Vector
```

For example:

```text
Chunk A → [0.13, -0.72, 0.44, ...]
Chunk B → [0.51,  0.22, 0.19, ...]
```

All vectors are stored in:

```text
embeddings.npy
```

---

# 15. Stage 4 — Dense Retrieval

When the user asks:

> "What are the limitations of RAG?"

Convert the question into an embedding.

Then:

```text
Question embedding
       ↓
Compare with all chunk embeddings
       ↓
Cosine similarity
       ↓
Rank chunks
       ↓
Top K
```

Start with:

```text
Top K = 10–20
```

Do not immediately send all of these to the LLM.

They will later be reranked.

---

# 16. Stage 5 — BM25

Add sparse retrieval using BM25.

BM25 is useful for exact terms such as:

* Model names
* Dataset names
* Acronyms
* Numbers
* Technical terminology
* Specific phrases

Example:

```text
Question:
"What learning rate was used in Paper X?"

BM25 can strongly match:
"learning rate"
"1e-5"
"5e-5"
```

---

# 17. Stage 6 — Hybrid Retrieval

Run both:

```text
             Question
                 │
        ┌────────┴────────┐
        ▼                 ▼
      Dense             BM25
        │                 │
        └────────┬────────┘
                 ▼
           Merge Results
                 │
                 ▼
         Candidate Evidence
```

Possible implementation:

```text
Dense → Top 20
BM25 → Top 20
        ↓
Union
        ↓
~20–40 candidates
```

Then rerank them.

---

# 18. Stage 7 — Reranking

Use a Hugging Face **Cross-Encoder**.

Input:

```text
(question, chunk)
```

The reranker produces a relevance score.

Pipeline:

```text
Hybrid Retrieval
      ↓
20–40 candidates
      ↓
Cross-Encoder
      ↓
Relevance scores
      ↓
Top 5
```

The top 5 chunks become the primary context for answer generation.

---

# 19. Stage 8 — Basic RAG Generation

This is the first working RAG system.

Input to Gemini:

```text
QUESTION

RETRIEVED CONTEXT

Context 1:
...

Context 2:
...

Context 3:
...

Context 4:
...

Context 5:
...
```

Prompt instructions should include:

* Answer using the supplied evidence
* Do not invent unsupported facts
* If evidence is insufficient, say so
* Mention sources
* Include page numbers where available

The lab specifically identifies confident invention as a common RAG failure and recommends instructing the model to say when the context lacks the answer.

---

# 20. Basic Answer Format

Aim for:

```text
Answer:
...

Sources:
1. Paper X — Page 4
2. Paper Y — Page 7
```

This makes the system visibly grounded.

---

# 21. Stage 9 — Intent Detection

After the basic RAG works, introduce the LLM router.

Use:

**Gemini 2.5 Flash**

The router determines:

### Intent

Possible categories:

```text
LOCAL_FACT
COMPARISON
FIGURE
CURRENT_INFORMATION
BROAD_RESEARCH
UNKNOWN
```

### Source

```text
LOCAL
WEB
BOTH
```

### Modality

```text
TEXT
IMAGE
TEXT_AND_IMAGE
```

Example:

```text
Question:
"What limitations of RAG are discussed in the papers?"

intent = LOCAL_FACT
source = LOCAL
modality = TEXT
```

---

# 22. Local vs Web Routing

There are two mechanisms.

## Mechanism A — Intent

If the question clearly requires fresh information:

```text
"latest Gemini model"
"current OpenAI API pricing"
"recent RAG research"
```

→ WEB

## Mechanism B — Retrieval confidence

For questions that might be answerable locally:

```text
Question
   ↓
Local retrieval
   ↓
Reranker
   ↓
Confidence
```

If evidence is strong:

```text
LOCAL
```

If evidence is weak:

```text
WEB
```

This avoids blindly trusting the LLM's judgment.

---

# 23. Confidence Threshold

Start with a threshold experimentally.

For example, investigate:

```text
0.60
0.70
0.75
0.80
0.85
```

Do not claim that one value is correct without measuring it.

The threshold should be selected based on your evaluation questions.

---

# 24. Stage 10 — Web Search

Use:

**Tavily API**

Flow:

```text
Question
   ↓
Router
   ↓
WEB required
   ↓
Generate search query
   ↓
Tavily
   ↓
Search results
   ↓
Extract relevant content
   ↓
Rerank
   ↓
Gemini
```

Do not simply pass every search result directly to the LLM.

Treat web results as another evidence source.

---

# 25. Local + Web Mode

Some questions require both.

Example:

> "How does the RAG approach described in these papers compare with current approaches?"

Pipeline:

```text
                Question
                    ↓
                  Router
                    ↓
                 BOTH
              ┌─────┴─────┐
              ▼           ▼
           Local         Web
           RAG          Tavily
              │           │
              └─────┬─────┘
                    ↓
              Evidence Pool
                    ↓
                 Reranker
                    ↓
                  Gemini
                    ↓
                  Answer
```

The final answer should clearly distinguish:

```text
According to the local papers...
According to current web sources...
```

---

# 26. Stage 11 — Figure Extraction

Use **PyMuPDF**.

For each PDF:

```text
Page
 ↓
Extract images
 ↓
Filter extremely small/irrelevant images
 ↓
Save useful figures
```

Store:

```text
paper_id
figure_id
page
caption
image_path
```

Example:

```text
paper_07_fig_03.png
```

---

# 27. Figure Metadata

Example:

```json
{
    "paper_id": "paper_07",
    "figure_id": "fig_03",
    "page": 6,
    "caption": "Architecture of the proposed retrieval system",
    "image_path": "figures/paper_07_fig_03.png"
}
```

The caption is particularly important because it gives you searchable text associated with the image.

---

# 28. Stage 12 — Figure Routing

If the user asks:

> "What does Figure 3 show?"

Router:

```text
intent = FIGURE
source = LOCAL
modality = IMAGE
```

Then:

```text
Question
 ↓
Identify paper
 ↓
Identify figure
 ↓
Retrieve figure
 ↓
Retrieve caption
 ↓
Retrieve relevant surrounding text
```

---

# 29. Multimodal Answering

Send Gemini:

```text
Question
+
Figure
+
Caption
+
Relevant text
```

Gemini can then interpret the visual evidence.

Example questions:

* "What does Figure 3 show?"
* "Explain the architecture in Figure 2."
* "What is the trend shown in this graph?"
* "According to Figure 4, which method performs best?"
* "How does the diagram represent the RAG pipeline?"

---

# 30. Do NOT Build Image Embeddings Initially

For this project, don't initially build:

```text
Image
 ↓
Image embedding
 ↓
Image vector database
```

It adds significant complexity without being necessary.

Instead:

```text
Figure caption
      ↓
Text retrieval
      ↓
Find relevant figure
      ↓
Gemini vision
```

This is much more realistic within the time constraint.

---

# 31. Final Routing Logic

The overall routing should approximately be:

```text
QUESTION
   │
   ▼
LLM ROUTER
   │
   ├── Current/fresh question?
   │        │
   │       YES ─────────► WEB
   │
   ├── Figure question?
   │        │
   │       YES ─────────► FIGURE RETRIEVAL
   │
   └── Otherwise
            │
            ▼
       LOCAL SEARCH
            │
            ▼
        RERANKING
            │
            ▼
      Is evidence strong?
          │       │
         YES      NO
          │       │
          ▼       ▼
        LOCAL    WEB
                  │
                  ▼
               Answer
```

---

# 32. LLM — Gemini 2.5 Flash

Primary LLM:

**Gemini 2.5 Flash**

Use it for:

* Intent classification
* Routing
* Query understanding
* Query rewriting if needed
* Answer generation
* Figure interpretation
* Combining local and web evidence

Do not use the LLM for:

* Embeddings
* BM25
* Cosine similarity
* Reranking

Those should be handled by dedicated/local tools.

---

# 33. Why Gemini?

The project requires:

* Good general reasoning
* Structured output
* Multimodal input
* API access
* Reasonable cost/free availability

Gemini 2.5 Flash is a good fit because the same model can handle both normal text reasoning and figure understanding.

---

# 34. Embedding Model

Use a local:

**Sentence Transformers**

model.

Advantages:

* No API dependency
* No embedding API costs
* Fast enough for 20–30 papers
* Easy to reproduce
* Good enough for the corpus size

---

# 35. Reranker

Use a lightweight Hugging Face **Cross-Encoder**.

Purpose:

```text
Initial retrieval:
"possibly relevant"

Reranker:
"actually relevant"
```

This should be treated as an experiment, not just an extra feature.

---

# 36. Web Search

Use:

**Tavily**

Purpose:

* Fresh information
* Questions outside local corpus
* Current information
* External comparison

Tavily should be called only when the routing logic determines that web information is necessary.

---

# 37. Suggested Answer Prompt

The final generation prompt should conceptually contain:

```text
You are a research assistant.

Answer the user's question using ONLY the supplied evidence.

If the evidence does not contain enough information,
say that the available evidence is insufficient.

Do not invent facts.

For every important claim, identify the supporting source.

Question:
{question}

Evidence:
{retrieved_evidence}
```

For multimodal questions:

```text
Question:
{question}

Text evidence:
{retrieved_text}

Figure:
{figure}

Figure caption:
{caption}
```

---

# 38. Evaluation Dataset

Create at least:

**30 questions**

Suggested breakdown:

```text
10 — Local text questions
5  — Questions requiring multiple papers
5  — Web-required questions
5  — Figure questions
5  — Insufficient/unanswerable questions
```

The lab requires at least 10 evaluation queries, but 30 gives you a much stronger evaluation.

---

# 39. Ground Truth

For each test question, record:

```text
question
expected_source
expected_document
expected_page
expected_answer
requires_figure
requires_web
```

Example:

```text
Question:
"What limitation of RAG is discussed in Paper 4?"

Expected source:
LOCAL

Expected document:
Paper 4

Expected page:
7

Requires figure:
NO

Requires web:
NO
```

---

# 40. Retrieval Evaluation

Do NOT only evaluate the final answer.

Evaluate retrieval separately.

For each question:

```text
Did the correct paper appear in Top 1?
Did it appear in Top 5?
Did it appear in Top 10?
```

Metrics:

* Recall@1
* Recall@5
* Recall@10
* MRR

The lab explicitly recommends separating retrieval evaluation from generation and recording which document should answer each question.

---

# 41. Main Retrieval Experiment

Compare:

```text
BM25
   vs
Dense Retrieval
   vs
Hybrid
   vs
Hybrid + Reranker
```

Example results table:

| Method            | Recall@5 | MRR |
| ----------------- | -------: | --: |
| BM25              |      XX% |  XX |
| Dense             |      XX% |  XX |
| Hybrid            |      XX% |  XX |
| Hybrid + Reranker |      XX% |  XX |

Do not fabricate numbers.

Run the experiments and report the actual results.

---

# 42. Chunking Experiment

Try at least two configurations.

Example:

```text
Experiment A:
500 words / 50 overlap

Experiment B:
800 words / 100 overlap
```

Evaluate both using the same questions.

Potential finding:

```text
500/50 → better retrieval
800/100 → better final answer
```

Or the opposite.

Either result is valid if measured honestly.

---

# 43. Routing Evaluation

Measure whether the agent chose correctly.

For example:

| Question | Expected | Predicted | Correct |
| -------- | -------- | --------- | ------- |
| Q1       | LOCAL    | LOCAL     | ✓       |
| Q2       | WEB      | WEB       | ✓       |
| Q3       | LOCAL    | WEB       | ✗       |

Calculate:

```text
Routing Accuracy
=
Correct routing decisions / Total questions
```

Also measure:

### Unnecessary web searches

System used web even though local evidence was sufficient.

### Missed web searches

System stayed local even though local evidence was insufficient.

These are important because a routing system should not just maximize web usage.

---

# 44. Multimodal Evaluation

Compare:

```text
Text-only RAG
```

against:

```text
Text + Figure RAG
```

using figure-dependent questions.

For example:

| System        | Figure Question Accuracy |
| ------------- | -----------------------: |
| Text-only     |                      XX% |
| Text + Figure |                      XX% |

The goal is to determine whether adding visual evidence actually improves answers.

---

# 45. Final Answer Evaluation

Evaluate:

### Correctness

Did the answer actually answer the question?

### Grounding

Is the answer supported by retrieved evidence?

### Hallucination

Did the model introduce unsupported claims?

### Abstention

Does it correctly say when the available evidence is insufficient?

---

# 46. Important Failure Cases to Test

Create questions where:

### Case 1 — Local answer exists

Expected:

```text
LOCAL
```

### Case 2 — Local answer does not exist

Expected:

```text
WEB
```

### Case 3 — Figure contains necessary information

Expected:

```text
IMAGE
```

### Case 4 — Text + figure required

Expected:

```text
TEXT + IMAGE
```

### Case 5 — Nobody has the answer

Expected:

```text
ABSTAIN
```

This will make the demo much more convincing.

---

# 47. Baselines

The rubric requires a baseline.

Use:

## Baseline 1

**Dense-only RAG**

```text
Question
 ↓
Dense retrieval
 ↓
Top 5
 ↓
Gemini
```

Then compare it with:

## Your system

```text
Hybrid
+
Reranking
+
Routing
+
Web fallback
+
Multimodal
```

This allows you to say:

> "The advanced system improved X compared with the basic RAG baseline."

---

# 48. Important: Inspect Retrieved Chunks

During development, always print:

```text
Question

Retrieved Chunk 1
Retrieved Chunk 2
Retrieved Chunk 3
Retrieved Chunk 4
Retrieved Chunk 5
```

before looking at the final answer.

The lab specifically warns that retrieval failures can easily be mistaken for generation/prompt failures.

---

# 49. What to Log

For every query, save:

```text
question
intent
selected_source
selected_modality

retrieved_chunks
retrieval_scores
reranker_scores

web_results (if used)

selected_figures (if used)

final_answer
sources
```

This will make debugging and evaluation much easier.

---

# 50. Six-Hour Execution Plan

The official structure is:

```text
Hour 1:
Crude end-to-end system

Hours 2–4:
Improve the part that measurably limits quality

Hour 5:
Evaluation + plots + README

Hour 6:
Presentation rehearsal
```

The lab explicitly says to stop building in Hour 6.

---

# 51. Hour 1 — MVP

Goal:

> **A question produces an answer from local research papers.**

Implement:

* [ ] Load PDFs
* [ ] Extract text
* [ ] Chunk text
* [ ] Generate embeddings
* [ ] Cosine similarity
* [ ] Top 5 chunks
* [ ] Gemini answer generation
* [ ] Basic source information

Figures can be extracted now but don't need to be used yet.

---

# 52. Day 1 End State

You should be able to demonstrate:

```text
Question
 ↓
Local research corpus
 ↓
Relevant chunks
 ↓
Gemini
 ↓
Answer
```

And tell the professor:

> "The basic local RAG pipeline works end-to-end. I have also extracted the figures and will use them in the multimodal extension."

The lab says the Day 1 hard requirement is an end-to-end version running, however crude.

---

# 53. Before Leaving Day 1

Save:

```text
chunks.json
embeddings.npy
metadata.json
figures/
```

Do NOT make yourself regenerate everything on Day 2.

The lab specifically recommends persisting expensive artifacts such as the chunked corpus and embedding matrix.

Also:

```text
git add .
git commit
git push
```

Your work should exist somewhere other than one laptop.

---

# 54. Hour 2 — Improve Retrieval

First investigate:

```text
Why are retrieval results wrong?
```

Look at the retrieved chunks.

Possible improvement:

```text
Dense
   ↓
BM25
   ↓
Hybrid
```

Measure before and after.

Do not simply assume hybrid is better.

---

# 55. Hour 3 — Reranking

Implement:

```text
Hybrid retrieval
      ↓
20 candidates
      ↓
Cross-Encoder
      ↓
Top 5
```

Measure:

```text
Hybrid
vs
Hybrid + Reranker
```

Choose whichever actually performs better.

---

# 56. Hour 4 — Agentic Features

Add:

### Router

```text
LOCAL
WEB
BOTH
```

### Web fallback

```text
Local retrieval
      ↓
Weak evidence
      ↓
Tavily
```

### Figure routing

```text
Text question → Text
Figure question → Figure
Mixed question → Both
```

If behind schedule, prioritize **local → web fallback** over additional fancy agent behavior.

---

# 57. Hour 5 — Evaluation

Do not add major new functionality.

Complete:

* [ ] 30 evaluation questions
* [ ] Retrieval metrics
* [ ] Routing accuracy
* [ ] Answer evaluation
* [ ] Multimodal evaluation
* [ ] Baseline comparison
* [ ] Plots
* [ ] README
* [ ] Results table

The lab specifically assigns Hour 5 to evaluation, plots and README.

---

# 58. Hour 6 — Presentation

STOP BUILDING.

Prepare your five-minute presentation.

Know:

1. What problem are you solving?
2. What did you build?
3. What did you compare?
4. What actually improved?
5. What failed?
6. What would you change with more time?

The lab explicitly says Hour 6 is for presentation rehearsal rather than further building.

---

# 59. Presentation Structure

## Slide 1 — Problem

> Research papers contain information across text, figures and multiple documents, while fresh information may exist outside the local corpus.

## Slide 2 — System

Show:

```text
Question
 ↓
Router
 ↓
Local / Web
 ↓
Hybrid Retrieval
 ↓
Reranker
 ↓
Text / Figure
 ↓
Gemini
 ↓
Answer
```

## Slide 3 — Experiments

Show:

```text
BM25
Dense
Hybrid
Hybrid + Reranker
```

## Slide 4 — Results

Show your actual numbers.

## Slide 5 — Conclusion

Answer:

> Did adaptive retrieval actually improve the system?

---

# 60. README Questions to Answer

The lab expects the README to clearly explain:

### What did you build?

Adaptive multimodal RAG over research papers.

### Why?

To investigate whether selecting the appropriate evidence source and modality improves research-paper QA.

### What did you measure?

Retrieval performance, routing accuracy, answer quality and multimodal performance.

### What did you learn?

State the actual findings, including failures.

---

# 61. Example Final Headline

Before the final presentation, create a one-sentence result.

Template:

> **"Our adaptive RAG system improved [metric] from [baseline] to [result], with the biggest improvement coming from [component]."**

Or, if something didn't help:

> **"Hybrid retrieval improved recall, but reranking produced little additional benefit on our corpus."**

Negative results are completely acceptable.

The lab explicitly values honest negative/null findings.

---

# 62. What a Strong Final Result Looks Like

Ideally you can demonstrate:

```text
                    Question
                       │
                       ▼
                 ┌───────────┐
                 │  Router   │
                 └─────┬─────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      LOCAL           WEB           BOTH
        │              │              │
        ▼              ▼              ▼
   Hybrid Search    Tavily          Both
        │
        ▼
     Reranker
        │
        ▼
   Text / Figure
        │
        ▼
      Gemini
        │
        ▼
 Grounded Answer
```

with evidence showing that each major component was actually useful.

---

# 63. Final Checklist

## Corpus

* [ ] ≥20 papers
* [ ] PDFs organized
* [ ] Paper metadata available

## Ingestion

* [ ] Text extracted
* [ ] Figures extracted
* [ ] Captions stored
* [ ] Page numbers stored

## Basic RAG

* [ ] Chunking
* [ ] Overlap
* [ ] Embeddings
* [ ] NumPy storage
* [ ] Cosine retrieval
* [ ] Gemini generation

## Retrieval Improvements

* [ ] BM25
* [ ] Dense retrieval
* [ ] Hybrid retrieval
* [ ] Cross-encoder reranking

## Agent

* [ ] Intent detection
* [ ] Local/Web routing
* [ ] Retrieval confidence
* [ ] Web fallback
* [ ] Local + Web mode

## Multimodal

* [ ] Figure retrieval
* [ ] Figure captions
* [ ] Figure questions
* [ ] Gemini multimodal answering

## Evaluation

* [ ] ≥10 questions
* [ ] Preferably ~30 questions
* [ ] Ground truth
* [ ] Recall@K
* [ ] MRR
* [ ] Routing accuracy
* [ ] Answer quality
* [ ] Grounding/hallucination
* [ ] Baseline comparison
* [ ] Plots

## Deliverables

* [ ] GitHub repository
* [ ] Reproducible code
* [ ] README
* [ ] Evaluation notebook
* [ ] Results
* [ ] Presentation
* [ ] Saved embeddings
* [ ] Saved processed corpus

---

# 64. Final Project in One Sentence

> **An adaptive multimodal RAG research assistant that learns to choose between local papers and fresh web information, retrieves the most relevant textual evidence, incorporates figures when necessary, and generates grounded answers with measurable retrieval and routing performance.**

---

# 65. Priority Order

If time becomes limited, implement in this exact order:

### Priority 1 — MUST WORK

```text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
Cosine retrieval
 ↓
Gemini
 ↓
Answer
```

### Priority 2

```text
BM25
 ↓
Hybrid retrieval
```

### Priority 3

```text
Cross-encoder reranking
```

### Priority 4

```text
Local → Web fallback
```

### Priority 5

```text
Figure extraction
 ↓
Figure understanding
```

### Priority 6

```text
LLM routing
```

### Priority 7

```text
Streamlit UI
```

If you're behind, **remove the UI first**, not evaluation.

---

# 66. Most Important Rule

The project is not successful merely because:

> "The chatbot gives good answers."

The project is successful if you can demonstrate:

> **"I built a RAG system, established a baseline, measured retrieval, changed the retrieval strategy, measured again, evaluated routing and multimodal evidence, and can explain which components actually improved the result."**

That matches the central emphasis of Track A: separate retrieval evaluation from generation, inspect retrieved chunks, and measure rather than blindly tune prompts.
