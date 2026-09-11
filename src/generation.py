"""
generation.py

Gemini-based answer generation.

Two generation modes:
  1. generate_answer(question, chunks)
       Standard text-only RAG answer from retrieved chunks.
       Chunks may come from local retrieval or web search (or both).

  2. generate_answer_with_figure(question, text_chunks, figure_path, caption)
       Multimodal answer: sends the question + retrieved text + figure image
       + caption to Gemini for visual reasoning.

Both functions include source citations (paper title + page) in the prompt
and instruct the model to abstain rather than hallucinate.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Add it to your .env file."
    )

client     = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite"  # fallback kept for reference
MODEL_NAME = "gemini-3.6-flash"


# ======================================================================
# Shared helpers
# ======================================================================

def _format_chunk(i: int, chunk: dict) -> str:
    """Format a single chunk as a labelled evidence block."""

    source_label = chunk.get("paper_title", "Unknown")
    page_label   = chunk.get("page", "N/A")

    # Web results include a URL instead of a page number
    if chunk.get("retrieval_type") == "web":
        source_label = chunk.get("paper_title", "Web Source")
        page_label   = chunk.get("url", "N/A")

    return (
        f"SOURCE {i}\n"
        f"Paper/Source : {source_label}\n"
        f"Page/URL     : {page_label}\n"
        f"\nContent:\n{chunk['text']}\n"
    )


def _build_context(chunks: list[dict]) -> str:
    parts = [_format_chunk(i, c) for i, c in enumerate(chunks, start=1)]
    return "\n---\n".join(parts)


# ======================================================================
# 1. Text-only generation
# ======================================================================

_TEXT_PROMPT_TEMPLATE = """\
You are a research-paper question-answering assistant.

Answer the user's question using ONLY the supplied evidence.

IMPORTANT RULES:
1. Do not invent facts or use outside knowledge.
2. If the evidence does not contain enough information to answer,
   explicitly say: "The available evidence is insufficient to answer this question."
3. Cite every important claim with the paper title and page number,
   e.g. (Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, Page 5).
4. If multiple papers provide evidence, distinguish their contributions clearly.
5. For web sources, cite the source title and URL.
6. Be concise and technically precise.
7. At the end of your answer, list all sources used.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Provide a grounded, well-cited answer.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = _build_context(retrieved_chunks)
    return _TEXT_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )


def generate_answer(
    question:         str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate a text-only grounded answer from retrieved chunks.

    Parameters
    ----------
    question         : user question string
    retrieved_chunks : list of chunk dicts (local or web)

    Returns
    -------
    Generated answer string (or error message on failure)
    """

    prompt = build_prompt(question, retrieved_chunks)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return f"[Generation error] {e}"


# ======================================================================
# 2. Multimodal generation (text + figure)
# ======================================================================

_MULTIMODAL_PROMPT_TEMPLATE = """\
You are a research-paper question-answering assistant with vision capabilities.

Answer the user's question using the supplied text evidence AND the figure image provided.

IMPORTANT RULES:
1. Describe what the figure shows when relevant to the question.
2. Use both the figure and the text evidence together.
3. Do not invent facts not visible in the figure or present in the text.
4. If the evidence or figure is insufficient, say so explicitly.
5. Cite sources (paper title + page) for text evidence.
6. Be concise and technically precise.

USER QUESTION:
{question}

FIGURE CAPTION:
{caption}

TEXT EVIDENCE:
{context}

Now provide a grounded answer that addresses both the text and the visual evidence.
"""


def generate_answer_with_figure(
    question:    str,
    text_chunks: list[dict],
    figure_path: str | Path,
    caption:     str = "",
) -> str:
    """
    Generate a multimodal answer using text chunks + a figure image.

    Parameters
    ----------
    question    : user question string
    text_chunks : list of chunk dicts for text context
    figure_path : path to the figure image file (PNG, JPG, JPEG)
    caption     : figure caption string (empty string if unavailable)

    Returns
    -------
    Generated answer string (or error message on failure)
    """

    figure_path = Path(figure_path)

    if not figure_path.exists():
        # Fall back to text-only if figure file is missing
        print(f"[Generation] Figure not found at {figure_path} — falling back to text-only.")
        return generate_answer(question, text_chunks)

    context = _build_context(text_chunks) if text_chunks else "(No text evidence retrieved.)"
    caption = caption or "(No caption available.)"

    prompt_text = _MULTIMODAL_PROMPT_TEMPLATE.format(
        question=question,
        caption=caption,
        context=context,
    )

    # Read image bytes
    try:
        image_bytes = figure_path.read_bytes()
    except Exception as e:
        print(f"[Generation] Could not read figure file: {e} — falling back to text-only.")
        return generate_answer(question, text_chunks)

    # Detect MIME type from extension
    ext_to_mime = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    mime_type = ext_to_mime.get(figure_path.suffix.lower(), "image/png")

    try:
        # Gemini multimodal input: list of [text, image_part]
        image_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data":      image_bytes,
            }
        }

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt_text, image_part],
        )

        return response.text

    except Exception as e:
        print(f"[Generation] Multimodal generation failed ({e}) — falling back to text-only.")
        return generate_answer(question, text_chunks)