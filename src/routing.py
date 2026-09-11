"""
routing.py

LLM-based query router using Gemini.

For each user question, the router returns three decisions:
  - intent   : LOCAL_FACT | COMPARISON | FIGURE | CURRENT_INFORMATION |
                BROAD_RESEARCH | UNKNOWN
  - source   : LOCAL | WEB | BOTH
  - modality : TEXT | IMAGE | TEXT_AND_IMAGE

The router is intentionally conservative — it defaults to LOCAL when
uncertain, and only routes to WEB when the question clearly requires
fresh or external information.

A confidence-based fallback also exists in main.py: if local retrieval
yields a weak reranker score, it additionally queries the web regardless
of what the router said.
"""

import json
import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment. "
        "Add it to your .env file."
    )

_client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"


# ------------------------------------------------------------------
# Valid categories
# ------------------------------------------------------------------

VALID_INTENTS   = {
    "LOCAL_FACT",
    "COMPARISON",
    "FIGURE",
    "CURRENT_INFORMATION",
    "BROAD_RESEARCH",
    "UNKNOWN",
}

VALID_SOURCES   = {"LOCAL", "WEB", "BOTH"}
VALID_MODALITIES = {"TEXT", "IMAGE", "TEXT_AND_IMAGE"}


# ------------------------------------------------------------------
# Router prompt
# ------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """
You are a query-routing assistant for a research-paper RAG system.

The local corpus contains 30 academic papers on RAG systems, LLMs,
retrieval methods, transformers, and related AI/ML topics.

Your job is to classify the user's question into three dimensions.

=== INTENT ===
LOCAL_FACT         - The question asks about a specific fact, method, or
                     result that is likely covered in the local corpus.
COMPARISON         - The question asks to compare methods, papers, or results.
FIGURE             - The question explicitly asks about a figure, diagram,
                     chart, table, or visual in a paper.
CURRENT_INFORMATION - The question asks about something recent, real-time,
                      or outside the scope of the local papers
                      (e.g. "latest", "current", "today's", pricing, news).
BROAD_RESEARCH     - The question requires synthesis from both local and
                     external sources.
UNKNOWN            - The question is unclear or does not fit any category.

=== SOURCE ===
LOCAL   - Answer can be found in the local research papers.
WEB     - Answer requires live web search (recent news, external data).
BOTH    - Answer would benefit from both local papers and web search.

=== MODALITY ===
TEXT              - The answer only needs text passages.
IMAGE             - The answer specifically requires a figure or diagram.
TEXT_AND_IMAGE    - The answer benefits from both text and a figure.

=== RULES ===
- Default to LOCAL unless there is a clear signal for WEB.
- Questions with words like "latest", "current", "today", "now",
  "price", "stock", "news", or specific recent years like "2025" or "2026" → WEB or BOTH.
- If the user asks about a specific paper title that is clearly not one of the foundational LLM/RAG papers, or mentions a reproduction/analysis paper from 2025+, strongly prefer BOTH or WEB.
- Questions about "Figure X", "the diagram in", "the graph shows",
  "the architecture in" → FIGURE intent, IMAGE or TEXT_AND_IMAGE modality.
- When in doubt, choose LOCAL / TEXT.

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no markdown, no explanation:

{
  "intent": "LOCAL_FACT",
  "source": "LOCAL",
  "modality": "TEXT",
  "reasoning": "One short sentence explaining your decision."
}
"""


# ------------------------------------------------------------------
# Main classification function
# ------------------------------------------------------------------

def classify_query(question: str) -> dict:
    """
    Classify a user question into routing decisions.

    Returns a dict with keys:
      intent, source, modality, reasoning

    On any failure (API error, JSON parse error, invalid value),
    returns a safe default routing to LOCAL / TEXT.
    """

    default = {
        "intent":    "LOCAL_FACT",
        "source":    "LOCAL",
        "modality":  "TEXT",
        "reasoning": "Default fallback — could not classify.",
    }

    prompt = (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Respond with JSON only."
    )

    try:

        response = _client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        raw = response.text.strip()

        # Strip any accidental markdown code fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate and normalise
        intent   = str(parsed.get("intent",   "LOCAL_FACT")).upper()
        source   = str(parsed.get("source",   "LOCAL")).upper()
        modality = str(parsed.get("modality", "TEXT")).upper()
        reasoning = str(parsed.get("reasoning", ""))

        if intent   not in VALID_INTENTS:    intent   = "LOCAL_FACT"
        if source   not in VALID_SOURCES:    source   = "LOCAL"
        if modality not in VALID_MODALITIES: modality = "TEXT"

        return {
            "intent":    intent,
            "source":    source,
            "modality":  modality,
            "reasoning": reasoning,
        }

    except Exception as e:
        print(f"[Router] Classification failed ({e}), using default routing.")
        return default


# ------------------------------------------------------------------
# Confidence-based fallback helper
# ------------------------------------------------------------------

def needs_web_fallback(
    reranked_chunks: list[dict],
    threshold: float = 0.0,
) -> bool:
    """
    Return True if the top reranked chunk's score is below `threshold`.

    Cross-encoder scores are raw logits (no fixed range).
    A score below 0.0 generally indicates little or no relevance.

    Tune this threshold using the evaluation notebook.
    """

    if not reranked_chunks:
        return True

    top_score = reranked_chunks[0].get("rerank_score", -999)

    return top_score < threshold
