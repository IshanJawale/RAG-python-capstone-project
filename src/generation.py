import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Put it inside your .env file."
    )


client = genai.Client(api_key=API_KEY)


MODEL_NAME = "gemini-3.5-flash"


def build_prompt(question, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
SOURCE {i}
Paper: {chunk['paper_title']}
Page: {chunk['page']}
Retrieval score: {chunk['score']:.4f}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a research-paper question answering assistant.

Answer the user's question using ONLY the supplied
research-paper evidence.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the supplied evidence is insufficient,
   explicitly say that there is not enough evidence.
4. Cite the paper title and page number for important claims.
5. Prefer precise technical explanations.
6. If multiple papers provide evidence, distinguish them.
7. Do not mention retrieval scores unless useful.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Now provide a concise but technically useful answer.
"""

    return prompt


def generate_answer(question, retrieved_chunks):

    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text