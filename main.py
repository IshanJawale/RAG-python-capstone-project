import json
import numpy as np

from src.retrieval import Retriever, print_results
from src.generation import generate_answer


# ==========================================================
# SAVED ARTIFACT PATHS
# ==========================================================

CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/embeddings.npy"


# ==========================================================
# LOAD SAVED CHUNKS
# ==========================================================

def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 80)
    print("ADAPTIVE MULTIMODAL RAG")
    print("=" * 80)

    # ======================================================
    # STEP 1: LOAD SAVED DATA
    # ======================================================

    print("\n[1/2] LOADING SAVED CORPUS")

    chunks = load_chunks()

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(f"Loaded chunks     : {len(chunks)}")
    print(f"Loaded embeddings : {embeddings.shape}")

    # ======================================================
    # STEP 2: CREATE RETRIEVER
    # ======================================================

    print("\n[2/2] LOADING RETRIEVER")

    retriever = Retriever(
        embeddings,
        chunks
    )

    print("\n")
    print("=" * 80)
    print("RAG SYSTEM READY")
    print("=" * 80)

    # ======================================================
    # QUESTION LOOP
    # ======================================================

    while True:

        print("\n")

        question = input(
            "Ask a question (or type 'exit'): "
        ).strip()

        # --------------------------------------------------
        # EXIT
        # --------------------------------------------------

        if question.lower() == "exit":
            print("\nExiting RAG system...")
            break

        if not question:
            continue

        # ==================================================
        # RETRIEVAL
        # ==================================================

        results = retriever.search(
            question,
            top_k=5
        )

        # IMPORTANT:
        # Inspect retrieved chunks
        print_results(
            question,
            results
        )

        # ==================================================
        # GENERATION
        # ==================================================

        print("\n")
        print("=" * 80)
        print("GENERATING ANSWER...")
        print("=" * 80)

        answer = generate_answer(
            question,
            results
        )

        # ==================================================
        # FINAL ANSWER
        # ==================================================

        print("\n")
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(answer)

        print("\n")


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
