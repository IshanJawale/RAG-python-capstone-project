from src.ingestion import ingest_corpus
from src.chunking import create_chunks
from src.embeddings import create_embeddings


def main():

    print("\n")
    print("=" * 80)
    print("ADAPTIVE MULTIMODAL RAG - CORPUS INGESTION")
    print("=" * 80)

    # ==========================================================
    # STEP 1: INGEST PAPERS
    # ==========================================================

    print("\n[1/3] INGESTING PAPERS")

    pages, figures = ingest_corpus()

    # ==========================================================
    # STEP 2: CREATE CHUNKS
    # ==========================================================

    print("\n[2/3] CREATING CHUNKS")

    chunks = create_chunks(
        pages,
        chunk_size=600,
        overlap=100
    )

    # ==========================================================
    # STEP 3: CREATE EMBEDDINGS
    # ==========================================================

    print("\n[3/3] CREATING EMBEDDINGS")

    embeddings = create_embeddings(chunks)

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print("\n")
    print("=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(f"\nPapers processed : {len(set(p['paper_id'] for p in pages))}")
    print(f"Pages extracted  : {len(pages)}")
    print(f"Figures extracted: {len(figures)}")
    print(f"Chunks created   : {len(chunks)}")
    print(f"Embedding shape  : {embeddings.shape}")

    print("\nSaved artifacts:")
    print("  data/chunks.json")
    print("  data/figures.json")
    print("  data/figures/")
    print("  data/embeddings.npy")

    print("\nIngestion is finished.")
    print("You can now run:")
    print("  python main.py")

    print("\n")


if __name__ == "__main__":
    main()
