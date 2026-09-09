import json
from pathlib import Path


DATA_DIR = Path("data")


def chunk_text(text, chunk_size=600, overlap=100):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(pages, chunk_size=600, overlap=100):
    """
    Convert extracted pages into searchable chunks.
    """

    all_chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in page_chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "paper_id": page["paper_id"],
                "paper_title": page["paper_title"],
                "page": page["page"],
                "text": chunk
            })

            chunk_id += 1

    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("--------------------------------")
    print("Chunking complete")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_path}")
    print("--------------------------------\n")

    return all_chunks