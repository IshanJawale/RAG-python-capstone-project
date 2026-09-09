import json
import re
from pathlib import Path

import pymupdf


PAPERS_DIR = Path("papers")
DATA_DIR = Path("data")
FIGURES_DIR = DATA_DIR / "figures"


def clean_text(text):
    """Clean basic PDF extraction artifacts."""

    # Replace multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


def extract_pdf(pdf_path):
    """
    Extract text page-by-page from a PDF.
    Also extract embedded images.
    """

    doc = pymupdf.open(pdf_path)

    pages = []
    figures = []

    paper_id = pdf_path.stem

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        # -------------------------
        # TEXT
        # -------------------------

        text = page.get_text("text")
        text = clean_text(text)

        if text:
            pages.append({
                "paper_id": paper_id,
                "paper_title": pdf_path.stem,
                "page": page_number,
                "text": text
            })

        # -------------------------
        # IMAGES
        # -------------------------

        image_list = page.get_images(full=True)

        for image_index, image_info in enumerate(image_list, start=1):

            xref = image_info[0]

            try:
                image_data = doc.extract_image(xref)

                image_bytes = image_data["image"]
                image_ext = image_data["ext"]

                figure_name = (
                    f"{paper_id}_page_{page_number}"
                    f"_figure_{image_index}.{image_ext}"
                )

                figure_path = FIGURES_DIR / figure_name

                with open(figure_path, "wb") as f:
                    f.write(image_bytes)

                figures.append({
                    "figure_id": f"{paper_id}_page_{page_number}_figure_{image_index}",
                    "paper_id": paper_id,
                    "paper_title": pdf_path.stem,
                    "page": page_number,
                    "path": str(figure_path)
                })

            except Exception as e:
                print(
                    f"Warning: could not extract image "
                    f"from {pdf_path.name}, page {page_number}: {e}"
                )

    doc.close()

    return pages, figures


def ingest_corpus():
    """
    Process all PDFs in papers/.
    """

    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the papers/ directory."
        )

    all_pages = []
    all_figures = []

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for i, pdf_path in enumerate(pdf_files, start=1):

        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}")

        pages, figures = extract_pdf(pdf_path)

        all_pages.extend(pages)
        all_figures.extend(figures)

    # Save extracted figure metadata
    with open(DATA_DIR / "figures.json", "w", encoding="utf-8") as f:
        json.dump(
            all_figures,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n--------------------------------")
    print("PDF ingestion complete")
    print(f"Pages extracted : {len(all_pages)}")
    print(f"Figures extracted: {len(all_figures)}")
    print("--------------------------------\n")

    return all_pages, all_figures