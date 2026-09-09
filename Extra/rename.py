import os

folder = "research_papers"

titles = {
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2312.10997": "Retrieval-Augmented Generation for Large Language Models: A Survey",
    "2404.10981": "A Survey on Retrieval-Augmented Text Generation for Large Language Models",
    "2405.06211": "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models",
    "2506.00054": "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers",

    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "2002.08909": "REALM: Retrieval-Augmented Language Model Pre-Training",
    "2007.01282": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
    "2112.04426": "Improving Language Models by Retrieving from Trillions of Tokens",
    "2208.03299": "Atlas: Few-shot Learning with Retrieval Augmented Language Models",
    "2212.10496": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
    "2301.12652": "REPLUG: Retrieval-Augmented Black-Box Language Models",

    "2310.11511": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "2401.15884": "Corrective Retrieval Augmented Generation",
    "2401.18059": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
    "2406.15319": "LongRAG: A Dual-Perspective Retrieval-Augmented Generation Framework for Long-Context Question Answering",
    "2408.08067": "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation",
    "2311.09476": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems",
    "2309.15217": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
    "2404.16130": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",

    "2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2302.04761": "Toolformer: Language Models Can Teach Themselves to Use Tools",
    "2201.11903": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
    "2309.11495": "Chain-of-Verification Reduces Hallucination in Large Language Models",

    "1706.03762": "Attention Is All You Need",
    "2005.14165": "Language Models are Few-Shot Learners",
    "2203.02155": "Training Language Models to Follow Instructions with Human Feedback",
    "2106.09685": "LoRA: Low-Rank Adaptation of Large Language Models",
    "2302.13971": "LLaMA: Open and Efficient Foundation Language Models"
}

for filename in os.listdir(folder):

    if not filename.lower().endswith(".pdf"):
        continue

    # Find arXiv ID in the filename
    arxiv_id = None

    for paper_id in titles:
        if paper_id in filename:
            arxiv_id = paper_id
            break

    if arxiv_id is None:
        print(f"⚠️ No title found for: {filename}")
        continue

    old_path = os.path.join(folder, filename)

    # Replace characters that Windows doesn't allow in filenames
    title = titles[arxiv_id]
    invalid_chars = '<>:"/\\|?*'
    
    for char in invalid_chars:
        title = title.replace(char, "")

    new_filename = f"{title}.pdf"
    new_path = os.path.join(folder, new_filename)

    os.rename(old_path, new_path)

    print(f"✓ {filename}")
    print(f"  → {new_filename}")

print("\nDone! All papers have been renamed.")