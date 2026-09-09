import requests
import time
import os

papers = {
    "RAG": "2005.11401",
    "RAG_Survey": "2312.10997",
    "RAG_Survey_2": "2404.10981",
    "RAG_Meeting_LLMs": "2405.06211",
    "RAG_Comprehensive_Survey": "2506.00054",

    "DPR": "2004.04906",
    "REALM": "2002.08909",
    "FiD": "2007.01282",
    "RETRO": "2112.04426",
    "Atlas": "2208.03299",
    "HyDE": "2212.10496",
    "REPLUG": "2301.12652",

    "Self_RAG": "2310.11511",
    "CRAG": "2401.15884",
    "RAPTOR": "2401.18059",
    "LongRAG": "2406.15319",
    "RAGChecker": "2408.08067",
    "ARES": "2311.09476",
    "RAGAS": "2309.15217",
    "GraphRAG": "2404.16130",

    "ReAct": "2210.03629",
    "Toolformer": "2302.04761",
    "Chain_of_Thought": "2201.11903",
    "Lost_in_the_Middle": "2307.03172",
    "Chain_of_Verification": "2309.11495",

    "Attention_Is_All_You_Need": "1706.03762",
    "GPT3": "2005.14165",
    "InstructGPT": "2203.02155",
    "LoRA": "2106.09685",
    "LLaMA": "2302.13971"
}

os.makedirs("research_papers", exist_ok=True)

for name, arxiv_id in papers.items():

    url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"

    print(f"Downloading {name}...")

    response = requests.get(url)

    if response.status_code == 200:
        filename = f"research_papers/{name}_{arxiv_id}.pdf"

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"✓ Saved: {filename}")
    else:
        print(f"✗ Failed: {name} ({response.status_code})")

    # Be polite to arXiv's servers
    time.sleep(3)

print("\nDone!")