
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load page-by-page report data
with open("data/stl_pages.json", "r", encoding="utf-8") as file:
    pages = json.load(file)

# 2. Split each page separately and preserve its PDF page number
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = []
for page in pages:
    page_chunks = splitter.split_text(page["text"])

    for chunk_text in page_chunks:
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "pdf_page": page["pdf_page"]
            })

print(f"Loaded {len(pages)} pages and created {len(chunks)} chunks.")

# 3. Prepare semantic search
chunk_texts = [chunk["text"] for chunk in chunks]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 4. Prepare keyword search (BM25)
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_chunks = [tokenize(chunk) for chunk in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

# 5. Combine rankings using Reciprocal Rank Fusion
def hybrid_search(query, top_k=3, candidate_k=10, rrf_k=60):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    _, semantic_ids = index.search(
        query_embedding, min(candidate_k, len(chunks))
    )

    keyword_scores = bm25.get_scores(tokenize(query))
    keyword_ids = np.argsort(keyword_scores)[::-1][
        :min(candidate_k, len(chunks))
    ]

    fused_scores = {}

    for rank, chunk_id in enumerate(semantic_ids[0]):
        if chunk_id >= 0:
            chunk_id = int(chunk_id)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank + 1)
            )

    for rank, chunk_id in enumerate(keyword_ids):
        chunk_id = int(chunk_id)
        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (rrf_k + rank + 1)
        )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": chunk_id,
            "score": fused_scores[chunk_id],
            "text": chunks[chunk_id]["text"],
            "pdf_page": chunks[chunk_id]["pdf_page"]
        }
        for chunk_id in ranked_ids
    ]

# 6. Test hybrid search
if __name__ == "__main__":
    query = input("Enter your question about STL: ")
    results = hybrid_search(query)

    print("\n--- Hybrid Search Results ---")
    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank: {rank} | "
            f"Chunk: {result['chunk_id']} | "
            f"PDF page: {result['pdf_page']} | "
            f"RRF score: {result['score']:.4f}"
        )
        print(result["text"])
        print("-" * 60)
