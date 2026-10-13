
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load the annual report text
with open("data/stl_annual_report_text.txt", "r", encoding="utf-8") as file:
    report_text = file.read()

# 2. Split the report into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_text(report_text)
print(f"Total chunks: {len(chunks)}")

# 3. Load a sentence embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 4. Convert report chunks into vectors
print("Creating embeddings...")
embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# 5. Create a FAISS vector index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype("float32"))

print(f"Vectors stored in FAISS: {index.ntotal}")

# 6. Ask a question
query = input("\nAsk a question about STL: ")

# 7. Convert the question into a vector
query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# 8. Find the top 3 semantically similar chunks
scores, indices = index.search(query_embedding, k=3)

print("\n--- Semantic Search Results ---")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    print(f"\nRank: {rank} | Similarity score: {score:.4f}")
    print(chunks[idx][:1000])
    print("-" * 60)
