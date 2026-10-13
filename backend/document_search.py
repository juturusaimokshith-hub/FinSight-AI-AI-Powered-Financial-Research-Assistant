
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. CONFIGURATION
# ==================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)


# ==================================================
# 2. TEXT TOKENIZATION
# ==================================================

def tokenize(text):
    """
    Tokenize text for BM25.

    Preserves financial numbers with commas and decimals,
    percentages, and common alphanumeric financial terms.
    """
    text = str(text).lower()

    tokens = re.findall(
        r"\d[\d,]*(?:\.\d+)?%?|[a-z]+(?:[a-z0-9]*[a-z])?",
        text,
    )

    # Also make hyphenated financial terms searchable.
    # Example: FY2024-25 can match FY2024 and 25.
    expanded_tokens = []

    for token in tokens:
        expanded_tokens.append(token)

        if "-" in token:
            expanded_tokens.extend(
                part for part in token.split("-") if part
            )

    return expanded_tokens


# ==================================================
# 3. BUILD DOCUMENT INDEX
# ==================================================

def build_document_index(pages, document_name):
    """
    Build a hybrid-search index for one uploaded PDF.

    Input:
        pages = [
            {"pdf_page": 1, "text": "..."},
            ...
        ]

    Returns:
        chunks, index, bm25, document_name
    """
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of page dictionaries.")

    if not pages:
        raise ValueError(
            f"No pages were provided for {document_name}."
        )

    chunks = []
    seen_chunks = set()

    for page in pages:
        if not isinstance(page, dict):
            continue

        page_text = str(page.get("text", "")).strip()

        if not page_text:
            continue

        page_number = page.get("pdf_page")

        if page_number is None:
            continue

        for chunk_text in splitter.split_text(page_text):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            # Remove exact duplicate text while preserving the
            # first occurrence and its original page metadata.
            normalized_text = re.sub(
                r"\s+", " ", chunk_text
            ).strip().casefold()

            if normalized_text in seen_chunks:
                continue

            seen_chunks.add(normalized_text)

            chunks.append({
                "text": chunk_text,
                "pdf_page": int(page_number),
                "document_name": str(document_name),
            })

    if not chunks:
        raise ValueError(
            f"No extractable text found in {document_name}. "
            "The PDF may be scanned or contain no readable text."
        )

    texts = [chunk["text"] for chunk in chunks]

    # Semantic embeddings.
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Inner product on normalized embeddings gives cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Keyword index.
    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    return {
        "chunks": chunks,
        "index": index,
        "bm25": bm25,
        "document_name": str(document_name),
    }


# ==================================================
# 4. HYBRID SEARCH
# ==================================================

def search_document(
    query,
    document_index,
    top_k=3,
    candidate_k=15,
    rrf_k=60,
):
    """
    Search an uploaded PDF using:
        1. FAISS semantic retrieval
        2. BM25 keyword retrieval
        3. Reciprocal Rank Fusion

    The returned structure remains compatible with rag_answer.py.
    """

    query = str(query).strip()

    if not query:
        return []

    if not isinstance(document_index, dict):
        raise TypeError("document_index must be a dictionary.")

    required_keys = {"chunks", "index", "bm25"}

    if not required_keys.issubset(document_index):
        raise ValueError(
            "Invalid document index. Expected chunks, index and bm25."
        )

    chunks = document_index["chunks"]
    index = document_index["index"]
    bm25 = document_index["bm25"]

    if not chunks:
        return []

    if top_k < 1 or candidate_k < 1 or rrf_k < 1:
        raise ValueError(
            "top_k, candidate_k and rrf_k must all be positive."
        )

    candidate_count = min(candidate_k, len(chunks))

    # ----------------------------------------------
    # A. Semantic retrieval
    # ----------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_scores, semantic_ids = index.search(
        query_embedding,
        candidate_count,
    )

    # ----------------------------------------------
    # B. Keyword retrieval
    # ----------------------------------------------

    query_tokens = tokenize(query)
    keyword_scores = bm25.get_scores(query_tokens)

    # Stable sorting gives deterministic tie-breaking.
    keyword_ids = np.argsort(
        -keyword_scores,
        kind="stable",
    )[:candidate_count]

    # ----------------------------------------------
    # C. Reciprocal Rank Fusion
    # ----------------------------------------------

    fused_scores = {}
    semantic_score_map = {}
    keyword_score_map = {}

    for rank, chunk_id in enumerate(semantic_ids[0], start=1):
        chunk_id = int(chunk_id)

        if chunk_id < 0:
            continue

        semantic_score_map[chunk_id] = float(
            semantic_scores[0][rank - 1]
        )

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        chunk_id = int(chunk_id)

        keyword_score_map[chunk_id] = float(
            keyword_scores[chunk_id]
        )

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    # Highest fused score first; chunk ID breaks exact ties.
    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            chunk_id,
        ),
    )

    # ----------------------------------------------
    # D. Return ranked evidence
    # ----------------------------------------------

    results = []

    for chunk_id in ranked_ids:
        chunk = chunks[chunk_id]

        results.append({
            "chunk_id": chunk_id,
            "score": float(fused_scores[chunk_id]),
            "semantic_score": semantic_score_map.get(chunk_id),
            "keyword_score": keyword_score_map.get(chunk_id),
            "text": chunk["text"],
            "pdf_page": chunk["pdf_page"],
            "document_name": chunk["document_name"],
        })

        if len(results) >= top_k:
            break

    return results
