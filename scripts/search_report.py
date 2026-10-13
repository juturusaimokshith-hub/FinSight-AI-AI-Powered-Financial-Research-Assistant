
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the extracted annual report
with open("data/stl_annual_report_text.txt", "r", encoding="utf-8") as file:
    report_text = file.read()

# 2. Split the report into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_text(report_text)

print(f"Total chunks: {len(chunks)}")

# 3. Ask the user a question
query = input("\nAsk a question about STL: ").lower()

# 4. Remove common words that are not useful for searching
stop_words = {
    "what", "are", "the", "and", "for", "with",
    "how", "does", "its", "is", "in", "of", "to",
    "a", "an", "on", "about", "do"
}

keywords = {
    word.strip(".,?!:;()")
    for word in query.split()
    if len(word.strip(".,?!:;()")) > 2
    and word.strip(".,?!:;()") not in stop_words
}

print(f"\nSearch keywords: {sorted(keywords)}")

# 5. Score each chunk based on matching keywords
scored_chunks = []

for chunk in chunks:
    chunk_lower = chunk.lower()

    matched_words = [
        word for word in keywords
        if word in chunk_lower
    ]

    score = len(matched_words)

    if score > 0:
        scored_chunks.append((score, matched_words, chunk))

# 6. Sort chunks by relevance score
scored_chunks.sort(key=lambda item: item[0], reverse=True)

# 7. Display the three most relevant chunks
print("\n--- Most relevant text ---")

if not scored_chunks:
    print("No matching text found. Try different keywords.")
else:
    for score, matched_words, chunk in scored_chunks[:3]:
        print(f"\nRelevance score: {score}")
        print(f"Matched keywords: {matched_words}")
        print("-" * 50)
        print(chunk[:1000])
