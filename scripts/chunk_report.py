
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_path = "data/stl_annual_report_text.txt"
output_path = "data/stl_chunks.txt"

with open(input_path, "r", encoding="utf-8") as file:
    report_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(report_text)

with open(output_path, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"\n\n--- CHUNK {i + 1} ---\n\n")
        file.write(chunk)

print("Total characters:", len(report_text))
print("Total chunks:", len(chunks))
print("First chunk:\n")
print(chunks[0])
print(f"\nChunks saved to: {output_path}")
