
import json
import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pdf_path = "data/stl_annual_report_2024_25.pdf"
output_path = "data/stl_pages.json"

doc = pymupdf.open(pdf_path)
pages = []

for i, page in enumerate(doc):
    print(f"Processing page {i + 1}/{len(doc)}...")

    text = page.get_text().strip()

    if not text:
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )
        image = Image.open(BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image).strip()

    pages.append({
        "pdf_page": i + 1,
        "text": text
    })

doc.close()

with open(output_path, "w", encoding="utf-8") as output:
    json.dump(pages, output, ensure_ascii=False, indent=2)

print(f"\nExtraction complete! Saved {len(pages)} pages to {output_path}")
