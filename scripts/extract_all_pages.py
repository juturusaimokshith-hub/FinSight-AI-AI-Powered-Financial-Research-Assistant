
import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pdf_path = "data/stl_annual_report_2024_25.pdf"
output_path = "data/stl_annual_report_text.txt"

doc = pymupdf.open(pdf_path)

with open(output_path, "w", encoding="utf-8") as output:
    for i, page in enumerate(doc):
        print(f"Processing page {i + 1}/{len(doc)}...")

        text = page.get_text().strip()

        if not text:
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )
            image = Image.open(BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image)

        output.write(f"\n\n--- PAGE {i + 1} ---\n\n")
        output.write(text)

doc.close()
print(f"\nExtraction complete! Text saved to: {output_path}")
