
import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pdf_path = "data/stl_annual_report_2024_25.pdf"
doc = pymupdf.open(pdf_path)

# Render the first page as an image
page = doc[9]
pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

# Convert the rendered image into a format OCR can read
image = Image.open(BytesIO(pix.tobytes("png")))

# Recognize text
text = pytesseract.image_to_string(image)

print("OCR extracted characters:", len(text))
print("\n--- First page OCR text ---\n")
print(text[:2000] if text.strip() else "No text recognized.")

doc.close()
