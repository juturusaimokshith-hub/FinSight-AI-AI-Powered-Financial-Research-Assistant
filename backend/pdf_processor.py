
import fitz


def extract_pdf_pages(pdf_bytes):
    """
    Extract text from each page of an uploaded PDF.

    Returns a list of dictionaries with page numbers
    and extracted text, compatible with our RAG pipeline.
    """
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "pdf_page": page_number,
                    "text": text
                })

    return pages
