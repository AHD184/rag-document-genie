# This file reads a pdf (binary file) and splits it into chunks
from pypdf import PdfReader

def extract_text_from_pdf(file) -> str:    # Function to extract text from uploaded PDF file
    reader = PdfReader(file)              # Create PDF reader object
    all_text = []

    # Some PDFs may have pages that fail extraction; we safely handle that
    for page in reader.pages:                 # Iterate through each page
        text = page.extract_text() or ""      # Extract text from page, if nothing could be extracted return empty string, to avoid errors from None Type
        all_text.append(text)

    return "\n".join(all_text)           # Returns a string, with all pages' text combined but seperated by new lines

# Function to split text into chunks, started by choosing 1200 characters (a few paragraphs), overlap to maintain context and logic
# 1200 characters is a few hundred tokens, token is a piece of text the model inputs
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.replace("\r", "\n")      # To avoid bugs and be consistent
    chunks = []

    start = 0
    n = len(text)

    # Safety: if someone passes overlap >= chunk_size, the loop would never progress
    if overlap >= chunk_size:
        overlap = chunk_size // 4        # pick a safe default overlap

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        step = chunk_size - overlap
        start += step

    return chunks