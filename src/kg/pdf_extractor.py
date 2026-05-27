# pdf_extractor.py
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

def extract_page_data(pdf_path, page_num):
    """Extract text and image from a specific page"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Extract text
    text = page.get_text("text")
    
    # Extract image (first image on page, or render page as image)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))  # Lower resolution (was 2.0)
    img_bytes = pix.tobytes("png")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    doc.close()
    return {
        "page_number": page_num + 1,
        "text": text.strip(),
        "image_base64": img_base64
    }

def extract_all_pages(pdf_path):
    """Extract data from all pages"""
    doc = fitz.open(pdf_path)
    pages = []
    
    for i in range(len(doc)):
        pages.append(extract_page_data(pdf_path, i))
    
    doc.close()
    return pages