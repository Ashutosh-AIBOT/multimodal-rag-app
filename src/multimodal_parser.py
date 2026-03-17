"""
multimodal_parser.py — Extracts text, images, and tables from PDF documents.
Uses pdfplumber for text/tables and PyMuPDF (fitz) for reliable image extraction.
"""

from __future__ import annotations
import io
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from src.config import IMAGES_DIR, TABLES_DIR


@dataclass
class ParsedDocument:
    file_name: str
    text_blocks: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)


def parse_document(file_path: str) -> ParsedDocument:
    """
    Parse a PDF and extract text blocks, images, and tables.
    Falls back gracefully if any extraction step fails.
    """
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)

    file_name = Path(file_path).stem
    text_blocks: list[str] = []
    image_paths: list[str] = []
    tables: list[dict] = []

    # ── Text & Tables via pdfplumber ─────────────────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_blocks.append(page_text.strip())
                except Exception as e:
                    print(f"[parser] Text extraction failed page {page_num}: {e}")

                try:
                    for t_idx, raw_table in enumerate(page.extract_tables() or []):
                        clean_rows = [
                            [cell if cell is not None else "" for cell in row]
                            for row in raw_table
                        ]
                        if clean_rows:
                            tables.append({
                                "rows": clean_rows,
                                "page": page_num,
                                "table_index": t_idx,
                            })
                except Exception as e:
                    print(f"[parser] Table extraction failed page {page_num}: {e}")
    except Exception as e:
        print(f"[parser] pdfplumber failed: {e}\n{traceback.format_exc()}")

    # ── Images via PyMuPDF ────────────────────────────────────────────────────
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_idx, img_ref in enumerate(page.get_images(full=True)):
                try:
                    xref = img_ref[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image.get("ext", "png")

                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Skip tiny images (likely icons/decorations)
                    if pil_img.width < 50 or pil_img.height < 50:
                        continue

                    img_fname = f"{file_name}_p{page_num+1}_img{img_idx}.png"
                    img_path = os.path.join(IMAGES_DIR, img_fname)
                    pil_img.save(img_path, format="PNG")
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"[parser] Image extraction failed p{page_num+1} img{img_idx}: {e}")
        doc.close()
    except Exception as e:
        print(f"[parser] PyMuPDF image extraction failed: {e}")

    print(
        f"[parser] '{file_name}': "
        f"{len(text_blocks)} text blocks, "
        f"{len(image_paths)} images, "
        f"{len(tables)} tables."
    )
    return ParsedDocument(
        file_name=file_name,
        text_blocks=text_blocks,
        image_paths=image_paths,
        tables=tables,
    )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
