"""
indexer.py — Indexes text, images, and tables into ChromaDB.
All three modalities use the same sentence-transformer embedding space.
"""

from __future__ import annotations
import csv
import os
import traceback
from pathlib import Path

from src.chroma_store import upsert_documents, collection_count, delete_collection
from src.config import (
    COLLECTION_TEXT,
    COLLECTION_IMAGES,
    COLLECTION_TABLES,
    TABLES_DIR,
)
from src.multimodal_parser import chunk_text


# ── Text Indexing ────────────────────────────────────────────────────────────

def index_text(text_blocks: list[str], doc_name: str, reset: bool = False) -> int:
    """Chunk and index text blocks into ChromaDB. Returns count of indexed chunks."""
    try:
        if reset:
            delete_collection(COLLECTION_TEXT)

        all_ids, all_docs, all_metas = [], [], []
        chunk_global = 0

        for block_idx, block in enumerate(text_blocks):
            chunks = chunk_text(block, chunk_size=400, overlap=50)
            for c_idx, chunk in enumerate(chunks):
                doc_id = f"{doc_name}_b{block_idx}_c{c_idx}"
                all_ids.append(doc_id)
                all_docs.append(chunk)
                all_metas.append({
                    "doc_name": doc_name,
                    "block_idx": block_idx,
                    "chunk_idx": c_idx,
                    "modality": "text",
                })
                chunk_global += 1

        if all_ids:
            upsert_documents(COLLECTION_TEXT, all_ids, all_docs, all_metas)
        print(f"[indexer] Indexed {chunk_global} text chunks for '{doc_name}'")
        return chunk_global
    except Exception as e:
        print(f"[indexer] Text indexing failed: {e}\n{traceback.format_exc()}")
        return 0


# ── Image Indexing ────────────────────────────────────────────────────────────

def index_images(image_data: list[dict], doc_name: str) -> int:
    """Index image captions into ChromaDB."""
    try:
        ids, docs, metas = [], [], []
        for i, item in enumerate(image_data):
            doc_id = f"{doc_name}_img_{i}"
            ids.append(doc_id)
            docs.append(item["caption"])
            metas.append({
                "doc_name": doc_name,
                "image_path": item.get("image_path", ""),
                "image_type": item.get("image_type", "figure"),
                "modality": "image",
            })
        if ids:
            upsert_documents(COLLECTION_IMAGES, ids, docs, metas)
        print(f"[indexer] Indexed {len(ids)} image captions for '{doc_name}'")
        return len(ids)
    except Exception as e:
        print(f"[indexer] Image indexing failed: {e}")
        return 0


# ── Table Indexing ────────────────────────────────────────────────────────────

def index_tables(table_data: list[dict], doc_name: str) -> int:
    """Index table descriptions into ChromaDB."""
    try:
        ids, docs, metas = [], [], []
        for item in table_data:
            table_id = item["table_id"]
            ids.append(f"{doc_name}_{table_id}")
            docs.append(item["description"])
            metas.append({
                "doc_name": doc_name,
                "table_id": table_id,
                "csv_path": item.get("csv_path", ""),
                "page": item.get("page", 0),
                "modality": "table",
            })
        if ids:
            upsert_documents(COLLECTION_TABLES, ids, docs, metas)
        print(f"[indexer] Indexed {len(ids)} table descriptions for '{doc_name}'")
        return len(ids)
    except Exception as e:
        print(f"[indexer] Table indexing failed: {e}")
        return 0


# ── Table Processing ──────────────────────────────────────────────────────────

def _format_table_as_text(table: list[list]) -> str:
    return "\n".join(" | ".join(str(c) for c in row) for row in table)


def process_tables(tables: list[dict], llm, doc_name: str) -> list[dict]:
    """Convert extracted tables to natural-language descriptions via LLM."""
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for idx, table_meta in enumerate(tables):
        raw_rows = table_meta.get("rows", [])
        page = table_meta.get("page", 0)
        table_id = f"table_p{page}_{table_meta.get('table_index', idx)}"
        csv_path = os.path.join(TABLES_DIR, f"{doc_name}_{table_id}.csv")

        # Save CSV
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(raw_rows)
        except Exception as e:
            print(f"[indexer] CSV save failed: {e}")

        # Generate description
        description = _format_table_as_text(raw_rows)  # fallback
        if llm and raw_rows:
            try:
                table_str = _format_table_as_text(raw_rows)
                prompt = (
                    "Convert this table to a natural language description for semantic search. "
                    "Be concise but include all key values and structure.\n\n"
                    f"Table:\n{table_str}\n\nDescription:"
                )
                from src.llm_clients import call_llm
                description = call_llm(llm, prompt, fallback=table_str)
            except Exception as e:
                print(f"[indexer] Table description LLM call failed: {e}")

        results.append({
            "table_id": table_id,
            "csv_path": csv_path,
            "description": description,
            "raw_table": raw_rows,
            "page": page,
        })
    return results


# ── Image Captioning ──────────────────────────────────────────────────────────

def caption_images_with_gemini(image_paths: list[str], gemini_api_key: str) -> list[dict]:
    """Caption images using Gemini vision (replaces GPT-4V)."""
    import base64
    import io as _io
    from PIL import Image as PILImage

    results = []
    for idx, img_path in enumerate(image_paths):
        print(f"[indexer] Captioning image {idx+1}/{len(image_paths)}: {img_path}")
        caption = f"[Image: {os.path.basename(img_path)}]"
        image_type = "figure"

        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            pil_img = PILImage.open(img_path).convert("RGB")
            buf = _io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            prompt = (
                "Describe this image in detail for a document search system. "
                "Include: what the image shows, any visible text or data, "
                "type of visualization (chart/diagram/photo/table), and key insights."
            )
            response = model.generate_content([prompt, PILImage.open(img_path)])
            caption = response.text.strip()
            image_type = _infer_image_type(caption)
        except Exception as e:
            print(f"[indexer] Image captioning failed for {img_path}: {e}")

        results.append({
            "image_path": img_path,
            "caption": caption,
            "image_type": image_type,
        })
    return results


def _infer_image_type(caption: str) -> str:
    cl = caption.lower()
    if any(w in cl for w in ("chart", "bar", "pie", "line graph", "plot", "histogram")):
        return "chart"
    if any(w in cl for w in ("diagram", "flowchart", "architecture", "uml", "network")):
        return "diagram"
    if any(w in cl for w in ("table", "matrix", "grid", "spreadsheet")):
        return "table_image"
    if any(w in cl for w in ("photo", "photograph", "picture", "image of")):
        return "photo"
    return "figure"
