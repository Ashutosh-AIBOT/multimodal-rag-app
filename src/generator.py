"""
generator.py — Final answer generation using Gemini.
Gemini is used here for its superior reasoning on mixed multimodal context.
Groq is used for all intermediate steps (routing, descriptions, etc.).
"""

from __future__ import annotations
from src.llm_clients import call_llm


def generate_answer(
    query: str,
    retrieved_results: list[dict],
    gemini_llm,
    groq_llm=None,
) -> str:
    """
    Generate a grounded, source-cited answer from multimodal retrieved context.
    Uses Gemini for final generation; falls back to Groq if Gemini unavailable.
    """
    # Separate by modality
    text_chunks, image_captions, table_descriptions = [], [], []
    image_refs = []

    for result in retrieved_results:
        modality = result.get("modality", "text")
        content = result.get("content", "").strip()
        if modality == "text":
            text_chunks.append(content)
        elif modality == "image":
            image_captions.append(content)
            path = result.get("metadata", {}).get("image_path", "")
            if path:
                image_refs.append(path)
        elif modality == "table":
            table_descriptions.append(content)

    text_section = "\n\n".join(text_chunks) or "No text context retrieved."
    image_section = "\n\n".join(image_captions) or "No image context retrieved."
    table_section = "\n\n".join(table_descriptions) or "No table context retrieved."

    prompt = f"""You are an expert research assistant. Answer the question below using ONLY the provided context.
Cite which modality (text/image/table) informed each part of your answer.
Be precise, structured, and thorough.

=== TEXT CONTEXT ===
{text_section}

=== IMAGE DESCRIPTIONS ===
{image_section}

=== TABLE DATA ===
{table_section}

=== QUESTION ===
{query}

=== ANSWER ===
Provide a comprehensive answer. At the end, state which content types (Text/Image/Table) were used."""

    # Try Gemini first, fall back to Groq
    llm = gemini_llm if gemini_llm is not None else groq_llm
    answer = call_llm(llm, prompt, fallback="Could not generate answer — check API keys.")

    # Append image references
    if image_refs:
        refs = "\n".join(f"📷 {p}" for p in image_refs)
        answer = f"{answer}\n\n**Referenced Images:**\n{refs}"

    return answer
