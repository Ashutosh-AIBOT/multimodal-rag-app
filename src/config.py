"""
config.py — Central configuration for the Multimodal RAG Research Assistant.
Reads API keys from environment / Streamlit secrets with safe fallbacks.
"""

import os
import streamlit as st


def get_config() -> dict:
    """Load all configuration from env vars or st.secrets."""
    config = {}

    # Try Streamlit secrets first (HuggingFace Spaces / Streamlit Cloud)
    try:
        config["groq_api_key"] = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        config["gemini_api_key"] = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        config["github_token"] = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
        config["youtube_api_key"] = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))
    except Exception:
        config["groq_api_key"] = os.getenv("GROQ_API_KEY", "")
        config["gemini_api_key"] = os.getenv("GEMINI_API_KEY", "")
        config["github_token"] = os.getenv("GITHUB_TOKEN", "")
        config["youtube_api_key"] = os.getenv("YOUTUBE_API_KEY", "")

    return config


# Model settings
GROQ_MODEL = "llama3-70b-8192"          # Fast, 14k req/day free
GROQ_FAST_MODEL = "llama3-8b-8192"      # Even faster for routing/classification
GEMINI_MODEL = "gemini-1.5-flash"       # For final answer generation

# ChromaDB settings
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_TEXT = "text_chunks"
COLLECTION_IMAGES = "image_captions"
COLLECTION_TABLES = "table_descriptions"

# Extraction dirs
IMAGES_DIR = "./data/extracted/images"
TABLES_DIR = "./data/extracted/tables"
UPLOADS_DIR = "./data/uploads"

# Search result limits
MAX_PAPERS = 20
MAX_BOOKS = 10
MAX_REPOS = 20
TOP_DISPLAY = 5

EMBED_MODEL = "all-MiniLM-L6-v2"
