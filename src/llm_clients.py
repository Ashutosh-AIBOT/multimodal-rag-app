"""
llm_clients.py — LLM client factory.

Strategy:
  • Groq (llama3-70b-8192)  → fast routing, classification, intermediate steps
  • Gemini (gemini-1.5-flash) → final answer generation (better reasoning)

Both initialised lazily and cached in session state.
"""

from __future__ import annotations
import traceback
from typing import Optional


def get_groq_llm(api_key: str, model: str = "llama3-70b-8192"):
    """Return a LangChain ChatGroq instance."""
    try:
        from langchain_groq import ChatGroq
        if not api_key:
            return None
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=0.1,
            max_tokens=2048,
            # Removed proxies parameter that was causing the error
        )
    except Exception as e:
        print(f"[llm_clients] Groq init failed: {e}")
        return None


def get_gemini_llm(api_key: str, model: str = "gemini-1.5-flash"):
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not api_key:
            return None
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0.2,
            max_output_tokens=4096,
            convert_system_message_to_human=True,
        )
    except Exception as e:
        print(f"[llm_clients] Gemini init failed: {e}")
        return None


def call_llm(llm, prompt: str, fallback: str = "") -> str:
    """
    Safe LLM call with error handling.
    Returns fallback string on any failure.
    """
    try:
        if llm is None:
            return fallback or "[LLM not initialised — check API keys]"
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[llm_clients] LLM call failed: {e}\n{tb}")
        return fallback or f"[LLM error: {e}]"
