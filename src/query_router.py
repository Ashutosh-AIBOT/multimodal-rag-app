"""
query_router.py — Classifies queries to select relevant ChromaDB collections.
Uses Groq (fast, free) for classification to save Gemini quota for final answers.
"""

from __future__ import annotations
import json
import re
from enum import Enum


class QueryType(Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    ALL = "ALL"


_CLASSIFICATION_PROMPT = """\
Classify this query to determine which document content types to search.

Query: {query}

Options:
- TEXT: Answer in text paragraphs
- IMAGE: Requires visual/diagram/chart/photo
- TABLE: Requires numerical/tabular data
- ALL: Multiple content types needed

Patterns:
- "show", "diagram", "figure", "visual", "look like" → IMAGE
- "revenue", "statistics", "percentage", "how many", "numbers", "data" → TABLE
- "explain", "what is", "describe", "summarise" → TEXT
- broad/complex questions → ALL

Respond ONLY with valid JSON: {{"types": ["TEXT"]}}
"""


def classify_query(query: str, groq_llm) -> list[QueryType]:
    """
    Classify a query using Groq (fast, cheap). Falls back to ALL on error.
    """
    try:
        if groq_llm is None:
            return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]

        prompt = _CLASSIFICATION_PROMPT.format(query=query)
        from src.llm_clients import call_llm
        raw = call_llm(groq_llm, prompt, fallback='{"types": ["ALL"]}')

        # Extract JSON robustly
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in response")

        parsed = json.loads(json_match.group())
        type_strings: list[str] = parsed.get("types", ["ALL"])

        query_types = []
        for t in type_strings:
            t_upper = t.upper().strip()
            if t_upper == "ALL":
                return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
            try:
                query_types.append(QueryType(t_upper))
            except ValueError:
                pass

        return query_types if query_types else [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]

    except Exception as e:
        print(f"[query_router] Classification failed ({e}) — defaulting to ALL")
        return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
