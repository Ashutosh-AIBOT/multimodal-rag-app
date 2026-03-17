"""
rag_pipeline.py — Orchestrates the full multimodal RAG pipeline using LangGraph.

Flow:
  upload_doc → parse → [index_text, caption_images, process_tables] → ready
  query → route → retrieve → generate_answer
"""

from __future__ import annotations
import traceback
from typing import TypedDict, Optional


# ── RAG State ─────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    file_path: str
    doc_name: str
    text_blocks: list
    image_paths: list
    tables: list
    image_data: list
    table_data: list
    text_count: int
    image_count: int
    table_count: int
    query: str
    query_types: list
    retrieved_results: list
    answer: str
    error: str
    skip_images: bool
    skip_tables: bool
    groq_api_key: str
    gemini_api_key: str


def _default_rag_state(
    file_path: str = "",
    doc_name: str = "",
    groq_api_key: str = "",
    gemini_api_key: str = "",
    skip_images: bool = False,
    skip_tables: bool = False,
) -> RAGState:
    return RAGState(
        file_path=file_path,
        doc_name=doc_name,
        text_blocks=[],
        image_paths=[],
        tables=[],
        image_data=[],
        table_data=[],
        text_count=0,
        image_count=0,
        table_count=0,
        query="",
        query_types=[],
        retrieved_results=[],
        answer="",
        error="",
        skip_images=skip_images,
        skip_tables=skip_tables,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
    )


# ── Indexing Nodes ────────────────────────────────────────────────────────────

def node_parse(state: RAGState) -> RAGState:
    """Parse PDF into text, images, tables."""
    try:
        from src.multimodal_parser import parse_document
        doc = parse_document(state["file_path"])
        state["text_blocks"] = doc.text_blocks
        state["image_paths"] = doc.image_paths
        state["tables"] = doc.tables
    except Exception as e:
        state["error"] = f"Parse failed: {e}"
        print(f"[rag_pipeline] Parse failed: {e}\n{traceback.format_exc()}")
    return state


def node_index_text(state: RAGState) -> RAGState:
    """Index text chunks into ChromaDB."""
    try:
        from src.indexer import index_text
        count = index_text(state["text_blocks"], state["doc_name"], reset=True)
        state["text_count"] = count
    except Exception as e:
        print(f"[rag_pipeline] Text indexing failed: {e}")
    return state


def node_caption_images(state: RAGState) -> RAGState:
    """Caption and index images using Gemini vision."""
    if state.get("skip_images") or not state["image_paths"]:
        return state
    try:
        from src.indexer import caption_images_with_gemini, index_images
        image_data = caption_images_with_gemini(state["image_paths"], state["gemini_api_key"])
        state["image_data"] = image_data
        count = index_images(image_data, state["doc_name"])
        state["image_count"] = count
    except Exception as e:
        print(f"[rag_pipeline] Image captioning failed: {e}")
    return state


def node_process_tables(state: RAGState) -> RAGState:
    """Process and index tables."""
    if state.get("skip_tables") or not state["tables"]:
        return state
    try:
        from src.indexer import process_tables, index_tables
        from src.llm_clients import get_groq_llm
        groq_llm = get_groq_llm(state["groq_api_key"], model="llama3-8b-8192")
        table_data = process_tables(state["tables"], groq_llm, state["doc_name"])
        state["table_data"] = table_data
        count = index_tables(table_data, state["doc_name"])
        state["table_count"] = count
    except Exception as e:
        print(f"[rag_pipeline] Table processing failed: {e}")
    return state


# ── Query Nodes ───────────────────────────────────────────────────────────────

def node_route_query(state: RAGState) -> RAGState:
    """Classify the query using Groq."""
    try:
        from src.query_router import classify_query, QueryType
        from src.llm_clients import get_groq_llm
        groq_llm = get_groq_llm(state["groq_api_key"], model="llama3-8b-8192")
        query_types = classify_query(state["query"], groq_llm)
        state["query_types"] = query_types
    except Exception as e:
        print(f"[rag_pipeline] Query routing failed: {e}")
        from src.query_router import QueryType
        state["query_types"] = [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
    return state


def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant context from ChromaDB."""
    try:
        from src.retriever import retrieve_all
        results = retrieve_all(state["query"], state["query_types"], k=4)
        state["retrieved_results"] = results
    except Exception as e:
        print(f"[rag_pipeline] Retrieval failed: {e}")
        state["retrieved_results"] = []
    return state


def node_generate(state: RAGState) -> RAGState:
    """Generate final answer using Gemini."""
    try:
        from src.generator import generate_answer
        from src.llm_clients import get_gemini_llm, get_groq_llm
        gemini_llm = get_gemini_llm(state["gemini_api_key"])
        groq_llm = get_groq_llm(state["groq_api_key"])
        state["answer"] = generate_answer(
            state["query"],
            state["retrieved_results"],
            gemini_llm,
            groq_llm,
        )
    except Exception as e:
        state["answer"] = f"Answer generation failed: {e}"
        print(f"[rag_pipeline] Generation failed: {e}")
    return state


# ── Graph builders ────────────────────────────────────────────────────────────

def build_indexing_graph():
    """Build LangGraph for document indexing."""
    try:
        from langgraph.graph import StateGraph, END
        builder = StateGraph(RAGState)
        builder.add_node("parse", node_parse)
        builder.add_node("index_text", node_index_text)
        builder.add_node("caption_images", node_caption_images)
        builder.add_node("process_tables", node_process_tables)
        builder.set_entry_point("parse")
        builder.add_edge("parse", "index_text")
        builder.add_edge("parse", "caption_images")
        builder.add_edge("parse", "process_tables")
        builder.add_edge("index_text", END)
        builder.add_edge("caption_images", END)
        builder.add_edge("process_tables", END)
        return builder.compile()
    except Exception as e:
        print(f"[rag_pipeline] Indexing graph build failed: {e}")
        return None


def build_query_graph():
    """Build LangGraph for query answering."""
    try:
        from langgraph.graph import StateGraph, END
        builder = StateGraph(RAGState)
        builder.add_node("route", node_route_query)
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("generate", node_generate)
        builder.set_entry_point("route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()
    except Exception as e:
        print(f"[rag_pipeline] Query graph build failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def index_document(
    file_path: str,
    doc_name: str,
    groq_api_key: str,
    gemini_api_key: str,
    skip_images: bool = False,
    skip_tables: bool = False,
) -> dict:
    """
    Parse and index a PDF document. Returns stats dict.
    Falls back to sequential if LangGraph fails.
    """
    state = _default_rag_state(
        file_path=file_path,
        doc_name=doc_name,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
        skip_images=skip_images,
        skip_tables=skip_tables,
    )

    try:
        graph = build_indexing_graph()
        if graph:
            state = graph.invoke(state)
        else:
            raise RuntimeError("Graph build failed")
    except Exception as e:
        print(f"[rag_pipeline] Graph indexing failed ({e}), running sequentially")
        state = node_parse(state)
        state = node_index_text(state)
        state = node_caption_images(state)
        state = node_process_tables(state)

    return {
        "text_count": state.get("text_count", 0),
        "image_count": state.get("image_count", 0),
        "table_count": state.get("table_count", 0),
        "error": state.get("error", ""),
        "text_blocks": len(state.get("text_blocks", [])),
        "image_paths": len(state.get("image_paths", [])),
        "tables": len(state.get("tables", [])),
    }


def answer_query(
    query: str,
    groq_api_key: str,
    gemini_api_key: str,
) -> dict:
    """
    Answer a query against indexed documents.
    Returns answer + retrieved_results + query_types.
    """
    state = _default_rag_state(
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
    )
    state["query"] = query

    try:
        graph = build_query_graph()
        if graph:
            state = graph.invoke(state)
        else:
            raise RuntimeError("Graph build failed")
    except Exception as e:
        print(f"[rag_pipeline] Query graph failed ({e}), running sequentially")
        state = node_route_query(state)
        state = node_retrieve(state)
        state = node_generate(state)

    return {
        "answer": state.get("answer", "No answer generated."),
        "retrieved_results": state.get("retrieved_results", []),
        "query_types": [qt.value if hasattr(qt, "value") else str(qt) for qt in state.get("query_types", [])],
    }
