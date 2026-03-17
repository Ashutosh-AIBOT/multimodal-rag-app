"""
research_agent.py — LangGraph-based research agent.

Graph flow:
  START → classify_query → [parallel: search_papers, search_books, search_repos,
                             search_websites, search_videos] → rank_results → END

Groq handles classification and intermediate steps (fast, 14k req/day free).
Gemini handles final answer synthesis.
"""

from __future__ import annotations
import traceback
import operator
from typing import TypedDict, Optional, Annotated, Any
from langgraph.graph.message import add_messages


# ── State definition ─────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    topic: str
    priority_channel: Optional[str]
    priority_repo_url: Optional[str]
    priority_paper_url: Optional[str]
    max_papers: int
    max_books: int
    max_repos: int
    papers: list
    books: list
    repos: list
    websites: list
    videos: list
    summary: str
    error: str
    groq_api_key: str
    gemini_api_key: str
    github_token: str
    youtube_api_key: str


def _default_state(
    query: str,
    groq_api_key: str = "",
    gemini_api_key: str = "",
    github_token: str = "",
    youtube_api_key: str = "",
    priority_channel: str = "",
    priority_repo_url: str = "",
    priority_paper_url: str = "",
    max_papers: int = 20,
    max_books: int = 10,
    max_repos: int = 20,
) -> ResearchState:
    return {
        "query": query,
        "topic": query,
        "priority_channel": priority_channel or None,
        "priority_repo_url": priority_repo_url or None,
        "priority_paper_url": priority_paper_url or None,
        "max_papers": max_papers,
        "max_books": max_books,
        "max_repos": max_repos,
        "papers": [],
        "books": [],
        "repos": [],
        "websites": [],
        "videos": [],
        "summary": "",
        "error": "",
        "groq_api_key": groq_api_key,
        "gemini_api_key": gemini_api_key,
        "github_token": github_token,
        "youtube_api_key": youtube_api_key,
    }


# ── Node functions ────────────────────────────────────────────────────────────

def node_refine_query(state: ResearchState) -> ResearchState:
    """Use Groq to extract a clean topic and ML-focused search terms."""
    try:
        from src.llm_clients import get_groq_llm, call_llm
        llm = get_groq_llm(state["groq_api_key"], model="llama3-8b-8192")
        if llm:
            prompt = (
                f"Extract the core ML/AI/Data Science topic from this query. "
                f"Return ONLY the refined topic (3-8 words), no explanation.\n\nQuery: {state['query']}"
            )
            refined = call_llm(llm, prompt, fallback=state["query"])
            refined = refined.strip().strip('"').strip("'")
            state["topic"] = refined if len(refined) > 3 else state["query"]
    except Exception as e:
        print(f"[research_agent] Query refinement failed: {e}")
        state["topic"] = state["query"]
    return state


def node_search_papers(state: ResearchState) -> ResearchState:
    """Search ArXiv + Semantic Scholar for research papers."""
    try:
        from src.tools.arxiv_tool import search_arxiv, search_semantic_scholar, merge_and_rank_papers

        query = state["topic"]
        priority = state.get("priority_paper_url")

        arxiv_papers = search_arxiv(query, max_results=state["max_papers"], priority_query=priority)
        ss_papers = search_semantic_scholar(query, max_results=10)
        merged = merge_and_rank_papers(arxiv_papers, ss_papers, max_results=state["max_papers"])
        state["papers"] = merged
    except Exception as e:
        print(f"[research_agent] Paper search failed: {e}\n{traceback.format_exc()}")
        state["papers"] = []
    return state


def node_search_books(state: ResearchState) -> ResearchState:
    """Search for books (free + paid)."""
    try:
        from src.tools.book_tool import search_books
        state["books"] = search_books(state["topic"], max_results=state["max_books"])
    except Exception as e:
        print(f"[research_agent] Book search failed: {e}")
        state["books"] = []
    return state


def node_search_repos(state: ResearchState) -> ResearchState:
    """Search GitHub repositories."""
    try:
        from src.tools.github_tool import search_github
        state["repos"] = search_github(
            state["topic"],
            max_results=state["max_repos"],
            github_token=state.get("github_token") or None,
            priority_repo_url=state.get("priority_repo_url") or None,
        )
    except Exception as e:
        print(f"[research_agent] Repo search failed: {e}")
        state["repos"] = []
    return state


def node_search_websites(state: ResearchState) -> ResearchState:
    """Search for authoritative websites and resources."""
    try:
        from src.tools.website_tool import search_websites
        state["websites"] = search_websites(
            state["topic"],
            max_results=10,
            priority_url=state.get("priority_repo_url") or None,
        )
    except Exception as e:
        print(f"[research_agent] Website search failed: {e}")
        state["websites"] = []
    return state


def node_search_videos(state: ResearchState) -> ResearchState:
    """Search YouTube for educational videos."""
    try:
        from src.tools.youtube_tool import search_youtube_videos
        state["videos"] = search_youtube_videos(
            state["topic"],
            max_results=10,
            priority_channel=state.get("priority_channel") or None,
            youtube_api_key=state.get("youtube_api_key") or None,
        )
    except Exception as e:
        print(f"[research_agent] Video search failed: {e}")
        state["videos"] = []
    return state


def node_generate_summary(state: ResearchState) -> ResearchState:
    """Use Gemini to generate a structured research summary."""
    try:
        from src.llm_clients import get_gemini_llm, get_groq_llm, call_llm

        # Build context for summary
        papers_ctx = "\n".join(
            f"- {p.title} ({p.published}) by {', '.join(p.authors[:2])} — {p.abstract[:150]}"
            for p in state["papers"][:5]
        ) or "No papers found."

        books_ctx = "\n".join(
            f"- {b.title} by {', '.join(b.authors[:2])} ({'Free' if b.is_free else 'Paid'})"
            for b in state["books"][:5]
        ) or "No books found."

        repos_ctx = "\n".join(
            f"- {r.full_name} ⭐{r.stars} — {r.description[:100]}"
            for r in state["repos"][:5]
        ) or "No repos found."

        prompt = f"""You are an expert ML/AI research assistant. 
Generate a concise, structured research overview for the topic: "{state['topic']}"

Based on these findings:

TOP PAPERS:
{papers_ctx}

TOP BOOKS:
{books_ctx}

TOP REPOS:
{repos_ctx}

Write a 3-4 paragraph summary covering:
1. What this topic is about and why it matters
2. Key research directions and landmark papers
3. Practical tools and implementations available
4. Recommended learning path for a data scientist/ML engineer

Be specific, cite the papers/books/repos above where relevant."""

        # Try Gemini first for better synthesis
        gemini_llm = get_gemini_llm(state["gemini_api_key"])
        groq_llm = get_groq_llm(state["groq_api_key"])
        llm = gemini_llm if gemini_llm else groq_llm

        state["summary"] = call_llm(
            llm, prompt,
            fallback=f"Research summary for '{state['topic']}': Found {len(state['papers'])} papers, {len(state['books'])} books, {len(state['repos'])} repositories."
        )
    except Exception as e:
        print(f"[research_agent] Summary generation failed: {e}\n{traceback.format_exc()}")
        state["summary"] = f"Found {len(state['papers'])} papers, {len(state['books'])} books, {len(state['repos'])} repos for '{state['topic']}'."
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_research_graph():
    """Build and compile the LangGraph research workflow."""
    try:
        from langgraph.graph import StateGraph, END

        builder = StateGraph(ResearchState)

        # Add nodes
        builder.add_node("refine_query", node_refine_query)
        builder.add_node("search_papers", node_search_papers)
        builder.add_node("search_books", node_search_books)
        builder.add_node("search_repos", node_search_repos)
        builder.add_node("search_websites", node_search_websites)
        builder.add_node("search_videos", node_search_videos)
        builder.add_node("generate_summary", node_generate_summary)

        # Flow: refine → all searches → summary
        builder.set_entry_point("refine_query")
        builder.add_edge("refine_query", "search_papers")
        builder.add_edge("refine_query", "search_books")
        builder.add_edge("refine_query", "search_repos")
        builder.add_edge("refine_query", "search_websites")
        builder.add_edge("refine_query", "search_videos")
        
        # Use parallel execution
        builder.add_edge(["search_papers", "search_books", "search_repos", "search_websites", "search_videos"], "generate_summary")
        builder.add_edge("generate_summary", END)

        return builder.compile()

    except Exception as e:
        print(f"[research_agent] Graph build failed: {e}\n{traceback.format_exc()}")
        return None


def run_research(
    query: str,
    groq_api_key: str = "",
    gemini_api_key: str = "",
    github_token: str = "",
    youtube_api_key: str = "",
    priority_channel: str = "",
    priority_repo_url: str = "",
    priority_paper_url: str = "",
    max_papers: int = 20,
    max_books: int = 10,
    max_repos: int = 20,
) -> ResearchState:
    """
    Run the full research pipeline for a query.
    Falls back to sequential execution if LangGraph fails.
    """
    state = _default_state(
        query=query,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
        github_token=github_token,
        youtube_api_key=youtube_api_key,
        priority_channel=priority_channel,
        priority_repo_url=priority_repo_url,
        priority_paper_url=priority_paper_url,
        max_papers=max_papers,
        max_books=max_books,
        max_repos=max_repos,
    )

    try:
        graph = build_research_graph()
        if graph:
            result = graph.invoke(state)
            return result
        else:
            raise RuntimeError("Graph compilation failed")
    except Exception as e:
        print(f"[research_agent] Graph execution failed ({e}), running sequentially")
        # Sequential fallback
        state = node_refine_query(state)
        state = node_search_papers(state)
        state = node_search_books(state)
        state = node_search_repos(state)
        state = node_search_websites(state)
        state = node_search_videos(state)
        state = node_generate_summary(state)
        return state
