"""
tools/arxiv_tool.py — Search ArXiv for research papers.
Returns ranked results with open-access PDF links.
Focuses on ML, AI, Data Science papers.
"""

from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str
    published: str
    categories: list[str]
    citation_count: int = 0
    relevance_score: float = 0.0


def search_arxiv(
    query: str,
    max_results: int = 20,
    priority_query: Optional[str] = None,
) -> list[Paper]:
    """
    Search ArXiv for papers matching the query.

    Args:
        query: Main search query
        max_results: Maximum papers to fetch (up to 20)
        priority_query: Optional channel/repo hint to prioritise certain papers

    Returns:
        List of Paper dataclass instances, sorted by relevance/recency.
    """
    papers = []
    try:
        import arxiv

        # Build search query — add ML/AI context if not present
        search_query = query
        ml_terms = ["machine learning", "deep learning", "neural", "ai ", "data science", "nlp"]
        if not any(t in query.lower() for t in ml_terms):
            search_query = f"{query} machine learning"

        if priority_query:
            search_query = f"{priority_query} {search_query}"

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=min(max_results, 30),
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            try:
                paper = Paper(
                    title=result.title.strip(),
                    authors=[str(a) for a in result.authors[:5]],
                    abstract=result.summary[:500] + ("..." if len(result.summary) > 500 else ""),
                    url=result.entry_id,
                    pdf_url=result.pdf_url or "",
                    published=str(result.published.date()) if result.published else "Unknown",
                    categories=result.categories[:3],
                )
                papers.append(paper)
            except Exception as e:
                print(f"[arxiv_tool] Error parsing paper: {e}")
                continue

    except Exception as e:
        print(f"[arxiv_tool] ArXiv search failed: {e}\n{traceback.format_exc()}")

    return papers[:max_results]


def search_semantic_scholar(query: str, max_results: int = 10) -> list[Paper]:
    """
    Search Semantic Scholar API (free, no key needed for basic search).
    Provides citation counts for ranking.
    """
    papers = []
    try:
        import requests
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 10),
            "fields": "title,authors,abstract,year,citationCount,externalIds,openAccessPdf",
        }
        headers = {"User-Agent": "ResearchAssistant/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("data", []):
                try:
                    ext_ids = item.get("externalIds", {})
                    arxiv_id = ext_ids.get("ArXiv", "")
                    doi = ext_ids.get("DOI", "")
                    paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (
                        f"https://doi.org/{doi}" if doi else ""
                    )
                    pdf_url = ""
                    if item.get("openAccessPdf"):
                        pdf_url = item["openAccessPdf"].get("url", "")

                    papers.append(Paper(
                        title=item.get("title", "Unknown Title"),
                        authors=[a.get("name", "") for a in item.get("authors", [])[:5]],
                        abstract=(item.get("abstract") or "")[:500],
                        url=paper_url,
                        pdf_url=pdf_url,
                        published=str(item.get("year", "Unknown")),
                        categories=[],
                        citation_count=item.get("citationCount", 0),
                    ))
                except Exception as e:
                    print(f"[semantic_scholar] Paper parse error: {e}")
    except Exception as e:
        print(f"[semantic_scholar] Search failed: {e}")
    return papers


def merge_and_rank_papers(
    arxiv_papers: list[Paper],
    ss_papers: list[Paper],
    max_results: int = 20,
) -> list[Paper]:
    """Merge ArXiv + Semantic Scholar results, deduplicate, rank by citations + recency."""
    all_papers: dict[str, Paper] = {}

    for p in arxiv_papers:
        key = p.title.lower()[:60]
        all_papers[key] = p

    for p in ss_papers:
        key = p.title.lower()[:60]
        if key in all_papers:
            # Enrich existing with citation count
            all_papers[key].citation_count = max(all_papers[key].citation_count, p.citation_count)
            if p.pdf_url and not all_papers[key].pdf_url:
                all_papers[key].pdf_url = p.pdf_url
        else:
            all_papers[key] = p

    # Sort: citation count desc, then by year desc
    result = sorted(all_papers.values(), key=lambda x: (x.citation_count, x.published), reverse=True)
    return result[:max_results]
