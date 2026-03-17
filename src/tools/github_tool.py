"""
tools/github_tool.py — Search GitHub for relevant open-source repositories.
Uses GitHub Search API. Optional token for higher rate limits.
Prioritises official repos, high stars, and active maintenance.
"""

from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class GitHubRepo:
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    forks: int
    language: str
    topics: list[str]
    last_updated: str
    is_official: bool
    license_name: str
    open_issues: int
    relevance_score: float = 0.0


def _compute_relevance(repo: GitHubRepo, query: str) -> float:
    """Score a repo for relevance: stars, forks, recency, official status."""
    score = 0.0
    query_terms = query.lower().split()

    # Name/description match
    desc_lower = (repo.description or "").lower()
    name_lower = repo.name.lower()
    for term in query_terms:
        if term in name_lower:
            score += 3.0
        if term in desc_lower:
            score += 1.0
        if any(term in t.lower() for t in repo.topics):
            score += 2.0

    # Popularity
    score += min(repo.stars / 1000, 5.0)
    score += min(repo.forks / 500, 3.0)

    # Official bonus
    if repo.is_official:
        score += 5.0

    return score


def _is_official(repo_data: dict) -> bool:
    """Heuristic: owned by known ML orgs or has very high stars."""
    official_orgs = {
        "google", "microsoft", "openai", "anthropic", "meta-llama",
        "huggingface", "pytorch", "tensorflow", "scikit-learn",
        "keras-team", "apache", "nvidia", "deepmind", "facebookresearch",
        "google-research", "aws", "amazon", "alibaba", "baidu",
        "stanfordnlp", "allenai", "explosion",
    }
    owner = repo_data.get("owner", {}).get("login", "").lower()
    stars = repo_data.get("stargazers_count", 0)
    return owner in official_orgs or stars > 10000


def search_github(
    query: str,
    max_results: int = 20,
    github_token: Optional[str] = None,
    priority_repo_url: Optional[str] = None,
) -> list[GitHubRepo]:
    """
    Search GitHub repositories by query.
    
    Args:
        query: Search query (ML/AI/DS topic)
        max_results: Max repos to return (up to 20)
        github_token: Optional GitHub PAT for higher rate limits
        priority_repo_url: Optional specific repo URL to prioritise
    """
    repos = []
    try:
        import requests

        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ResearchAssistant/1.0"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        # Build search query — add language:python for ML focus
        search_q = f"{query} topic:machine-learning OR topic:deep-learning OR topic:data-science"

        url = "https://api.github.com/search/repositories"
        params = {
            "q": search_q,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_results + 5, 30),
        }

        resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 403:
            # Rate limited — try without topic filter
            params["q"] = query
            resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 200:
            items = resp.json().get("items", [])

            # If priority repo given, fetch and prepend
            if priority_repo_url:
                try:
                    repo_path = priority_repo_url.replace("https://github.com/", "")
                    api_url = f"https://api.github.com/repos/{repo_path.strip('/')}"
                    pr = requests.get(api_url, headers=headers, timeout=10)
                    if pr.status_code == 200:
                        items.insert(0, pr.json())
                except Exception:
                    pass

            for item in items[:max_results + 5]:
                try:
                    license_name = ""
                    if item.get("license"):
                        license_name = item["license"].get("spdx_id", "") or item["license"].get("name", "")

                    repo = GitHubRepo(
                        name=item.get("name", ""),
                        full_name=item.get("full_name", ""),
                        description=item.get("description") or "No description",
                        url=item.get("html_url", ""),
                        stars=item.get("stargazers_count", 0),
                        forks=item.get("forks_count", 0),
                        language=item.get("language") or "Unknown",
                        topics=item.get("topics", [])[:6],
                        last_updated=item.get("updated_at", "")[:10],
                        is_official=_is_official(item),
                        license_name=license_name,
                        open_issues=item.get("open_issues_count", 0),
                    )
                    repo.relevance_score = _compute_relevance(repo, query)
                    repos.append(repo)
                except Exception as e:
                    print(f"[github_tool] Repo parse error: {e}")

        elif resp.status_code == 422:
            print("[github_tool] Invalid query for GitHub search")
        else:
            print(f"[github_tool] GitHub API returned {resp.status_code}")

    except Exception as e:
        print(f"[github_tool] Search failed: {e}\n{traceback.format_exc()}")

    # Sort: official first, then by relevance score
    repos.sort(key=lambda r: (r.is_official, r.relevance_score, r.stars), reverse=True)

    # Remove duplicates
    seen = set()
    unique = []
    for r in repos:
        if r.full_name not in seen:
            seen.add(r.full_name)
            unique.append(r)

    return unique[:max_results]
