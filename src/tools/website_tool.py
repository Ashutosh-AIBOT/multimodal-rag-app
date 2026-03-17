"""
tools/website_tool.py — Search and return authoritative ML/AI/DS websites
and online resources relevant to a given topic.
Uses DuckDuckGo (free, no API key) + curated site list.
"""

from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class WebResource:
    title: str
    url: str
    description: str
    site_type: str  # "documentation", "tutorial", "course", "blog", "tool"
    is_free: bool
    priority: int = 0  # higher = more important


# Curated authoritative ML/AI/DS websites
CURATED_SITES = {
    "documentation": [
        {"title": "PyTorch Documentation", "url": "https://pytorch.org/docs/stable/", "description": "Official PyTorch deep learning framework docs", "is_free": True},
        {"title": "TensorFlow Documentation", "url": "https://www.tensorflow.org/", "description": "Official TensorFlow ML platform docs", "is_free": True},
        {"title": "Scikit-learn Documentation", "url": "https://scikit-learn.org/stable/", "description": "ML algorithms and tools in Python", "is_free": True},
        {"title": "HuggingFace Docs", "url": "https://huggingface.co/docs", "description": "Transformers, datasets, and ML hub", "is_free": True},
        {"title": "LangChain Docs", "url": "https://python.langchain.com/docs/", "description": "LLM application framework", "is_free": True},
        {"title": "Keras Documentation", "url": "https://keras.io/", "description": "High-level neural networks API", "is_free": True},
    ],
    "course": [
        {"title": "fast.ai Courses", "url": "https://www.fast.ai/", "description": "Practical deep learning courses — free", "is_free": True},
        {"title": "deeplearning.ai", "url": "https://www.deeplearning.ai/", "description": "Andrew Ng's ML/DL courses", "is_free": False},
        {"title": "Coursera ML Specialisation", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "description": "Stanford ML course by Andrew Ng", "is_free": False},
        {"title": "MIT OpenCourseWare AI", "url": "https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/", "description": "Free MIT AI course materials", "is_free": True},
        {"title": "Stanford CS229 ML", "url": "https://cs229.stanford.edu/", "description": "Stanford Machine Learning course materials", "is_free": True},
        {"title": "Kaggle Learn", "url": "https://www.kaggle.com/learn", "description": "Free micro-courses on ML, Python, deep learning", "is_free": True},
    ],
    "blog": [
        {"title": "Papers With Code", "url": "https://paperswithcode.com/", "description": "ML papers + implementations + leaderboards", "is_free": True},
        {"title": "Distill.pub", "url": "https://distill.pub/", "description": "Interactive ML research articles", "is_free": True},
        {"title": "Google AI Blog", "url": "https://ai.googleblog.com/", "description": "Research from Google AI", "is_free": True},
        {"title": "OpenAI Blog", "url": "https://openai.com/blog", "description": "OpenAI research updates", "is_free": True},
        {"title": "Towards Data Science", "url": "https://towardsdatascience.com/", "description": "ML/DS articles and tutorials", "is_free": True},
        {"title": "Sebastian Ruder's Blog", "url": "https://ruder.io/", "description": "NLP research blog", "is_free": True},
        {"title": "Lilian Weng's Blog", "url": "https://lilianweng.github.io/", "description": "Deep dives into ML topics", "is_free": True},
        {"title": "The Gradient", "url": "https://thegradient.pub/", "description": "ML research perspectives", "is_free": True},
    ],
    "tool": [
        {"title": "Weights & Biases", "url": "https://wandb.ai/", "description": "ML experiment tracking and collaboration", "is_free": False},
        {"title": "Hugging Face Hub", "url": "https://huggingface.co/", "description": "Pre-trained models, datasets, spaces", "is_free": True},
        {"title": "Papers With Code Methods", "url": "https://paperswithcode.com/methods", "description": "ML methods taxonomy", "is_free": True},
        {"title": "Connected Papers", "url": "https://www.connectedpapers.com/", "description": "Visual graph of related papers", "is_free": True},
        {"title": "Semantic Scholar", "url": "https://www.semanticscholar.org/", "description": "AI-powered academic search engine", "is_free": True},
        {"title": "Arxiv Sanity Preserver", "url": "https://arxiv-sanity-lite.com/", "description": "ArXiv ML paper recommendations", "is_free": True},
    ],
}


def search_duckduckgo(query: str, max_results: int = 10) -> list[WebResource]:
    """Search DuckDuckGo for web resources (no API key needed)."""
    resources = []
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
            "Accept": "text/html",
        }
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}+machine+learning+tutorial"
        resp = requests.get(search_url, headers=headers, timeout=15)

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.find_all("div", class_="result")[:max_results]

            for r in results:
                try:
                    title_el = r.find("a", class_="result__a")
                    snippet_el = r.find("a", class_="result__snippet")
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    url = title_el.get("href", "")
                    description = snippet_el.get_text(strip=True) if snippet_el else ""

                    resources.append(WebResource(
                        title=title,
                        url=url,
                        description=description[:200],
                        site_type="web",
                        is_free=True,
                    ))
                except Exception:
                    pass

    except Exception as e:
        print(f"[website_tool] DuckDuckGo search failed: {e}")

    return resources


def get_curated_resources(query: str, max_per_type: int = 3) -> list[WebResource]:
    """Return curated resources matching the query across all site types."""
    query_lower = query.lower()
    matched = []

    for site_type, sites in CURATED_SITES.items():
        for s in sites:
            score = sum(
                1 for word in query_lower.split()
                if word in s["title"].lower() or word in s["description"].lower()
            )
            matched.append((score, WebResource(
                title=s["title"],
                url=s["url"],
                description=s["description"],
                site_type=site_type,
                is_free=s["is_free"],
                priority=score,
            )))

    matched.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in matched[:max_per_type * len(CURATED_SITES)]]


def search_websites(
    query: str,
    max_results: int = 10,
    priority_url: Optional[str] = None,
) -> list[WebResource]:
    """
    Comprehensive website search combining curated list + DuckDuckGo.
    """
    all_resources: dict[str, WebResource] = {}

    # Curated resources (highest priority)
    for r in get_curated_resources(query):
        all_resources[r.url] = r

    # If priority URL given, try to fetch info about it
    if priority_url:
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(priority_url, timeout=10, headers={"User-Agent": "ResearchBot/1.0"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.find("title")
                desc = soup.find("meta", attrs={"name": "description"})
                all_resources[priority_url] = WebResource(
                    title=title.get_text() if title else priority_url,
                    url=priority_url,
                    description=desc.get("content", "")[:200] if desc else "",
                    site_type="priority",
                    is_free=True,
                    priority=10,
                )
        except Exception:
            pass

    # DuckDuckGo search
    ddg_results = search_duckduckgo(query, max_results=max_results)
    for r in ddg_results:
        if r.url not in all_resources:
            all_resources[r.url] = r

    # Sort by priority, then free, then type
    result = sorted(all_resources.values(), key=lambda r: (r.priority, r.is_free), reverse=True)
    return result[:max_results]
