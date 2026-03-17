"""
tools/book_tool.py — Search for free/open-source books on ML, AI, Data Science.
Uses Open Library and Google Books APIs (both free, no key needed for basic use).
"""

from __future__ import annotations
import traceback
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Book:
    title: str
    authors: list[str]
    description: str
    year: str
    url: str
    preview_url: str
    is_free: bool
    source: str
    cover_url: str = ""
    rating: float = 0.0


# Known free ML/AI/DS books
FREE_BOOKS_DB = [
    {
        "title": "Deep Learning",
        "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
        "url": "https://www.deeplearningbook.org/",
        "year": "2016",
        "description": "Comprehensive deep learning textbook covering fundamentals to advanced topics.",
        "is_free": True,
    },
    {
        "title": "Pattern Recognition and Machine Learning",
        "authors": ["Christopher Bishop"],
        "url": "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
        "year": "2006",
        "description": "Classic ML textbook covering probabilistic graphical models, neural networks, and more.",
        "is_free": True,
    },
    {
        "title": "Mathematics for Machine Learning",
        "authors": ["Marc Peter Deisenroth", "A. Aldo Faisal", "Cheng Soon Ong"],
        "url": "https://mml-book.github.io/",
        "year": "2020",
        "description": "Mathematical foundations for machine learning including linear algebra, calculus, and statistics.",
        "is_free": True,
    },
    {
        "title": "Neural Networks and Deep Learning",
        "authors": ["Michael Nielsen"],
        "url": "http://neuralnetworksanddeeplearning.com/",
        "year": "2019",
        "description": "Free online book explaining neural networks and deep learning concepts visually.",
        "is_free": True,
    },
    {
        "title": "Dive into Deep Learning",
        "authors": ["Aston Zhang", "Zachary Lipton", "Mu Li", "Alexander Smola"],
        "url": "https://d2l.ai/",
        "year": "2023",
        "description": "Interactive deep learning book with code examples in PyTorch, JAX, TensorFlow.",
        "is_free": True,
    },
    {
        "title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
        "authors": ["Aurélien Géron"],
        "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/",
        "year": "2022",
        "description": "Practical ML guide covering Scikit-Learn and TensorFlow/Keras.",
        "is_free": False,
    },
    {
        "title": "An Introduction to Statistical Learning",
        "authors": ["Gareth James", "Daniela Witten", "Trevor Hastie", "Robert Tibshirani"],
        "url": "https://www.statlearning.com/",
        "year": "2021",
        "description": "Statistical learning methods with applications in R and Python. Free PDF available.",
        "is_free": True,
    },
    {
        "title": "The Elements of Statistical Learning",
        "authors": ["Trevor Hastie", "Robert Tibshirani", "Jerome Friedman"],
        "url": "https://hastie.su.domains/ElemStatLearn/",
        "year": "2009",
        "description": "Advanced statistical learning theory and methods. Free PDF available.",
        "is_free": True,
    },
    {
        "title": "Reinforcement Learning: An Introduction",
        "authors": ["Richard Sutton", "Andrew Barto"],
        "url": "http://incompleteideas.net/book/the-book-2nd.html",
        "year": "2018",
        "description": "Definitive reinforcement learning textbook. Free PDF available.",
        "is_free": True,
    },
    {
        "title": "Natural Language Processing with Python",
        "authors": ["Steven Bird", "Ewan Klein", "Edward Loper"],
        "url": "https://www.nltk.org/book/",
        "year": "2019",
        "description": "NLP with NLTK — free online book for text processing.",
        "is_free": True,
    },
]


def search_open_library(query: str, max_results: int = 10) -> list[Book]:
    """Search Open Library API for books."""
    books = []
    try:
        import requests
        url = "https://openlibrary.org/search.json"
        params = {"q": query, "limit": min(max_results, 10), "fields": "key,title,author_name,first_publish_year,subject"}
        resp = requests.get(url, params=params, timeout=15, headers={"User-Agent": "ResearchAssistant/1.0"})
        if resp.status_code == 200:
            for item in resp.json().get("docs", [])[:max_results]:
                try:
                    ol_key = item.get("key", "")
                    book_url = f"https://openlibrary.org{ol_key}" if ol_key else ""
                    books.append(Book(
                        title=item.get("title", "Unknown Title"),
                        authors=item.get("author_name", ["Unknown"])[:3],
                        description=", ".join(item.get("subject", [])[:5]) or "No description",
                        year=str(item.get("first_publish_year", "Unknown")),
                        url=book_url,
                        preview_url=book_url,
                        is_free=False,
                        source="Open Library",
                    ))
                except Exception:
                    pass
    except Exception as e:
        print(f"[book_tool] Open Library search failed: {e}")
    return books


def search_google_books(query: str, max_results: int = 10) -> list[Book]:
    """Search Google Books API (free tier, no key needed for basic search)."""
    books = []
    try:
        import requests
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {"q": query, "maxResults": min(max_results, 10), "printType": "books"}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            for item in resp.json().get("items", [])[:max_results]:
                try:
                    vol = item.get("volumeInfo", {})
                    access = item.get("accessInfo", {})
                    sale = item.get("saleInfo", {})

                    is_free = (
                        access.get("viewability") in ("ALL_PAGES", "PARTIAL")
                        or sale.get("saleability") == "FREE"
                    )
                    preview_url = vol.get("previewLink", "")
                    epub = access.get("epub", {})
                    pdf = access.get("pdf", {})
                    download_url = epub.get("downloadLink") or pdf.get("downloadLink") or preview_url

                    cover = ""
                    if vol.get("imageLinks"):
                        cover = vol["imageLinks"].get("thumbnail", "")

                    books.append(Book(
                        title=vol.get("title", "Unknown Title"),
                        authors=vol.get("authors", ["Unknown"])[:3],
                        description=(vol.get("description") or "")[:300],
                        year=vol.get("publishedDate", "Unknown")[:4],
                        url=vol.get("infoLink", ""),
                        preview_url=download_url,
                        is_free=is_free,
                        source="Google Books",
                        cover_url=cover,
                        rating=vol.get("averageRating", 0.0),
                    ))
                except Exception as e:
                    print(f"[book_tool] Google Books item parse error: {e}")
    except Exception as e:
        print(f"[book_tool] Google Books search failed: {e}")
    return books


def get_curated_free_books(query: str) -> list[Book]:
    """Match query against curated list of free ML/AI books."""
    query_lower = query.lower()
    matched = []
    for b in FREE_BOOKS_DB:
        score = sum(
            1 for word in query_lower.split()
            if word in b["title"].lower() or word in b.get("description", "").lower()
        )
        if score > 0 or len(query_lower.split()) <= 2:
            matched.append((score, Book(
                title=b["title"],
                authors=b["authors"],
                description=b["description"],
                year=b["year"],
                url=b["url"],
                preview_url=b["url"],
                is_free=b["is_free"],
                source="Curated Free Books",
            )))
    matched.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in matched]


def search_books(query: str, max_results: int = 10, priority_url: Optional[str] = None) -> list[Book]:
    """
    Comprehensive book search combining multiple sources.
    Prioritises free books. Returns up to max_results books.
    """
    all_books: dict[str, Book] = {}

    # Curated free books (highest priority for ML/AI topics)
    for book in get_curated_free_books(query):
        key = book.title.lower()[:50]
        all_books[key] = book

    # Open Library
    try:
        for book in search_open_library(f"{query} machine learning", max_results=5):
            key = book.title.lower()[:50]
            if key not in all_books:
                all_books[key] = book
    except Exception:
        pass

    # Google Books
    try:
        for book in search_google_books(f"{query} data science AI", max_results=5):
            key = book.title.lower()[:50]
            if key not in all_books:
                all_books[key] = book
    except Exception:
        pass

    # Sort: free first, then by rating
    result = sorted(all_books.values(), key=lambda b: (b.is_free, b.rating), reverse=True)
    return result[:max_results]
