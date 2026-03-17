"""
ui_components.py — Reusable Streamlit UI components for displaying research results.
"""

from __future__ import annotations
import streamlit as st
from typing import Optional


# ── Papers ────────────────────────────────────────────────────────────────────

def render_paper_card(paper, idx: int, expanded: bool = False):
    """Render a single paper as a Streamlit expander card."""
    badge = "🏆" if idx < 3 else f"#{idx+1}"
    free_badge = "🆓 Open Access" if paper.pdf_url else "📄 Abstract"
    citation_str = f"📊 {paper.citation_count} citations" if paper.citation_count else ""
    
    title_display = f"{badge} {paper.title[:80]}{'...' if len(paper.title)>80 else ''}"
    
    with st.expander(title_display, expanded=(idx == 0 and expanded)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Authors:** {', '.join(paper.authors[:4])}")
            st.markdown(f"**Published:** {paper.published}")
            if paper.categories:
                cats = " · ".join(paper.categories[:3])
                st.markdown(f"**Categories:** `{cats}`")
        with col2:
            if citation_str:
                st.info(citation_str)
            st.caption(free_badge)

        st.markdown(f"**Abstract:** {paper.abstract}")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if paper.url:
                st.link_button("📖 View Paper", paper.url, use_container_width=True)
        with btn_col2:
            if paper.pdf_url:
                st.link_button("📥 Download PDF", paper.pdf_url, use_container_width=True)


def render_papers_section(papers: list, top_n: int = 5):
    """Render the full papers section with show-more toggle."""
    if not papers:
        st.info("No papers found. Try adjusting your query.")
        return

    st.markdown(f"**Found {len(papers)} papers** (showing top {min(top_n, len(papers))})")

    # Top N always visible
    for i, paper in enumerate(papers[:top_n]):
        render_paper_card(paper, i)

    # Show more toggle
    if len(papers) > top_n:
        remaining = papers[top_n:]
        with st.expander(f"📚 Show {len(remaining)} more papers"):
            for i, paper in enumerate(remaining, start=top_n):
                render_paper_card(paper, i)


# ── Books ─────────────────────────────────────────────────────────────────────

def render_book_card(book, idx: int):
    """Render a book card."""
    free_icon = "🆓" if book.is_free else "💰"
    rating_str = f"⭐ {book.rating:.1f}" if book.rating else ""
    title_short = book.title[:70] + ('...' if len(book.title) > 70 else '')
    
    with st.expander(f"{free_icon} {title_short}", expanded=(idx == 0)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Authors:** {', '.join(book.authors[:3])}")
            st.markdown(f"**Year:** {book.year}")
            st.markdown(f"**Source:** {book.source}")
        with col2:
            if rating_str:
                st.info(rating_str)
            status = "🆓 Free" if book.is_free else "💰 Paid"
            st.caption(status)

        if book.description:
            st.markdown(f"**About:** {book.description[:300]}")

        if book.url:
            label = "📖 Read Free" if book.is_free else "🔗 View Book"
            st.link_button(label, book.url, use_container_width=True)


def render_books_section(books: list, top_n: int = 5):
    if not books:
        st.info("No books found.")
        return

    st.markdown(f"**Found {len(books)} books** (showing top {min(top_n, len(books))})")
    for i, book in enumerate(books[:top_n]):
        render_book_card(book, i)

    if len(books) > top_n:
        with st.expander(f"📚 Show {len(books) - top_n} more books"):
            for i, book in enumerate(books[top_n:], start=top_n):
                render_book_card(book, i)


# ── Repos ─────────────────────────────────────────────────────────────────────

def render_repo_card(repo, idx: int):
    """Render a GitHub repo card."""
    official = "✅ Official" if repo.is_official else ""
    lang = f"`{repo.language}`" if repo.language and repo.language != "Unknown" else ""
    title_short = repo.full_name[:60] + ('...' if len(repo.full_name) > 60 else '')

    with st.expander(f"{'🏅' if repo.is_official else '📦'} {title_short}", expanded=(idx == 0)):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⭐ Stars", f"{repo.stars:,}")
        with col2:
            st.metric("🔀 Forks", f"{repo.forks:,}")
        with col3:
            st.metric("🐛 Issues", f"{repo.open_issues:,}")

        info_parts = []
        if official:
            info_parts.append(official)
        if lang:
            info_parts.append(f"Language: {lang}")
        if repo.license_name:
            info_parts.append(f"License: `{repo.license_name}`")
        if repo.last_updated:
            info_parts.append(f"Updated: {repo.last_updated}")
        if info_parts:
            st.markdown(" · ".join(info_parts))

        if repo.description:
            st.markdown(f"**Description:** {repo.description[:200]}")

        if repo.topics:
            tags = " ".join(f"`{t}`" for t in repo.topics[:6])
            st.markdown(f"**Topics:** {tags}")

        if repo.url:
            st.link_button("🔗 View on GitHub", repo.url, use_container_width=True)


def render_repos_section(repos: list, top_n: int = 5):
    if not repos:
        st.info("No repositories found.")
        return

    st.markdown(f"**Found {len(repos)} repositories** (showing top {min(top_n, len(repos))})")
    for i, repo in enumerate(repos[:top_n]):
        render_repo_card(repo, i)

    if len(repos) > top_n:
        with st.expander(f"📂 Show {len(repos) - top_n} more repositories"):
            for i, repo in enumerate(repos[top_n:], start=top_n):
                render_repo_card(repo, i)


# ── Websites ──────────────────────────────────────────────────────────────────

def render_website_card(resource):
    """Render a web resource."""
    icon_map = {
        "documentation": "📘",
        "course": "🎓",
        "blog": "✍️",
        "tool": "🔧",
        "web": "🌐",
        "priority": "⭐",
    }
    icon = icon_map.get(resource.site_type, "🔗")
    free_badge = "🆓" if resource.is_free else "💰"

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"{icon} **[{resource.title}]({resource.url})**")
        if resource.description:
            st.caption(resource.description[:150])
    with col2:
        st.caption(f"{free_badge} {resource.site_type.title()}")
    st.divider()


def render_websites_section(websites: list, top_n: int = 8):
    if not websites:
        st.info("No web resources found.")
        return

    st.markdown(f"**Found {len(websites)} web resources**")
    for resource in websites[:top_n]:
        render_website_card(resource)

    if len(websites) > top_n:
        with st.expander(f"🌐 Show {len(websites) - top_n} more websites"):
            for resource in websites[top_n:]:
                render_website_card(resource)


# ── Videos ────────────────────────────────────────────────────────────────────

def render_video_card(video, idx: int):
    """Render a YouTube video card."""
    priority_badge = "⭐ " if video.is_priority_channel else ""
    title_short = video.title[:65] + ('...' if len(video.title) > 65 else '')

    with st.expander(f"▶️ {priority_badge}{title_short}", expanded=(idx == 0)):
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown(f"**Channel:** {video.channel}")
            if video.published and video.published != "N/A":
                st.markdown(f"**Published:** {video.published}")
            if video.duration and video.duration != "N/A":
                st.markdown(f"**Duration:** {video.duration}")
            if video.view_count and video.view_count != "N/A":
                st.markdown(f"**Views:** {video.view_count}")
        with cols[1]:
            if video.thumbnail:
                try:
                    st.image(video.thumbnail, use_column_width=True)
                except Exception:
                    pass

        if video.description:
            st.caption(video.description[:200])

        if video.url:
            st.link_button("▶️ Watch on YouTube", video.url, use_container_width=True)


def render_videos_section(videos: list, top_n: int = 5):
    if not videos:
        st.info("No videos found.")
        return

    st.markdown(f"**Found {len(videos)} videos** (showing top {min(top_n, len(videos))})")
    for i, video in enumerate(videos[:top_n]):
        render_video_card(video, i)

    if len(videos) > top_n:
        with st.expander(f"📺 Show {len(videos) - top_n} more videos"):
            for i, video in enumerate(videos[top_n:], start=top_n):
                render_video_card(video, i)


# ── RAG Results ───────────────────────────────────────────────────────────────

def render_retrieved_context(results: list):
    """Render retrieved RAG context with modality badges."""
    if not results:
        st.caption("No context retrieved.")
        return

    modality_icon = {"text": "📝", "image": "🖼️", "table": "📊"}
    modality_color = {"text": "blue", "image": "green", "table": "orange"}

    for i, r in enumerate(results[:6]):
        mod = r.get("modality", "text")
        icon = modality_icon.get(mod, "📄")
        score = r.get("score", 0)
        content_preview = r["content"][:200] + ("..." if len(r["content"]) > 200 else "")

        with st.expander(f"{icon} [{mod.upper()}] Result #{i+1} (score: {score:.3f})"):
            st.markdown(content_preview)
            meta = r.get("metadata", {})
            if meta:
                st.caption(f"Metadata: {meta}")


# ── Status helpers ────────────────────────────────────────────────────────────

def render_api_status(groq_key: str, gemini_key: str):
    """Show API key status in sidebar."""
    col1, col2 = st.columns(2)
    with col1:
        if groq_key:
            st.success("✅ Groq")
        else:
            st.error("❌ Groq")
    with col2:
        if gemini_key:
            st.success("✅ Gemini")
        else:
            st.error("❌ Gemini")


def render_index_stats(text_count: int, image_count: int, table_count: int):
    """Show indexing statistics."""
    c1, c2, c3 = st.columns(3)
    c1.metric("📝 Text Chunks", text_count)
    c2.metric("🖼️ Images", image_count)
    c3.metric("📊 Tables", table_count)
