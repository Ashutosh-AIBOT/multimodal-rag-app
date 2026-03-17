"""
app.py — Multimodal RAG Research Assistant
==========================================
Streamlit application for HuggingFace Spaces deployment.

Two modes:
  1. 📄 RAG Chat — Upload PDF, ask questions, get multimodal answers
  2. 🔬 Research Explorer — Search papers, books, repos, videos on any ML/AI topic

Stack:
  • Groq (llama3-70b-8192) — Fast routing, table descriptions, query classification
  • Gemini (gemini-1.5-flash) — Final answer synthesis, image captioning
  • ChromaDB — Persistent vector store (text + image + table)
  • LangGraph — Orchestration for both RAG indexing and research pipelines
"""

import os
import sys
import traceback
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure src is importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Create required directories ───────────────────────────────────────────────
for d in ["./data/chroma_db", "./data/uploads", "./data/extracted/images", "./data/extracted/tables"]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_keys() -> dict:
    """Load API keys from Streamlit secrets or session state inputs."""
    keys = {
        "groq": st.session_state.get("groq_key_input", ""),
        "gemini": st.session_state.get("gemini_key_input", ""),
        "github": st.session_state.get("github_key_input", ""),
        "youtube": st.session_state.get("youtube_key_input", ""),
    }
    # Override with Streamlit secrets if available
    try:
        if st.secrets.get("GROQ_API_KEY"):
            keys["groq"] = st.secrets["GROQ_API_KEY"]
        if st.secrets.get("GEMINI_API_KEY"):
            keys["gemini"] = st.secrets["GEMINI_API_KEY"]
        if st.secrets.get("GITHUB_TOKEN"):
            keys["github"] = st.secrets["GITHUB_TOKEN"]
        if st.secrets.get("YOUTUBE_API_KEY"):
            keys["youtube"] = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return keys


def validate_keys(keys: dict) -> tuple[bool, str]:
    """Check that required API keys are present."""
    missing = []
    if not keys["groq"]:
        missing.append("Groq API Key")
    if not keys["gemini"]:
        missing.append("Gemini API Key")
    if missing:
        return False, f"Missing required keys: {', '.join(missing)}"
    return True, ""


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2rem; }
    .main-header p { color: #a8b2d8; margin: 0.5rem 0 0; }
    .stat-box {
        background: #0f3460;
        border: 1px solid #e94560;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #e94560;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .section-header {
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
        color: #1a1a2e;
        font-weight: 700;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stButton > button:hover { background-color: #c73652; }
    .priority-badge {
        background: #ffd700;
        color: #000;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(keys: dict):
    with st.sidebar:
        st.markdown("## 🔑 API Keys")
        st.caption("Keys are stored only in your session.")

        keys_from_secrets = bool(
            (hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"))
        )
        if keys_from_secrets:
            st.success("✅ Keys loaded from Spaces secrets")
        else:
            st.session_state["groq_key_input"] = st.text_input(
                "Groq API Key *", value=keys["groq"],
                type="password", help="Required. Get free at console.groq.com"
            )
            st.session_state["gemini_key_input"] = st.text_input(
                "Gemini API Key *", value=keys["gemini"],
                type="password", help="Required. Get free at aistudio.google.com"
            )
            st.session_state["github_key_input"] = st.text_input(
                "GitHub Token (optional)", value=keys["github"],
                type="password", help="Optional. Increases GitHub API rate limits."
            )
            st.session_state["youtube_key_input"] = st.text_input(
                "YouTube API Key (optional)", value=keys["youtube"],
                type="password", help="Optional. Improves YouTube search quality."
            )

        st.divider()
        st.markdown("## ℹ️ About")
        st.markdown("""
**Multimodal RAG Research Assistant**

**RAG Mode:**
- Upload any PDF
- Extracts text, images & tables
- ChromaDB vector storage
- Gemini answers + Groq routing

**Research Mode:**
- ArXiv + Semantic Scholar papers
- Free & paid books
- GitHub repositories (ranked by stars)
- YouTube educational videos
- Authoritative ML websites

**Models:**
- 🚀 Groq `llama3-70b` — routing & tables
- 🤖 Gemini `1.5-flash` — final answers
- 🔍 `all-MiniLM-L6-v2` — embeddings
        """)
        st.divider()
        st.caption("Built with LangGraph · ChromaDB · Streamlit")


# ── RAG Tab ───────────────────────────────────────────────────────────────────

def render_rag_tab(keys: dict):
    st.markdown('<h2 class="section-header">📄 Multimodal Document Q&A</h2>', unsafe_allow_html=True)
    st.markdown("Upload a PDF and ask questions. The system searches across **text, images, and tables**.")

    valid, err = validate_keys(keys)
    if not valid:
        st.warning(f"⚠️ {err}")

    col_upload, col_options = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📂 Upload PDF Document",
            type=["pdf"],
            help="Upload any PDF — reports, papers, books, manuals",
        )

    with col_options:
        st.markdown("**Indexing Options**")
        skip_images = st.checkbox(
            "⏭️ Skip image captioning",
            value=False,
            help="Faster but won't understand charts/diagrams. Uses Gemini vision API."
        )
        skip_tables = st.checkbox(
            "⏭️ Skip table processing",
            value=False,
            help="Faster but won't search table data."
        )
        retrieval_k = st.slider("Results per modality (k)", 1, 6, 3)

    if uploaded_file is not None:
        doc_name = Path(uploaded_file.name).stem

        # Save uploaded file
        upload_path = f"./data/uploads/{uploaded_file.name}"
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Index button
        already_indexed = st.session_state.get(f"indexed_{doc_name}", False)
        index_label = "✅ Re-index Document" if already_indexed else "🔄 Index Document"

        col_btn1, col_btn2 = st.columns([2, 3])
        with col_btn1:
            do_index = st.button(index_label, type="primary", use_container_width=True)

        if do_index:
            if not valid:
                st.error(f"Cannot index: {err}")
            else:
                with st.spinner("🔄 Parsing and indexing document... (may take a minute)"):
                    try:
                        from src.rag_pipeline import index_document
                        stats = index_document(
                            file_path=upload_path,
                            doc_name=doc_name,
                            groq_api_key=keys["groq"],
                            gemini_api_key=keys["gemini"],
                            skip_images=skip_images,
                            skip_tables=skip_tables,
                        )
                        st.session_state[f"indexed_{doc_name}"] = True
                        st.session_state["current_doc"] = doc_name

                        if stats.get("error"):
                            st.warning(f"⚠️ Partial index: {stats['error']}")

                        st.success(f"✅ Document indexed successfully!")
                        from src.ui_components import render_index_stats
                        render_index_stats(
                            stats.get("text_count", 0),
                            stats.get("image_count", 0),
                            stats.get("table_count", 0),
                        )
                    except Exception as e:
                        st.error(f"❌ Indexing failed: {e}")
                        st.code(traceback.format_exc())

        # Q&A section
        st.divider()
        if already_indexed or st.session_state.get(f"indexed_{doc_name}"):
            st.markdown("### 💬 Ask Questions")
            st.caption(f"Querying: **{doc_name}**")

            # Chat history
            if "rag_messages" not in st.session_state:
                st.session_state["rag_messages"] = []

            # Display history
            for msg in st.session_state["rag_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input
            query = st.chat_input("Ask anything about the document...")
            if query:
                if not valid:
                    st.error(f"Cannot answer: {err}")
                else:
                    st.session_state["rag_messages"].append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Searching across text, images, and tables..."):
                            try:
                                from src.rag_pipeline import answer_query
                                result = answer_query(
                                    query=query,
                                    groq_api_key=keys["groq"],
                                    gemini_api_key=keys["gemini"],
                                )
                                answer = result["answer"]
                                qt = result.get("query_types", [])
                                retrieved = result.get("retrieved_results", [])

                                # Display modality badges
                                if qt:
                                    badges = " ".join(f"`{t}`" for t in qt)
                                    st.caption(f"🎯 Searched modalities: {badges}")

                                st.markdown(answer)

                                # Show retrieved context
                                if retrieved:
                                    with st.expander("🔎 View Retrieved Context"):
                                        from src.ui_components import render_retrieved_context
                                        render_retrieved_context(retrieved)

                                st.session_state["rag_messages"].append({
                                    "role": "assistant",
                                    "content": answer
                                })

                            except Exception as e:
                                err_msg = f"❌ Error: {e}"
                                st.error(err_msg)
                                st.code(traceback.format_exc())
                                st.session_state["rag_messages"].append({
                                    "role": "assistant",
                                    "content": err_msg
                                })

            # Clear chat button
            if st.session_state.get("rag_messages"):
                if st.button("🗑️ Clear Chat History"):
                    st.session_state["rag_messages"] = []
                    st.rerun()
        else:
            st.info("👆 Upload a PDF and click **Index Document** to begin.")


# ── Research Tab ──────────────────────────────────────────────────────────────

def render_research_tab(keys: dict):
    st.markdown('<h2 class="section-header">🔬 ML/AI Research Explorer</h2>', unsafe_allow_html=True)
    st.markdown("Search for the best papers, books, repos, and videos on any ML/AI/Data Science topic.")

    # ── Search form ───────────────────────────────────────────────────────────
    with st.form("research_form"):
        st.markdown("### 🔍 Search Query")
        query = st.text_input(
            "Topic or question",
            placeholder="e.g. 'Transformer attention mechanisms', 'Graph Neural Networks for drug discovery'",
            help="Be specific for better results. ML/AI/DS topics work best.",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Priority Hints** *(optional)*")
            priority_channel = st.text_input(
                "📺 YouTube Channel",
                placeholder="e.g. 3Blue1Brown, Yannic Kilcher",
                help="Prioritise videos from this channel",
            )
            priority_repo = st.text_input(
                "💻 GitHub Repo URL",
                placeholder="e.g. https://github.com/huggingface/transformers",
                help="Prioritise this repository in results",
            )
            priority_paper = st.text_input(
                "📄 Paper URL / Keyword",
                placeholder="e.g. 'Attention is All You Need' or arxiv link",
                help="Prioritise this paper or author",
            )

        with col2:
            st.markdown("**Result Limits**")
            max_papers = st.slider("Max papers", 5, 20, 10)
            max_repos = st.slider("Max repositories", 5, 20, 10)
            max_books = st.slider("Max books", 3, 10, 5)
            top_display = st.slider("Top results shown (expandable for more)", 3, 10, 5)

        submitted = st.form_submit_button("🚀 Search Everything", type="primary", use_container_width=True)

    # ── Run research ──────────────────────────────────────────────────────────
    if submitted and query.strip():
        valid, err = validate_keys(keys)
        if not valid:
            st.warning(f"⚠️ {err} — some features (AI summary) will be limited.")

        progress_bar = st.progress(0, text="Starting research pipeline...")
        status_text = st.empty()

        try:
            from src.research_agent import run_research

            status_text.text("🔍 Refining query with Groq...")
            progress_bar.progress(10, text="Refining query...")

            result = None
            with st.spinner("⚙️ Running full research pipeline (LangGraph)..."):
                result = run_research(
                    query=query.strip(),
                    groq_api_key=keys["groq"],
                    gemini_api_key=keys["gemini"],
                    github_token=keys["github"],
                    youtube_api_key=keys["youtube"],
                    priority_channel=priority_channel,
                    priority_repo_url=priority_repo,
                    priority_paper_url=priority_paper,
                    max_papers=max_papers,
                    max_books=max_books,
                    max_repos=max_repos,
                )

            progress_bar.progress(100, text="✅ Research complete!")
            status_text.empty()

            if result is None:
                st.error("Research pipeline returned no results.")
                return

            # ── Overview ──────────────────────────────────────────────────────
            refined_topic = result.get("topic", query)
            st.markdown(f"## 📊 Research Results: *{refined_topic}*")

            # Stats row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📄 Papers", len(result.get("papers", [])))
            c2.metric("📚 Books", len(result.get("books", [])))
            c3.metric("💻 Repos", len(result.get("repos", [])))
            c4.metric("🌐 Websites", len(result.get("websites", [])))
            c5.metric("▶️ Videos", len(result.get("videos", [])))

            # ── AI Summary ────────────────────────────────────────────────────
            if result.get("summary"):
                st.markdown("### 🤖 AI Research Summary")
                st.info(result["summary"])

            st.divider()

            # ── Tabbed results ────────────────────────────────────────────────
            from src.ui_components import (
                render_papers_section, render_books_section,
                render_repos_section, render_websites_section,
                render_videos_section,
            )

            tab_papers, tab_books, tab_repos, tab_websites, tab_videos = st.tabs([
                f"📄 Papers ({len(result.get('papers', []))})",
                f"📚 Books ({len(result.get('books', []))})",
                f"💻 Repos ({len(result.get('repos', []))})",
                f"🌐 Websites ({len(result.get('websites', []))})",
                f"▶️ Videos ({len(result.get('videos', []))})",
            ])

            with tab_papers:
                st.markdown("#### Research Papers")
                st.caption("Sources: ArXiv + Semantic Scholar · Ranked by citations + relevance")
                render_papers_section(result.get("papers", []), top_n=top_display)

            with tab_books:
                st.markdown("#### Books & Learning Materials")
                st.caption("Curated free books prioritised · Also includes Open Library + Google Books")
                render_books_section(result.get("books", []), top_n=top_display)

            with tab_repos:
                st.markdown("#### GitHub Repositories")
                st.caption("Official repos prioritised · Ranked by stars, forks, relevance")
                render_repos_section(result.get("repos", []), top_n=top_display)

            with tab_websites:
                st.markdown("#### Websites & Online Resources")
                st.caption("Documentation, courses, blogs, and tools")
                render_websites_section(result.get("websites", []), top_n=8)

            with tab_videos:
                st.markdown("#### YouTube Educational Videos")
                st.caption("Educational content · Priority channel shown first if specified")
                render_videos_section(result.get("videos", []), top_n=top_display)

            # ── Save to session ───────────────────────────────────────────────
            st.session_state["last_research"] = result
            st.session_state["last_query"] = query

        except Exception as e:
            progress_bar.progress(0)
            status_text.empty()
            st.error(f"❌ Research pipeline error: {e}")
            with st.expander("🔍 Debug Info"):
                st.code(traceback.format_exc())

    elif submitted and not query.strip():
        st.warning("Please enter a search query.")

    # ── Show last results if available ────────────────────────────────────────
    elif not submitted and st.session_state.get("last_research"):
        st.info(f"💾 Showing previous results for: **{st.session_state.get('last_query', 'Unknown')}**")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Multimodal RAG Research Assistant</h1>
        <p>Powered by Groq (llama3-70b) · Gemini (1.5-flash) · ChromaDB · LangGraph</p>
        <p style="font-size: 0.85rem; color: #7f8c8d;">
            Ask questions over PDFs (text + images + tables) · Search papers, books, repos, videos
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load keys
    keys = get_api_keys()

    # Sidebar
    render_sidebar(keys)

    # Main tabs
    tab_rag, tab_research = st.tabs([
        "📄 Document Q&A (RAG)",
        "🔬 Research Explorer",
    ])

    with tab_rag:
        render_rag_tab(keys)

    with tab_research:
        render_research_tab(keys)


if __name__ == "__main__":
    main()
