# 🔬 Multimodal RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=flat-square&logo=huggingface)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Stage-Live-brightgreen?style=flat-square)

---

🤗 **Live Demo:** [huggingface.co/spaces/Ashutosh1975/multimodal-rag-app](https://huggingface.co/spaces/Ashutosh1975/multimodal-rag-app)
👤 **Author:** [Ashutosh — GitHub](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/)
💼 **Portfolio:** [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/)

---

## 📋 Table of Contents

- [What This Does](#-what-this-does)
- [Live Demo](#-live-demo)
- [Sources Supported](#-sources-supported)
- [Architecture](#-architecture)
- [What I Built](#-what-i-built)
- [RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [Tech Stack](#-tech-stack)
- [Project Status](#-project-status)
- [Links](#-links)
- [Author](#-author)

---

## 🧠 What This Does

A production-grade multi-source RAG system that lets you
research and discuss content from 5 different source types
— all in one unified chat interface.

1. **Problem** — Researchers and analysts waste hours
   switching between PDFs, YouTube, GitHub, and papers
   trying to find and connect information manually
2. **Solution** — Single RAG assistant that ingests any
   source, embeds it into a vector store, and lets you
   ask questions across all sources simultaneously
3. **For** — GenAI / RAG Engineer / LLM Developer hiring
   managers looking for real multi-source RAG proof

---

## 🌐 Live Demo

👉 **[huggingface.co/spaces/Ashutosh1975/multimodal-rag-app](https://huggingface.co/spaces/Ashutosh1975/multimodal-rag-app)**

> Upload a PDF, paste a YouTube URL, or enter a GitHub repo
> and start asking questions immediately.
> All processing happens in real time.

---

## 📂 Sources Supported

| Source Type | What You Can Do | How It Works |
|-------------|----------------|-------------|
| 📄 PDF Documents | Upload and chat with any PDF | PyPDF2 extraction → chunking → embedding |
| 🎥 YouTube Videos | Ask questions about any video | Transcript extraction → RAG pipeline |
| 🐙 GitHub Repos | Explore and discuss any codebase | Repo cloning → code parsing → embedding |
| 📚 Research Papers | Find and discuss arxiv papers | PDF/URL ingestion → RAG retrieval |
| 🌐 Any Topic | General research with cited answers | Web-augmented retrieval |

---

## 🏗️ Architecture
```
User Query + Source Input
        ↓
Source Router
  → Detects source type: PDF / YouTube / GitHub / Web
  → Routes to correct ingestion pipeline
        ↓
Document Ingestion
  ├── PDF    → PyPDF2 → raw text extraction
  ├── YouTube → youtube-transcript-api → transcript
  ├── GitHub  → GitPython → code file parsing
  └── Web    → requests + BeautifulSoup → content
        ↓
Text Chunking
  → RecursiveCharacterTextSplitter
  → Chunk size: 1000 tokens
  → Overlap: 200 tokens
        ↓
Embedding Generation
  → HuggingFace sentence-transformers
  → all-MiniLM-L6-v2 model
  → Dense vector per chunk
        ↓
Vector Store (FAISS)
  → Index all chunk embeddings
  → Persist for session reuse
        ↓
RAG Retrieval
  → User query embedded
  → Top-k similar chunks retrieved (k=5)
  → Context assembled from chunks
        ↓
LLM Response Generation
  → Context + query sent to LLM
  → Response generated with citations
  → Source references included
        ↓
Streamlit Chat UI
  → Response displayed with source attribution
  → Conversation history maintained
```

---

## 🔨 What I Built

### 1. Source Router
- Detects input type automatically (PDF upload vs URL vs topic)
- Routes to the correct ingestion class
- Handles errors gracefully for unsupported formats
- Supports mixed-source sessions (PDF + YouTube together)

### 2. PDF Ingestion Pipeline
- PyPDF2 extracts raw text from uploaded PDFs
- Handles multi-page documents of any length
- Preserves page structure for citation tracking
- Supports scanned PDFs via OCR fallback

### 3. YouTube Pipeline
- youtube-transcript-api fetches auto-generated transcripts
- Handles both manual and auto captions
- Timestamps preserved for citation references
- Supports any public YouTube video URL

### 4. GitHub Repo Pipeline
- GitPython clones target repository
- Parses Python, JavaScript, and markdown files
- Filters out binary files and dependencies
- Code chunked with function-level awareness

### 5. Text Chunking Strategy
- RecursiveCharacterTextSplitter for smart chunking
- 1000 token chunks with 200 token overlap
- Overlap prevents context loss at chunk boundaries
- Metadata attached per chunk (source, page, timestamp)

### 6. Embedding + Vector Store
- HuggingFace all-MiniLM-L6-v2 for dense embeddings
- FAISS index for fast similarity search
- Top-5 most relevant chunks retrieved per query
- Vector store persists within session for speed

### 7. LLM Response with Citations
- Retrieved chunks assembled into context window
- LLM instructed to cite sources in every response
- Source type and location included in citations
- Handles "I don't know" gracefully when context is missing

### 8. Streamlit Chat Interface
- Multi-source input panel (upload + URL fields)
- Chat interface with full conversation history
- Source documents panel showing what was ingested
- Citation display below each response
- Deployed live on HuggingFace Spaces

---

## 🔍 RAG Pipeline Deep Dive
```
Query: "What does this paper say about transformer attention?"

Step 1 — Query Embedding
  "What does this paper say about transformer attention?"
  → [0.23, -0.11, 0.87, ...] (384-dim vector)

Step 2 — FAISS Similarity Search
  Compare query vector against all chunk vectors
  → Top 5 most similar chunks retrieved

Step 3 — Context Assembly
  Chunk 1: "...multi-head attention allows the model to..."
  Chunk 2: "...attention weights are computed as softmax..."
  Chunk 3: "...positional encoding added to input embeddings..."

Step 4 — LLM Prompt
  System: "Answer using only the provided context. Cite sources."
  Context: [Chunk 1 + Chunk 2 + Chunk 3]
  Query: "What does this paper say about transformer attention?"

Step 5 — Response
  "According to the paper (Page 4), transformer attention
   uses multi-head attention to allow the model to jointly
   attend to information from different representation
   subspaces..."
```

---

## ⚡ Quick Start

**Prerequisites:** Python 3.11+, Git
```bash
# 1. Clone the repo
git clone https://github.com/Ashutosh-AIBOT/multimodal-rag-research-assistant.git
cd multimodal-rag-research-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your API keys to .env

# 5. Run the app
streamlit run app.py

# 6. Open browser
# http://localhost:8501
```

---

## 🔑 Environment Variables

| Variable | What It Is | Where To Get |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | LLM API key | [platform.openai.com](https://platform.openai.com/) |
| `HUGGINGFACE_TOKEN` | HuggingFace API token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `APP_ENV` | Environment flag | `development` or `production` |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| LangChain | RAG framework and chain management |
| FAISS | Vector similarity search |
| HuggingFace Transformers | Embedding model (all-MiniLM-L6-v2) |
| PyPDF2 | PDF text extraction |
| youtube-transcript-api | YouTube transcript fetching |
| GitPython | GitHub repo cloning and parsing |
| Streamlit | Chat UI and file upload interface |
| HuggingFace Spaces | Live deployment and hosting |
| Git | Version control |

---

## 📁 Repository Structure
```
multimodal-rag-research-assistant/
│
├── app.py                          # Main Streamlit app
│
├── rag/
│   ├── __init__.py
│   ├── router.py                   # Source type detection + routing
│   ├── ingestion/
│   │   ├── pdf_loader.py           # PDF extraction pipeline
│   │   ├── youtube_loader.py       # YouTube transcript pipeline
│   │   ├── github_loader.py        # GitHub repo parsing pipeline
│   │   └── web_loader.py           # Web content extraction
│   ├── chunker.py                  # Text chunking strategy
│   ├── embedder.py                 # HuggingFace embedding generation
│   ├── vectorstore.py              # FAISS index management
│   └── retriever.py                # Query + retrieval + LLM response
│
├── .env.example                    # Environment variable template
├── requirements.txt
└── README.md
```

---

## 📊 Project Status

| Deliverable | Status |
|-------------|--------|
| PDF Ingestion Pipeline | ✅ Complete |
| YouTube Transcript Pipeline | ✅ Complete |
| GitHub Repo Pipeline | ✅ Complete |
| Text Chunking Strategy | ✅ Complete |
| FAISS Vector Store | ✅ Complete |
| HuggingFace Embeddings | ✅ Complete |
| LLM Response with Citations | ✅ Complete |
| Streamlit Chat UI | ✅ Complete |
| HuggingFace Spaces Deployment | ✅ Live |
| Research Paper Pipeline | 🔄 In Progress |

---

## 🌐 Links

| Resource | URL |
|----------|-----|
| 🤗 Live Demo | [huggingface.co/spaces/Ashutosh1975/multimodal-rag-app](https://huggingface.co/spaces/Ashutosh1975/multimodal-rag-app) |
| 💼 Portfolio | [ashutosh-portfolio-kappa.vercel.app](https://ashutosh-portfolio-kappa.vercel.app/) |
| 🐙 GitHub | [github.com/Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT) |
| 🔗 LinkedIn | [linkedin.com/in/ashutosh1975271](https://www.linkedin.com/in/ashutosh1975271/) |

---

## 👤 Author

**Ashutosh**
B.Tech Electronics Engineering · CGPA 7.5 · Batch 2026
[GitHub](https://github.com/Ashutosh-AIBOT) · [LinkedIn](https://www.linkedin.com/in/ashutosh1975271/) · [Portfolio](https://ashutosh-portfolio-kappa.vercel.app/)

---

> *"PDF. YouTube. GitHub. Research papers.*
> *One assistant. Every source. Zero switching."*
>
> — Ashutosh, building this from zero.
```
