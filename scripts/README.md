# Legal RAG Data Pipeline Scripts

This folder contains Python scripts for building a Legal RAG (Retrieval-Augmented Generation) system for Indian laws.

## 📋 Prerequisites

```bash
pip install -r requirements.txt
```

Set the `GROQ_API_KEY` environment variable for LLM-powered explanations:
```bash
export GROQ_API_KEY="your-api-key"
```

## 🔄 Pipeline Overview

```
1. download_acts.py       → Download act HTML pages
2. fetch_full_sections.py → Fetch section content via API
3. parse_indiacode_html.py → Parse HTML to JSON
4. normalize_sections.py  → Add metadata
5. clean_sections.py      → Dedupe & filter
6. chunk_sections.py      → Split into chunks
7. build_faiss_index.py   → Create vector index
8. query_faiss.py         → Query (basic)
9. query_and_explain.py   → Query (with AI explanation)
```

## 🚀 Quick Start

Run the full pipeline:
```bash
python download_acts.py
python fetch_full_sections.py
python parse_indiacode_html.py
python normalize_sections.py
python clean_sections.py
python chunk_sections.py
python build_faiss_index.py
```

Query the system:
```bash
python query_faiss.py              # Basic search
python query_and_explain.py        # With AI explanation
```

## 📁 File Structure

| File | Description |
|------|-------------|
| `config.py` | Centralized configuration |
| `utils.py` | Shared utilities |
| `download_acts.py` | Download act pages |
| `fetch_full_sections.py` | Fetch section content |
| `parse_indiacode_html.py` | Parse HTML to JSON |
| `normalize_sections.py` | Normalize with metadata |
| `clean_sections.py` | Clean and deduplicate |
| `chunk_sections.py` | Chunk for RAG |
| `build_faiss_index.py` | Build FAISS index |
| `query_faiss.py` | Interactive query CLI |
| `query_and_explain.py` | Query + LLM explanation |

## ⚙️ Configuration

All paths and settings are in `config.py`. Key settings:
- `EMBEDDING_MODEL`: Sentence transformer model
- `CHUNK_SIZE_TOKENS`: Target chunk size
- `FAISS_TOP_K`: Number of search results
