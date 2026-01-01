---
title: Nyay Sathi - Indian Legal Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ⚖️ Nyay Sathi - AI Indian Legal Assistant

An AI-powered legal assistant for Indian citizens using **Retrieval-Augmented Generation (RAG)**. Ask questions about Indian laws, IPC sections, legal procedures, and get accurate answers sourced from actual legal documents.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/nyay-sathi)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

## ✨ Features

- 📚 **Legal Database** - FAISS-indexed Indian Penal Code, BNS, and other acts
- 🤖 **Agentic RAG** - LLM decides when to search database vs web
- 🌐 **Web Fallback** - Searches trusted gov.in sources when needed
- 🔐 **Secure API** - Bearer token authentication & rate limiting
- ⚡ **Fast** - GPU support when available, optimized embeddings
- 🖥️ **Beautiful CLI** - Claude Code-like streaming interface

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip
- [Groq API Key](https://console.groq.com/keys)

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nyay-sathi.git
cd nyay-sathi

# Install dependencies with UV (10-100x faster than pip)
uv sync

# Create environment file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start the backend server
cd backend
uv run uvicorn main:app --port 10000 --reload

# In another terminal, run the CLI
cd cli
uv run python nyay_cli.py
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# Optional (defaults shown)
GROQ_MODEL=llama-3.3-70b-versatile
API_SECRET_KEYS=nyay-sathi-local-dev-key
RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
```

## 🐳 Docker

### Build & Run Locally

```bash
# Build the image
docker build -t nyay-sathi .

# Run the container
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_groq_key \
  -e API_SECRET_KEYS=your_secret_key \
  nyay-sathi
```

### Image Optimization

The Dockerfile is optimized for production:
- **Multi-stage build** - Only runtime dependencies in final image
- **CPU-only PyTorch** - Saves ~2GB (no CUDA libraries)
- **UV package manager** - 10-100x faster builds
- **Non-root user** - Security best practice
- **Final size**: ~1.5GB (vs 3GB+ with full PyTorch)

## ☁️ Deploy to HuggingFace Spaces

### Step 1: Create a HuggingFace Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: `nyay-sathi`
   - **SDK**: `Docker`
   - **Hardware**: `CPU Basic` (free tier works!)
   - **Visibility**: Public or Private

### Step 2: Configure Secrets

In your Space settings, add these **Secrets**:

| Secret Name | Value |
|-------------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `API_SECRET_KEYS` | Your API authentication key(s) |

### Step 3: Push Code

```bash
# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/nyay-sathi

# Push to HuggingFace
git push hf main
```

Or use the HuggingFace web UI to upload files directly.

### Step 4: Wait for Build

- Build takes ~5-10 minutes
- Check the "Logs" tab for progress
- Once "Running", your API is live!

**Your API URL**: `https://YOUR_USERNAME-nyay-sathi.hf.space`

## 🔗 Connect CLI to HuggingFace

Once deployed, connect your local CLI to the cloud API:

```bash
# Set environment variables
export NYAY_SATHI_API_URL=https://YOUR_USERNAME-nyay-sathi.hf.space
export NYAY_SATHI_API_KEY=your_secret_key

# Or create cli/.env file
echo "NYAY_SATHI_API_URL=https://YOUR_USERNAME-nyay-sathi.hf.space" > cli/.env
echo "NYAY_SATHI_API_KEY=your_secret_key" >> cli/.env

# Run CLI
cd cli
uv run python nyay_cli.py
```

## 📡 API Reference

### Health Check

```bash
curl https://YOUR_USERNAME-nyay-sathi.hf.space/health
```

Response:
```json
{
  "status": "ok",
  "service": "Nyay Sathi Backend",
  "version": "2.0.0",
  "vectors_loaded": 1234,
  "device": "cpu"
}
```

### Ask a Question

```bash
curl -X POST https://YOUR_USERNAME-nyay-sathi.hf.space/ask \
  -H "Authorization: Bearer your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft in India?"}'
```

Response:
```json
{
  "mode": "grounded",
  "confidence": "high",
  "answer": "According to Section 379 of the Indian Penal Code...",
  "tokens_in": 1200,
  "tokens_out": 450,
  "local_sources": [
    {"act": "Indian Penal Code", "section": "379", "score": 0.89}
  ],
  "web_sources": [],
  "disclaimer": "This information is for educational purposes only..."
}
```

### Streaming (SSE)

```bash
curl -X POST https://YOUR_USERNAME-nyay-sathi.hf.space/ask/stream \
  -H "Authorization: Bearer your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is defamation?"}'
```

Returns Server-Sent Events with live progress updates.

## 📁 Project Structure

```
nyay-sathi/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── agent.py            # Agentic tool calling
│   ├── rag_engine.py       # FAISS retrieval
│   ├── tools.py            # Tool definitions
│   ├── browser.py          # Web scraping
│   ├── config.py           # Configuration
│   └── ...
├── cli/                     # Rich terminal client
│   ├── nyay_cli.py         # Main CLI
│   ├── ui.py               # Streaming display
│   └── config.py           # CLI config
├── data/
│   ├── raw/                # Source HTML/JSON (dev only)
│   └── processed/          # FAISS index (production)
├── scripts/                 # Data pipeline (dev only)
├── Dockerfile              # Production container
├── pyproject.toml          # Dependencies
└── README.md
```

## �️ Development

### Building the FAISS Index

If you need to rebuild the legal database:

```bash
cd scripts

# Download acts from India Code
uv run python download_acts.py

# Parse and chunk sections
uv run python chunk_sections.py

# Build FAISS index
uv run python build_faiss_index.py
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy backend/
```

## 🔒 Security

- **Authentication**: Bearer token required for all `/ask` endpoints
- **Rate Limiting**: 60 requests/minute per IP (configurable)
- **Input Sanitization**: All queries sanitized before processing
- **Domain Whitelist**: Web search restricted to trusted gov.in domains
- **Non-root Container**: Production container runs as unprivileged user

## 📜 Legal Disclaimer

> **This tool is for educational and informational purposes only.**
> 
> The information provided by Nyay Sathi does not constitute legal advice. For specific legal matters, please consult a qualified lawyer or legal professional.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Groq** - Fast LLM inference
- **HuggingFace** - Model hosting & Spaces
- **India Code** - Legal database source
- **Sentence Transformers** - Embeddings
- **FAISS** - Vector search
