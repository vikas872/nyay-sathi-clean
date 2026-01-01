# Nyay Sathi - Legal RAG System

AI-powered legal information assistant for Indian laws using Retrieval-Augmented Generation.

## 🚀 Quick Start

### 1. Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Configure
```bash
# Copy template and add your API key
cp .env.example .env
# Edit .env and add GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com/keys

### 3. Run
```bash
cd backend
uvicorn main:app --reload --port 10000
```

API: http://localhost:10000

## 📁 Project Structure
```
nyay-sathi-clean/
├── backend/              # FastAPI backend
│   ├── main.py           # API endpoints + CORS
│   ├── rag_engine.py     # FAISS search + LLM
│   ├── config.py         # Settings
│   └── logger.py         # Logging
├── data/
│   ├── raw/              # Source HTML (dev only)
│   └── processed/        # FAISS index (runtime)
├── scripts/              # Data pipeline (dev only)
├── docs/                 # Documentation + PDFs
├── Dockerfile            # Production build
└── .env                  # API keys (git-ignored)
```

## 🐳 Docker

```bash
# Build
docker build -t nyay-sathi .

# Run
docker run -p 10000:10000 -e GROQ_API_KEY=your_key nyay-sathi
```

## 📋 API

**POST /ask**
```json
{"question": "What is the punishment for theft?"}
```

**Response**
```json
{
  "mode": "rag",
  "confidence": "high", 
  "answer": "...",
  "sources": [...]
}
```
