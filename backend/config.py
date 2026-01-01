"""
Backend configuration module.

Centralizes all settings, paths, and environment variables.
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Backend is at: backend/config.py → parent.parent = project root
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data" / "processed"

FAISS_INDEX_PATH: Final[Path] = DATA_DIR / "faiss.index"
FAISS_META_PATH: Final[Path] = DATA_DIR / "faiss_meta.pkl"

# =============================================================================
# API KEYS
# =============================================================================

GROQ_API_KEY: Final[str] = os.getenv("GROQ_API_KEY", "")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

# Groq models ranked by capability:
# - llama-3.3-70b-versatile: 128K context, best reasoning (recommended)
# - llama-3.1-70b-versatile: 128K context, great reasoning
# - llama-3.1-8b-instant: 8K context, fast but limited (previous default)
GROQ_MODEL: Final[str] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# =============================================================================
# RAG SETTINGS
# =============================================================================

TOP_K: Final[int] = 5
# Higher threshold = more confident/relevant results only
CONFIDENCE_THRESHOLD: Final[float] = 0.60

# =============================================================================
# SERVER SETTINGS
# =============================================================================

SERVER_HOST: Final[str] = os.getenv("HOST", "0.0.0.0")
SERVER_PORT: Final[int] = int(os.getenv("PORT", "10000"))

# CORS - Add your frontend URLs here
CORS_ORIGINS: Final[list[str]] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Add production URLs as needed
    # "https://your-frontend.vercel.app",
]

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# =============================================================================
# SECURITY
# =============================================================================

# List of valid API keys (Bearer tokens)
# In production, these should be loaded from env vars or a secure secret manager
_keys_str = os.getenv("API_SECRET_KEYS", "nyay-sathi-local-dev-key")
API_SECRET_KEYS: Final[list[str]] = [k.strip() for k in _keys_str.split(",") if k.strip()]

# Rate limit (requests per minute per IP)
RATE_LIMIT_PER_MINUTE: Final[int] = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# =============================================================================
# WEB SEARCH FALLBACK
# =============================================================================

WEB_SEARCH_ENABLED: Final[bool] = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_TIMEOUT: Final[float] = float(os.getenv("WEB_SEARCH_TIMEOUT", "10.0"))
WEB_SEARCH_MAX_RESULTS: Final[int] = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

# =============================================================================
# GPU / DEVICE CONFIGURATION
# =============================================================================

# Auto-detect GPU availability
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return "cpu"

DEVICE: Final[str] = os.getenv("DEVICE", _detect_device())


