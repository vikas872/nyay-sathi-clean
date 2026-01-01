"""
Configuration module for the Legal RAG data pipeline.

This module centralizes all configuration constants, paths, and settings
used across the data processing and query scripts.
"""

import os
from pathlib import Path
from typing import Final

# =============================================================================
# BASE PATHS
# =============================================================================

# Project root (parent of scripts folder)
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

# Raw data subdirectories
ACTS_HTML_DIR: Final[Path] = RAW_DIR / "acts_html"
ACTS_PDF_DIR: Final[Path] = RAW_DIR / "acts_pdf"
SECTIONS_HTML_DIR: Final[Path] = RAW_DIR / "sections_html"
METADATA_DIR: Final[Path] = RAW_DIR / "metadata"

# Processed data files
SECTIONS_JSON_DIR: Final[Path] = PROCESSED_DIR / "sections_json"
NORMALIZED_FILE: Final[Path] = PROCESSED_DIR / "normalized_sections.json"
CLEAN_FILE: Final[Path] = PROCESSED_DIR / "sections_clean.json"
CHUNKS_FILE: Final[Path] = PROCESSED_DIR / "sections_chunks.json"
FAISS_INDEX_FILE: Final[Path] = PROCESSED_DIR / "faiss.index"
FAISS_META_FILE: Final[Path] = PROCESSED_DIR / "faiss_meta.pkl"
METADATA_FILE: Final[Path] = METADATA_DIR / "acts_metadata.json"

# =============================================================================
# API CONFIGURATION
# =============================================================================

INDIACODE_BASE_URL: Final[str] = "https://www.indiacode.nic.in"
INDIACODE_SHOW_DATA_URL: Final[str] = f"{INDIACODE_BASE_URL}/show-data"
INDIACODE_API_ENDPOINT: Final[str] = f"{INDIACODE_BASE_URL}/SectionPageContent"

# =============================================================================
# HTTP REQUEST SETTINGS
# =============================================================================

USER_AGENT: Final[str] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQUEST_HEADERS: Final[dict] = {
    "User-Agent": USER_AGENT,
}
REQUEST_TIMEOUT: Final[int] = 30  # seconds
REQUEST_DELAY: Final[float] = 0.1  # seconds between requests

# =============================================================================
# TEXT PROCESSING SETTINGS
# =============================================================================

MIN_TEXT_LENGTH: Final[int] = 40  # Minimum characters for valid section
CHUNK_SIZE_TOKENS: Final[int] = 450  # Target tokens per chunk
CHUNK_OVERLAP_TOKENS: Final[int] = 50  # Overlap between chunks
CHARS_PER_TOKEN: Final[float] = 4.0  # Approximate character to token ratio

# =============================================================================
# EMBEDDING & FAISS SETTINGS
# =============================================================================

EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_TOP_K: Final[int] = 5

# =============================================================================
# LLM SETTINGS (for query_and_explain.py)
# =============================================================================

GROQ_MODEL: Final[str] = "llama3-8b-8192"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        ACTS_HTML_DIR,
        ACTS_PDF_DIR,
        SECTIONS_HTML_DIR,
        METADATA_DIR,
        SECTIONS_JSON_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
