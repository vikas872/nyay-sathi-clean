"""
Utility functions for the Legal RAG data pipeline.

This module provides common utilities for text processing, logging setup,
and other shared functionality across scripts.
"""

import logging
import re
import sys
import unicodedata
from typing import Optional

from bs4 import BeautifulSoup

from config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Optional log level override.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
    
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text with normalized whitespace.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text by cleaning HTML, whitespace, and Unicode.

    Args:
        text: Raw text (potentially with HTML) to normalize.

    Returns:
        Normalized, clean text.
    """
    if not text:
        return ""
    
    # Remove zero-width spaces
    text = text.replace('\u200b', '')
    
    # Convert HTML breaks to newlines
    text = re.sub(r'</?br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<hr[^>]*>', ' ', text, flags=re.IGNORECASE)
    
    # Parse HTML and extract text
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator="\n")
    
    # Normalize Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Clean up lines
    lines = [line.strip() for line in text.split('\n')]
    text = "\n".join(line for line in lines if line)
    
    return text


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> float:
    """
    Estimate the number of tokens in text.

    Args:
        text: Text to estimate tokens for.
        chars_per_token: Approximate characters per token ratio.

    Returns:
        Estimated token count.
    """
    return len(text) / chars_per_token


def is_valid_section_text(text: str, min_length: int = 40) -> bool:
    """
    Validate if text is a valid legal section.

    Args:
        text: Section text to validate.
        min_length: Minimum character length for valid text.

    Returns:
        True if text is valid, False otherwise.
    """
    if not text:
        return False
    
    if len(text) < min_length:
        return False
    
    # Check for truncated endings
    invalid_endings = ("by", "of", "and", ":", "the", "or", "for")
    stripped_text = text.strip().lower()
    if stripped_text.endswith(invalid_endings):
        return False
    
    return True


def safe_filename(name: str, max_length: int = 100) -> str:
    """
    Create a safe filename from a string.

    Args:
        name: Original name to convert.
        max_length: Maximum length for the filename.

    Returns:
        Sanitized filename.
    """
    # Replace non-alphanumeric characters with underscores
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Remove consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Trim and lowercase
    safe = safe.strip('_').lower()
    return safe[:max_length]


def extract_year_from_text(text: str) -> int:
    """
    Extract a 4-digit year from text.

    Args:
        text: Text potentially containing a year.

    Returns:
        Extracted year or 0 if not found.
    """
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return int(match.group(0)) if match else 0
