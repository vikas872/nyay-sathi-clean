"""
Content sanitization for Nyay Sathi API.

Protects against prompt injection, XSS, and malicious content.
"""

import html
import re
from typing import Optional

from logger import app_logger as logger


# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"new\s+instructions?:",
    r"system\s*prompt:",
    r"<\s*script",
    r"javascript:",
    r"data:\s*text/html",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: The raw user input.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text.
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = " ".join(text.split())

    # HTML escape
    text = html.escape(text)

    return text.strip()


def detect_prompt_injection(text: str) -> bool:
    """
    Detect potential prompt injection attempts.

    Args:
        text: The user input to check.

    Returns:
        True if injection attempt detected.
    """
    text_lower = text.lower()

    for pattern in COMPILED_PATTERNS:
        if pattern.search(text_lower):
            logger.warning(f"Potential prompt injection detected: {text[:50]}...")
            return True

    return False


def sanitize_web_content(html_content: str, max_length: int = 10000) -> str:
    """
    Sanitize web content retrieved from external sources.

    Removes scripts, styles, and potentially dangerous content.

    Args:
        html_content: The raw HTML content.
        max_length: Maximum allowed length.

    Returns:
        Sanitized plain text.
    """
    if not html_content:
        return ""

    # Remove script tags and content
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags and content
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Truncate
    return text[:max_length].strip()


def validate_query(query: str) -> tuple[bool, str, Optional[str]]:
    """
    Validate a user query for safety and suitability.

    Args:
        query: The user's question.

    Returns:
        Tuple of (is_valid, sanitized_query, error_message).
    """
    if not query or not query.strip():
        return False, "", "Query cannot be empty"

    sanitized = sanitize_user_input(query)

    if len(sanitized) < 3:
        return False, sanitized, "Query too short (minimum 3 characters)"

    if detect_prompt_injection(sanitized):
        return False, sanitized, "Query contains potentially harmful content"

    return True, sanitized, None
