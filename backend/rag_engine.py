"""
RAG Engine for Nyay Sathi.

Handles FAISS-based retrieval and LLM-powered explanations
for legal questions.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

# Force CPU mode for Railway/free-tier compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
from groq import Groq

from config import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    GROQ_API_KEY,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)
from logger import rag_logger as logger


# =============================================================================
# GLOBAL STATE (Lazy Loading)
# =============================================================================

_index: Optional[faiss.Index] = None
_metadata: Optional[list[dict]] = None
_embedder: Any = None
_client: Optional[Groq] = None


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_rag() -> int:
    """
    Initialize the RAG system.

    Loads FAISS index and metadata. Embedding model is loaded lazily.

    Returns:
        Number of vectors in the index.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    global _index, _metadata, _client

    logger.info("Initializing RAG system...")
    logger.debug(f"FAISS path: {FAISS_INDEX_PATH}")

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

    if not FAISS_META_PATH.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {FAISS_META_PATH}")

    # Load FAISS index
    _index = faiss.read_index(str(FAISS_INDEX_PATH))
    logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")

    # Load metadata
    with open(FAISS_META_PATH, "rb") as f:
        _metadata = pickle.load(f)
    logger.debug(f"Loaded {len(_metadata)} metadata records")

    # Initialize Groq client
    if GROQ_API_KEY:
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized")
    else:
        logger.warning("GROQ_API_KEY not set - LLM explanations disabled")

    return _index.ntotal


def _get_embedder():
    """
    Lazy-load the embedding model.

    This prevents OOM on free-tier hosting during startup.
    """
    global _embedder
    
    if _embedder is None:
        logger.info("Loading embedding model (first query)...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        logger.info("Embedding model loaded")
    
    return _embedder


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_sections(query: str) -> list[dict]:
    """
    Retrieve relevant legal sections for a query.

    Args:
        query: The user's question.

    Returns:
        List of matching sections with scores.
    """
    if _index is None or _metadata is None:
        logger.error("RAG not initialized")
        return []

    embedder = _get_embedder()

    # Encode query
    query_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Search
    scores, indices = _index.search(query_vec, TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_metadata):
            continue
        record = _metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)

    logger.debug(f"Retrieved {len(results)} sections for query")
    return results


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT_GROUNDED = """You are Nyay Sathi, a helpful Indian legal assistant.
MODE: RAG-BACKED (HIGH CONFIDENCE).

INSTRUCTIONS:
1. Use ONLY the provided legal text.
2. Mention Act Name and Section Number if present.
3. Explain in simple, clear English.
4. If text does not answer the question, say so clearly.
5. Do NOT invent laws or punishments.
6. Do NOT give legal advice.

MANDATORY DISCLAIMER:
End with: "Disclaimer: This information is for educational purposes only and does not constitute legal advice."
"""

SYSTEM_PROMPT_FALLBACK = """You are Nyay Sathi, a helpful Indian legal assistant.
MODE: GENERAL FALLBACK.

INSTRUCTIONS:
1. No specific legal section matched the query.
2. Do NOT cite specific Acts or Sections.
3. Give only high-level educational explanation.
4. Suggest the user rephrase their question.
5. Do NOT give legal advice.

MANDATORY DISCLAIMER:
End with: "Disclaimer: This information is for educational purposes only and does not constitute legal advice."
"""


# =============================================================================
# LLM EXPLANATION
# =============================================================================

def explain_with_llm(
    query: str,
    retrieved: list[dict],
) -> tuple[str, str, float]:
    """
    Generate an LLM explanation for retrieved sections.

    Args:
        query: The user's question.
        retrieved: List of retrieved sections.

    Returns:
        Tuple of (mode, explanation, top_score).
    """
    # Determine mode based on confidence
    if not retrieved:
        mode = "fallback"
        top_score = 0.0
    else:
        top_score = retrieved[0]["score"]
        mode = "grounded" if top_score >= CONFIDENCE_THRESHOLD else "fallback"

    logger.debug(f"LLM mode: {mode}, top_score: {top_score:.3f}")

    # Build prompt
    if mode == "grounded":
        context = ""
        for r in retrieved:
            context += (
                f"---\n"
                f"Act: {r.get('act_name', 'Unknown')}\n"
                f"Section: {r.get('section_number', 'Unknown')}\n"
                f"Text: {r.get('text', '')}\n"
            )
        user_content = f"USER QUESTION:\n{query}\n\nLEGAL TEXT:\n{context}"
        system_prompt = SYSTEM_PROMPT_GROUNDED
    else:
        user_content = f"USER QUESTION:\n{query}\n\n(No relevant legal text found)"
        system_prompt = SYSTEM_PROMPT_FALLBACK

    # Call LLM
    if _client is None:
        logger.warning("Groq client not available")
        return (
            "fallback",
            "LLM service not available. Please check API key configuration.\n\n"
            "Disclaimer: This information is for educational purposes only.",
            0.0,
        )

    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        explanation = response.choices[0].message.content.strip()
        logger.debug("LLM response generated successfully")
        return mode, explanation, top_score

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return (
            "fallback",
            "An error occurred while generating the explanation. "
            "Please try again later.\n\n"
            "Disclaimer: This information is for educational purposes only.",
            0.0,
        )
