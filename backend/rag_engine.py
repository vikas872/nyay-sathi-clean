"""
RAG Engine for Nyay Sathi v2.0.

Handles FAISS-based retrieval, LLM-powered explanations,
and web search fallback for legal questions.

Features:
- GPU acceleration when available
- Parallel embedding generation
- Hybrid search (local + web fallback)
- Confidence-based routing
"""

from __future__ import annotations

import asyncio
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

# Force environment before torch import
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
    DEVICE,
    WEB_SEARCH_ENABLED,
)
from logger import rag_logger as logger


# =============================================================================
# GLOBAL STATE (Lazy Loading for Memory Efficiency)
# =============================================================================

_index: Optional[faiss.Index] = None
_metadata: Optional[list[dict]] = None
_embedder: Any = None
_client: Optional[Groq] = None
_executor: Optional[ThreadPoolExecutor] = None


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_rag() -> int:
    """
    Initialize the RAG system.

    Loads FAISS index and metadata. Embedding model is loaded lazily on first query.

    Returns:
        Number of vectors in the index.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    global _index, _metadata, _client, _executor

    logger.info(f"Initializing RAG system (device: {DEVICE})...")
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

    # Initialize thread pool for parallel operations
    _executor = ThreadPoolExecutor(max_workers=4)
    logger.debug("Thread pool initialized")

    return _index.ntotal


def get_vectors_count() -> int:
    """Return the number of vectors in the index."""
    if _index is None:
        return 0
    return _index.ntotal


def _get_embedder():
    """
    Lazy-load the embedding model.

    Uses the configured device (GPU if available).
    """
    global _embedder

    if _embedder is None:
        logger.info(f"Loading embedding model on {DEVICE}...")
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        logger.info(f"Embedding model loaded on {DEVICE}")

    return _embedder


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_sections(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve relevant legal sections for a query.

    Args:
        query: The user's question.
        top_k: Number of results to retrieve.

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
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_metadata):
            continue
        record = _metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)

    logger.debug(f"Retrieved {len(results)} sections (top score: {results[0]['score']:.3f})" if results else "No results")
    return results


# Alias for agent.py
retrieve = retrieve_sections





# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT_GROUNDED = """You are Nyay Sathi, a helpful Indian legal assistant.

You have been given legal text from Indian laws. Use ONLY this information to answer.

RULES:
1. Use numbered citations like [1], [2] when referencing sources.
2. Mention the Act Name and Section when citing.
3. Explain in simple English a layperson can understand.
4. If the sources don't answer the question, say so.
5. Do NOT invent information not in the sources.
6. Keep your answer concise and focused.

End with: "Disclaimer: For educational purposes only, not legal advice."
"""

SYSTEM_PROMPT_HYBRID = """You are Nyay Sathi, a helpful Indian legal assistant.

You have legal text from a database AND verified web sources. Synthesize both.

RULES:
1. Use numbered citations like [1], [2] when referencing sources.
2. Prioritize official legal text over web sources.
3. Mention source names (Act, Section, or website).
4. Explain in simple English.
5. Keep your answer concise.

End with: "Disclaimer: For educational purposes only, not legal advice."
"""

SYSTEM_PROMPT_FALLBACK = """You are Nyay Sathi, a helpful Indian legal assistant.

No specific legal section matched this query. Be honest about limitations.

RULES:
1. Acknowledge you don't have specific legal text for this.
2. Do NOT cite specific Acts or Sections.
3. Give only general educational information.
4. Suggest rephrasing the question.
5. Keep it brief.

End with: "Disclaimer: For educational purposes only, not legal advice."
"""



# =============================================================================
# LLM EXPLANATION
# =============================================================================

def explain_with_llm(
    query: str,
    local_results: list[dict],
    web_results: Optional[list] = None,
    source_mode: str = "local",
) -> tuple[str, str, float]:
    """
    Generate an LLM explanation for retrieved sections.

    Args:
        query: The user's question.
        local_results: List of locally retrieved sections.
        web_results: Optional list of web search results.
        source_mode: One of "local", "hybrid", "fallback".

    Returns:
        Tuple of (mode, explanation, confidence_score).
    """
    # Determine confidence and mode
    if not local_results and not web_results:
        mode = "fallback"
        top_score = 0.0
    else:
        top_score = local_results[0]["score"] if local_results else 0.0
        if source_mode == "hybrid":
            mode = "hybrid"
        elif top_score >= CONFIDENCE_THRESHOLD:
            mode = "grounded"
        else:
            mode = "fallback"

    logger.debug(f"LLM mode: {mode}, top_score: {top_score:.3f}")

    # Build context with numbered citations
    context_parts = []

    # Only use top 3 sources with decent scores
    relevant_local = [r for r in local_results if r.get("score", 0) >= 0.5][:3]

    if relevant_local and mode in ("grounded", "hybrid"):
        context_parts.append("SOURCES:")
        for i, r in enumerate(relevant_local, 1):
            context_parts.append(
                f"[{i}] {r.get('act_name', 'Unknown')} - Section {r.get('section_number', 'Unknown')}\n"
                f"    {r.get('text', '')[:800]}\n"
            )

    if web_results and mode == "hybrid":
        start_idx = len(relevant_local) + 1
        for i, wr in enumerate(web_results[:2], start_idx):
            context_parts.append(
                f"[{i}] {wr.title} ({wr.source_domain})\n"
                f"    {wr.snippet[:400]}\n"
            )

    context = "\n".join(context_parts) if context_parts else "(No relevant sources found)"

    # Select prompt
    if mode == "grounded":
        system_prompt = SYSTEM_PROMPT_GROUNDED
    elif mode == "hybrid":
        system_prompt = SYSTEM_PROMPT_HYBRID
    else:
        system_prompt = SYSTEM_PROMPT_FALLBACK

    user_content = f"USER QUESTION:\n{query}\n\nAVAILABLE INFORMATION:\n{context}"

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
            max_tokens=800,
        )
        explanation = response.choices[0].message.content.strip()
        
        # Extract token usage
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        
        logger.debug(f"LLM response: {tokens_in}→{tokens_out} tokens")
        return mode, explanation, top_score, tokens_in, tokens_out

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return (
            "fallback",
            "An error occurred while generating the explanation. "
            "Please try again later.\n\n"
            "Disclaimer: This information is for educational purposes only.",
            0.0,
            0,
            0,
        )



