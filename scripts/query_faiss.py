"""
Query FAISS index for relevant legal sections.

This script provides an interactive CLI to search for legal
sections based on semantic similarity.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    EMBEDDING_MODEL,
    FAISS_TOP_K,
)
from utils import setup_logger

logger = setup_logger(__name__)


def load_faiss() -> tuple[faiss.Index, list[dict]]:
    """
    Load FAISS index and metadata.

    Returns:
        Tuple of (FAISS index, metadata list).
    """
    if not FAISS_INDEX_FILE.exists():
        logger.error(f"FAISS index not found: {FAISS_INDEX_FILE}")
        sys.exit(1)
    
    if not FAISS_META_FILE.exists():
        logger.error(f"Metadata file not found: {FAISS_META_FILE}")
        sys.exit(1)
    
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    
    with open(FAISS_META_FILE, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Loaded {len(metadata)} metadata records")
    
    return index, metadata


def load_model() -> SentenceTransformer:
    """
    Load the embedding model.

    Returns:
        SentenceTransformer model.
    """
    logger.info(f"Loading model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = FAISS_TOP_K,
) -> list[dict]:
    """
    Search for relevant sections.

    Args:
        query: Search query string.
        model: Embedding model.
        index: FAISS index.
        metadata: Chunk metadata.
        top_k: Number of results to return.

    Returns:
        List of matching results with scores.
    """
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        
        record = metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)
    
    return results


def display_results(results: list[dict]) -> None:
    """
    Display search results.

    Args:
        results: List of result dictionaries.
    """
    if not results:
        print("\nNo relevant sections found.\n")
        return
    
    print("\n" + "=" * 80)
    print("RELEVANT LEGAL SECTIONS")
    print("=" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r.get('act_name', 'Unknown Act')}")
        print(f"   Section: {r.get('section_number', 'N/A')}")
        print(f"   Category: {r.get('category', 'N/A')}")
        print(f"   Relevance: {r['score']:.3f}")
        print("-" * 40)
        
        text = r.get("text", "")
        print(f"   {text[:600]}{'...' if len(text) > 600 else ''}")
    
    print("\n" + "=" * 80)
    print("DISCLAIMER: This information is for educational purposes only")
    print("and does not constitute legal advice.")
    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point for query interface."""
    print("\n" + "=" * 50)
    print("  LEGAL RAG SYSTEM - Section Query")
    print("=" * 50 + "\n")
    
    index, metadata = load_faiss()
    model = load_model()
    
    print("\nReady! Type your legal question below.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        results = search(query, model, index, metadata)
        display_results(results)


if __name__ == "__main__":
    main()
