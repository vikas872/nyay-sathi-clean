"""
Build FAISS index from chunked section data.

This script generates embeddings for all chunks and builds a FAISS
index for fast similarity search.
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_FILE,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    EMBEDDING_MODEL,
    ensure_directories,
)
from utils import setup_logger

logger = setup_logger(__name__)


def load_chunks() -> list[dict]:
    """
    Load chunked section data.

    Returns:
        List of chunk dictionaries.
    """
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        sys.exit(1)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Generate embeddings for texts.

    Args:
        texts: List of text strings.
        model_name: Name of the embedding model.

    Returns:
        Numpy array of embeddings.
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity with IP index
    )
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Numpy array of embeddings.

    Returns:
        FAISS index.
    """
    dimension = embeddings.shape[1]
    
    # IndexFlatIP with normalized embeddings = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    logger.info(f"Index built with {index.ntotal} vectors")
    return index


def save_outputs(index: faiss.Index, chunks: list[dict]) -> None:
    """
    Save FAISS index and metadata.

    Args:
        index: FAISS index to save.
        chunks: Chunk metadata to save.
    """
    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving index to {FAISS_INDEX_FILE}")
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    
    logger.info(f"Saving metadata to {FAISS_META_FILE}")
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(chunks, f)


def main() -> None:
    """Main entry point for building FAISS index."""
    logger.info("Starting FAISS index build...")
    ensure_directories()
    
    # Load data
    chunks = load_chunks()
    if not chunks:
        logger.error("No chunks to process")
        return
    
    texts = [chunk.get("text", "") for chunk in chunks]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    
    # Build index
    index = build_index(embeddings)
    
    # Save outputs
    save_outputs(index, chunks)
    
    logger.info("FAISS index build complete!")
    logger.info(f"  Index: {FAISS_INDEX_FILE}")
    logger.info(f"  Metadata: {FAISS_META_FILE}")
    logger.info(f"  Total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
