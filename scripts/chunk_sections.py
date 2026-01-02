"""
Chunk sections into smaller pieces for RAG.

This script splits long section texts into smaller chunks with
overlap for better retrieval performance.
"""

import json
from pathlib import Path

from config import (
    CLEAN_FILE,
    CHUNKS_FILE,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHARS_PER_TOKEN,
    ensure_directories,
)
from utils import setup_logger, estimate_tokens

logger = setup_logger(__name__)


def split_into_chunks(
    text: str,
    max_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split text into chunks respecting sentence boundaries.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.

    Returns:
        List of text chunks.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    # Split into sentences (simple approximation)
    sentences = text.replace(";", ".").split(".")
    
    chunks = []
    current_chunk: list[str] = []
    current_length = 0.0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence = sentence + "."
        sent_len = estimate_tokens(sentence)
        
        # If adding this sentence exceeds limit, finalize current chunk
        if current_length + sent_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            overlap_len = 0.0
            new_start = []
            for s in reversed(current_chunk):
                s_len = estimate_tokens(s)
                if overlap_len + s_len <= overlap_tokens:
                    new_start.insert(0, s)
                    overlap_len += s_len
                else:
                    break
            
            current_chunk = new_start
            current_length = overlap_len
        
        current_chunk.append(sentence)
        current_length += sent_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_section(record: dict) -> list[dict]:
    """
    Chunk a single section record.

    Args:
        record: Section record with text.

    Returns:
        List of chunk records.
    """
    text = record.get("text", "")
    text_chunks = split_into_chunks(text)
    
    chunk_records = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_record = {
            "chunk_id": f"{record['id']}_chunk_{i + 1}",
            "parent_id": record["id"],
            "act_name": record["act_name"],
            "act_year": record["act_year"],
            "category": record["category"],
            "section_number": record["section_number"],
            "text": chunk_text,
            "source": record.get("source", "India Code"),
        }
        chunk_records.append(chunk_record)
    
    return chunk_records


def main() -> None:
    """Main entry point for chunking."""
    logger.info("Starting section chunking...")
    ensure_directories()
    
    if not CLEAN_FILE.exists():
        logger.error(f"Input file not found: {CLEAN_FILE}")
        return
    
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_sections = len(data)
    all_chunks = []
    
    for record in data:
        all_chunks.extend(chunk_section(record))
    
    # Write output
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    avg_size = (
        sum(estimate_tokens(c["text"]) for c in all_chunks) / len(all_chunks)
        if all_chunks else 0
    )
    
    logger.info(f"Processed {total_sections} sections")
    logger.info(f"Generated {len(all_chunks)} chunks (avg {avg_size:.1f} tokens)")
    logger.info(f"Output: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
