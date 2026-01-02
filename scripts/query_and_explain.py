"""
Query FAISS and explain results using LLM.

This script combines semantic search with LLM-powered explanations
to provide user-friendly answers to legal questions.
"""

import os
import pickle
import sys
from typing import Optional

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    EMBEDDING_MODEL,
    FAISS_TOP_K,
    GROQ_MODEL,
    GROQ_API_KEY,
)
from utils import setup_logger

logger = setup_logger(__name__)


def load_faiss() -> tuple[faiss.Index, list[dict]]:
    """Load FAISS index and metadata."""
    if not FAISS_INDEX_FILE.exists() or not FAISS_META_FILE.exists():
        logger.error("FAISS files not found. Run build_faiss_index.py first.")
        sys.exit(1)
    
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(FAISS_META_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded {index.ntotal} vectors")
    return index, metadata


def retrieve_sections(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = FAISS_TOP_K,
) -> list[dict]:
    """Retrieve relevant sections from FAISS index."""
    query_vec = model.encode([query]).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        record = metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)
    
    return results


def explain_with_llm(
    query: str,
    retrieved: list[dict],
    client: Groq,
) -> str:
    """
    Generate LLM explanation for retrieved sections.

    Args:
        query: User's question.
        retrieved: Retrieved section results.
        client: Groq API client.

    Returns:
        LLM-generated explanation.
    """
    if not retrieved:
        return (
            "No relevant legal provisions were found in the knowledge base.\n\n"
            "This does not mean no law exists—only that it is not in the loaded data.\n\n"
            "Disclaimer: This is for educational purposes only and is not legal advice."
        )
    
    # Build context from retrieved sections
    context = ""
    for r in retrieved:
        context += (
            f"Act: {r.get('act_name', 'Unknown')}\n"
            f"Section: {r.get('section_number', 'Unknown')}\n"
            f"Text: {r.get('text', '')}\n\n"
        )
    
    prompt = f"""You are a legal information assistant for Indian laws.

RULES:
- Use ONLY the legal text provided.
- Do NOT add new laws.
- Do NOT give legal advice.
- Explain in simple, user-friendly language.
- Mention Act name and Section number.
- End with a disclaimer.

USER QUESTION:
{query}

LEGAL TEXT:
{context}

ANSWER:
"""
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        # Fallback response
        fallback = "Based on the retrieved legal provisions:\n\n"
        for r in retrieved:
            fallback += (
                f"- {r.get('act_name', 'Unknown')} "
                f"(Section {r.get('section_number', 'Unknown')})\n"
            )
        fallback += (
            "\nA detailed explanation could not be generated.\n\n"
            "Disclaimer: This is for educational purposes only "
            "and does not constitute legal advice."
        )
        return fallback


def display_results(results: list[dict]) -> None:
    """Display retrieved sections."""
    print("\n" + "-" * 60)
    print("RELEVANT SECTIONS:")
    print("-" * 60)
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('act_name', 'Unknown')} - Section {r.get('section_number', 'N/A')}")
        print(f"   Score: {r['score']:.3f}")
        print(f"   {r.get('text', '')[:300]}...")
        print()


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  LEGAL RAG SYSTEM - AI-Powered Answers")
    print("=" * 60 + "\n")
    
    # Check API key
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. LLM explanations will be disabled.")
        client = None
    else:
        client = Groq(api_key=api_key)
    
    # Load resources
    index, metadata = load_faiss()
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Ready! Type your legal question below.")
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
        
        # Retrieve
        results = retrieve_sections(query, model, index, metadata)
        display_results(results)
        
        # Explain
        print("\n" + "=" * 60)
        print("AI EXPLANATION:")
        print("=" * 60 + "\n")
        
        if client:
            explanation = explain_with_llm(query, results, client)
        else:
            explanation = "LLM explanations disabled (no API key)."
        
        print(explanation)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
