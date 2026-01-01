# ================= RAILWAY / FREE-TIER SAFE RAG ENGINE =================
import os

# ---- HARD SAFETY FLAGS (must be first) ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ---------------- ENV ----------------
load_dotenv()

# ---------------- PATH RESOLUTION ----------------
# File is: backend/rag_engine.py
# Repo root is one level up from backend/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_META_PATH = DATA_DIR / "faiss_meta.pkl"

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.50

# ---------------- GLOBALS (LAZY) ----------------
index = None
metadata = None
embedder = None
client = None

# ======================================================================
# INITIALIZATION (LIGHTWEIGHT ONLY)
# ======================================================================

def initialize_rag():
    """
    Initialize ONLY what is required for startup.
    DO NOT load embedding model here (memory heavy).
    """
    global index, metadata, client

    print("Starting Nyay Sathi Backend (Railway Safe Mode)")
    print(f"FAISS path: {FAISS_INDEX_PATH}")

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")

    if not FAISS_META_PATH.exists():
        raise FileNotFoundError(f"FAISS metadata not found at {FAISS_META_PATH}")

    # Load FAISS index (read-only, CPU)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"FAISS loaded with {index.ntotal} vectors")

    # Load metadata
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Init Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("RAG system initialized successfully (light mode)")

# ======================================================================
# LAZY EMBEDDING MODEL LOADER
# ======================================================================

def get_embedder():
    """
    Load SentenceTransformer ONLY when first query comes.
    Prevents Railway free-tier OOM during startup.
    """
    global embedder
    if embedder is None:
        print("Loading embedding model lazily...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return embedder

# ======================================================================
# RETRIEVAL
# ======================================================================

def retrieve_sections(query: str):
    embedder_local = get_embedder()

    query_vec = embedder_local.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        record = metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)

    return results

# ======================================================================
# PROMPTS
# ======================================================================

SYSTEM_PROMPT_A = """You are Nyay Sathi, a helpful Indian legal assistant.
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

SYSTEM_PROMPT_B = """You are Nyay Sathi, a helpful Indian legal assistant.
MODE: GENERAL FALLBACK.

INSTRUCTIONS:
1. No specific legal section matched.
2. Do NOT cite Acts or Sections.
3. Give only high-level educational explanation.
4. Encourage rephrasing the question.
5. Do NOT give legal advice.

MANDATORY DISCLAIMER:
End with: "Disclaimer: This information is for educational purposes only and does not constitute legal advice."
"""

# ======================================================================
# LLM EXPLANATION
# ======================================================================

def explain_with_llm(query, retrieved):
    if not retrieved:
        mode = "fallback"
        top_score = 0.0
    else:
        top_score = retrieved[0]["score"]
        mode = "grounded" if top_score >= CONFIDENCE_THRESHOLD else "fallback"

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
        system_prompt = SYSTEM_PROMPT_A
    else:
        user_content = f"USER QUESTION:\n{query}\n\n(No relevant legal text found)"
        system_prompt = SYSTEM_PROMPT_B

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        return mode, response.choices[0].message.content.strip(), top_score

    except Exception as e:
        print("LLM error:", e)
        return (
            "fallback",
            "System error occurred. "
            "Disclaimer: This information is for educational purposes only and does not constitute legal advice.",
            0.0,
        )
