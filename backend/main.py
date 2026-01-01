"""
Nyay Sathi API - Legal RAG Backend.

FastAPI application providing legal question-answering using
Retrieval-Augmented Generation.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import CORS_ORIGINS
from logger import app_logger as logger
from rag_engine import initialize_rag, retrieve_sections, explain_with_llm


# =============================================================================
# SCHEMAS
# =============================================================================

class AskRequest(BaseModel):
    """Request schema for asking a legal question."""
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    """A source citation from the knowledge base."""
    act: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    """Response schema for legal question answers."""
    mode: str  # "rag" | "fallback"
    confidence: str  # "high" | "low"
    answer: str
    sources: List[Source]
    disclaimer: str = "This information is for educational purposes only and does not constitute legal advice."


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    vectors_loaded: int


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting Nyay Sathi Backend...")
    try:
        vector_count = initialize_rag()
        app.state.vectors_loaded = vector_count
        logger.info(f"RAG system ready with {vector_count} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        app.state.vectors_loaded = 0
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nyay Sathi Backend...")


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Nyay Sathi API",
    description="Legal information assistant for Indian laws",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="Nyay Sathi Backend",
        vectors_loaded=getattr(request.app.state, "vectors_loaded", 0),
    )


@app.get("/health", response_model=HealthResponse)
def detailed_health(request: Request) -> HealthResponse:
    """Detailed health check for monitoring."""
    return HealthResponse(
        status="ok",
        service="Nyay Sathi Backend",
        vectors_loaded=getattr(request.app.state, "vectors_loaded", 0),
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Answer a legal question using RAG.

    Args:
        request: The question request.

    Returns:
        Answer with sources and confidence level.
    """
    query = request.question.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info(f"Processing question: {query[:50]}...")
    
    # 1. Retrieve relevant sections
    raw_results = retrieve_sections(query)
    logger.debug(f"Retrieved {len(raw_results)} sections")
    
    # 2. Generate explanation with LLM
    rag_mode, explanation, score = explain_with_llm(query, raw_results)
    
    # Map internal mode to API mode
    api_mode = "rag" if rag_mode == "grounded" else "fallback"
    confidence_str = "high" if api_mode == "rag" else "low"
    
    logger.info(f"Response mode: {api_mode}, confidence: {confidence_str}")
    
    # 3. Format sources
    sources = []
    if api_mode == "rag":
        for r in raw_results:
            sources.append(Source(
                act=r.get("act_name", "Unknown"),
                section=str(r.get("section_number", "Unknown")),
                text=r.get("text", "")[:500],  # Truncate for response size
                score=round(r.get("score", 0.0), 3),
            ))
    
    return AskResponse(
        mode=api_mode,
        confidence=confidence_str,
        answer=explanation,
        sources=sources,
    )
