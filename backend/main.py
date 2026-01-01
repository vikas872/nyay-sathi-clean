"""
Nyay Sathi API v2.0 - Legal RAG Backend.

FastAPI application providing legal question-answering using
Retrieval-Augmented Generation with security and web fallback.

Features:
- Bearer token authentication
- Rate limiting
- Input sanitization
- Web search fallback for trusted sources
- Server-Sent Events for streaming status updates
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from auth import verify_api_key
from config import CORS_ORIGINS, RATE_LIMIT_PER_MINUTE
from logger import app_logger as logger
from rag_engine import initialize_rag, process_query
from rate_limiter import RateLimitMiddleware
from sanitizer import validate_query


# =============================================================================
# SCHEMAS
# =============================================================================

class AskRequest(BaseModel):
    """Request schema for asking a legal question."""
    question: str = Field(..., min_length=3, max_length=2000)


class LocalSource(BaseModel):
    """A source from the local legal database."""
    act: str
    section: str
    text: str
    score: float


class WebSource(BaseModel):
    """A source from trusted web search."""
    url: str
    title: str
    domain: str


class AskResponse(BaseModel):
    """Response schema for legal question answers."""
    mode: str  # "grounded" | "hybrid" | "fallback"
    confidence: str  # "high" | "medium" | "low"
    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    local_sources: List[LocalSource]
    web_sources: List[WebSource]
    disclaimer: str = "This information is for educational purposes only and does not constitute legal advice."


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    vectors_loaded: int
    device: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting Nyay Sathi Backend v2.0...")
    try:
        from config import DEVICE
        vector_count = initialize_rag()
        app.state.vectors_loaded = vector_count
        app.state.device = DEVICE
        logger.info(f"RAG system ready with {vector_count} vectors on {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        app.state.vectors_loaded = 0
        app.state.device = "unavailable"

    yield

    # Shutdown
    logger.info("Shutting down Nyay Sathi Backend...")


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Nyay Sathi API",
    description="Indian Legal RAG System - AI-powered legal information assistant",
    version="2.0.0",
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, limit=RATE_LIMIT_PER_MINUTE)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


# =============================================================================
# PUBLIC ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Health check endpoint (public)."""
    return HealthResponse(
        status="ok",
        service="Nyay Sathi Backend",
        version="2.0.0",
        vectors_loaded=getattr(request.app.state, "vectors_loaded", 0),
        device=getattr(request.app.state, "device", "unknown"),
    )


@app.get("/health", response_model=HealthResponse)
def detailed_health(request: Request) -> HealthResponse:
    """Detailed health check for monitoring (public)."""
    return HealthResponse(
        status="ok",
        service="Nyay Sathi Backend",
        version="2.0.0",
        vectors_loaded=getattr(request.app.state, "vectors_loaded", 0),
        device=getattr(request.app.state, "device", "unknown"),
    )


# =============================================================================
# PROTECTED ENDPOINTS
# =============================================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    _token: str = Depends(verify_api_key),
) -> AskResponse:
    """
    Answer a legal question using RAG.

    Requires Bearer token authentication.

    Args:
        request: The question request.

    Returns:
        Answer with sources and confidence level.
    """
    # Validate and sanitize input
    is_valid, sanitized_query, error = validate_query(request.question)

    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    logger.info(f"Processing query: {sanitized_query[:50]}...")

    # Process query through agentic pipeline
    from agent import run_agent
    result = await run_agent(sanitized_query)

    # Determine confidence from mode
    mode = result.get("mode", "fallback")
    if mode == "grounded":
        confidence_str = "high"
    elif mode == "hybrid":
        confidence_str = "medium"
    else:
        confidence_str = "low"

    # Agent returns tools_used, convert to sources
    local_sources = []
    web_sources = []
    
    for tool in result.get("tools_used", []):
        if tool["name"] == "rag_search" and tool.get("result") == "success":
            # Extract local sources
            data = tool.get("data", [])
            if isinstance(data, list):
                for item in data:
                    local_sources.append(LocalSource(
                        act=item.get("act", "Unknown"),
                        section=str(item.get("section", "Unknown")),
                        text=item.get("text", "")[:500],
                        score=item.get("score", 0.0),
                    ))
        
        elif tool["name"] == "web_search" and tool.get("result") == "success":
            # Extract web sources
            data = tool.get("data", [])
            if isinstance(data, list):
                for item in data:
                    web_sources.append(WebSource(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        domain=item.get("domain", ""),
                    ))

    logger.info(f"Response: mode={mode}, tools={[t['name'] for t in result.get('tools_used', [])]}")

    return AskResponse(
        mode=mode,
        confidence=confidence_str,
        answer=result.get("answer", "No answer"),
        tokens_in=result.get("tokens_in", 0),
        tokens_out=result.get("tokens_out", 0),
        local_sources=local_sources,
        web_sources=web_sources,
    )


@app.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    _token: str = Depends(verify_api_key),
):
    """
    Stream answer with real-time status updates using Server-Sent Events.
    
    Events:
    - status: Progress updates (thinking, searching, etc.)
    - tool: Tool execution details
    - answer: Final answer
    - sources: Source information
    - done: Completion signal
    """
    # Validate and sanitize input
    is_valid, sanitized_query, error = validate_query(request.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    logger.info(f"[Stream] Processing: {sanitized_query[:50]}...")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        from agent import run_agent_streaming
        
        try:
            async for event in run_agent_streaming(sanitized_query):
                event_type = event.get("type", "status")
                data = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
            
            yield f"event: done\ndata: {{}}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/sources")
async def list_sources(_token: str = Depends(verify_api_key)) -> dict:
    """
    List available legal sources in the database.

    Requires Bearer token authentication.
    """
    # This would query the metadata for unique acts
    # Placeholder for now
    return {
        "message": "Source listing coming soon",
        "hint": "Use /ask to query the legal database",
    }
