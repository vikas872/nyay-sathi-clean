# =============================================================================
# Nyay Sathi - Production Dockerfile for HuggingFace Spaces
# =============================================================================
# Optimized for:
# - Fast builds (UV package manager)
# - Small image size (~1.5GB vs 3GB+)
# - HuggingFace Spaces compatibility
# - CPU inference (no CUDA overhead)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies with UV
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install UV (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV for optimal builds
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1

# Create venv and install ONLY production dependencies
# Using CPU-only torch to save ~2GB
RUN uv venv /build/.venv && \
    uv pip install --python=/build/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    torch && \
    uv pip install --python=/build/.venv/bin/python \
    fastapi==0.115.* \
    uvicorn[standard]==0.32.* \
    pydantic==2.* \
    httpx==0.28.* \
    sentence-transformers==3.* \
    faiss-cpu==1.9.* \
    numpy==2.* \
    groq==0.15.* \
    python-dotenv==1.* \
    duckduckgo-search==7.*

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/processed && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy backend code
COPY --chown=appuser:appuser backend/*.py ./

# Copy ONLY the processed FAISS data (not raw data)
COPY --chown=appuser:appuser data/processed/faiss.index ./data/processed/
COPY --chown=appuser:appuser data/processed/faiss_meta.pkl ./data/processed/

# Switch to non-root user
USER appuser

# Production environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVICE=cpu
ENV LOG_LEVEL=INFO

# HuggingFace Spaces port
EXPOSE 7860

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start with optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
