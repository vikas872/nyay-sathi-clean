"""
Rate limiting middleware for Nyay Sathi API.

Prevents abuse by limiting the number of requests per IP address.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config import RATE_LIMIT_PER_MINUTE
from logger import app_logger as logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    
    Tracks request counts per IP address within a minute window.
    """

    def __init__(self, app, limit: int = RATE_LIMIT_PER_MINUTE):
        super().__init__(app)
        self.limit = limit
        # Dictionary to store request timestamps: {ip: [timestamp1, timestamp2, ...]}
        self.requests: Dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get client IP
        client_ip = request.client.host or "unknown"
        
        # Current time
        now = time.time()
        
        # Clean up old requests (older than 60 seconds)
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < 60]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS
            )
            
        # Record this request
        self.requests[client_ip].append(now)
        
        # Proceed
        return await call_next(request)
