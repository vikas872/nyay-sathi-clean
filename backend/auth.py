"""
Authentication middleware for Nyay Sathi API.

Handles Bearer token validation to secure the endpoints.
"""

from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import API_SECRET_KEYS
from logger import app_logger as logger

# Bearer token scheme
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify the Bearer token against the configured secret keys.

    Args:
        credentials: The HTTP authorization credentials.

    Returns:
        The valid token.

    Raises:
        HTTPException: If the token is invalid or missing.
    """
    token = credentials.credentials
    
    if not API_SECRET_KEYS:
        # If no keys configured, warn but allow (or fail safe - let's fail safe)
        logger.warning("No API_SECRET_KEYS configured! All requests will fail.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: No API keys defined",
        )

    if token not in API_SECRET_KEYS:
        logger.warning(f"Invalid API key attempt: {token[:4]}***")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
