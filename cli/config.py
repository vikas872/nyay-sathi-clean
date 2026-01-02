"""
CLI Configuration for Nyay Sathi.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API endpoint - change this to your HuggingFace deployment URL
API_BASE_URL: str = os.getenv(
    "NYAY_SATHI_API_URL",
    "http://localhost:10000"
)

# API key for authentication
API_KEY: str = os.getenv(
    "NYAY_SATHI_API_KEY",
    "nyay-sathi-local-dev-key"
)

# =============================================================================
# CLI SETTINGS
# =============================================================================

# Maximum query length
MAX_QUERY_LENGTH: int = 2000

# Request timeout (seconds)
REQUEST_TIMEOUT: float = 60.0

# History file
HISTORY_FILE: Path = Path.home() / ".nyay_sathi_history"

# Maximum history entries
MAX_HISTORY: int = 100
