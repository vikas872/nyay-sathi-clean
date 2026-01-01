"""
Download legal acts from India Code website.

This script downloads HTML pages for specified legal acts from the
India Code portal for subsequent processing.
"""

import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    ACTS_HTML_DIR,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    INDIACODE_BASE_URL,
    ensure_directories,
)
from utils import setup_logger, safe_filename

logger = setup_logger(__name__)

# India Code Act URLs (handles that resolve to actual act pages)
ACT_URLS: list[str] = [
    f"{INDIACODE_BASE_URL}/handle/123456789/20062?locale=en",  # Bharatiya Nyaya Sanhita
    f"{INDIACODE_BASE_URL}/handle/123456789/20061?locale=en",  # Bharatiya Nagarik Suraksha
    f"{INDIACODE_BASE_URL}/handle/123456789/1657?locale=en",   # Indian Penal Code
    f"{INDIACODE_BASE_URL}/handle/123456789/1537?locale=en",   # Code of Criminal Procedure
    f"{INDIACODE_BASE_URL}/handle/123456789/2340?locale=en",   # Indian Evidence Act
    f"{INDIACODE_BASE_URL}/handle/123456789/3002?locale=en",   # Protection of Women
    f"{INDIACODE_BASE_URL}/handle/123456789/4101?locale=en",   # POCSO Act
    f"{INDIACODE_BASE_URL}/handle/123456789/2263?locale=en",   # IT Act
]


def get_real_act_url(handle_url: str, session: requests.Session) -> Optional[str]:
    """
    Resolve a handle URL to the actual act page URL.

    Args:
        handle_url: India Code handle URL.
        session: Requests session for connection pooling.

    Returns:
        Resolved URL or None if failed.
    """
    try:
        response = session.get(handle_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for link in soup.find_all("a", href=True):
            text = link.get_text(strip=True).lower()
            if "view" in text or "act" in text:
                return requests.compat.urljoin(handle_url, link["href"])
        
        # Fallback: use the handle page itself
        return handle_url
        
    except requests.RequestException as e:
        logger.error(f"Failed to resolve URL {handle_url}: {e}")
        return None


def download_act(
    url: str,
    index: int,
    session: requests.Session,
    output_dir: Path = ACTS_HTML_DIR,
) -> bool:
    """
    Download a single act HTML page.

    Args:
        url: URL of the act to download.
        index: Index number for filename.
        session: Requests session for connection pooling.
        output_dir: Directory to save the HTML file.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Processing act [{index}]: {url}")
    
    real_url = get_real_act_url(url, session)
    if not real_url:
        return False
    
    logger.debug(f"Resolved to: {real_url}")
    
    try:
        response = session.get(real_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        filename = f"{index:02d}_act.html"
        filepath = output_dir / filename
        
        filepath.write_text(response.text, encoding="utf-8")
        logger.info(f"Saved: {filepath.name}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Failed to download act: {e}")
        return False


def main() -> None:
    """Main entry point for downloading acts."""
    logger.info("Starting act download...")
    ensure_directories()
    
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    
    success_count = 0
    
    for idx, url in enumerate(ACT_URLS, start=1):
        if download_act(url, idx, session):
            success_count += 1
        
        if idx < len(ACT_URLS):
            time.sleep(REQUEST_DELAY * 60)  # Polite delay
    
    logger.info(f"Download complete. Success: {success_count}/{len(ACT_URLS)}")


if __name__ == "__main__":
    main()
