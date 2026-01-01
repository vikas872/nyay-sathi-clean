"""
Fetch full section content from India Code API.

This script iterates through act HTML files and fetches the complete
text content for each section using India Code's internal API.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    ACTS_HTML_DIR,
    SECTIONS_HTML_DIR,
    INDIACODE_SHOW_DATA_URL,
    INDIACODE_API_ENDPOINT,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    ensure_directories,
)
from utils import setup_logger

logger = setup_logger(__name__)

MAX_CONSECUTIVE_ERRORS: int = 20
MAX_ORDERNO: int = 2000


def extract_act_id(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the act ID from HTML meta tags or links.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        Act ID string or None if not found.
    """
    # Try meta tag first
    for meta in soup.find_all("meta", attrs={"name": "DC.identifier"}):
        content = meta.get("content", "")
        if content.startswith("AC_") or content.startswith("act"):
            return content
    
    # Fallback to link search
    for link in soup.find_all("a", href=True):
        if "show-data" in link["href"]:
            qs = parse_qs(urlparse(link["href"]).query)
            for key, value in qs.items():
                if key.lower() == "actid":
                    return value[0]
    
    return None


def fetch_section_content(
    session: requests.Session,
    act_id: str,
    section_id: str,
) -> Optional[dict]:
    """
    Fetch section content from India Code API.

    Args:
        session: Requests session.
        act_id: Act identifier.
        section_id: Section identifier.

    Returns:
        JSON response or None if failed.
    """
    headers = session.headers.copy()
    headers["X-Requested-With"] = "XMLHttpRequest"
    
    params = {"actid": act_id, "sectionID": section_id}
    
    try:
        response = session.get(
            INDIACODE_API_ENDPOINT,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.debug(f"API request failed: {e}")
    
    return None


def process_act(
    act_file: Path,
    session: requests.Session,
    output_dir: Path,
) -> int:
    """
    Process a single act file and fetch all sections.

    Args:
        act_file: Path to act HTML file.
        session: Requests session.
        output_dir: Directory to save section JSON files.

    Returns:
        Number of sections successfully fetched.
    """
    logger.info(f"Processing: {act_file.name}")
    act_slug = act_file.stem
    
    with open(act_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    act_id = extract_act_id(soup)
    if not act_id:
        logger.warning(f"Could not find act ID for {act_file.name}")
        return 0
    
    logger.debug(f"Act ID: {act_id}")
    
    sections_fetched = 0
    orderno = 1
    error_count = 0
    
    while orderno <= MAX_ORDERNO:
        page_url = f"{INDIACODE_SHOW_DATA_URL}?actid={act_id}&orderno={orderno}"
        
        try:
            response = session.get(page_url, timeout=REQUEST_TIMEOUT)
            
            if "Invalid URL" in response.text or response.status_code != 200:
                error_count += 1
                if error_count > MAX_CONSECUTIVE_ERRORS:
                    logger.debug(f"Max errors reached at orderno {orderno}")
                    break
                orderno += 1
                continue
            
            # Extract section ID from page
            match = re.search(r"secId\s*=\s*'(\d+)';", response.text)
            if not match:
                match = re.search(r"sectionId\s*=\s*'(\d+)';", response.text)
            
            if not match:
                error_count += 1
                if error_count > MAX_CONSECUTIVE_ERRORS:
                    break
                orderno += 1
                continue
            
            section_id = match.group(1)
            filename = f"{act_slug}_ord_{orderno}_sec_{section_id}.json"
            filepath = output_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                orderno += 1
                continue
            
            # Fetch section content from API
            content = fetch_section_content(session, act_id, section_id)
            
            if content:
                content["_meta"] = {
                    "act_slug": act_slug,
                    "act_id": act_id,
                    "section_id": section_id,
                    "orderno": orderno,
                    "url": page_url,
                }
                
                filepath.write_text(
                    json.dumps(content, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.debug(f"Saved: {filename}")
                sections_fetched += 1
                error_count = 0
            
        except requests.RequestException as e:
            logger.debug(f"Request error at orderno {orderno}: {e}")
            error_count += 1
            if error_count > MAX_CONSECUTIVE_ERRORS:
                break
        
        orderno += 1
        time.sleep(REQUEST_DELAY)
    
    logger.info(f"Fetched {sections_fetched} sections from {act_file.name}")
    return sections_fetched


def main() -> None:
    """Main entry point for fetching sections."""
    logger.info("Starting section fetch...")
    ensure_directories()
    
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    
    total_sections = 0
    
    for act_file in sorted(ACTS_HTML_DIR.glob("*.html")):
        total_sections += process_act(act_file, session, SECTIONS_HTML_DIR)
    
    logger.info(f"Complete. Total sections fetched: {total_sections}")


if __name__ == "__main__":
    main()
