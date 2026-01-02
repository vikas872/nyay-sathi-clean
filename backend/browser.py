"""
Browser automation for Nyay Sathi web search.

Uses httpx for lightweight API-based search.
Strict whitelist enforcement for trusted domains only.
"""

import asyncio
from urllib.parse import urlparse, quote_plus

import httpx
from dataclasses import dataclass

from logger import rag_logger as logger
from sanitizer import sanitize_web_content


# =============================================================================
# TRUSTED DOMAINS - WHITELIST ONLY
# =============================================================================

TRUSTED_DOMAINS = {
    # Government
    "indiacode.nic.in",
    "legislative.gov.in",
    "lawmin.gov.in",
    "india.gov.in",
    "doj.gov.in",
    "main.sci.gov.in",
    "niti.gov.in",
    "prsindia.org",
    
    # Legal resources
    "indiankanoon.org",
    "legalserviceindia.com",
    
    # Encyclopedia
    "en.wikipedia.org",
}


@dataclass
class SearchResult:
    """A web search result."""
    url: str
    title: str
    snippet: str
    domain: str
    source: str = "web_search"


@dataclass
class PageContent:
    """Content extracted from a webpage."""
    url: str
    title: str
    text: str
    domain: str


def is_trusted_domain(url: str) -> bool:
    """Check if URL is from a trusted domain."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Allow any gov.in or nic.in
        if domain.endswith(".gov.in") or domain.endswith(".nic.in"):
            return True
        
        # Check whitelist
        for trusted in TRUSTED_DOMAINS:
            if trusted in domain:
                return True
        return False
    except:
        return False


async def web_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """
    Search the web using SearXNG public API (no browser needed).
    Falls back gracefully if search fails.
    """
    results = []
    
    # Use SearXNG public instance
    encoded_query = quote_plus(f"{query} site:gov.in OR site:indiankanoon.org")
    search_url = f"https://searx.be/search?q={encoded_query}&format=json&categories=general"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(search_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", [])[:max_results * 3]:
                    url = item.get("url", "")
                    if not is_trusted_domain(url):
                        continue
                    
                    results.append(SearchResult(
                        url=url,
                        title=item.get("title", "")[:100],
                        snippet=item.get("content", "")[:300],
                        domain=urlparse(url).netloc,
                        source="web_search"
                    ))
                    
                    if len(results) >= max_results:
                        break
                
                logger.info(f"Web search found {len(results)} trusted results")
            else:
                logger.warning(f"Search API returned {response.status_code}")
                
    except Exception as e:
        logger.error(f"Web search error: {e}")
    
    return results


async def read_url(url: str) -> PageContent | None:
    """
    Read content from a trusted URL using httpx.
    """
    if not is_trusted_domain(url):
        logger.warning(f"Blocked untrusted URL: {url}")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            if response.status_code == 200:
                # Simple content extraction
                text = response.text
                
                # Try to extract title
                import re
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', text, re.IGNORECASE)
                title = title_match.group(1) if title_match else urlparse(url).netloc
                
                # Strip HTML tags for body
                body = re.sub(r'<[^>]+>', ' ', text)
                body = re.sub(r'\s+', ' ', body)[:3000]
                
                return PageContent(
                    url=url,
                    title=sanitize_web_content(title, 200),
                    text=sanitize_web_content(body, 3000),
                    domain=urlparse(url).netloc,
                )
                
    except Exception as e:
        logger.error(f"Error reading page {url}: {e}")
    
    return None
