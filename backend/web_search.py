"""
Web search fallback for Nyay Sathi.

Searches Indian legal sources when local RAG confidence is low.
Uses duckduckgo-search with STRICT SafeSearch.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from config import WEB_SEARCH_MAX_RESULTS
from logger import rag_logger as logger
from sanitizer import sanitize_web_content


# =============================================================================
# DOMAIN WHITELIST - ONLY THESE DOMAINS ALLOWED
# =============================================================================

WHITELISTED_DOMAINS = {
    # Government
    "indiacode.nic.in",
    "legislative.gov.in", 
    "lawmin.gov.in",
    "india.gov.in",
    "doj.gov.in",
    "main.sci.gov.in",
    "niti.gov.in",
    
    # Legal resources
    "indiankanoon.org",
    "legalserviceindia.com",
    "lawrato.com",
    "vakilsearch.com",
    
    # Encyclopedia
    "en.wikipedia.org",
    "britannica.com",
}


@dataclass
class WebSearchResult:
    """A single web search result with citation."""
    url: str
    title: str
    snippet: str
    source_domain: str
    relevance_score: float = 0.0


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return ""


def is_whitelisted(domain: str) -> bool:
    """Check if domain is in whitelist."""
    for allowed in WHITELISTED_DOMAINS:
        if allowed in domain or domain.endswith(f".{allowed.split('.')[0]}.gov.in"):
            return True
    # Allow any gov.in domain
    if domain.endswith(".gov.in") or domain.endswith(".nic.in"):
        return True
    return False


async def search_trusted_sources(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> list[WebSearchResult]:
    """
    Search for legal information using DuckDuckGo with SafeSearch.
    ONLY returns results from whitelisted domains.
    """
    results: list[WebSearchResult] = []
    
    # Build query focused on Indian law
    full_query = f"{query} site:gov.in OR site:nic.in OR site:indiankanoon.org"
    
    logger.info(f"Web search for: {query[:40]}...")
    
    try:
        ddgs = DDGS()
        search_results = ddgs.text(
            full_query,
            region="in-en",
            safesearch="on",  # STRICT SafeSearch
            max_results=max_results * 5,
        )
        
        for r in search_results:
            url = r.get("href", "")
            domain = get_domain(url)
            
            # STRICT: Only whitelisted domains
            if not is_whitelisted(domain):
                continue
            
            results.append(WebSearchResult(
                url=url,
                title=sanitize_web_content(r.get("title", domain), 200),
                snippet=sanitize_web_content(r.get("body", ""), 400),
                source_domain=domain,
                relevance_score=1.0 - (len(results) * 0.1),
            ))
            
            if len(results) >= max_results:
                break
        
        logger.info(f"Found {len(results)} whitelisted results")
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
    
    return results


def format_citations(results: list[WebSearchResult]) -> str:
    """Format search results as citations."""
    if not results:
        return ""
    return "\n".join(f"[{i}] {r.title} - {r.url}" for i, r in enumerate(results, 1))
