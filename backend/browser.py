"""
Browser automation for Nyay Sathi web search.

Uses Playwright with headless Chromium for safe web scraping.
Strict whitelist enforcement for trusted domains only.
"""

import asyncio
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, Browser
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


class BrowserSearch:
    """Browser-based search using Playwright."""
    
    def __init__(self):
        self.browser: Browser = None
        self.playwright = None
    
    async def start(self):
        """Start the browser."""
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )
            logger.info("Browser started")
    
    async def _create_context(self):
        """Create a stealthy browser context."""
        return await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            java_script_enabled=True,
            has_touch=False,
            is_mobile=False,
        )
    
    async def stop(self):
        """Stop the browser."""
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
            self.browser = None
            logger.info("Browser stopped")
    
    async def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        """
        Search using Google (with SafeSearch) and filter to trusted domains.
        """
        await self.start()
        results = []
        
        # Build site-restricted query
        sites = " OR ".join(f"site:{d}" for d in list(TRUSTED_DOMAINS)[:5])
        safe_query = f"{query} ({sites})"
        
        try:
            context = await self._create_context()
            page = await context.new_page()
            
            # Anti-detection script
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            await page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            })
            
            # Use Google with SafeSearch
            search_url = f"https://www.google.com/search?q={safe_query}&safe=active&hl=en"
            await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
            
            # Wait for results
            await page.wait_for_selector("div#search", timeout=10000)
            
            # Extract results
            links = await page.query_selector_all("a[href^='http']")
            
            for link in links[:max_results * 3]:
                try:
                    href = await link.get_attribute("href")
                    if not href or not is_trusted_domain(href):
                        continue
                    
                    # Get title
                    title_elem = await link.query_selector("h3")
                    title = await title_elem.inner_text() if title_elem else urlparse(href).netloc
                    
                    # Get snippet (parent text)
                    parent = await link.evaluate_handle("el => el.closest('div')")
                    snippet = ""
                    try:
                        snippet = await parent.inner_text()
                        snippet = sanitize_web_content(snippet, 300)
                    except:
                        pass
                    
                    domain = urlparse(href).netloc
                    
                    results.append(SearchResult(
                        url=href,
                        title=sanitize_web_content(title, 200),
                        snippet=snippet,
                        domain=domain,
                    ))
                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    continue
            
            await context.close()
            logger.info(f"Browser search found {len(results)} trusted results")
            
        except Exception as e:
            logger.error(f"Browser search error: {e}")
        
        return results
    
    async def read_page(self, url: str) -> PageContent | None:
        """
        Read content from a trusted URL.
        """
        if not is_trusted_domain(url):
            logger.warning(f"Blocked untrusted URL: {url}")
            return None
        
        await self.start()
        
        try:
            context = await self._create_context()
            page = await context.new_page()
            
            # Anti-detection script
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            await page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            })
            
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            # Get title
            title = await page.title()
            
            # Get main content (try common selectors)
            content = ""
            for selector in ["main", "article", "#content", ".content", "body"]:
                try:
                    elem = await page.query_selector(selector)
                    if elem:
                        content = await elem.inner_text()
                        break
                except:
                    continue
            
            await context.close()
            
            domain = urlparse(url).netloc
            
            return PageContent(
                url=url,
                title=sanitize_web_content(title, 200),
                text=sanitize_web_content(content, 3000),
                domain=domain,
            )
            
        except Exception as e:
            logger.error(f"Error reading page {url}: {e}")
            return None


# Global browser instance
_browser: BrowserSearch = None


async def get_browser() -> BrowserSearch:
    """Get or create browser instance."""
    global _browser
    if _browser is None:
        _browser = BrowserSearch()
    return _browser


async def web_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """Search the web using browser."""
    browser = await get_browser()
    return await browser.search(query, max_results)


async def read_url(url: str) -> PageContent | None:
    """Read a URL using browser."""
    browser = await get_browser()
    return await browser.read_page(url)
