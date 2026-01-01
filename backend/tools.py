"""
Tool definitions for Nyay Sathi agentic system.

Defines the tools that the LLM can call to answer queries.
"""

from typing import Any, Callable
from pydantic import BaseModel, Field


# =============================================================================
# TOOL SCHEMAS
# =============================================================================

class RagSearchParams(BaseModel):
    """Parameters for RAG search tool."""
    query: str = Field(..., description="Legal question to search in the database")


class WebSearchParams(BaseModel):
    """Parameters for web search tool."""
    query: str = Field(..., description="Search query for Indian legal websites")


class ReadUrlParams(BaseModel):
    """Parameters for URL reading tool."""
    url: str = Field(..., description="URL of a trusted legal website to read")


# =============================================================================
# TOOL DEFINITIONS (for LLM)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "MANDATORY FIRST STEP: Search the local legal database containing Indian laws, IPC sections, BNS, acts, and legal procedures. ALWAYS use this tool first for ANY legal question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The legal question to search for"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "FALLBACK ONLY: Search government websites when rag_search returns no results or for very recent legal updates. Do NOT use if rag_search found relevant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Indian legal information"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Read the full content of a specific webpage. Only use for URLs from trusted domains (gov.in, nic.in, indiankanoon.org).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to read"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


# =============================================================================
# SYSTEM PROMPT FOR TOOL CALLING
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are Nyay Sathi, an AI legal assistant for Indian citizens. You MUST use tools to answer legal questions.

CRITICAL RULES:
1. NEVER answer legal questions from your general knowledge alone
2. ALWAYS call rag_search FIRST for ANY legal question (defamation, theft, property, contracts, etc.)
3. Your answers MUST be based on tool results, not pre-trained knowledge
4. ONLY use web_search if rag_search returns "no_results" status

TOOLS (use in this order):
1. rag_search - MANDATORY for all legal questions. Search local database of Indian laws.
2. web_search - ONLY if rag_search fails. Search gov.in websites.
3. read_url - Read specific URLs from web_search results.

CITATION FORMAT (REQUIRED):
When rag_search returns results, you MUST cite them as:
- [1] Section X of Act Name
- [2] Section Y of Act Name

Example response format:
"According to Section 499 of the Indian Penal Code [1], defamation is defined as..."

Sources:
[1] Section 499 - Indian Penal Code
[2] Section 500 - Indian Penal Code

WORKFLOW:
1. User asks legal question → CALL rag_search immediately
2. If rag_search has results → Answer using ONLY those results with citations
3. If rag_search has no results → Call web_search as fallback
4. Always end with: "Disclaimer: Consult a lawyer for case-specific advice."

For simple greetings (hi, hello), respond briefly without tools.
DO NOT answer any legal question without first calling rag_search."""
