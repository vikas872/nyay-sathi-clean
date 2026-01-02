"""
Agentic tool calling system for Nyay Sathi.

The LLM decides which tools to use based on the query.
"""

import json
from typing import Any, AsyncGenerator

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from logger import rag_logger as logger
from tools import TOOLS, AGENT_SYSTEM_PROMPT

# Lazy imports to avoid circular dependencies
_rag_engine = None
_browser = None


def _get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        from rag_engine import retrieve, get_vectors_count
        _rag_engine = {"retrieve": retrieve, "count": get_vectors_count}
    return _rag_engine


async def _get_browser():
    global _browser
    if _browser is None:
        from browser import web_search, read_url
        _browser = {"search": web_search, "read": read_url}
    return _browser


def _is_greeting(query: str) -> bool:
    """Check if the query is just a greeting (no legal question)."""
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", 
                 "namaste", "namaskar", "thanks", "thank you", "bye", "goodbye"}
    normalized = query.lower().strip().rstrip("!.,?")
    return normalized in greetings or len(normalized) < 4


# Tool display names for better UX
TOOL_DISPLAY_INFO = {
    "rag_search": {
        "name": "Legal Database",
        "icon": "📚",
        "searching": "Searching legal database",
        "detail": "Indian Penal Code, BNS, Acts & Sections"
    },
    "web_search": {
        "name": "Web Search",
        "icon": "🌐", 
        "searching": "Searching trusted legal websites",
        "detail": "gov.in, indiankanoon.org"
    },
    "read_url": {
        "name": "Reading Page",
        "icon": "📄",
        "searching": "Reading webpage content",
        "detail": ""
    }
}


# =============================================================================
# TOOL EXECUTION
# =============================================================================

async def execute_tool(name: str, args: dict) -> dict:
    """
    Execute a tool and return the result.
    """
    logger.info(f"Executing tool: {name} with args: {args}")
    
    if name == "rag_search":
        rag = _get_rag_engine()
        query = args.get("query", "")
        
        # Run synchronous RAG in thread pool to avoid blocking event loop
        import asyncio
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, rag["retrieve"], query)
        
        # Format results for LLM - truncate text to prevent context bloat
        if not results:
            return {"status": "no_results", "data": []}
        
        formatted = []
        for i, r in enumerate(results[:3], 1):
            formatted.append({
                "index": i,
                "act": r.get("act_name", "Unknown"),
                "section": r.get("section_number", ""),
                "text": r.get("text", "")[:800],  # Increased for better context
                "score": round(r.get("score", 0), 3),
            })
        
        return {"status": "success", "data": formatted}
    
    elif name == "web_search":
        browser = await _get_browser()
        query = args.get("query", "")
        results = await browser["search"](query)
        
        if not results:
            return {"status": "no_results", "data": []}
        
        formatted = [
            {"index": i, "title": r.title, "snippet": r.snippet, "url": r.url, "domain": r.domain}
            for i, r in enumerate(results, 1)
        ]
        
        return {"status": "success", "data": formatted}
    
    elif name == "read_url":
        browser = await _get_browser()
        url = args.get("url", "")
        content = await browser["read"](url)
        
        if not content:
            return {"status": "blocked", "reason": "URL not from trusted domain"}
        
        return {
            "status": "success",
            "data": {
                "title": content.title,
                "text": content.text[:2000],
                "domain": content.domain,
            }
        }
    
    else:
        return {"status": "error", "reason": f"Unknown tool: {name}"}


# =============================================================================
# AGENT LOOP
# =============================================================================

async def run_agent(query: str, max_iterations: int = 5) -> dict:
    """
    Run the agentic loop with tool calling.
    
    Returns:
        dict with answer, tools_used, tokens
    """
    if not GROQ_API_KEY:
        return {
            "answer": "API key not configured.",
            "mode": "error",
            "tools_used": [],
            "tokens_in": 0,
            "tokens_out": 0,
        }
    
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    
    tools_used = []
    total_tokens_in = 0
    total_tokens_out = 0
    
    for iteration in range(max_iterations):
        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")
        
        try:
            # Force tool use on first iteration for legal questions
            # This ensures RAG is always consulted first
            if iteration == 0 and not _is_greeting(query):
                current_tool_choice = {"type": "function", "function": {"name": "rag_search"}}
            else:
                current_tool_choice = "auto"
            
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=current_tool_choice,
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Track tokens
            if response.usage:
                total_tokens_in += response.usage.prompt_tokens
                total_tokens_out += response.usage.completion_tokens
            
            message = response.choices[0].message
            
            # Check if LLM wants to call tools (Structured)
            if message.tool_calls:
                # ... existing structured handling ...
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except:
                        args = {}
                    
                    try:
                        result = await execute_tool(name, args)
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        result = {"status": "error", "reason": str(e)}
                    tools_used.append({"name": name, "args": args, "result": result["status"], "data": result.get("data")})
                    
                    # Truncate tool result to prevent context bloat
                    result_str = json.dumps(result, ensure_ascii=False)
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + '..."}'
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
                continue
            
            # Check for XML-style tool calls (Fallback with loose Regex)
            content = message.content or ""
            import re
            
            tool_matches = []
            for tool_def in TOOLS:
                t_name = tool_def["function"]["name"]
                # Match <t_name>... and any closing tag or just end of string
                # This handles <rag_search>...{"query": "..."}</function> or </rag_search>
                pattern = f"<{t_name}>(.*?)(?:</{t_name}>|</function>|$)"
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for m in matches:
                    tool_matches.append((t_name, m.group(1)))
            
            if tool_matches:
                # Add assistant message
                messages.append({"role": "assistant", "content": content})
                
                tool_results = []
                for name, args_str in tool_matches:
                    try:
                        # Try parsing JSON
                        args = json.loads(args_str)
                    except:
                        # Clean up string and try again
                        cleaned = args_str.strip().replace("'", '"')
                        try:
                            args = json.loads(cleaned)
                        except:
                            # Fallback
                            args = {"query": args_str.strip(), "url": args_str.strip()}
                    
                    try:
                        result = await execute_tool(name, args)
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        result = {"status": "error", "reason": str(e)}
                    # Include data in tools_used so main.py can extract sources
                    tools_used.append({"name": name, "args": args, "result": result["status"], "data": result.get("data")})
                    tool_results.append(f"Result of {name}: {json.dumps(result, ensure_ascii=False)}")
                
                # Feed results back
                messages.append({
                    "role": "user", 
                    "content": "Tool Output:\n" + "\n".join(tool_results) + "\n\nBased on these results, please provide the final answer."
                })
                continue

            else:
                # No tool calls - LLM is done
                answer = message.content or "I couldn't generate a response."
                
                # Determine mode based on tools used
                if any(t["name"] == "web_search" for t in tools_used):
                    mode = "hybrid"
                elif any(t["name"] == "rag_search" for t in tools_used):
                    mode = "grounded"
                else:
                    mode = "fallback"
                
                return {
                    "answer": answer,
                    "mode": mode,
                    "tools_used": tools_used,
                    "tokens_in": total_tokens_in,
                    "tokens_out": total_tokens_out,
                }
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "mode": "error",
                "tools_used": tools_used,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
            }
    
    # Max iterations reached
    return {
        "answer": "I couldn't complete the request within the allowed steps.",
        "mode": "fallback",
        "tools_used": tools_used,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
    }


# =============================================================================
# STREAMING AGENT (for live status updates)
# =============================================================================

async def run_agent_streaming(query: str, max_iterations: int = 5) -> AsyncGenerator[dict, None]:
    """
    Run the agentic loop with streaming status updates.
    
    Yields events like:
    - {"type": "status", "message": "...", "icon": "..."}
    - {"type": "tool_start", "tool": "rag_search", "query": "..."}
    - {"type": "tool_result", "tool": "...", "status": "success", "count": N}
    - {"type": "thinking", "message": "..."}
    - {"type": "answer", "text": "...", "mode": "...", "confidence": "..."}
    - {"type": "sources", "local": [...], "web": [...]}
    """
    if not GROQ_API_KEY:
        yield {"type": "error", "message": "API key not configured"}
        return
    
    # Initial status
    yield {
        "type": "status",
        "message": "Understanding your question",
        "icon": "🤔",
        "detail": query[:100] + "..." if len(query) > 100 else query
    }
    
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    
    tools_used = []
    total_tokens_in = 0
    total_tokens_out = 0
    
    for iteration in range(max_iterations):
        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")
        
        # Show thinking status
        if iteration > 0:
            yield {
                "type": "thinking",
                "message": f"Analyzing results (step {iteration + 1})",
                "icon": "💭"
            }
        
        try:
            # Force tool use on first iteration for legal questions
            if iteration == 0 and not _is_greeting(query):
                current_tool_choice = {"type": "function", "function": {"name": "rag_search"}}
            else:
                current_tool_choice = "auto"
            
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=current_tool_choice,
                temperature=0.1,
                max_tokens=2048,
            )
            
            if response.usage:
                total_tokens_in += response.usage.prompt_tokens
                total_tokens_out += response.usage.completion_tokens
            
            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except:
                        args = {}
                    
                    # Get tool display info
                    tool_info = TOOL_DISPLAY_INFO.get(name, {
                        "name": name, "icon": "🔧", "searching": f"Running {name}", "detail": ""
                    })
                    
                    # Emit tool start event
                    yield {
                        "type": "tool_start",
                        "tool": name,
                        "display_name": tool_info["name"],
                        "icon": tool_info["icon"],
                        "message": tool_info["searching"],
                        "detail": tool_info["detail"],
                        "query": args.get("query", args.get("url", ""))
                    }
                    
                    try:
                        result = await execute_tool(name, args)
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        result = {"status": "error", "reason": str(e)}
                    
                    # Count results
                    result_count = 0
                    if result.get("status") == "success":
                        data = result.get("data", [])
                        result_count = len(data) if isinstance(data, list) else 1
                    
                    # Emit tool result event
                    yield {
                        "type": "tool_result",
                        "tool": name,
                        "display_name": tool_info["name"],
                        "icon": "✓" if result["status"] == "success" else "✗",
                        "status": result["status"],
                        "count": result_count,
                        "message": f"Found {result_count} results" if result["status"] == "success" else "No results"
                    }
                    
                    tools_used.append({
                        "name": name, 
                        "args": args, 
                        "result": result["status"], 
                        "data": result.get("data")
                    })
                    
                    result_str = json.dumps(result, ensure_ascii=False)
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + '..."}'
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
                continue
            
            # Handle XML-style tool calls (fallback)
            content = message.content or ""
            import re
            
            tool_matches = []
            for tool_def in TOOLS:
                t_name = tool_def["function"]["name"]
                pattern = f"<{t_name}>(.*?)(?:</{t_name}>|</function>|$)"
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for m in matches:
                    tool_matches.append((t_name, m.group(1)))
            
            if tool_matches:
                messages.append({"role": "assistant", "content": content})
                
                tool_results = []
                for name, args_str in tool_matches:
                    try:
                        args = json.loads(args_str)
                    except:
                        cleaned = args_str.strip().replace("'", '"')
                        try:
                            args = json.loads(cleaned)
                        except:
                            args = {"query": args_str.strip(), "url": args_str.strip()}
                    
                    tool_info = TOOL_DISPLAY_INFO.get(name, {
                        "name": name, "icon": "🔧", "searching": f"Running {name}", "detail": ""
                    })
                    
                    yield {
                        "type": "tool_start",
                        "tool": name,
                        "display_name": tool_info["name"],
                        "icon": tool_info["icon"],
                        "message": tool_info["searching"],
                        "query": args.get("query", args.get("url", ""))
                    }
                    
                    try:
                        result = await execute_tool(name, args)
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        result = {"status": "error", "reason": str(e)}
                    
                    result_count = len(result.get("data", [])) if isinstance(result.get("data"), list) else (1 if result.get("data") else 0)
                    
                    yield {
                        "type": "tool_result",
                        "tool": name,
                        "status": result["status"],
                        "count": result_count
                    }
                    
                    tools_used.append({
                        "name": name,
                        "args": args,
                        "result": result["status"],
                        "data": result.get("data")
                    })
                    tool_results.append(f"Result of {name}: {json.dumps(result, ensure_ascii=False)}")
                
                messages.append({
                    "role": "user",
                    "content": "Tool Output:\n" + "\n".join(tool_results) + "\n\nBased on these results, please provide the final answer."
                })
                continue
            
            else:
                # Final answer - LLM is done
                yield {
                    "type": "status",
                    "message": "Generating response",
                    "icon": "✍️"
                }
                
                answer = message.content or "I couldn't generate a response."
                
                # Determine mode
                if any(t["name"] == "web_search" for t in tools_used):
                    mode = "hybrid"
                elif any(t["name"] == "rag_search" for t in tools_used):
                    mode = "grounded"
                else:
                    mode = "fallback"
                
                # Confidence
                if mode == "grounded":
                    confidence = "high"
                elif mode == "hybrid":
                    confidence = "medium"
                else:
                    confidence = "low"
                
                # Extract sources
                local_sources = []
                web_sources = []
                
                for tool in tools_used:
                    if tool["name"] == "rag_search" and tool.get("result") == "success":
                        data = tool.get("data", [])
                        if isinstance(data, list):
                            for item in data:
                                local_sources.append({
                                    "act": item.get("act", "Unknown"),
                                    "section": str(item.get("section", "")),
                                    "text": item.get("text", "")[:300],
                                    "score": item.get("score", 0)
                                })
                    
                    elif tool["name"] == "web_search" and tool.get("result") == "success":
                        data = tool.get("data", [])
                        if isinstance(data, list):
                            for item in data:
                                web_sources.append({
                                    "url": item.get("url", ""),
                                    "title": item.get("title", ""),
                                    "domain": item.get("domain", "")
                                })
                
                # Emit sources
                if local_sources or web_sources:
                    yield {
                        "type": "sources",
                        "local": local_sources,
                        "web": web_sources
                    }
                
                # Emit final answer
                yield {
                    "type": "answer",
                    "text": answer,
                    "mode": mode,
                    "confidence": confidence,
                    "tokens_in": total_tokens_in,
                    "tokens_out": total_tokens_out
                }
                return
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }
            return
    
    # Max iterations reached
    yield {
        "type": "answer",
        "text": "I couldn't complete the request within the allowed steps.",
        "mode": "fallback",
        "confidence": "low",
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out
    }
