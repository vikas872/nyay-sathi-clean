#!/usr/bin/env python3
"""
Nyay Sathi CLI - Indian Legal Assistant.

A beautiful command-line interface for querying Indian legal information.
Connects to the Nyay Sathi API (local or HuggingFace deployment).

Features:
- Claude Code-like streaming status updates
- Live tool execution display
- Collapsible progress panels

Usage:
    python nyay_cli.py                    # Interactive mode
    python nyay_cli.py "your question"    # Single query mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from rich.prompt import Prompt

from config import (
    API_BASE_URL,
    API_KEY,
    HISTORY_FILE,
    MAX_HISTORY,
    MAX_QUERY_LENGTH,
    REQUEST_TIMEOUT,
)
from ui import (
    clear_screen,
    console,
    get_prompt,
    print_answer,
    print_error,
    print_goodbye,
    print_header,
    print_help,
    print_success,
    print_warning,
    StreamingDisplay,
    print_streaming_result,
    StatusDisplay,
)


# =============================================================================
# API CLIENT
# =============================================================================

class NyaySathiClient:
    """Client for the Nyay Sathi API with streaming support."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self) -> bool:
        """Check if the API is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def ask(self, question: str) -> dict:
        """
        Send a question to the API (non-streaming).

        Args:
            question: The legal question to ask.

        Returns:
            API response dictionary.

        Raises:
            Exception: On API errors.
        """
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{self.base_url}/ask",
                headers=self.headers,
                json={"question": question},
            )

            if response.status_code == 401:
                raise Exception("Authentication failed. Check your API key.")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded. Please wait a moment.")
            elif response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")

            return response.json()

    def ask_streaming(self, question: str, display: StreamingDisplay):
        """
        Send a question and process streaming response with live updates.
        
        Args:
            question: The legal question to ask.
            display: StreamingDisplay instance for live updates.
        """
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/ask/stream",
                headers=self.headers,
                json={"question": question},
            ) as response:
                if response.status_code == 401:
                    raise Exception("Authentication failed. Check your API key.")
                elif response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please wait a moment.")
                elif response.status_code != 200:
                    raise Exception(f"API error: {response.status_code}")
                
                # Process Server-Sent Events
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk
                    
                    # Parse SSE events
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        self._process_sse_event(event_str, display)
    
    def _process_sse_event(self, event_str: str, display: StreamingDisplay):
        """Parse and process a single SSE event."""
        event_type = "message"
        data = None
        
        for line in event_str.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except:
                    data = {"raw": line[6:]}
        
        if data is None:
            return
        
        # Handle different event types
        if event_type == "status":
            display.update_status(
                message=data.get("message", "Processing..."),
                icon=data.get("icon", "🔄"),
                detail=data.get("detail", "")
            )
        
        elif event_type == "tool_start":
            display.add_tool_start(
                tool=data.get("tool", "unknown"),
                display_name=data.get("display_name", data.get("tool", "Tool")),
                icon=data.get("icon", "🔧"),
                message=data.get("message", "Running..."),
                query=data.get("query", ""),
                detail=data.get("detail", "")
            )
        
        elif event_type == "tool_result":
            display.update_tool_result(
                tool=data.get("tool", "unknown"),
                status=data.get("status", "error"),
                count=data.get("count", 0)
            )
        
        elif event_type == "thinking":
            display.set_thinking(data.get("message", "Thinking..."))
        
        elif event_type == "sources":
            display.set_sources(
                local=data.get("local", []),
                web=data.get("web", [])
            )
        
        elif event_type == "answer":
            display.set_answer(
                text=data.get("text", ""),
                mode=data.get("mode", "fallback"),
                confidence=data.get("confidence", "low"),
                tokens_in=data.get("tokens_in", 0),
                tokens_out=data.get("tokens_out", 0)
            )
        
        elif event_type == "error":
            display.set_error(data.get("message", "Unknown error"))
        
        elif event_type == "done":
            pass  # Stream complete


# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def load_history() -> list[str]:
    """Load query history from file."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())[-MAX_HISTORY:]
        except Exception:
            return []
    return []


def save_history(history: list[str]):
    """Save query history to file."""
    try:
        HISTORY_FILE.write_text(json.dumps(history[-MAX_HISTORY:]))
    except Exception:
        pass


def add_to_history(query: str, history: list[str]) -> list[str]:
    """Add a query to history."""
    if query and query not in history[-5:]:  # Avoid duplicates
        history.append(query)
    return history


# =============================================================================
# MAIN CLI
# =============================================================================

def process_command(cmd: str, history: list[str]) -> bool:
    """
    Process a CLI command.

    Returns:
        True to continue, False to exit.
    """
    cmd_lower = cmd.lower().strip()

    if cmd_lower in ("exit", "quit", "q"):
        return False

    elif cmd_lower == "help":
        print_help()

    elif cmd_lower == "clear":
        clear_screen()
        print_header()

    elif cmd_lower == "history":
        if history:
            console.print("\n[bold]Recent Queries:[/bold]")
            for i, q in enumerate(history[-10:], 1):
                console.print(f"  {i}. {q[:60]}...")
            console.print()
        else:
            print_warning("No history yet.")

    else:
        print_warning(f"Unknown command: {cmd}")
        console.print("[dim]Type 'help' for available commands.[/dim]")

    return True


def run_interactive(client: NyaySathiClient):
    """Run the interactive CLI session."""
    clear_screen()
    print_header()

    # Check API connection
    console.print("[dim]Connecting to API...[/dim]")
    if not client.health_check():
        print_error(
            "Cannot connect to Nyay Sathi API",
            f"Make sure the server is running at {client.base_url}"
        )
        console.print("[dim]Tip: Run 'cd backend && uvicorn main:app --port 10000'[/dim]\n")
        return

    print_success(f"Connected to {client.base_url}")
    console.print()

    # Load history
    history = load_history()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask(get_prompt()).strip()

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/") or user_input.lower() in ("help", "clear", "history", "exit", "quit", "q"):
                cmd = user_input.lstrip("/")
                if not process_command(cmd, history):
                    break
                continue

            # Validate query length
            if len(user_input) > MAX_QUERY_LENGTH:
                print_warning(f"Query too long. Maximum {MAX_QUERY_LENGTH} characters.")
                continue

            # Add to history
            history = add_to_history(user_input, history)
            save_history(history)

            # Query the API with streaming status updates
            display = StreamingDisplay()
            display.start()
            
            try:
                client.ask_streaming(user_input, display)
            except Exception as e:
                display.set_error(str(e))
            finally:
                display.stop()
            
            # Show the final result
            print_streaming_result(display)

        except KeyboardInterrupt:
            console.print("\n")
            break
        except EOFError:
            break

    # Goodbye
    print_goodbye()


def run_single_query(client: NyaySathiClient, question: str):
    """Run a single query with streaming and exit."""
    # Check API connection
    if not client.health_check():
        print_error(
            "Cannot connect to Nyay Sathi API",
            f"Make sure the server is running at {client.base_url}"
        )
        sys.exit(1)

    # Use streaming display
    display = StreamingDisplay()
    display.start()
    
    try:
        client.ask_streaming(question, display)
    except Exception as e:
        display.set_error(str(e))
    finally:
        display.stop()
    
    # Show the final result
    print_streaming_result(display)
    
    if display.state.error:
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Nyay Sathi CLI - Indian Legal Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Legal question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="API key for authentication",
    )

    args = parser.parse_args()

    # Create client
    client = NyaySathiClient(args.api_url, args.api_key)

    if args.question:
        run_single_query(client, args.question)
    else:
        run_interactive(client)


if __name__ == "__main__":
    main()
