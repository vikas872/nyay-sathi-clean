"""
Terminal UI components for Nyay Sathi CLI.

Claude Code-inspired interface with live streaming status updates,
collapsible tool panels, and beautiful formatting.
"""

import os
import re
import sys
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.markdown import Markdown
from rich.box import ROUNDED, SIMPLE
from rich.style import Style
from rich.columns import Columns

console = Console()


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    """Print a minimal header."""
    console.print()
    console.print("[bold cyan]⚖️  NYAY SATHI[/bold cyan] [dim]- Indian Legal Assistant[/dim]")
    console.print("[dim]Type your question. Commands: help, clear, exit[/dim]")
    console.print()


def print_help():
    """Print help."""
    console.print()
    console.print("[bold]Commands:[/bold] help, clear, history, exit")
    console.print("[dim]Just type your legal question to get started.[/dim]")
    console.print()


# =============================================================================
# STREAMING STATUS DISPLAY (Claude Code-like)
# =============================================================================

@dataclass
class ToolStep:
    """Represents a tool execution step."""
    tool: str
    display_name: str
    icon: str
    message: str
    query: str = ""
    detail: str = ""
    status: str = "running"  # running, success, error
    count: int = 0
    collapsed: bool = False


@dataclass  
class StreamingState:
    """State for streaming display."""
    current_status: str = "Starting..."
    current_icon: str = "🔄"
    steps: List[ToolStep] = field(default_factory=list)
    is_thinking: bool = False
    thinking_message: str = ""
    final_answer: str = ""
    mode: str = ""
    confidence: str = ""
    local_sources: List[Dict] = field(default_factory=list)
    web_sources: List[Dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    done: bool = False


class StreamingDisplay:
    """
    Claude Code-like streaming display with collapsible tool panels.
    
    Shows live progress as the agent thinks and uses tools.
    """
    
    def __init__(self):
        self.state = StreamingState()
        self.live: Optional[Live] = None
        self.start_time = time.time()
    
    def _build_step_panel(self, step: ToolStep, index: int) -> Panel:
        """Build a panel for a single tool step."""
        if step.status == "running":
            # Running state - show spinner
            content = Text()
            content.append(f"{step.icon} ", style="bold")
            content.append(step.message, style="cyan")
            if step.query:
                content.append(f"\n   ", style="dim")
                content.append(f'"{step.query[:60]}{"..." if len(step.query) > 60 else ""}"', style="dim italic")
            if step.detail:
                content.append(f"\n   {step.detail}", style="dim")
            
            return Panel(
                Group(Spinner("dots", text=""), content),
                border_style="cyan",
                box=ROUNDED,
                padding=(0, 1),
            )
        
        elif step.status == "success":
            # Success state - collapsible
            header = Text()
            header.append(f"✓ ", style="green bold")
            header.append(f"{step.display_name}", style="green")
            header.append(f" • ", style="dim")
            header.append(f"{step.count} result{'s' if step.count != 1 else ''}", style="dim")
            
            if step.collapsed:
                # Collapsed view - just the header
                return Panel(
                    header,
                    border_style="dim green",
                    box=SIMPLE,
                    padding=(0, 1),
                )
            else:
                # Expanded view
                content = Text()
                content.append(header)
                if step.query:
                    content.append(f"\n  Query: ", style="dim")
                    content.append(f'"{step.query[:80]}"', style="dim italic")
                
                return Panel(
                    content,
                    border_style="green",
                    box=ROUNDED,
                    padding=(0, 1),
                )
        
        else:
            # Error state
            header = Text()
            header.append(f"✗ ", style="red bold")
            header.append(f"{step.display_name}", style="red")
            header.append(f" • No results", style="dim")
            
            return Panel(
                header,
                border_style="dim red",
                box=SIMPLE,
                padding=(0, 1),
            )
    
    def _build_display(self) -> Group:
        """Build the complete display."""
        elements = []
        
        # Current status (if not done)
        if not self.state.done and not self.state.final_answer:
            status_text = Text()
            status_text.append(f"{self.state.current_icon} ", style="bold")
            status_text.append(self.state.current_status, style="cyan")
            elements.append(status_text)
            elements.append(Text())  # Spacer
        
        # Tool steps (show all, collapse completed ones)
        for i, step in enumerate(self.state.steps):
            # Auto-collapse completed steps except the most recent
            if step.status != "running" and i < len(self.state.steps) - 1:
                step.collapsed = True
            elements.append(self._build_step_panel(step, i))
        
        # Thinking indicator
        if self.state.is_thinking:
            thinking = Text()
            thinking.append("💭 ", style="bold")
            thinking.append(self.state.thinking_message, style="yellow")
            elements.append(Text())  # Spacer
            elements.append(thinking)
        
        # Error
        if self.state.error:
            elements.append(Text())
            elements.append(Text(f"❌ Error: {self.state.error}", style="red"))
        
        return Group(*elements)
    
    def start(self):
        """Start the live display."""
        self.live = Live(
            self._build_display(),
            console=console,
            refresh_per_second=10,
            transient=True,  # Will be replaced by final output
        )
        self.live.__enter__()
    
    def update_status(self, message: str, icon: str = "🔄", detail: str = ""):
        """Update the current status message."""
        self.state.current_status = message
        self.state.current_icon = icon
        if self.live:
            self.live.update(self._build_display())
    
    def add_tool_start(self, tool: str, display_name: str, icon: str, message: str, 
                       query: str = "", detail: str = ""):
        """Add a new tool step in running state."""
        step = ToolStep(
            tool=tool,
            display_name=display_name,
            icon=icon,
            message=message,
            query=query,
            detail=detail,
            status="running"
        )
        self.state.steps.append(step)
        self.state.is_thinking = False
        if self.live:
            self.live.update(self._build_display())
    
    def update_tool_result(self, tool: str, status: str, count: int = 0):
        """Update a tool step with its result."""
        for step in reversed(self.state.steps):
            if step.tool == tool and step.status == "running":
                step.status = status
                step.count = count
                break
        if self.live:
            self.live.update(self._build_display())
    
    def set_thinking(self, message: str):
        """Show thinking indicator."""
        self.state.is_thinking = True
        self.state.thinking_message = message
        if self.live:
            self.live.update(self._build_display())
    
    def set_error(self, error: str):
        """Set error state."""
        self.state.error = error
        if self.live:
            self.live.update(self._build_display())
    
    def set_sources(self, local: List[Dict], web: List[Dict]):
        """Set source information."""
        self.state.local_sources = local
        self.state.web_sources = web
    
    def set_answer(self, text: str, mode: str, confidence: str, 
                   tokens_in: int = 0, tokens_out: int = 0):
        """Set the final answer."""
        self.state.final_answer = text
        self.state.mode = mode
        self.state.confidence = confidence
        self.state.tokens_in = tokens_in
        self.state.tokens_out = tokens_out
        self.state.done = True
    
    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.__exit__(None, None, None)
            self.live = None


# =============================================================================
# SIMPLE STATUS DISPLAY (Fallback for non-streaming)
# =============================================================================

class StatusDisplay:
    """Simple context manager for showing status with spinner."""
    
    def __init__(self, message: str = "Searching"):
        self.message = message
        self.live = None
    
    def __enter__(self):
        self.live = Live(
            Spinner("dots", text=f" {self.message}..."),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        self.live.__enter__()
        return self
    
    def update(self, message: str):
        """Update the status message."""
        if self.live:
            self.live.update(Spinner("dots", text=f" {message}..."))
    
    def __exit__(self, *args):
        if self.live:
            self.live.__exit__(*args)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def colorize_citations(text: str) -> str:
    """Color citation numbers like [1], [2] in the text."""
    return re.sub(r'\[(\d+)\]', r'[cyan][\1][/cyan]', text)


def stream_text(text: str, delay: float = 0.003):
    """Stream text with a typing effect."""
    with Live(console=console, refresh_per_second=30, transient=False) as live:
        display_text = Text()
        words = text.split(" ")
        
        for i, word in enumerate(words):
            # Check for citations to color
            if re.match(r'\[\d+\]', word):
                display_text.append(word + " ", style="cyan")
            else:
                display_text.append(word + " ")
            
            live.update(display_text)
            time.sleep(delay)
    
    console.print()


def print_answer(
    answer: str,
    mode: str,
    confidence: str,
    local_sources: list,
    web_sources: list,
    tokens_in: int = 0,
    tokens_out: int = 0,
    stream: bool = True,
):
    """Print the answer with formatted output."""
    console.print()
    
    # Print answer with colored citations and streaming
    if stream:
        stream_text(answer)
    else:
        console.print(colorize_citations(answer))
        console.print()
    
    # Sources section - only show sources that are actually cited in the answer
    import re
    cited_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
    
    # Filter to only sources that were cited
    relevant = []
    for i, src in enumerate(local_sources[:5], 1):  # Check up to 5
        if i in cited_numbers:
            relevant.append((i, src))
    
    if relevant and mode in ("grounded", "hybrid"):
        console.print()
        console.print("[dim]─── Sources ───[/dim]")
        for idx, src in relevant:
            act = src.get("act", "Unknown")
            section = src.get("section", "")
            score = src.get("score", 0)
            console.print(f"  [cyan][{idx}][/cyan] {act}, Section {section} [dim]({score:.0%})[/dim]")
    
    if web_sources and mode == "hybrid":
        console.print("[dim]─── Web Sources ───[/dim]")
        for i, src in enumerate(web_sources[:3], 1):
            title = src.get('title', '')[:50]
            url = src.get('url', '')
            domain = src.get('domain', '')
            console.print(f"  [cyan][W{i}][/cyan] {title}")
            if url:
                console.print(f"       [link={url}]{url}[/link]", style="dim")
    
    # Footer
    console.print()
    
    mode_desc = {
        "grounded": "📚 Database",
        "hybrid": "🌐 Database+Web",
        "fallback": "💭 General",
        "error": "❌ Error",
    }.get(mode, "?")
    
    conf_info = {
        "high": ("●", "green", "High"),
        "medium": ("◐", "yellow", "Medium"),
        "low": ("○", "red", "Low"),
    }.get(confidence, ("○", "dim", "Unknown"))
    
    conf_symbol, conf_color, conf_text = conf_info
    
    footer_parts = [
        f"[dim]{mode_desc}[/dim]",
        f"[{conf_color}]{conf_symbol} {conf_text}[/{conf_color}]",
    ]
    
    if tokens_in or tokens_out:
        footer_parts.append(f"[dim]{tokens_in}→{tokens_out} tokens[/dim]")
    
    footer = " │ ".join(footer_parts)
    
    width = console.width or 80
    console.print(f"{footer:>{width}}")
    console.print()


def print_streaming_result(display: StreamingDisplay):
    """Print the final result after streaming completes."""
    state = display.state
    
    if state.error:
        print_error(state.error)
        return
    
    # Print the answer
    print_answer(
        answer=state.final_answer,
        mode=state.mode,
        confidence=state.confidence,
        local_sources=state.local_sources,
        web_sources=state.web_sources,
        tokens_in=state.tokens_in,
        tokens_out=state.tokens_out,
        stream=True,
    )


def print_error(message: str, detail: Optional[str] = None):
    """Print error."""
    console.print(f"\n[red]Error: {message}[/red]")
    if detail:
        console.print(f"[dim]{detail}[/dim]")
    console.print()


def print_warning(message: str):
    """Print warning."""
    console.print(f"[yellow]⚠ {message}[/yellow]")


def print_success(message: str):
    """Print success."""
    console.print(f"[green]✓ {message}[/green]")


def get_prompt() -> str:
    """Input prompt."""
    return "[bold cyan]>[/bold cyan] "


def print_goodbye():
    """Goodbye."""
    console.print("\n[dim]Goodbye! 🙏[/dim]\n")
