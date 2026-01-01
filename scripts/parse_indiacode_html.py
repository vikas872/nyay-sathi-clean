"""
Parse India Code HTML files to extract section information.

This script parses downloaded act HTML files and extracts section
numbers and text content into structured JSON format.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from config import ACTS_HTML_DIR, SECTIONS_JSON_DIR, ensure_directories
from utils import setup_logger, clean_text

logger = setup_logger(__name__)


def extract_act_name(soup: BeautifulSoup) -> str:
    """
    Extract the act name from HTML.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        Act name string.
    """
    if soup.find("h1"):
        return clean_text(soup.find("h1").get_text())
    if soup.title:
        return clean_text(soup.title.get_text())
    return "Unknown Act"


def extract_sections(full_text: str) -> list[dict]:
    """
    Extract sections from act text using legal numbering patterns.

    Args:
        full_text: Full text content of the act.

    Returns:
        List of section dictionaries.
    """
    sections = []
    
    # Pattern matches: "1. Title", "1A. Title", "Section 1"
    section_pattern = re.compile(
        r"(?:Section\s+)?(\d+[A-Z]?)\.\s",
        re.IGNORECASE,
    )
    
    matches = list(section_pattern.finditer(full_text))
    
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        
        section_number = match.group(1)
        section_text = full_text[start:end]
        
        sections.append({
            "section_number": section_number,
            "section_text": clean_text(section_text),
        })
    
    return sections


def parse_act_file(html_file: Path, output_dir: Path) -> int:
    """
    Parse a single act HTML file.

    Args:
        html_file: Path to HTML file.
        output_dir: Directory to save JSON output.

    Returns:
        Number of sections extracted.
    """
    logger.info(f"Parsing: {html_file.name}")
    
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    act_name = extract_act_name(soup)
    full_text = clean_text(soup.get_text(separator=" ", strip=True))
    sections = extract_sections(full_text)
    
    output_data = {
        "act_name": act_name,
        "source_file": html_file.name,
        "sections_count": len(sections),
        "sections": sections,
    }
    
    output_file = output_dir / f"{html_file.stem}.json"
    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    logger.debug(f"Saved {len(sections)} sections to {output_file.name}")
    return len(sections)


def main() -> None:
    """Main entry point for parsing act files."""
    logger.info("Starting HTML parsing...")
    ensure_directories()
    
    html_files = list(ACTS_HTML_DIR.glob("*.html"))
    
    if not html_files:
        logger.warning("No HTML files found to parse")
        return
    
    total_sections = 0
    for html_file in html_files:
        total_sections += parse_act_file(html_file, SECTIONS_JSON_DIR)
    
    logger.info(f"Complete. Parsed {len(html_files)} files, {total_sections} total sections")


if __name__ == "__main__":
    main()
