"""
Clean and deduplicate section data.

This script deduplicates sections, removes invalid/short entries,
and ensures consistent data quality.
"""

import json
from pathlib import Path

from config import NORMALIZED_FILE, CLEAN_FILE, MIN_TEXT_LENGTH, ensure_directories
from utils import setup_logger, is_valid_section_text

logger = setup_logger(__name__)


def deduplicate_sections(data: list[dict]) -> tuple[dict, int]:
    """
    Deduplicate sections, keeping the longest text for each key.

    Args:
        data: List of section records.

    Returns:
        Tuple of (unique sections dict, duplicates count).
    """
    unique_sections = {}
    duplicates_removed = 0
    
    for record in data:
        key = (record["act_name"], str(record["section_number"]))
        
        if key in unique_sections:
            duplicates_removed += 1
            existing = unique_sections[key]
            # Keep the one with longer text
            if len(record.get("text", "")) > len(existing.get("text", "")):
                unique_sections[key] = record
        else:
            unique_sections[key] = record
    
    return unique_sections, duplicates_removed


def filter_and_clean(
    unique_sections: dict,
    min_length: int = MIN_TEXT_LENGTH,
) -> tuple[list[dict], int]:
    """
    Filter invalid sections and enforce schema.

    Args:
        unique_sections: Dictionary of unique sections.
        min_length: Minimum text length for valid section.

    Returns:
        Tuple of (clean records list, dropped count).
    """
    clean_records = []
    invalid_dropped = 0
    
    for record in unique_sections.values():
        text = record.get("text", "").strip()
        
        if not is_valid_section_text(text, min_length):
            invalid_dropped += 1
            continue
        
        # Enforce consistent schema
        clean_record = {
            "id": record["id"],
            "act_name": record["act_name"],
            "act_year": record["act_year"],
            "category": record["category"],
            "section_number": record["section_number"],
            "text": text,
            "source": record.get("source", "India Code"),
        }
        clean_records.append(clean_record)
    
    # Sort for deterministic output
    clean_records.sort(key=lambda x: (x["act_name"], x["id"]))
    
    return clean_records, invalid_dropped


def main() -> None:
    """Main entry point for data cleaning."""
    logger.info("Starting data cleaning...")
    ensure_directories()
    
    if not NORMALIZED_FILE.exists():
        logger.error(f"Input file not found: {NORMALIZED_FILE}")
        return
    
    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_input = len(data)
    logger.info(f"Loaded {total_input} records")
    
    # Deduplicate
    unique_sections, duplicates = deduplicate_sections(data)
    logger.info(f"Removed {duplicates} duplicates")
    
    # Filter and clean
    clean_records, invalid_count = filter_and_clean(unique_sections)
    logger.info(f"Dropped {invalid_count} invalid records")
    
    # Write output
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_FILE.write_text(
        json.dumps(clean_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    logger.info(f"Final clean count: {len(clean_records)}")
    logger.info(f"Output: {CLEAN_FILE}")


if __name__ == "__main__":
    main()
