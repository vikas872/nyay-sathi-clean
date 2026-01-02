"""
Normalize extracted sections with proper metadata.

This script reads section JSON files and normalizes them with
consistent metadata (act name, year, category) from the metadata file.
"""

import json
from pathlib import Path

from config import SECTIONS_JSON_DIR, METADATA_FILE, NORMALIZED_FILE, ensure_directories
from utils import setup_logger, safe_filename

logger = setup_logger(__name__)


def load_metadata() -> dict:
    """
    Load act metadata from JSON file.

    Returns:
        Dictionary mapping act titles to metadata.
    """
    if not METADATA_FILE.exists():
        logger.warning(f"Metadata file not found: {METADATA_FILE}")
        return {}
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_section(
    section: dict,
    meta: dict,
    act_title: str,
) -> dict | None:
    """
    Normalize a single section with metadata.

    Args:
        section: Raw section data.
        meta: Act metadata.
        act_title: Title of the act.

    Returns:
        Normalized section dict or None if invalid.
    """
    section_number = section.get("section_number")
    text = section.get("section_text")
    
    if not section_number or not text:
        return None
    
    act_name = meta.get("act_name", act_title)
    act_year = meta.get("year", 0)
    
    # Create stable ID
    safe_name = safe_filename(act_name)
    record_id = f"{safe_name}_{act_year}_{section_number}"
    
    return {
        "id": record_id,
        "act_name": act_name,
        "act_year": act_year,
        "category": meta.get("category", "Uncategorized"),
        "section_number": section_number,
        "text": text,
        "source": "India Code",
    }


def main() -> None:
    """Main entry point for normalization."""
    logger.info("Starting section normalization...")
    ensure_directories()
    
    act_metadata = load_metadata()
    normalized_records = []
    skipped_count = 0
    
    for json_file in SECTIONS_JSON_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        raw_title = data.get("act_name", "").strip()
        
        if "Invalid URL" in raw_title:
            continue
        
        meta = act_metadata.get(raw_title, {})
        
        if not meta:
            logger.debug(f"No metadata for: {raw_title}")
        
        for section in data.get("sections", []):
            record = normalize_section(section, meta, raw_title)
            if record:
                normalized_records.append(record)
            else:
                skipped_count += 1
    
    # Write output
    NORMALIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
    NORMALIZED_FILE.write_text(
        json.dumps(normalized_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    logger.info(f"Normalized {len(normalized_records)} sections (skipped {skipped_count})")
    logger.info(f"Output: {NORMALIZED_FILE}")


if __name__ == "__main__":
    main()
