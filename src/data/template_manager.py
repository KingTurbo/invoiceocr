import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "config" / "templates"

def _sanitize_filename(vendor_id: str) -> str:
    s = vendor_id.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_-]+', '_', s)
    return f"{s}.json"

def save_template(vendor_id: str, template_data: dict):
    if not vendor_id or not isinstance(template_data, dict):
        logger.error("Invalid arguments: vendor_id must be a non-empty string and template_data must be a dictionary.")
        return

    try:
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        
        filename = _sanitize_filename(vendor_id)
        final_path = TEMPLATES_DIR / filename
        temp_path = TEMPLATES_DIR / f"~{filename}.tmp"

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=4)
        
        os.replace(temp_path, final_path)
        logger.info(f"Successfully saved template for vendor '{vendor_id}' to {final_path}")

    except (IOError, OSError) as e:
        logger.critical(f"Failed to save template for vendor '{vendor_id}'. Error: {e}")

def load_template(vendor_id: str) -> dict | None:
    if not vendor_id:
        return None

    filename = _sanitize_filename(vendor_id)
    file_path = TEMPLATES_DIR / filename

    if not file_path.exists():
        logger.debug(f"Template not found for vendor '{vendor_id}' at {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded template for vendor '{vendor_id}'")
        return data
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse template for '{vendor_id}'. File may be corrupt. Error: {e}")
        return None

def check_template_exists(vendor_id: str) -> bool:
    if not vendor_id:
        return False
        
    filename = _sanitize_filename(vendor_id)
    file_path = TEMPLATES_DIR / filename
    return file_path.is_file()