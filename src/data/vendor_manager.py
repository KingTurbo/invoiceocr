import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
VENDORS_FILE = CONFIG_DIR / "vendors.json"

def _get_initial_vendors():
    return {
        "Staples Inc": ["staples", "staples Advantage"],
        "Amazon Web Services": ["amazon web services", "aws", "invoice.amazon.com"],
        "Google Workspace": ["google workspace", "g suite"],
        "Uline": ["uline", "uline.com", "800-295-5510"]
    }

def _ensure_vendors_file():
    if not VENDORS_FILE.exists():
        logger.info(f"Vendors file not found at {VENDORS_FILE}. Creating with default values.")
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(VENDORS_FILE, 'w', encoding='utf-8') as f:
                json.dump(_get_initial_vendors(), f, indent=4)
        except IOError as e:
            logger.critical(f"Could not create the initial vendors file. Error: {e}")

def load_vendor_identifiers() -> dict:
    _ensure_vendors_file()
    try:
        with open(VENDORS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse vendors file. Returning empty dict. Error: {e}")
        return {}

def get_vendor_list() -> list[str]:
    vendor_data = load_vendor_identifiers()
    return list(vendor_data.keys())

def add_or_update_vendor(vendor_name: str, keywords: list[str]):
    vendor_data = load_vendor_identifiers()
    vendor_data[vendor_name] = keywords
    try:
        with open(VENDORS_FILE, 'w', encoding='utf-8') as f:
            json.dump(vendor_data, f, indent=4)
        logger.info(f"Successfully added or updated vendor: {vendor_name}")
    except IOError as e:
        logger.error(f"Failed to save updated vendor data. Error: {e}")