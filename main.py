# ... (sys and path boilerplate at the top) ...
import time
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import logging
import os
import shutil
import re
from PIL import Image
import imagehash

from src.core.document_processor import process_document, split_multipage_pdf


from src.core.logger_config import setup_logging
from src.core.document_processor import process_document
from src.data import template_manager
from src.ocr import engine
from src.gui.learning_interface import start_learning_gui
from config.config import FIELDS_CONFIG
from src.output_handler import save_to_excel

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ARCHIVE_DIR = BASE_DIR / "archive"
TEMP_DIR = BASE_DIR / "temp_processed_images"

EXCEL_OUTPUT_FILE = OUTPUT_DIR / "invoice_records.xlsx"
MULTIPAGE_ARCHIVE_DIR = ARCHIVE_DIR / "multipage_originals"

HASH_MATCH_THRESHOLD = 5

# ... (setup_directories and identify_vendor_via_hashing are the same) ...
def setup_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    MULTIPAGE_ARCHIVE_DIR.mkdir(exist_ok=True)

def identify_vendor_via_hashing(processed_image_path: Path) -> tuple[str | None, dict | None]:
    logger.debug("Identifying vendor by checking all available template hashes.")
    templates_dir = BASE_DIR / "config" / "templates"
    if not templates_dir.exists(): return None, None
    try:
        new_invoice_image = Image.open(processed_image_path)
    except Exception as e:
        logger.error(f"Could not open processed image for hashing: {e}")
        return None, None
    for template_file in templates_dir.glob("*.json"):
        template = template_manager.load_template(template_file.stem)
        if not template or "identifier_area" not in template or "identifier_hash" not in template:
            continue
        area = template["identifier_area"]
        box = (area['x'], area['y'], area['x'] + area['width'], area['y'] + area['height'])
        new_invoice_crop = new_invoice_image.crop(box)
        new_hash = imagehash.phash(new_invoice_crop)
        saved_hash_str = template["identifier_hash"]
        distance = new_hash - imagehash.hex_to_hash(saved_hash_str)
        logger.debug(f"Comparing with '{template['vendor_name']}': hash distance is {distance}")
        if distance <= HASH_MATCH_THRESHOLD:
            vendor_name = template["vendor_name"]
            logger.info(f"MATCH FOUND! Vendor identified as '{vendor_name}' with hash distance {distance}.")
            return vendor_name, template
    logger.warning("Could not identify vendor using any existing template hashes.")
    return None, None

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]',"", name)

def handle_production_path(template, ocr_results, original_file_path):
    vendor_name = template["vendor_name"]
    logger.info(f"Production Path: Processing template for '{vendor_name}'.")
    extracted_data = {"vendor_name": vendor_name}
    for field in template['fields']:
        field_name = field.get("field_name")
        coords = field.get("coordinates")
        if field_name and coords:
            text = engine.filter_text_in_area(ocr_results, tuple(coords.values()))
            extracted_data[field_name] = text.strip()
            logger.debug(f"Extracted '{field_name}': '{text.strip()}'")
    logger.info(f"Final extracted data for {original_file_path.name}: {extracted_data}")
    save_to_excel(extracted_data, EXCEL_OUTPUT_FILE)
    invoice_num = extracted_data.get("invoice_number", "UNKNOWN_INV")
    safe_vendor_name = _sanitize_filename(vendor_name)
    safe_invoice_num = _sanitize_filename(invoice_num)
    new_filename = f"{safe_vendor_name}_{safe_invoice_num}{original_file_path.suffix}"
    logger.info(f"Archiving original file as '{new_filename}'")
    shutil.move(original_file_path, ARCHIVE_DIR / new_filename)

def start_interactive_session(unidentified_tasks):
    logger.info("--- Automated Processing Complete ---")
    if not unidentified_tasks:
        logger.info("No items require manual intervention.")
        return
    logger.info("Starting interactive session for all pending documents.")
    for task in unidentified_tasks:
        suggested_name = task['original_path'].stem
        logger.info(f"Launching GUI with suggested name: '{suggested_name}'")
        new_template = start_learning_gui(
            image_path=str(task['processed_image_path']),
            suggested_vendor_name=suggested_name,
            fields_config=FIELDS_CONFIG
        )
        if new_template:
            final_vendor_name = new_template.get("vendor_name")
            template_manager.save_template(final_vendor_name, new_template)
            logger.info(f"New template for '{final_vendor_name}' has been saved.")
            logger.info(f"Immediately processing '{task['original_path'].name}' with the new template.")
            handle_production_path(new_template, task['ocr_results'], task['original_path'])
        else:
            logger.warning(f"GUI was cancelled for '{suggested_name}'. No template was saved.")
            logger.debug(f"Archiving unprocessed file: {task['original_path'].name}")
            shutil.move(task['original_path'], ARCHIVE_DIR / task['original_path'].name)
        processed_image_path = task['processed_image_path']
        if processed_image_path and processed_image_path.exists():
            os.remove(processed_image_path)


# --- MODIFIED: The main function now has a pre-processing stage ---
def main():
    logger.info("--- Starting Document Processing Run ---")
    setup_directories()
    engine.initialize_reader()

    # --- NEW: Pre-processing stage for splitting PDFs ---
    logger.info("--- Stage 1: Pre-processing and splitting multi-page PDFs ---")
    initial_files = list(INPUT_DIR.glob('*'))
    for file_path in initial_files:
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            split_files = split_multipage_pdf(file_path)
            # If the file was split (more than one result), archive the original
            if len(split_files) > 1:
                logger.info(f"Archiving original multi-page file: {file_path.name}")
                shutil.move(file_path, MULTIPAGE_ARCHIVE_DIR / file_path.name)
    
    logger.info("--- Stage 2: Processing all single-page documents ---")
    files_to_process = [f for f in INPUT_DIR.iterdir() if f.is_file()]
    if not files_to_process:
        logger.info("No files found in input directory to process.")
        logger.info("--- Document Processing Run Finished ---")
        return

    unidentified_queue = []

    for file_path in files_to_process:
        logger.info(f"--- Processing file: {file_path.name} ---")
        processed_image_path = process_document(file_path)
        if not processed_image_path:
            shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            continue
        
        full_ocr_results = engine.get_all_text_and_boxes(str(processed_image_path))
        vendor_name, template_data = identify_vendor_via_hashing(processed_image_path)

        if vendor_name and template_data:
            handle_production_path(template_data, full_ocr_results, file_path)
            if processed_image_path.exists():
                os.remove(processed_image_path)
        else:
            logger.info(f"Queueing '{file_path.name}' for interactive learning.")
            unidentified_queue.append({
                "original_path": file_path,
                "processed_image_path": processed_image_path,
                "ocr_results": full_ocr_results
            })

    if unidentified_queue:
        start_interactive_session(unidentified_queue)

    logger.info("--- Document Processing Run Finished ---")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.critical("An unhandled exception occurred during the run.", exc_info=True)