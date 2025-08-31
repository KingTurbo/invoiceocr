# ... (sys and path boilerplate at the top) ...
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import time
import logging
import os
import shutil
import re
import uuid
from PIL import Image
import imagehash

from src.core.logger_config import setup_logging
from src.core.document_processor import process_document, split_multipage_pdf
from src.data import template_manager
from src.ocr import engine
from src.gui.learning_interface import start_learning_gui
from config.config import FIELDS_CONFIG
from src.output_handler import save_to_excel

setup_logging()
logger = logging.getLogger(__name__)

# ... (directory and config definitions are the same) ...
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ARCHIVE_DIR = BASE_DIR / "archive"
TEMP_DIR = BASE_DIR / "temp_processed_images"
EXCEL_OUTPUT_FILE = OUTPUT_DIR / "invoice_records.xlsx"
HASH_MATCH_THRESHOLD = 5

def setup_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

# --- MODIFIED: Now takes an in-memory image object ---
def identify_vendor_via_hashing(processed_image_obj: Image.Image) -> tuple[str | None, dict | None]:
    logger.debug("Identifying vendor by checking all available template hashes.")
    templates_dir = BASE_DIR / "config" / "templates"
    if not templates_dir.exists(): return None, None

    for template_file in templates_dir.glob("*.json"):
        template = template_manager.load_template(template_file.stem)
        if not template or "identifier_area" not in template or "identifier_hash" not in template:
            continue
        
        area = template["identifier_area"]
        box = (area['x'], area['y'], area['x'] + area['width'], area['y'] + area['height'])
        # Crop the in-memory object directly
        new_invoice_crop = processed_image_obj.crop(box)

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

# --- MODIFIED: Now takes an image object, not full OCR results ---
def handle_production_path(template, processed_image_obj: Image.Image, original_file_path):
    vendor_name = template["vendor_name"]
    logger.info(f"Production Path: Processing template for '{vendor_name}'.")
    extracted_data = {"vendor_name": vendor_name}
    for field in template['fields']:
        field_name = field.get("field_name")
        coords = field.get("coordinates")
        if field_name and coords:
            # --- CORE CHANGE: Use the new targeted OCR function ---
            text = engine.extract_text_from_area(processed_image_obj, tuple(coords.values()))
            extracted_data[field_name] = text.strip()
            logger.debug(f"Extracted '{field_name}': '{text.strip()}'")

    logger.info(f"Final extracted data for {original_file_path.name}: {extracted_data}")
    save_to_excel(extracted_data, EXCEL_OUTPUT_FILE)
    
    unique_id = str(uuid.uuid4()).split('-')[0]
    invoice_num = extracted_data.get("invoice_number", "UNKNOWN_INV")
    safe_vendor_name = _sanitize_filename(vendor_name)
    safe_invoice_num = _sanitize_filename(invoice_num)
    new_filename = f"{safe_vendor_name}_{safe_invoice_num}_{unique_id}{original_file_path.suffix}"
    
    logger.info(f"Archiving original file as '{new_filename}'")
    shutil.move(original_file_path, ARCHIVE_DIR / new_filename)

# ... (start_interactive_session is the same) ...
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
            handle_production_path(new_template, task['processed_image_obj'], task['original_path'])
        else:
            logger.warning(f"GUI was cancelled for '{suggested_name}'. No template was saved.")
            unique_id = str(uuid.uuid4()).split('-')[0]
            new_filename = f"UNPROCESSED_{suggested_name}_{unique_id}{task['original_path'].suffix}"
            logger.debug(f"Archiving unprocessed file as: {new_filename}")
            shutil.move(task['original_path'], ARCHIVE_DIR / new_filename)
        processed_image_path = task['processed_image_path']
        if processed_image_path and processed_image_path.exists():
            os.remove(processed_image_path)

# --- MODIFIED: The main workflow is completely inverted here ---
def main():
    logger.info("--- Starting Document Processing Run ---")
    setup_directories()
    engine.initialize_reader()

    logger.info("--- Stage 1: Pre-processing and splitting multi-page PDFs ---")
    # ... (splitting logic is the same)
    initial_files = list(INPUT_DIR.glob('*'))
    for file_path in initial_files:
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            split_files = split_multipage_pdf(file_path)
            if len(split_files) > 1:
                unique_id = str(uuid.uuid4()).split('-')[0]
                new_filename = f"SOURCE_MULTIPAGE_{file_path.stem}_{unique_id}{file_path.suffix}"
                logger.info(f"Archiving original multi-page file as: {new_filename}")
                shutil.move(file_path, ARCHIVE_DIR / new_filename)
    
    logger.info("--- Stage 2: Processing all single-page documents ---")
    files_to_process = [f for f in INPUT_DIR.iterdir() if f.is_file()]
    if not files_to_process:
        logger.info("No files found in input directory to process.")
        logger.info("--- Document Processing Run Finished ---")
        return

    unidentified_queue = []

    for file_path in files_to_process:
        logger.info(f"--- Processing file: {file_path.name} ---")
        processed_image_obj = process_document(file_path) # Now an in-memory object
        if not processed_image_obj:
            shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            continue
        
        vendor_name, template_data = identify_vendor_via_hashing(processed_image_obj)

        if vendor_name and template_data:
            # --- OPTIMIZED PATH: NO FULL OCR ---
            handle_production_path(template_data, processed_image_obj, file_path)
            # No temp files to clean up!
        else:
            # --- LEARNING PATH: ONLY NOW DO WE PERFORM FULL OCR ---
            logger.info(f"Vendor not identified. Performing full OCR for Learning Path.")
            
            # Since the GUI needs a path, we save a temp file only when needed
            temp_image_path = TEMP_DIR / f"{file_path.stem}.png"
            processed_image_obj.save(temp_image_path)

            full_ocr_results = engine.get_all_text_and_boxes(str(temp_image_path))
            
            logger.info(f"Queueing '{file_path.name}' for interactive learning.")
            unidentified_queue.append({
                "original_path": file_path,
                "processed_image_path": temp_image_path, # Path to the temp file
                "processed_image_obj": processed_image_obj, # Pass the object too
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