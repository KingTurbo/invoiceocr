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
# Note: defaultdict is no longer strictly necessary for clustering, but we keep the import clean.

from src.core.logger_config import setup_logging
from src.core.document_processor import process_document, split_multipage_pdf
from src.data import template_manager
from src.ocr import engine
from src.gui.learning_interface import start_learning_gui
from config.config import FIELDS_CONFIG, HASH_MATCH_THRESHOLD # TRIAGE_AREA_RATIO is now ignored
from src.output_handler import save_to_excel

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ARCHIVE_DIR = BASE_DIR / "archive"
TEMP_DIR = BASE_DIR / "temp_processed_images"

EXCEL_OUTPUT_FILE = OUTPUT_DIR / "invoice_records.xlsx"

# --- All helper functions remain unchanged as their logic is independent and correct ---

def setup_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

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

def handle_production_path(template, processed_image_obj: Image.Image, original_file_path):
    vendor_name = template["vendor_name"]
    logger.info(f"Production Path: Processing template for '{vendor_name}'.")
    extracted_data = {"vendor_name": vendor_name}
    for field in template['fields']:
        field_name = field.get("field_name")
        coords = field.get("coordinates")
        if field_name and coords:
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

# --- REWRITTEN: The dynamic "Seed and Propagate" interactive session ---
def start_interactive_session(unidentified_tasks: list):
    logger.info("--- Automated Processing Complete ---")
    if not unidentified_tasks:
        logger.info("No items require manual intervention.")
        return

    logger.info(f"Starting interactive learning session for {len(unidentified_tasks)} documents.")
    
    # Use a while loop to process the queue dynamically
    while unidentified_tasks:
        # Step A: Take the first document as the 'seed'
        seed_task = unidentified_tasks.pop(0)
        
        suggested_name = seed_task['original_path'].stem
        logger.info(f"\n--- Launching GUI for new vendor (Seed file: {suggested_name}) ---")
        
        # Save temp file only for GUI display, as before
        temp_image_path = TEMP_DIR / f"{suggested_name}_learning.png"
        seed_task['processed_image_obj'].save(temp_image_path)
        
        new_template = start_learning_gui(
            image_path=str(temp_image_path),
            suggested_vendor_name=suggested_name,
            fields_config=FIELDS_CONFIG,
            image_obj=seed_task['processed_image_obj']
        )
        
        os.remove(temp_image_path) # Clean up temp file

        if new_template:
            # Step B: Save the new template
            final_vendor_name = new_template.get("vendor_name")
            template_manager.save_template(final_vendor_name, new_template)
            logger.info(f"New template for '{final_vendor_name}' saved. Processing batch...")

            # Process the seed file immediately
            handle_production_path(new_template, seed_task['processed_image_obj'], seed_task['original_path'])

            # Step C: Propagate the new template to the rest of the queue
            unidentified_count = len(unidentified_tasks)
            processed_count = 0
            
            # Iterate backwards to safely remove items from the list
            for i in range(unidentified_count - 1, -1, -1):
                task_to_check = unidentified_tasks[i]
                
                # Check for propagation match using the new template's identifier area
                vendor_name_propagated, template_data_propagated = identify_vendor_via_hashing(task_to_check['processed_image_obj'])

                # A critical check: We only process if the hash matched the template we *just* created
                if template_data_propagated and template_data_propagated['vendor_name'] == final_vendor_name:
                    
                    logger.info(f"Propagated match found for: {task_to_check['original_path'].name}. Processing automatically.")
                    handle_production_path(template_data_propagated, task_to_check['processed_image_obj'], task_to_check['original_path'])
                    
                    # Remove processed task from the queue
                    unidentified_tasks.pop(i)
                    processed_count += 1
            
            logger.info(f"Propagation complete: Processed {processed_count} additional documents automatically.")

        else:
            # User canceled, archive the seed file without processing the remaining queue
            logger.warning(f"GUI was cancelled for {suggested_name}. Skipping propagation.")
            unique_id = str(uuid.uuid4()).split('-')[0]
            new_filename = f"UNPROCESSED_{suggested_name}_{unique_id}{seed_task['original_path'].suffix}"
            shutil.move(seed_task['original_path'], ARCHIVE_DIR / new_filename)

# ... (main function is mostly the same) ...
def main():
    logger.info("--- Starting Document Processing Run ---")
    setup_directories()
    engine.initialize_reader()

    logger.info("--- Stage 1: Pre-processing and splitting multi-page PDFs ---")
    initial_files = list(INPUT_DIR.glob('*'))
    for file_path in initial_files:
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            split_files = split_multipage_pdf(file_path)
            if len(split_files) > 1:
                unique_id = str(uuid.uuid4()).split('-')[0]
                new_filename = f"SOURCE_MULTIPAGE_{file_path.stem}_{unique_id}{file_path.suffix}"
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
        processed_image_obj = process_document(file_path)
        if not processed_image_obj:
            shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            continue
        
        vendor_name, template_data = identify_vendor_via_hashing(processed_image_obj)

        if vendor_name and template_data:
            handle_production_path(template_data, processed_image_obj, file_path)
        else:
            logger.info(f"Queueing '{file_path.name}' for interactive learning.")
            unidentified_queue.append({
                "original_path": file_path,
                "processed_image_obj": processed_image_obj
            })

    if unidentified_queue:
        start_interactive_session(unidentified_queue)

    logger.info("--- Document Processing Run Finished ---")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        logger.info(f"Total run time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.critical("An unhandled exception occurred during the run.", exc_info=True)