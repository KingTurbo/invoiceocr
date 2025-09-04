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
from pdf2image import convert_from_path # <-- CRITICAL IMPORT

from src.core.logger_config import setup_logging
# The import is now simplified, as split_multipage_pdf is gone.
from src.core.document_processor import process_document
from src.data import template_manager
from src.ocr import engine
from src.gui.learning_interface import start_learning_gui
from config.config import FIELDS_CONFIG, HASH_MATCH_THRESHOLD
from src.output_handler import save_to_excel

# All setup and configuration remains the same from previous steps...
setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ARCHIVE_DIR = BASE_DIR / "archive"
TEMP_DIR = BASE_DIR / "temp_processed_images"
EXCEL_OUTPUT_FILE = OUTPUT_DIR / "invoice_records.xlsx"

TRIAGE_HASH_SIZE = 8
TRIAGE_MATCH_THRESHOLD = 5
TEMPLATE_CACHE = []

# All functions from identify_vendor_via_hashing down to handle_production_path
# remain UNCHANGED from the previous step (Strategy 2). For brevity, they
# are included here without additional comments.

def _generate_triage_hash(image_obj: Image.Image) -> imagehash.ImageHash:
    return imagehash.average_hash(image_obj, hash_size=TRIAGE_HASH_SIZE)

def load_and_cache_templates():
    global TEMPLATE_CACHE
    TEMPLATE_CACHE = []
    logger.info("--- Initializing and warming template cache ---")
    templates_dir = BASE_DIR / "config" / "templates"
    if not templates_dir.exists():
        logger.warning("Template directory not found. Cache will be empty.")
        return
    loaded_count = 0
    for template_file in templates_dir.glob("*.json"):
        template = template_manager.load_template(template_file.stem)
        if (template and "identifier_area" in template and "identifier_hash" in template and "vendor_name" in template):
            try:
                precomputed_precise_hash = imagehash.hex_to_hash(template["identifier_hash"])
                cached_item = {
                    "vendor_name": template["vendor_name"], "identifier_area": template["identifier_area"],
                    "precomputed_precise_hash": precomputed_precise_hash, "full_template_data": template,
                    "precomputed_triage_hash": None
                }
                if "triage_hash" in template and template["triage_hash"]:
                    cached_item["precomputed_triage_hash"] = imagehash.hex_to_hash(template["triage_hash"])
                else:
                    logger.warning(f"Template '{template['vendor_name']}' is missing a triage hash. It will be checked against every document.")
                TEMPLATE_CACHE.append(cached_item)
                loaded_count += 1
            except (ValueError, TypeError) as e:
                logger.error(f"Could not parse hash for template '{template_file.name}'. Skipping. Error: {e}")
        else:
            logger.warning(f"Skipping invalid or incomplete template file: {template_file.name}")
    logger.info(f"--- Template cache warmed. Loaded {loaded_count} templates. ---")

def identify_vendor_via_hashing(processed_image_obj: Image.Image) -> tuple[str | None, dict | None]:
    if not TEMPLATE_CACHE: return None, None
    logger.debug("Starting Tier 1 (Triage) Filtering...")
    incoming_triage_hash = _generate_triage_hash(processed_image_obj)
    candidates = []
    for cached_template in TEMPLATE_CACHE:
        if cached_template["precomputed_triage_hash"] is None:
            candidates.append(cached_template)
            continue
        triage_distance = incoming_triage_hash - cached_template["precomputed_triage_hash"]
        if triage_distance <= TRIAGE_MATCH_THRESHOLD:
            logger.debug(f"Candidate found: '{cached_template['vendor_name']}' with triage distance {triage_distance}.")
            candidates.append(cached_template)
    logger.info(f"Tier 1 complete. Found {len(candidates)} candidates out of {len(TEMPLATE_CACHE)} total templates.")
    if not candidates: return None, None
    logger.debug("Starting Tier 2 (Precise) Filtering on candidates...")
    for candidate_template in candidates:
        area = candidate_template["identifier_area"]
        box = (area['x'], area['y'], area['x'] + area['width'], area['y'] + area['height'])
        try:
            new_invoice_crop = processed_image_obj.crop(box)
            new_precise_hash = imagehash.phash(new_invoice_crop)
        except IndexError:
            logger.warning(f"Identifier area for '{candidate_template['vendor_name']}' is out of bounds. Skipping candidate.")
            continue
        precise_distance = new_precise_hash - candidate_template["precomputed_precise_hash"]
        logger.debug(f"Tier 2 check for '{candidate_template['vendor_name']}': precise hash distance is {precise_distance}")
        if precise_distance <= HASH_MATCH_THRESHOLD:
            vendor_name = candidate_template["vendor_name"]
            logger.info(f"MATCH CONFIRMED! Vendor identified as '{vendor_name}' with precise distance {precise_distance}.")
            return vendor_name, candidate_template["full_template_data"]
    logger.warning("Could not identify vendor. All candidates failed Tier 2 precise matching.")
    return None, None

def setup_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]',"", name)



def handle_production_path(template, processed_image_obj: Image.Image, original_file_path):
    """
    Processes extracted data, saves to Excel, and RENAMES the already-archived
    source file to its final, descriptive name.
    """
    vendor_name = template["vendor_name"]
    logger.info(f"Production Path: Processing template for '{vendor_name}'.")
    extracted_data = {"vendor_name": vendor_name}
    for field in template['fields']:
        field_name = field.get("field_name")
        coords = field.get("coordinates")
        if field_name and coords:
            text = engine.extract_text_from_area(processed_image_obj, tuple(coords.values()))
            extracted_data[field_name] = text.strip()
    logger.info(f"Final extracted data for {original_file_path.name}: {extracted_data}")
    save_to_excel(extracted_data, EXCEL_OUTPUT_FILE)
    
    # --- FIX FOR STRATEGY 3 WORKFLOW ---
    # The original file was already moved to ARCHIVE_DIR during ingestion.
    # Our job now is to RENAME it from its original name to the new, structured name.
    
    # 1. Calculate the final, desired filename.
    unique_id = str(uuid.uuid4()).split('-')[0]
    invoice_num = extracted_data.get("invoice_number", "UNKNOWN_INV")
    safe_vendor_name = _sanitize_filename(vendor_name)
    safe_invoice_num = _sanitize_filename(invoice_num)
    new_filename = f"{safe_vendor_name}_{safe_invoice_num}_{unique_id}{original_file_path.suffix}"

    # 2. Define the source and destination paths WITHIN the archive directory.
    archived_original_path = ARCHIVE_DIR / original_file_path.name
    final_archive_path = ARCHIVE_DIR / new_filename

    # 3. Perform a safe rename operation.
    try:
        if archived_original_path.exists():
            logger.info(f"Renaming archived file to '{new_filename}'")
            os.rename(archived_original_path, final_archive_path)
        else:
            # This case handles multi-page PDFs where only the first page is identified.
            # The original file might have a different name if it was part of a split.
            # We log a warning but do not treat it as a critical error.
            logger.warning(f"Could not find {archived_original_path} to rename. File might have already been renamed by another page's processing.")

    except OSError as e:
        logger.error(f"Failed to rename archived file {archived_original_path}. Error: {e}")
    # --- END OF FIX ---


# The start_interactive_session function also remains UNCHANGED.
def start_interactive_session(unidentified_tasks: list):
    logger.info("--- Automated Processing Complete ---")
    if not unidentified_tasks:
        logger.info("No items require manual intervention.")
        return
    logger.info(f"Starting interactive learning session for {len(unidentified_tasks)} documents.")
    while unidentified_tasks:
        seed_task = unidentified_tasks.pop(0)
        suggested_name = seed_task['original_path'].stem
        logger.info(f"\n--- Launching GUI for new vendor (Seed file: {suggested_name}) ---")
        temp_image_path = TEMP_DIR / f"{suggested_name}_learning.png"
        seed_task['processed_image_obj'].save(temp_image_path)
        new_template = start_learning_gui(image_path=str(temp_image_path), suggested_vendor_name=suggested_name, fields_config=FIELDS_CONFIG, image_obj=seed_task['processed_image_obj'])
        os.remove(temp_image_path)
        if new_template:
            final_vendor_name = new_template.get("vendor_name")
            seed_image = seed_task['processed_image_obj']
            triage_hash = _generate_triage_hash(seed_image)
            new_template['triage_hash'] = str(triage_hash)
            logger.info(f"Generated and added triage hash '{triage_hash}' to new template '{final_vendor_name}'.")
            template_manager.save_template(final_vendor_name, new_template)
            logger.info(f"New template for '{final_vendor_name}' saved. Processing batch...")
            try:
                precomputed_precise_hash = imagehash.hex_to_hash(new_template["identifier_hash"])
                precomputed_triage_hash = imagehash.hex_to_hash(new_template["triage_hash"])
                cached_item = {
                    "vendor_name": new_template["vendor_name"], "identifier_area": new_template["identifier_area"],
                    "precomputed_precise_hash": precomputed_precise_hash, "precomputed_triage_hash": precomputed_triage_hash,
                    "full_template_data": new_template
                }
                TEMPLATE_CACHE.append(cached_item)
                logger.info(f"Live cache updated. Total templates: {len(TEMPLATE_CACHE)}")
            except (ValueError, TypeError) as e:
                logger.error(f"Could not add newly created template to cache. Error: {e}")
            handle_production_path(new_template, seed_task['processed_image_obj'], seed_task['original_path'])
            unidentified_count = len(unidentified_tasks)
            processed_count = 0
            for i in range(unidentified_count - 1, -1, -1):
                task_to_check = unidentified_tasks[i]
                vendor_name_propagated, template_data_propagated = identify_vendor_via_hashing(task_to_check['processed_image_obj'])
                if template_data_propagated and template_data_propagated['vendor_name'] == final_vendor_name:
                    logger.info(f"Propagated match found for: {task_to_check['original_path'].name}. Processing automatically.")
                    handle_production_path(template_data_propagated, task_to_check['processed_image_obj'], task_to_check['original_path'])
                    unidentified_tasks.pop(i)
                    processed_count += 1
            logger.info(f"Propagation complete: Processed {processed_count} additional documents automatically.")
        else:
            logger.warning(f"GUI was cancelled for {suggested_name}. Skipping propagation.")
            unique_id = str(uuid.uuid4()).split('-')[0]
            new_filename = f"UNPROCESSED_{suggested_name}_{unique_id}{seed_task['original_path'].suffix}"
            shutil.move(seed_task['original_path'], ARCHIVE_DIR / new_filename)

# ==============================================================================
# === STRATEGY 3 OPTIMIZATION: IN-MEMORY WORKFLOW ==============================
# ==============================================================================
def main():
    logger.info("--- Starting Document Processing Run ---")
    setup_directories()
    engine.initialize_reader()
    load_and_cache_templates()

    # --- STAGE 1: IN-MEMORY INGESTION ---
    # Build a queue of in-memory image objects, completely avoiding intermediate files.
    logger.info("--- Stage 1: In-Memory Document Ingestion ---")
    
    # The processing_queue holds tuples of: (PIL.Image, original_Path_object, display_name)
    processing_queue = []
    
    files_to_ingest = list(INPUT_DIR.glob('*'))
    if not files_to_ingest:
        logger.info("No files found in input directory to process.")
        logger.info("--- Document Processing Run Finished ---")
        return

    for file_path in files_to_ingest:
        if not file_path.is_file():
            continue

        file_suffix = file_path.suffix.lower()
        
        try:
            if file_suffix == '.pdf':
                # Use pdf2image to directly convert all pages to in-memory, grayscale PIL objects.
                # This is vastly more efficient than writing and re-reading temp files.
                logger.info(f"Ingesting PDF: {file_path.name}")
                images = convert_from_path(file_path, dpi=300, grayscale=True)
                for i, image_obj in enumerate(images):
                    display_name = f"{file_path.stem}_page_{i+1}"
                    processing_queue.append((image_obj, file_path, display_name))
                # Archive the original PDF immediately after successful conversion
                # Note: For multi-page, all pages are handled before this move.
                shutil.move(file_path, ARCHIVE_DIR / file_path.name)

            elif file_suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                logger.info(f"Ingesting Image: {file_path.name}")
                # Use the simplified process_document for image files
                image_obj = process_document(file_path)
                if image_obj:
                    processing_queue.append((image_obj, file_path, file_path.name))
                    shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            else:
                logger.warning(f"Unsupported file type '{file_suffix}'. Archiving as-is.")
                shutil.move(file_path, ARCHIVE_DIR / file_path.name)
        
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path.name}. Archiving. Error: {e}")
            # Ensure failed files are moved out of the input queue
            if file_path.exists():
                shutil.move(file_path, ARCHIVE_DIR / f"FAILED_{file_path.name}")

    # --- STAGE 2: PROCESS THE IN-MEMORY QUEUE ---
    logger.info(f"--- Stage 2: Processing {len(processing_queue)} pages from in-memory queue ---")
    unidentified_queue = []

    for processed_image_obj, original_path, display_name in processing_queue:
        logger.info(f"--- Processing item: {display_name} (from {original_path.name}) ---")
        
        # The image is already processed and in memory, no need for process_document() call.
        vendor_name, template_data = identify_vendor_via_hashing(processed_image_obj)

        if vendor_name and template_data:
            # We must handle archiving carefully. Since the original file is already
            # moved, we just pass its original path for logging and file naming.
            handle_production_path(template_data, processed_image_obj, original_path)
        else:
            logger.info(f"Queueing '{display_name}' for interactive learning.")
            unidentified_queue.append({
                "original_path": original_path, # Keep track of the source file
                "processed_image_obj": processed_image_obj
            })
    
    # The interactive session now works on the unidentified items from memory.
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