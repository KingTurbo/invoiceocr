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
from pdf2image import convert_from_path

Image.MAX_IMAGE_PIXELS = None

import base64
import numpy as np

from src.core.logger_config import setup_logging
from src.core.document_processor import process_document
from src.core.image_preprocessor import correct_skew
from src.core.feature_matcher import identify_vendor_via_features
from src.data import template_manager
from src.ocr import engine
from src.gui.learning_interface import start_learning_gui
from config.config import FIELDS_CONFIG, HASH_MATCH_THRESHOLD
from src.output_handler import save_to_excel

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
        
        # --- CORRECTED: Load keypoint coordinates into cache ---
        if (template and "anchor_features" in template and "vendor_name" in template):
            try:
                features = template["anchor_features"]
                des_bytes = base64.b64decode(features["descriptors_b64"])
                descriptors = np.frombuffer(des_bytes, dtype=np.uint8).reshape(-1, 32)
                
                # Load the keypoint coordinates
                keypoints_pts = features["keypoints_pts"]

                cached_item = {
                    "vendor_name": template["vendor_name"],
                    "template_type": "orb_features", 
                    "anchor_descriptors": descriptors,
                    "anchor_keypoints_pts": keypoints_pts, # Add coordinates to cache
                    "anchor_source_shape": features["source_shape"],
                    "identifier_area": template["identifier_area"],
                    "full_template_data": template,
                }
                TEMPLATE_CACHE.append(cached_item)
                loaded_count += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Could not parse new ORB template '{template_file.name}'. Skipping. Error: {e}", exc_info=True)
        # --- END CORRECTION ---
        
        elif (template and "identifier_hash" in template and "vendor_name" in template):
            try:
                precomputed_precise_hash = imagehash.hex_to_hash(template["identifier_hash"])
                cached_item = {
                    "vendor_name": template["vendor_name"],
                    "template_type": "phash", 
                    "identifier_area": template["identifier_area"],
                    "precomputed_precise_hash": precomputed_precise_hash,
                    "full_template_data": template,
                    "precomputed_triage_hash": None
                }
                if "triage_hash" in template and template["triage_hash"]:
                    cached_item["precomputed_triage_hash"] = imagehash.hex_to_hash(template["triage_hash"])
                TEMPLATE_CACHE.append(cached_item)
                loaded_count += 1
            except (ValueError, TypeError) as e:
                logger.error(f"Could not parse legacy hash for template '{template_file.name}'. Skipping. Error: {e}")
        else:
            logger.warning(f"Skipping invalid or incomplete template file: {template_file.name}")
            
    logger.info(f"--- Template cache warmed. Loaded {loaded_count} templates. ---")

def identify_vendor_via_hashing(processed_image_obj: Image.Image, legacy_templates: list) -> tuple[str | None, dict | None]:
    if not legacy_templates: return None, None
    logger.debug("Starting Legacy Tier 1 (Triage Hashing)...")
    incoming_triage_hash = _generate_triage_hash(processed_image_obj)
    candidates = []
    for cached_template in legacy_templates:
        if cached_template.get("precomputed_triage_hash") is None:
            candidates.append(cached_template)
            continue
        triage_distance = incoming_triage_hash - cached_template["precomputed_triage_hash"]
        if triage_distance <= TRIAGE_MATCH_THRESHOLD: candidates.append(cached_template)
    if not candidates: return None, None
    logger.debug("Starting Legacy Tier 2 (Precise) Filtering on candidates...")
    for candidate_template in candidates:
        area = candidate_template["identifier_area"]
        box = (area['x'], area['y'], area['x'] + area['width'], area['y'] + area['height'])
        try:
            new_precise_hash = imagehash.phash(processed_image_obj.crop(box))
            precise_distance = new_precise_hash - candidate_template["precomputed_precise_hash"]
            if precise_distance <= HASH_MATCH_THRESHOLD:
                vendor_name = candidate_template["vendor_name"]
                logger.info(f"LEGACY MATCH CONFIRMED! Vendor identified as '{vendor_name}'.")
                return vendor_name, candidate_template["full_template_data"]
        except IndexError: continue
    return None, None

def identify_vendor(processed_image_obj: Image.Image) -> tuple[str | None, dict | None, np.ndarray | None]:
    if not TEMPLATE_CACHE: return None, None, None
    orb_templates = [t for t in TEMPLATE_CACHE if t.get("template_type") == "orb_features"]
    vendor_name, template_data, anchor_bbox = identify_vendor_via_features(processed_image_obj, orb_templates)
    if vendor_name:
        return vendor_name, template_data, anchor_bbox
    legacy_templates = [t for t in TEMPLATE_CACHE if t.get("template_type") == "phash"]
    vendor_name, template_data = identify_vendor_via_hashing(processed_image_obj, legacy_templates)
    if vendor_name:
        return vendor_name, template_data, None
    logger.warning("Could not identify vendor. All identification methods failed.")
    return None, None, None

def setup_directories():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]',"", name)

def handle_production_path(template: dict, processed_image_obj: Image.Image, original_file_path: Path, anchor_bbox: np.ndarray | None = None):
    vendor_name = template["vendor_name"]
    logger.info(f"Production Path: Processing template for '{vendor_name}'.")
    if anchor_bbox is not None:
        logger.debug(f"Received dynamic anchor bounding box for '{vendor_name}'.")
    else:
        logger.debug(f"Processing '{vendor_name}' using legacy static coordinates.")
    extracted_data = {"vendor_name": vendor_name}
    for field in template['fields']:
        field_name = field.get("field_name")
        coords = field.get("coordinates")
        if field_name and coords:
            text = engine.extract_text_from_area(processed_image_obj, tuple(coords.values()))
            extracted_data[field_name] = text.strip()
    logger.info(f"Final extracted data for {original_file_path.name}: {extracted_data}")
    save_to_excel(extracted_data, EXCEL_OUTPUT_FILE)
    unique_id = str(uuid.uuid4()).split('-')[0]
    invoice_num = extracted_data.get("invoice_number", "UNKNOWN_INV")
    safe_vendor_name = _sanitize_filename(vendor_name)
    safe_invoice_num = _sanitize_filename(invoice_num)
    new_filename = f"{safe_vendor_name}_{safe_invoice_num}_{unique_id}{original_file_path.suffix}"
    archived_original_path = ARCHIVE_DIR / original_file_path.name
    final_archive_path = ARCHIVE_DIR / new_filename
    try:
        if archived_original_path.exists():
            os.rename(archived_original_path, final_archive_path)
        else:
            logger.warning(f"Could not find {archived_original_path} to rename.")
    except OSError as e:
        logger.error(f"Failed to rename archived file {archived_original_path}. Error: {e}")

def start_interactive_session(unidentified_tasks: list):
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
            template_manager.save_template(final_vendor_name, new_template)
            logger.info(f"New template for '{final_vendor_name}' saved. Reloading cache...")
            load_and_cache_templates()
            handle_production_path(new_template, seed_task['processed_image_obj'], seed_task['original_path'])
            logger.info("--- Starting Propagation: Applying new template to remaining queue... ---")
            processed_count = 0
            for i in range(len(unidentified_tasks) - 1, -1, -1):
                task_to_check = unidentified_tasks[i]
                vendor_name_propagated, template_data_propagated, anchor_bbox_propagated = identify_vendor(task_to_check['processed_image_obj'])
                if template_data_propagated and template_data_propagated['vendor_name'] == final_vendor_name:
                    logger.info(f"PROPAGATED MATCH FOUND for: {task_to_check['original_path'].name}.")
                    handle_production_path(template_data_propagated, task_to_check['processed_image_obj'], task_to_check['original_path'], anchor_bbox_propagated)
                    unidentified_tasks.pop(i)
                    processed_count += 1
            logger.info(f"--- Propagation Complete: Processed {processed_count} additional documents automatically. ---")
        else:
            logger.warning(f"GUI was cancelled for {suggested_name}. Skipping propagation.")
            unique_id = str(uuid.uuid4()).split('-')[0]
            new_filename = f"UNPROCESSED_{suggested_name}_{unique_id}{seed_task['original_path'].suffix}"
            shutil.move(ARCHIVE_DIR / seed_task['original_path'].name, ARCHIVE_DIR / new_filename)

def main():
    logger.info("--- Starting Document Processing Run ---")
    setup_directories()
    engine.initialize_reader()
    load_and_cache_templates()
    logger.info("--- Stage 1: In-Memory Document Ingestion ---")
    processing_queue = []
    files_to_ingest = list(INPUT_DIR.glob('*'))
    if not files_to_ingest:
        logger.info("No files found in input directory to process.")
        return
    for file_path in files_to_ingest:
        if not file_path.is_file(): continue
        file_suffix = file_path.suffix.lower()
        try:
            if file_suffix == '.pdf':
                images = convert_from_path(file_path, dpi=250, grayscale=True)
                for i, image_obj in enumerate(images):
                    corrected_image = correct_skew(image_obj)
                    processing_queue.append((corrected_image, file_path, f"{file_path.stem}_page_{i+1}"))
                shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            elif file_suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                image_obj = process_document(file_path)
                if image_obj:
                    corrected_image = correct_skew(image_obj)
                    processing_queue.append((corrected_image, file_path, file_path.name))
                    shutil.move(file_path, ARCHIVE_DIR / file_path.name)
            else:
                shutil.move(file_path, ARCHIVE_DIR / file_path.name)
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path.name}. Archiving. Error: {e}")
            if file_path.exists():
                shutil.move(file_path, ARCHIVE_DIR / f"FAILED_{file_path.name}")
    logger.info(f"--- Stage 2: Processing {len(processing_queue)} pages from in-memory queue ---")
    unidentified_queue = []
    for processed_image_obj, original_path, display_name in processing_queue:
        logger.info(f"--- Processing item: {display_name} (from {original_path.name}) ---")
        vendor_name, template_data, anchor_bbox = identify_vendor(processed_image_obj)
        if vendor_name and template_data:
            handle_production_path(template_data, processed_image_obj, original_path, anchor_bbox)
        else:
            logger.info(f"Queueing '{display_name}' for interactive learning.")
            unidentified_queue.append({"original_path": original_path, "processed_image_obj": processed_image_obj})
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