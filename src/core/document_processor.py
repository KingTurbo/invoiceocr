import logging
from pathlib import Path
from PIL import Image

# pdf2image and pypdf are no longer needed here, their roles are absorbed
# or replaced by the in-memory workflow in main.py.

logger = logging.getLogger(__name__)

# --- DELETED ---
# The I/O-heavy split_multipage_pdf function has been completely removed.

# --- SIMPLIFIED ---
def process_document(file_path: Path) -> Image.Image | None:
    """
    Processes a single-page IMAGE file (e.g., PNG, JPG) and returns a
    standardized, in-memory PIL Image object.
    
    NOTE: PDF processing is now handled directly in the main workflow
    for maximum efficiency and is no longer part of this function.
    """
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Input file does not exist or is not a file: {file_path}")
        return None

    file_suffix = file_path.suffix.lower()
    logger.debug(f"Processing image file: {file_path.name}")

    try:
        if file_suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            image_to_process = Image.open(file_path)
            
            # Ensure image is in a consistent mode for OCR
            logger.debug("Converting image to grayscale for better OCR accuracy.")
            grayscale_image = image_to_process.convert('L')
            
            logger.info(f"Successfully processed image file into an in-memory object.")
            return grayscale_image
        else:
            logger.warning(f"Unsupported file type '{file_suffix}' passed to process_document: {file_path.name}")
            return None

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path.name}. Error: {e}")
        return None