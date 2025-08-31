import logging
import easyocr
from PIL import Image

logger = logging.getLogger(__name__)

READER = None

def initialize_reader():
    global READER
    if READER is None:
        logger.info("Initializing EasyOCR reader for CPU... This may take a moment.")
        try:
            READER = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader initialized successfully on CPU.")
        except Exception as e:
            logger.critical(f"Failed to initialize EasyOCR reader. Error: {e}")
            raise

# --- NEW, OPTIMIZED FUNCTION ---
def extract_text_from_area(image_obj: Image.Image, coordinates: tuple) -> str:
    """
    Performs a targeted OCR operation on a small, cropped area of an in-memory image.
    This is the core of the optimized production path.
    """
    initialize_reader()
    if not READER:
        logger.error("OCR Reader is not available.")
        return ""
    
    try:
        # Unpack coordinates and crop the in-memory image object
        x, y, w, h = coordinates
        box = (x, y, x + w, y + h)
        cropped_image = image_obj.crop(box)
        
        # Run OCR only on the small, cropped image
        results = READER.readtext(cropped_image, detail=0, paragraph=True)
        
        text = ' '.join(results)
        logger.debug(f"Targeted OCR at {coordinates} found text: '{text}'")
        return text
    except Exception as e:
        logger.error(f"An error occurred during targeted OCR at {coordinates}. Error: {e}")
        return ""


def get_all_text_and_boxes(image_path: str) -> list:
    """Performs a full-page OCR. Only used for the Learning Path."""
    initialize_reader()
    if not READER: return []
    try:
        results = READER.readtext(image_path)
        return results
    except Exception as e:
        logger.error(f"An error occurred during full-page OCR for {image_path}. Error: {e}")
        return []

def filter_text_in_area(ocr_results: list, target_area: tuple) -> str:
    """Filters pre-existing full OCR results. Only used by the GUI now."""
    tx, ty, tw, th = target_area
    tx_end, ty_end = tx + tw, ty + th
    found_texts = []
    for box, text, confidence in ocr_results:
        center_x = (box[0][0] + box[2][0]) / 2
        center_y = (box[0][1] + box[2][1]) / 2
        if tx <= center_x <= tx_end and ty <= center_y <= ty_end:
            found_texts.append(text)
    return ' '.join(found_texts)

def get_full_text(ocr_results: list) -> str:
    """Joins pre-existing OCR results. Only used by the GUI."""
    return '\n'.join([result[1] for result in ocr_results])