import logging
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pypdf

logger = logging.getLogger(__name__)

# The TEMP_DIR is no longer needed by process_document, but we leave it for now
# as other parts of the code might still use it implicitly.

def split_multipage_pdf(file_path: Path) -> list[Path]:
    # ... (this function remains unchanged) ...
    new_file_paths = []
    try:
        reader = pypdf.PdfReader(file_path)
        num_pages = len(reader.pages)
        if num_pages <= 1:
            return [file_path]
        logger.info(f"Multi-page PDF '{file_path.name}' detected with {num_pages} pages. Splitting...")
        for i, page in enumerate(reader.pages):
            writer = pypdf.PdfWriter()
            writer.add_page(page)
            new_filename = f"{file_path.stem}_page_{i+1}{file_path.suffix}"
            new_filepath = file_path.parent / new_filename
            with open(new_filepath, "wb") as f:
                writer.write(f)
            new_file_paths.append(new_filepath)
            logger.info(f"Created split file: {new_filename}")
        return new_file_paths
    except Exception as e:
        logger.error(f"Failed to split PDF {file_path.name}. Error: {e}")
        return [file_path]

# --- MODIFIED: This function now returns an in-memory Image object ---
def process_document(file_path: Path) -> Image.Image | None:
    """
    Processes an input file (PDF or image) and returns a standardized,
    in-memory PIL Image object, ready for hashing or OCR.
    """
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Input file does not exist or is not a file: {file_path}")
        return None

    image_to_process = None
    file_suffix = file_path.suffix.lower()
    logger.info(f"Starting processing for document: {file_path.name}")

    try:
        if file_suffix == '.pdf':
            logger.debug("PDF file detected. Converting first page to image.")
            images = convert_from_path(file_path, dpi=300)
            if images:
                image_to_process = images[0]
        elif file_suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            logger.debug("Image file detected.")
            image_to_process = Image.open(file_path)
        else:
            logger.warning(f"Unsupported file type '{file_suffix}' for file: {file_path.name}")
            return None

        if image_to_process:
            logger.debug("Converting image to grayscale for better OCR accuracy.")
            grayscale_image = image_to_process.convert('L')
            logger.info(f"Successfully processed document into an in-memory object.")
            return grayscale_image # <-- Return the object, not a path

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path.name}. Error: {e}")
    
    return None