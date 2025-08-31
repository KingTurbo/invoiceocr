import logging
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pypdf

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = BASE_DIR / "temp_processed_images"

def split_multipage_pdf(file_path: Path) -> list[Path]:
    new_file_paths = []
    try:
        reader = pypdf.PdfReader(file_path)
        num_pages = len(reader.pages)

        if num_pages <= 1:
            logger.debug(f"'{file_path.name}' is a single-page PDF. No splitting needed.")
            return [file_path]

        logger.info(f"Multi-page PDF '{file_path.name}' detected with {num_pages} pages. Splitting...")
        
        for i, page in enumerate(reader.pages):
            writer = pypdf.PdfWriter()
            writer.add_page(page)
            
            # Construct a new, unique filename for each page
            new_filename = f"{file_path.stem}_page_{i+1}{file_path.suffix}"
            new_filepath = file_path.parent / new_filename
            
            with open(new_filepath, "wb") as f:
                writer.write(f)
            
            new_file_paths.append(new_filepath)
            logger.info(f"Created split file: {new_filename}")

        return new_file_paths

    except Exception as e:
        logger.error(f"Failed to split PDF {file_path.name}. Error: {e}")
        # Return the original path so it can be archived without crashing
        return [file_path]


def process_document(file_path: Path) -> Path | None:
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"Input file does not exist or is not a file: {file_path}")
        return None
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.critical(f"Could not create temporary processing directory at {TEMP_DIR}. Error: {e}")
        return None


    output_image_path = TEMP_DIR / f"{file_path.stem}.png"
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
            logger.debug(f"Saving processed image to {output_image_path}")
            grayscale_image.save(output_image_path, 'PNG')
            logger.info(f"Successfully processed document and saved as {output_image_path.name}")
            
            return output_image_path

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path.name}. Error: {e}")
    return None