import logging
from pathlib import Path
import openpyxl

logger = logging.getLogger(__name__)

# Define the headers for our Excel sheet. This ensures consistency.
HEADERS = ["Vendor Name", "Invoice Number", "Invoice Date", "Total Amount", "Processed On"]

def save_to_excel(extracted_data: dict, filepath: Path):
    """
    Saves or appends extracted invoice data to an Excel spreadsheet.
    Creates the file and adds headers if it doesn't exist.
    """
    from datetime import datetime

    try:
        if not filepath.exists():
            # Create a new workbook and add headers
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Invoice Records"
            sheet.append(HEADERS)
            logger.info(f"Created new Excel file: {filepath.name}")
        else:
            # Load the existing workbook
            workbook = openpyxl.load_workbook(filepath)
            sheet = workbook.active

        # Prepare the row data, getting values from the dict or using a blank
        row_data = [
            extracted_data.get("vendor_name", ""),
            extracted_data.get("invoice_number", ""),
            extracted_data.get("invoice_date", ""),
            extracted_data.get("total_amount", ""),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]

        sheet.append(row_data)
        workbook.save(filepath)
        logger.info(f"Successfully saved record for '{extracted_data.get('vendor_name')}' to Excel.")

    except (IOError, PermissionError) as e:
        logger.critical(f"Could not write to Excel file at {filepath}. Check file permissions. Error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Excel export: {e}")