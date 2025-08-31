import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    
    file_handler = RotatingFileHandler(
        'app.log', maxBytes=5*1024*1024, backupCount=3 #5 mb
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)