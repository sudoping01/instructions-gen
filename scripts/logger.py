
from logging.handlers import RotatingFileHandler
import logging

def setup_logger():
    logger = logging.getLogger('InstructionDatasetCreation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = RotatingFileHandler(
        'InstructionDatasetCreation.log', 
        maxBytes=100*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )

    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
