import logging
import sys

# Get the root logger for the library
logger = logging.getLogger('cosmo_analysis')
logger.setLevel(logging.INFO) # Default level

# Prevent the logger from propagating to the root logger
logger.propagate = False

# Default handler to avoid "No handler found" warnings
if not logger.handlers:
    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(default_handler)

def set_log_level(level):
    """Sets the logging level for the library."""
    logger.setLevel(level)

def add_file_handler(log_file, level=logging.INFO):
    """Adds a file handler to the logger."""
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
