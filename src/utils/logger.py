import logging

logger = logging.getLogger(__name__)
# Configure the logger settings
logger.setLevel(logging.INFO)

# Create a file handler and set the formatter
file_handler = logging.FileHandler("log_info.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

