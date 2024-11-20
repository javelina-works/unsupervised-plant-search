import logging

# This is all about re-using this great logger. 
# And thwarting the debug output of matplotlib.
# But mostly this great logger.

def setup_logger(name="unsupervised_logger", log_level=logging.INFO, log_file=None):
    """
    Set up a logger for the script with optional file logging.

    Args:
    - name (str): Name of the logger.
    - log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    - log_file (str): Optional file path to log to a file.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
