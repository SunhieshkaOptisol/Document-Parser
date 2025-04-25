import logging
# Set up logging
def setup_logger():
    """
    Sets up a logger with console output at INFO level.
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler to log to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)
    return logger
