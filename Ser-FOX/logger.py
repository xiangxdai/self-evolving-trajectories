import logging


def get_logger(filename, verbosity=1, name=None, file_mode='a', console=True):
    """
    Create and configure a logger with file and optional console output.

    Args:
        filename (str): Path to the log file.
        verbosity (int): Logging level (0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL).
        name (str, optional): Logger name. If None, use root logger.
        file_mode (str): File mode for FileHandler ('w' for overwrite, 'a' for append).
        console (bool): Whether to add a StreamHandler for console output.

    Returns:
        logging.Logger: Configured logger object.
    """
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL,
    }
    level = level_dict.get(verbosity, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    file_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(filename, mode=file_mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        console_formatter = logging.Formatter("%(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
