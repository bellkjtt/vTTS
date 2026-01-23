import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Simple python logger without lightning dependency."""
    return logging.getLogger(name)
