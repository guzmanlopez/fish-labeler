"""Application logging configuration."""

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name="sam3_labeler"):
    """Return a configured console logger with one shared handler per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(RichHandler(console=console, show_path=False, rich_tracebacks=True))
        logger.propagate = False
    return logger


logger = get_logger()
