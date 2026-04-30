import logging
from core.logger import get_logger

def test_get_logger():
    """Test logger initialization and singleton handler."""
    logger1 = get_logger("test_logger")
    assert isinstance(logger1, logging.Logger)
    assert len(logger1.handlers) == 1
    logger2 = get_logger("test_logger")
    assert len(logger2.handlers) == 1
    assert logger1 is logger2
