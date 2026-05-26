import sys
import logging
from datetime import datetime

try:
    from loguru import logger as _logger
except ImportError:
    _logger = None

from finacumen.ft.config import PROJECT_ROOT


_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if _logger is not None:
        _logger.remove()
        _logger.add(sys.stderr, level=print_level)
        _logger.add(logs_dir / f"{log_name}.log", level=logfile_level)
        return _logger

    # Fallback to standard logging when loguru is unavailable.
    logger = logging.getLogger("openmanus")
    logger.setLevel(getattr(logging, print_level.upper(), logging.INFO))
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(getattr(logging, print_level.upper(), logging.INFO))
    file_handler = logging.FileHandler(logs_dir / f"{log_name}.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, logfile_level.upper(), logging.DEBUG))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = define_log_level()


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
