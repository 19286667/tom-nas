"""
ToM-NAS Logging Configuration

Professional logging setup supporting:
- Local console output (development)
- File logging (all environments)
- Google Cloud Logging (production)
- Structured JSON logging (production)
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from functools import lru_cache

# Attempt to import Google Cloud Logging (optional)
try:
    from google.cloud import logging as cloud_logging
    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON for production environments.
    Compatible with Google Cloud Logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'severity': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add custom fields if present
        if hasattr(record, 'custom_fields'):
            log_entry.update(record.custom_fields)

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development console output.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        return super().format(record)


def setup_logging(
    level: str = 'INFO',
    enable_cloud_logging: bool = False,
    log_file: Optional[str] = None,
    structured: bool = False,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_cloud_logging: Enable Google Cloud Logging integration
        log_file: Optional file path for file logging
        structured: Use structured JSON format (for production)

    Example:
        >>> setup_logging(level='DEBUG', log_file='logs/tom_nas.log')
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        # Use colored formatter for development
        fmt = '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Google Cloud Logging (if enabled and available)
    if enable_cloud_logging and CLOUD_LOGGING_AVAILABLE:
        try:
            client = cloud_logging.Client()
            cloud_handler = cloud_logging.handlers.CloudLoggingHandler(client)
            cloud_handler.setLevel(getattr(logging, level.upper()))
            root_logger.addHandler(cloud_handler)
            root_logger.info('Google Cloud Logging enabled')
        except Exception as e:
            root_logger.warning(f'Failed to initialize Google Cloud Logging: {e}')

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ['urllib3', 'google', 'werkzeug', 'streamlit']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    root_logger.info(f'Logging initialized at {level} level')


@lru_cache(maxsize=100)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Uses caching for performance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Processing started')
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that allows adding context to log messages.

    Example:
        >>> logger = get_context_logger(__name__, agent_id=5, experiment='baseline')
        >>> logger.info('Agent initialized')  # Includes agent_id and experiment in output
    """

    def process(self, msg, kwargs):
        # Add extra fields to the log record
        extra = kwargs.get('extra', {})
        extra['custom_fields'] = self.extra
        kwargs['extra'] = extra
        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context fields.

    Args:
        name: Logger name
        **context: Key-value pairs to include in all log messages

    Returns:
        LoggerAdapter: Logger that includes context in all messages

    Example:
        >>> logger = get_context_logger(__name__, experiment='coevolution', run_id=42)
        >>> logger.info('Generation complete')
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
