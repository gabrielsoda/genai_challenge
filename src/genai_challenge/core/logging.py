"""
Structured logging configuration.

Provides JSON-formatted logging for production observability.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # add request_id if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging() -> None:
    """Configure structured logging for the application"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    # conf root logger
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    # set levels for specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# singleton logger instance
logger = logging.getLogger("genai-challenge")