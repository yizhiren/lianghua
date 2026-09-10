from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings


EXTRA_FIELDS = (
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "dialogue_id",
    "attempt",
    "max_attempts",
    "error_type",
    "error_detail",
    "model",
    "question_count",
    "rule_count",
    "security_id",
    "securities",
    "quotes",
    "processed",
    "bars",
)


request_id_context: ContextVar[str | None] = ContextVar("lianghua_request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if field == "request_id" and value is None:
                value = request_id_context.get()
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)[:2000]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("lianghua")
    if getattr(logger, "_lianghua_configured", False):
        return logger
    log_path = path or settings.backend_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger._lianghua_configured = True  # type: ignore[attr-defined]
    return logger


logger = configure_logging()
