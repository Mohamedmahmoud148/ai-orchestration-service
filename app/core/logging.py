"""
app/core/logging.py

Structured logging with correlation ID propagation.

Every log line in production emits a JSON object with:
  ts, level, logger, msg, correlation_id, user_id (when set),
  duration_ms (when set), intent, module.

In development the format stays human-readable for easier debugging.
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from .config import settings

# ── Context variables — carried per-request via Python's ContextVar ────────────
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_request_user_id: ContextVar[str] = ContextVar("request_user_id", default="-")
_request_start: ContextVar[float] = ContextVar("request_start", default=0.0)


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_request_user_id(uid: str) -> None:
    _request_user_id.set(uid)


def get_elapsed_ms() -> int:
    """Return ms elapsed since request started (0 if not set)."""
    start = _request_start.get()
    return int((time.monotonic() - start) * 1000) if start else 0


def mark_request_start() -> None:
    _request_start.set(time.monotonic())


# ── JSON formatter (production) ────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """
    Emits one JSON line per log record.

    Fields always present:
      ts           ISO-8601 timestamp
      level        DEBUG / INFO / WARNING / ERROR / CRITICAL
      logger       logger name (e.g. "ai_service")
      msg          the formatted message string
      correlation  X-Request-ID for the current request
      user_id      authenticated user (when available)
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":          self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":       record.levelname,
            "logger":      record.name,
            "msg":         record.getMessage(),
            "correlation": _correlation_id.get(),
            "user_id":     _request_user_id.get(),
        }
        # Attach exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Propagate any extra fields the caller attached via `extra={}`
        for key in ("intent", "module", "duration_ms", "stage", "tool"):
            val = record.__dict__.get(key)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


# ── Human-readable formatter (development) ────────────────────────────────────

_DEV_FORMAT = (
    "%(asctime)s [%(levelname)-8s] [%(correlation)s] %(name)s — %(message)s"
)


class _DevFormatter(logging.Formatter):
    """Injects correlation_id into every dev log record."""

    def format(self, record: logging.LogRecord) -> str:
        record.correlation = _correlation_id.get()  # type: ignore[attr-defined]
        return super().format(record)


# ── Public setup ───────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """
    Configure root + ai_service logger.

    Production → JSON lines to stdout (machine-parseable, Railway-friendly).
    Development → human-readable with correlation ID prefix.
    """
    is_prod = settings.ENVIRONMENT == "production"
    level   = logging.INFO if is_prod else logging.DEBUG

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JSONFormatter() if is_prod else _DevFormatter(fmt=_DEV_FORMAT, datefmt="%H:%M:%S")
    )

    # Configure root logger minimally (avoids double-printing)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers.clear()
        root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


logger = logging.getLogger("ai_service")
