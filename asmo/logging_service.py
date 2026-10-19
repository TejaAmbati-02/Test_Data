"""Centralised logging infrastructure for the ASMO Data pipeline.

This module defines the project's structured-logging backbone and is imported
by every other module via ``from .logging_service import logger, log_function``.
It provides:

* :class:`JsonFormatter` — renders each log record as a single-line JSON object
  with a stable set of keys (timestamp, level, request_id, message, latency, …)
  so logs can be ingested by log-aggregation tooling.
* :class:`LoggingService` — one-time configuration of the root logger with a
  rotating JSON file handler plus a human-readable console handler.
* :func:`set_log_context` / :func:`clear_log_context` — attach a per-thread
  ``request_id`` that is automatically folded into every record.
* :func:`log_function` — a decorator that times a callable and logs its start,
  success (with latency) or failure (with error details), transparently
  handling both coroutine and regular functions.

A single shared :data:`logging_service` instance, its :data:`logger`, and a
:data:`setup_logging` entry point are exposed at module level for the rest of
the application to use.

Note: nothing in the formatter or the ``log_function`` wrappers emits log
records of its own, which would risk infinite recursion while a record is being
formatted or a wrapped call is being logged.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime
from pathlib import Path


# Per-request context: thread-local storage that lets log records be tagged with
# the request they belong to without threading a request_id through every call.
_log_context = threading.local()


def set_log_context(request_id: str) -> None:
    """Bind ``request_id`` to the current thread for subsequent log records.

    The value is read back by :meth:`JsonFormatter.format` and emitted in the
    ``request_id`` field of every record logged on this thread until it is
    overwritten or cleared with :func:`clear_log_context`.
    """
    _log_context.request_id = request_id


def clear_log_context() -> None:
    """Reset the current thread's ``request_id`` to an empty string.

    Call this once a unit of work finishes so later, unrelated records on the
    same (possibly pooled) thread are not mislabelled with a stale request id.
    """
    _log_context.request_id = ""


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects with a fixed schema.

    Every record is rendered to the same set of keys regardless of which
    ``extra`` fields were supplied, so downstream parsers see a stable shape.
    Missing attributes fall back to sensible defaults, the ``request_id`` is
    pulled from the record or the thread-local context, and the ``environment``
    defaults to the ``ENV_NAME`` environment variable (``"local"`` if unset).
    This method never logs anything itself to avoid recursion during
    formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Inner helper: return `val` unchanged when it is JSON-serialisable,
        # otherwise fall back to its str() form so a non-serialisable extra can
        # never break the whole log line.
        def safe(val):
            try:
                json.dumps(val)
                return val
            except Exception:
                return str(val)

        entry = {
            "timestamp":        self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level":            record.levelname,
            "request_id":       getattr(record, "request_id", None) or getattr(_log_context, "request_id", ""),
            "message":          record.getMessage(),
            "search_query":     getattr(record, "search_query", ""),
            "language":         getattr(record, "language", ""),
            "channel":          getattr(record, "channel", ""),
            "function_name":    getattr(record, "function_name", ""),
            "input_data":       safe(getattr(record, "input_data", "")),
            "response_payload": safe(getattr(record, "response_payload", {})),
            "error_details":    safe(getattr(record, "error_details", "")),
            "status":           getattr(record, "status", "success"),
            "latency":          getattr(record, "latency", ""),
            "environment":      getattr(record, "environment", os.environ.get("ENV_NAME", "local")),
            "logger_name":      "[ASMO Data]",
        }
        return json.dumps(entry, ensure_ascii=False)


class LoggingService:
    """Owns one-time configuration of the application's logging stack.

    Resolves (and creates) a ``logs`` directory next to the project root, holds
    the application logger, and tracks whether :meth:`setup` has already run so
    repeated calls are no-ops. Instantiated once at module import as
    :data:`logging_service`.
    """

    def __init__(self) -> None:
        """Prepare the log directory and application logger without configuring handlers.

        Creates the ``logs`` directory (parents included) if it does not exist,
        caches its path, marks the service as not-yet-configured, and obtains
        the ``"app"`` logger. Handlers and levels are only attached later by
        :meth:`setup`.
        """
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self._configured = False
        self.logger = logging.getLogger("app")

    def setup(self, level: int = logging.INFO) -> None:
        """Configure the root logger once with JSON file and console handlers.

        Idempotent: returns immediately if already configured. On the first
        call it sets the root log level, attaches a
        :class:`~logging.handlers.TimedRotatingFileHandler` that writes
        JSON-formatted records to a timestamped file under ``logs`` (rotating at
        midnight, keeping 7 backups) and a console handler with a concise
        human-readable format, quietens noisy third-party loggers to WARNING,
        and records an initialisation message.

        Args:
            level: Minimum level for the root logger and both handlers.
        """
        if self._configured:
            return

        root = logging.getLogger()
        root.setLevel(level)

        log_file = self.log_dir / f"asmo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

        fh = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        fh.setFormatter(JsonFormatter())
        fh.setLevel(level)

        root.addHandler(fh)

        # Console handler so starting the app gives immediate, human-readable
        # feedback in the terminal/IDE (the file handler alone leaves the
        # console blank, making a running app look dead).
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        ch.setLevel(level)
        root.addHandler(ch)

        for name in ("urllib3", "httpx", "httpcore", "elastic_transport", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.WARNING)

        self._configured = True
        root.info(f"Logging initialised — file: {log_file}")

    def get_logger(self, name: str) -> logging.Logger:
        """Return the named logger from the standard logging hierarchy.

        A thin wrapper over :func:`logging.getLogger` so callers can obtain a
        child logger through the service rather than importing ``logging``
        directly.

        Args:
            name: The logger name (typically a module or component name).

        Returns:
            The :class:`logging.Logger` registered under ``name``.
        """
        return logging.getLogger(name)


def log_function(func):
    """Decorator that logs the entry, timed completion, and failure of ``func``.

    The returned wrapper logs a ``"→ <name> called"`` message on entry, then
    either a success message with the measured latency or, if the call raises,
    an error message carrying the elapsed time and exception detail before
    re-raising. Both the success and error records populate the standard
    ``extra`` keys (``function_name``, ``latency``, ``status`` and, on failure,
    ``error_details``).

    The wrapped callable's behaviour, arguments and return value are untouched.
    :func:`inspect.iscoroutinefunction` is used to pick the right wrapper:
    ``async_wrapper`` ``await``\\ s a coroutine function, while ``sync_wrapper``
    calls a regular function directly; otherwise the two are identical. Neither
    wrapper logs anything beyond these three messages, so decorating a function
    that the logging layer itself depends on cannot cause recursion.

    Args:
        func: The coroutine function or regular function to wrap.

    Returns:
        A wrapper of the matching kind that preserves ``func``'s metadata via
        :func:`functools.wraps`.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = datetime.now()
        logger.info(f"→ {func.__name__} called")
        try:
            result = await func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "success",
                },
            )
            return result
        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s — {exc}",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                },
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = datetime.now()
        logger.info(f"→ {func.__name__} called")
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "success",
                },
            )
            return result
        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s — {exc}",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                },
            )
            raise

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


# Module-level singletons shared across the application:
# `logging_service` is the single LoggingService instance; `logger` is its
# "app" logger that every module imports; `setup_logging` is a convenience
# alias for the one-time setup() entry point.
logging_service = LoggingService()
logger = logging_service.logger
setup_logging = logging_service.setup
