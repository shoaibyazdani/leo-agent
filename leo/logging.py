"""Structured JSON logging with credential redaction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from .config import Settings


class RedactingFilter(logging.Filter):
    def __init__(self, patterns: Iterable[str]) -> None:
        super().__init__()
        self._patterns = [re.compile(p) for p in patterns]

    def _redact_arg(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            # ``logger.info("%(token)s", {...})`` stores the mapping as a
            # single arg; its values must be redacted too.
            return {
                key: self.redact(val) if isinstance(val, str) else val
                for key, val in value.items()
            }
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = self._redact_arg(record.args)
            else:
                record.args = tuple(self._redact_arg(a) for a in record.args)
        return True

    def redact(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub("[REDACTED]", text)
        return text


class JsonFormatter(logging.Formatter):
    """Render log records as JSON, redacting secrets in every emitted field.

    A ``logging.Filter`` only sees ``msg``/``args``. Tracebacks and ``extra``
    fields are turned into text later, at format time, so redaction has to run
    here as well or credentials leak straight into the sink.
    """

    _EXTRA_KEYS = (
        "task_id",
        "agent_id",
        "tool",
        "event",
        "request_id",
        "correlation_id",
    )

    def __init__(self, patterns: Iterable[str] = (), *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._redactor = RedactingFilter(patterns).redact

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redactor(record.getMessage()),
        }
        for key in self._EXTRA_KEYS:
            if hasattr(record, key):
                value = getattr(record, key)
                payload[key] = self._redactor(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = self._redactor(
                self.formatException(record.exc_info)
            )
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings) -> logging.Logger:
    settings.ensure_dirs()
    root = logging.getLogger("leo")
    root.setLevel(logging.DEBUG if settings.environment == "dev" else logging.INFO)
    # Drop handlers properly instead of just clearing the list: repeated
    # configuration leaks file descriptors (and appenders) under long runs.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    flt = RedactingFilter(settings.security.redact_patterns)
    formatter = JsonFormatter(settings.security.redact_patterns)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(flt)
    root.addHandler(console)
    file_handler = logging.FileHandler(
        settings.resolve(settings.paths.logs) / "leo.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(flt)
    root.addHandler(file_handler)
    root.propagate = False
    return root


def get_logger(name: str = "leo") -> logging.Logger:
    return logging.getLogger(f"leo.{name}" if not name.startswith("leo") else name)
