import contextvars
import logging
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from epistemic_uq.redaction import Redactor
from epistemic_uq.settings import settings


_trace_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_redactor = Redactor(settings.redact_pattern_set)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_context.get()
        return True


class RedactingJsonFormatter(JsonFormatter):
    def process_log_record(self, log_record: dict[str, Any]) -> dict[str, Any]:
        return _redactor.redact_mapping(log_record)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(RedactingJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())


def bind_trace(value: str):
    return _trace_context.set(value)


def reset_trace(token) -> None:
    _trace_context.reset(token)
