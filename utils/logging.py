import logging
import os
import sys
from typing import Dict, Iterable

try:
    from rich.logging import RichHandler  # type: ignore
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


_STD_KEYS: set[str] = set(
    [
        "name",
        "msg",
        "message",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "stacklevel",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    ]
)


class ContextFormatter(logging.Formatter):
    """Formatter that appends extra context as key=value pairs.

    Works with both standard handlers and RichHandler.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        # Let handler decide basic formatting; we append context at the end
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for k, v in record.__dict__.items():
            if k in _STD_KEYS:
                continue
            if k.startswith("_"):
                continue
            # Skip internal LogRecord attrs Rich adds sometimes
            if k in ("markup",):
                continue
            try:
                val = str(v)
            except Exception:
                val = repr(v)
            if val == "" or val == "None":
                continue
            extras.append(f"{k}={val}")
        if extras:
            return f"{base} | " + " ".join(extras)
        return base


def _env_level() -> int:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


def _want_rich() -> bool:
    val = os.getenv("LOG_PRETTY", "1").lower()
    return val not in ("0", "false", "no") and _HAS_RICH and sys.stderr.isatty()


def setup_logging() -> None:
    """Configure root logger for pretty, contextual console logs.

    Honors env vars:
    - LOG_LEVEL (default INFO)
    - LOG_PRETTY (default 1; if 0, use plain formatter)
    """

    level = _env_level()

    # Clear existing handlers to avoid duplicates on reloads
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
            except Exception:
                pass

    if _want_rich():
        handler = RichHandler(rich_tracebacks=True, tracebacks_width=100, show_time=True, show_level=True, show_path=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    handler.setLevel(level)
    handler.setFormatter(ContextFormatter(fmt=fmt, datefmt="%H:%M:%S"))

    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


class BindAdapter(logging.LoggerAdapter):
    """Logger adapter that binds default contextual fields (like call_id)."""

    def process(self, msg, kwargs):
        # Allow standard logging keywords to pass through
        allowed = {"exc_info", "stack_info", "stacklevel", "extra"}
        extra = dict(self.extra)
        supplied_extra = kwargs.pop("extra", {}) or {}
        if supplied_extra:
            extra.update(supplied_extra)
        # Any non-standard kwargs are treated as contextual fields
        to_move = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in allowed}
        if to_move:
            extra.update(to_move)
        if extra:
            kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str | None = None, **ctx: str) -> logging.Logger:
    base = logging.getLogger(name or __name__)
    # Always return an adapter so keyword context works everywhere
    return BindAdapter(base, ctx)  # type: ignore[return-value]


def bind(logger: logging.Logger, **ctx: str) -> logging.Logger:
    """Return a logger that always includes given context fields."""
    if isinstance(logger, BindAdapter):
        # Merge into existing adapter
        logger.extra.update(ctx)
        return logger
    return BindAdapter(logger, ctx)  # type: ignore[return-value]


def exception(logger: logging.Logger, msg: str, **ctx: str) -> None:
    """Log an exception with stack trace and context."""
    logger.error(msg, exc_info=True, **({"extra": ctx} if ctx else {}))
