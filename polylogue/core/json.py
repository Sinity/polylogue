"""Central JSON utilities using orjson."""

from __future__ import annotations

import logging
from typing import Any

import orjson

logger = logging.getLogger(__name__)


def dumps(obj: Any, *, default: Any = None, option: int | None = None) -> str:
    """Dump object to JSON string."""
    # orjson.dumps returns bytes
    # OPT_INDENT_2 | OPT_SORT_KEYS equivalent to indent=2, sort_keys=True
    # But we usually want compact for logs/DB and pretty for files.
    # We'll stick to default minimal dump for now unless parameterized.
    # However, callers often want string.
    try:
        return orjson.dumps(obj, default=default, option=option).decode("utf-8")
    except TypeError:
        # Fallback for complex types not handled by orjson default if needed?
        # Usually orjson is stricter.
        pass
    import json

    return json.dumps(obj, default=default)


def loads(obj: str | bytes) -> Any:
    """Load object from JSON string or bytes."""
    return orjson.loads(obj)
