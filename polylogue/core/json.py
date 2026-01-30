"""Central JSON utilities using orjson."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import orjson

logger = logging.getLogger(__name__)

# Export JSONDecodeError for compatibility
JSONDecodeError = ValueError  # orjson.JSONDecodeError is not exposed, use ValueError


def _default_encoder(user_default: Callable[[Any], Any] | None = None) -> Callable[[Any], Any]:
    """Create a JSON encoder that handles Decimal values."""

    def _encoder(obj: Any) -> Any:
        if user_default is not None:
            try:
                return user_default(obj)
            except TypeError:
                pass
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return _encoder


def dumps(obj: Any, *, default: Callable[[Any], Any] | None = None, option: int | None = None) -> str:
    """Dump object to JSON string."""
    encoder = _default_encoder(default)
    kwargs: dict[str, Any] = {"default": encoder}
    if option is not None:
        kwargs["option"] = option
    try:
        return orjson.dumps(obj, **kwargs).decode("utf-8")
    except TypeError:
        pass
    import json

    return json.dumps(obj, default=encoder)


def loads(obj: str | bytes) -> Any:
    """Load object from JSON string or bytes."""
    return orjson.loads(obj)
