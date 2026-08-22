"""Validation for opaque maintenance operation identifiers."""

from __future__ import annotations

import ntpath
from pathlib import Path


def validate_operation_id(value: object) -> str:
    """Validate a caller-supplied operation ID before it reaches a path.

    Operation IDs are opaque names, not filesystem paths. Generation belongs
    to the operation entry point, so ``None`` is deliberately invalid here.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("operation_id must be a non-empty string")
    if "\x00" in value:
        raise ValueError("operation_id must not contain NUL")
    if "/" in value or "\\" in value:
        raise ValueError("operation_id must not contain path separators")
    if value in {".", ".."} or Path(value).is_absolute() or ntpath.isabs(value):
        raise ValueError("operation_id must be a relative opaque identifier")
    return value


__all__ = ["validate_operation_id"]
