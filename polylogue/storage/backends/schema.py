"""SQLite schema management: orchestration over the declared schema bands."""

from __future__ import annotations

from polylogue.storage.backends.schema_ddl import SCHEMA_DDL, SCHEMA_VERSION
from polylogue.storage.backends.schema_upgrade import (
    apply_current_schema_extensions,
)
from polylogue.storage.backends.schema_upgrade import (
    ensure_schema as _ensure_schema,
)
from polylogue.storage.backends.schema_upgrade import (
    ensure_vec0_table as _ensure_vec0_table,
)

__all__ = [
    "SCHEMA_DDL",
    "SCHEMA_VERSION",
    "_ensure_schema",
    "_ensure_vec0_table",
    "apply_current_schema_extensions",
]
