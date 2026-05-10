"""Polylogue error hierarchy.

All project exceptions inherit from PolylogueError, enabling:
- ``except PolylogueError`` at top-level boundaries (CLI, MCP server)
- Fine-grained catches deeper in the stack (``except DriveAuthError``)

Hierarchy (subclasses defined in their respective modules):
    PolylogueError                          # this module
    ├── ConfigError                         # config.py
    ├── DriveError                          # sources/drive_client.py
    │   ├── DriveAuthError
    │   └── DriveNotFoundError
    ├── DatabaseError                       # this module
    │   └── SqliteVecError                  # storage/search_providers/sqlite_vec.py
    └── UIError                             # ui/facade.py
"""

from __future__ import annotations

from http import HTTPStatus


class PolylogueError(Exception):
    """Base class for all Polylogue errors.

    Every derived error should set ``is_transient`` (whether retrying might
    succeed) and ``http_status_code`` (the best HTTP status when surfaced
    through the daemon API or MCP boundary).
    """

    is_transient: bool = False
    http_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR


class DatabaseError(PolylogueError):
    """Base class for database errors."""

    http_status_code: int = HTTPStatus.SERVICE_UNAVAILABLE


class SchemaIncompatibleError(DatabaseError):
    """Raised when the on-disk schema version cannot be served by this runtime.

    The runtime expects ``expected_version`` (the build-time ``SCHEMA_VERSION``
    constant). The database reports ``current_version``. There is no automatic
    migration path that the runtime is willing to apply for this transition.

    Both versions are exposed as attributes so call sites can format a
    structured message (and the daemon health surface can render it) without
    re-parsing the human-readable string.
    """

    def __init__(
        self,
        message: str,
        *,
        current_version: int,
        expected_version: int,
    ) -> None:
        super().__init__(message)
        self.current_version = current_version
        self.expected_version = expected_version


__all__ = ["DatabaseError", "PolylogueError", "SchemaIncompatibleError"]
