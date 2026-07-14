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


class SchemaVersionMismatchError(DatabaseError):
    """Raised when the on-disk schema version cannot be served by this runtime.

    The runtime expects ``expected_version`` (the build-time ``SCHEMA_VERSION``
    constant). The database reports ``current_version``. There is no automatic
    in-place upgrade path that the runtime is willing to apply for this
    transition.

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


class EmbeddingRetrievalNotReadyError(DatabaseError):
    """Raised when ``--similar``/``--semantic`` is asked for but vectors aren't ready.

    Carries an operator-actionable message naming the current readiness
    status and the next step (``polylogue ops embed status`` →
    ``polylogue ops embed backfill``/``enable``). Unlike a generic
    :class:`DatabaseError`, this class lets surfaces forward the message
    verbatim to the client because the contents are by construction free
    of secrets — the readiness status enum is a closed vocabulary
    (``ready``/``partial``/``pending``/``disabled``/``none``) and the
    follow-up command names are fixed strings, not user data.

    Used by the operations layer's
    ``_resolve_vector_provider_for_search`` so that CLI, MCP, and HTTP
    surfaces all surface the same actionable error instead of the CLI
    getting the message and MCP getting only the exception class name
    (#1503 AC4).
    """

    http_status_code: int = HTTPStatus.CONFLICT

    def __init__(self, message: str, *, readiness_status: str) -> None:
        super().__init__(message)
        self.readiness_status = readiness_status


__all__ = [
    "DatabaseError",
    "EmbeddingRetrievalNotReadyError",
    "PolylogueError",
    "SchemaVersionMismatchError",
]
