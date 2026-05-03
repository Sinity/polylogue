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


__all__ = ["DatabaseError", "PolylogueError"]
