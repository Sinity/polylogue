"""Polylogue error hierarchy.

All project exceptions inherit from PolylogueError, enabling:
- ``except PolylogueError`` at top-level boundaries (CLI, MCP server)
- Fine-grained catches deeper in the stack (``except DriveAuthError``)

Hierarchy:
    PolylogueError
    ├── ConfigError
    ├── DriveError
    │   ├── DriveAuthError
    │   └── DriveNotFoundError
    ├── DatabaseError
    │   └── SqliteVecError
    └── UIError
"""

from __future__ import annotations


class PolylogueError(Exception):
    """Base class for all Polylogue errors."""
