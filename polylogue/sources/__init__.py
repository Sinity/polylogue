"""Sources package — unified parsing from all AI providers.

- sources/parsers/: JSON → ParsedSession for each provider
- sources/providers/: Pydantic models for provider export formats
- sources/source_parsing.py: Source walking and parsed session iteration
- sources/source_acquisition.py: Raw source acquisition iteration
- sources/drive_*.py / drive.py: Google Drive auth, gateway, and source access

Public re-exports are lazy (PEP 562 module ``__getattr__``): merely importing
a *submodule* of this package (``polylogue.sources.parsers.base``,
``polylogue.sources.dispatch``, ...) -- which many callers only need for a
single lightweight type -- otherwise forces this package's own ``__init__``
to run first, which used to mean eagerly pulling in the entire Google Drive
subsystem (``.drive``, ``.drive.source``, ``tenacity``, ...) regardless of
whether Drive is ever touched (polylogue-8s70/h1wt: this was the single
largest remaining contributor to ``polylogue.storage.repair``'s and
``polylogue.storage.sqlite.archive_tiers.write``'s import cost). Submodule
imports (``from polylogue.sources import dispatch``) are unaffected by this
``__getattr__`` -- Python's import system falls back to importing the
submodule directly whenever an attribute lookup on the package fails.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .drive.source import DriveSourceAPI, DriveSourceClient, build_drive_source_client
    from .drive.types import DriveAuthError, DriveError, DriveFile, DriveNotFoundError
    from .parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
    from .source_parsing import iter_source_sessions


def __getattr__(name: str) -> object:
    lazy_exports = {
        "DriveAuthError": (".drive.types", "DriveAuthError"),
        "DriveError": (".drive.types", "DriveError"),
        "DriveFile": (".drive.types", "DriveFile"),
        "DriveNotFoundError": (".drive.types", "DriveNotFoundError"),
        "DriveSourceAPI": (".drive.source", "DriveSourceAPI"),
        "DriveSourceClient": (".drive.source", "DriveSourceClient"),
        "build_drive_source_client": (".drive.source", "build_drive_source_client"),
        "download_drive_files": (".drive", "download_drive_files"),
        "ParsedAttachment": (".parsers.base", "ParsedAttachment"),
        "ParsedMessage": (".parsers.base", "ParsedMessage"),
        "ParsedSession": (".parsers.base", "ParsedSession"),
        "iter_source_sessions": (".source_parsing", "iter_source_sessions"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DriveAuthError",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "DriveSourceAPI",
    "DriveSourceClient",
    "ParsedAttachment",
    "ParsedSession",
    "ParsedMessage",
    "build_drive_source_client",
    "download_drive_files",
    "iter_source_sessions",
]
