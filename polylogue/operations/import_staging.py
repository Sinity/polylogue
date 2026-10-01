"""Archive-owned staging for explicit imports.

``polylogue import PATH`` and ``POST /api/ingest`` copy the caller's material
into the archive and submit the declared ``ingest`` operation, which retains
the copy under the caller's ``source_path``. The copy lives in
``import-staging/``, which no watcher and no configured source reads. Staging
it in the watched ``inbox/`` made the live watcher and fair file intake acquire
the same bytes a second time, keyed on the inbox path, so one import produced
two raw revisions. The inbox remains the drop directory the daemon watches.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

IMPORT_STAGING_DIRECTORY = "import-staging"


def import_staging_root(archive_root: Path) -> Path:
    """The directory staged imports wait in until the ingest operation retains them."""
    return archive_root / IMPORT_STAGING_DIRECTORY


def stage_import_input(source: Path, archive_root: Path, *, check_stop: Callable[[], None]) -> Path:
    """Capture a surface's input through the sole source-staging owner."""
    from polylogue.sources.source_staging import stage_source_input

    return stage_source_input(source, import_staging_root(archive_root), check_stop=check_stop)


def resolve_staged_import(raw_path: object, archive_root: Path) -> tuple[Path | None, str | None]:
    """Resolve the exact staged coordinate, including private slot descendants.

    Relative coordinates start at the staging root; absolute coordinates must
    name that same namespace. Resolving a symlink must still stay inside it.
    A matching basename elsewhere does not select another staged input.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "missing_path"
    if "\0" in raw_path:
        return None, "invalid_path"
    staging = import_staging_root(archive_root)
    try:
        staging_root = staging.resolve(strict=True)
        requested = Path(raw_path)
        lexical = Path(os.path.abspath(requested if requested.is_absolute() else staging_root / requested))
        try:
            lexical.relative_to(staging_root)
        except ValueError:
            return None, "path_not_found"
        resolved = lexical.resolve(strict=True)
        try:
            resolved.relative_to(staging_root)
        except ValueError:
            return None, "invalid_path"
        if resolved == staging_root or not (resolved.is_file() or resolved.is_dir()):
            return None, "invalid_path"
        return resolved, None
    except (OSError, ValueError):
        return None, "path_not_found"


__all__ = [
    "IMPORT_STAGING_DIRECTORY",
    "import_staging_root",
    "resolve_staged_import",
    "stage_import_input",
]
