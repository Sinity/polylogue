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

from pathlib import Path, PurePath

IMPORT_STAGING_DIRECTORY = "import-staging"


def import_staging_root(archive_root: Path) -> Path:
    """The directory staged imports wait in until the ingest operation retains them."""
    return archive_root / IMPORT_STAGING_DIRECTORY


def resolve_staged_import(raw_path: object, archive_root: Path) -> tuple[Path | None, str | None]:
    """Resolve an ingest request to an existing entry in the import staging root.

    Only the final component of ``raw_path`` names the entry, and the resolved
    entry must stay inside the staging root, so the loopback HTTP surface
    never becomes an arbitrary local file copier. Returns the entry or an
    error token (``missing_path``, ``invalid_path``, ``path_not_found``).
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "missing_path"

    source_name = PurePath(raw_path).name
    if not source_name or source_name in {".", ".."}:
        return None, "invalid_path"

    staging = import_staging_root(archive_root)
    try:
        staging_root = staging.resolve()
        candidates = list(staging.iterdir())
    except OSError:
        return None, "path_not_found"

    for candidate in candidates:
        if candidate.name != source_name:
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(staging_root)
        except ValueError:
            return None, "invalid_path"
        return resolved, None

    return None, "path_not_found"


__all__ = ["IMPORT_STAGING_DIRECTORY", "import_staging_root", "resolve_staged_import"]
