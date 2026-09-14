"""Which file set an operation must read, resolved without opening anything.

Split out of ``operation_context`` deliberately. That module declares the
execution dependencies of a read and imports ``ArchiveStore`` and the query
execution machinery to do it; this question is answered from a configuration
object and two paths. A caller that only needs to know *where* to read -- the
operation kernel choosing a daemon socket, for one, on a path that may never
open an archive at all -- pays the whole substrate import to ask, and on a
shell-completion keystroke that cost is the entire latency.

This module is now the declared home of that question; ``operation_context``
owns execution dependencies, not addressing.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.archive_identity import archive_file_set_root


def operation_archive_root(config: object) -> Path:
    """Resolve the file set an operation must read for this configuration.

    ``config.db_path`` always names a concrete ``index.db`` — an explicit
    ``--db`` override or the resolved active generation (polylogue-yla8.1).
    When that index belongs to a different file set than ``config.archive_root``
    names, the pinned index is the operator's instruction: reading the active
    generation instead would answer from different rows than the one they
    named, while reporting success.
    """

    archive_root = Path(str(getattr(config, "archive_root", "") or ""))
    db_path = getattr(config, "db_path", None)
    if db_path is None:
        return archive_root
    return archive_file_set_root(archive_root=archive_root, db_path=Path(str(db_path)))


__all__ = ["operation_archive_root"]
