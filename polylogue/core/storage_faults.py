"""Archive storage faults: infrastructure failures that say nothing about the input.

A full disk, an I/O error, a read-only mount or a malformed database page
fails every write in the same way, whatever bytes were being written. Treating
such a failure as a verdict on the input -- a parse error on the raw, a failed
cursor that backs off into quarantine -- converts an operator-fixable storage
condition into permanently refused, perfectly good source files. This module
names the condition once so every ingest boundary can let it escape as
:class:`ArchiveStorageFaultError` instead.

Classification keys off SQLite's typed result codes and ``errno``, never off
message text, and follows the explicit ``__cause__``/``__context__`` chain so a
fault wrapped by an intermediate layer or grouped with cleanup failures is
still recognized.
"""

from __future__ import annotations

import errno
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from enum import StrEnum
from itertools import chain

from polylogue.core.errors import PolylogueError

_SQLITE_PRIMARY_RESULT_CODE_MASK = 0xFF


class StorageFaultKind(StrEnum):
    """What kind of storage failure stopped a write."""

    CAPACITY = "capacity"
    IO = "io"
    CORRUPT = "corrupt"
    READ_ONLY = "read_only"
    #: Bytes published before the writer reserved them were reclaimed by blob
    #: GC in between. The retained source still holds them; a retry
    #: republishes them.
    EVICTED = "evicted"


_SQLITE_FAULTS: dict[int, StorageFaultKind] = {
    sqlite3.SQLITE_FULL: StorageFaultKind.CAPACITY,
    sqlite3.SQLITE_IOERR: StorageFaultKind.IO,
    sqlite3.SQLITE_CORRUPT: StorageFaultKind.CORRUPT,
    sqlite3.SQLITE_NOTADB: StorageFaultKind.CORRUPT,
    sqlite3.SQLITE_READONLY: StorageFaultKind.READ_ONLY,
}

_ERRNO_FAULTS: dict[int, StorageFaultKind] = {
    errno.ENOSPC: StorageFaultKind.CAPACITY,
    errno.EDQUOT: StorageFaultKind.CAPACITY,
    errno.EIO: StorageFaultKind.IO,
    errno.EROFS: StorageFaultKind.READ_ONLY,
}


def _own_fault(exc: BaseException) -> StorageFaultKind | None:
    if isinstance(exc, ArchiveStorageFaultError):
        return exc.kind
    if isinstance(exc, sqlite3.Error):
        code = getattr(exc, "sqlite_errorcode", None)
        if isinstance(code, int):
            return _SQLITE_FAULTS.get(code & _SQLITE_PRIMARY_RESULT_CODE_MASK)
        return None
    if isinstance(exc, OSError) and isinstance(exc.errno, int):
        return _ERRNO_FAULTS.get(exc.errno)
    return None


def storage_fault_kind(exc: BaseException) -> StorageFaultKind | None:
    """Return a typed storage fault in the full causal/cleanup exception graph."""
    seen: set[int] = set()
    pending: list[Iterator[BaseException]] = [iter((exc,))]
    while pending:
        current = next(pending[-1], None)
        if current is None:
            pending.pop()
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        kind = _own_fault(current)
        if kind is not None:
            return kind
        cause = current.__cause__ if current.__cause__ is not None else current.__context__
        causes = () if cause is None else (cause,)
        children = current.exceptions if isinstance(current, BaseExceptionGroup) else ()
        pending.append(iter(chain(causes, children)))
    return None


class ArchiveStorageFaultError(PolylogueError):
    """Archive storage refused a write; the input being written is not at fault.

    Raised from an ingest boundary instead of recording a per-item failure. The
    original exception is the ``__cause__``. Callers report the affected items
    as retryable and leave their cursors and raw parse state untouched, so the
    same inputs are offered again once the storage condition is resolved.
    """

    def __init__(self, kind: StorageFaultKind, cause: BaseException) -> None:
        self.kind = kind
        super().__init__(f"archive storage fault ({kind.value}): {type(cause).__name__}: {cause}")

    @property
    def reason(self) -> str:
        """A stable event ``reason`` token for this fault."""
        return f"storage_fault.{self.kind.value}"


def raise_if_storage_fault(exc: BaseException, *, kinds: frozenset[StorageFaultKind] | None = None) -> None:
    """Re-raise ``exc`` as :class:`ArchiveStorageFaultError` when it is one.

    ``kinds`` narrows the recognized faults for a boundary where some errno is
    ambiguous -- a failed copy of a source file can report ``EIO`` for the
    source read, which is a per-file failure, while ``ENOSPC`` can only come
    from the archive side.
    """
    kind = storage_fault_kind(exc)
    if kind is None or (kinds is not None and kind not in kinds):
        return
    if isinstance(exc, ArchiveStorageFaultError):
        raise exc
    raise ArchiveStorageFaultError(kind, exc) from exc


#: Faults that can only originate on the archive (write) side of a copy.
ARCHIVE_SIDE_FAULTS = frozenset({StorageFaultKind.CAPACITY, StorageFaultKind.READ_ONLY})
#: The narrower set for a boundary whose source can itself report read-only
#: or I/O failures (a SQLite export opens the source database).
CAPACITY_FAULTS = frozenset({StorageFaultKind.CAPACITY})


__all__ = [
    "ARCHIVE_SIDE_FAULTS",
    "CAPACITY_FAULTS",
    "ArchiveStorageFaultError",
    "StorageFaultKind",
    "raise_if_storage_fault",
    "storage_fault_kind",
]
