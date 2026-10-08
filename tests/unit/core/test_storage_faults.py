"""Storage-fault classification: typed codes and errno, never message text."""

from __future__ import annotations

import errno
import sqlite3
from builtins import BaseExceptionGroup

import pytest

from polylogue.core.storage_faults import (
    ARCHIVE_SIDE_FAULTS,
    ArchiveStorageFaultError,
    StorageFaultKind,
    raise_if_storage_fault,
    storage_fault_kind,
)
from polylogue.pipeline.ingest_outcomes import classify_archive_write_exception


def _sqlite_error(code: int, message: str = "storage refused") -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError(message)
    exc.sqlite_errorcode = code
    return exc


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_sqlite_error(sqlite3.SQLITE_FULL), StorageFaultKind.CAPACITY),
        (_sqlite_error(sqlite3.SQLITE_IOERR | (3 << 8)), StorageFaultKind.IO),  # extended IOERR_WRITE
        (_sqlite_error(sqlite3.SQLITE_CORRUPT), StorageFaultKind.CORRUPT),
        (_sqlite_error(sqlite3.SQLITE_NOTADB), StorageFaultKind.CORRUPT),
        (_sqlite_error(sqlite3.SQLITE_READONLY), StorageFaultKind.READ_ONLY),
        (OSError(errno.ENOSPC, "No space left on device"), StorageFaultKind.CAPACITY),
        (OSError(errno.EDQUOT, "Disk quota exceeded"), StorageFaultKind.CAPACITY),
        (OSError(errno.EIO, "Input/output error"), StorageFaultKind.IO),
        (OSError(errno.EROFS, "Read-only file system"), StorageFaultKind.READ_ONLY),
    ],
)
def test_storage_faults_are_classified_by_code(exc: BaseException, kind: StorageFaultKind) -> None:
    assert storage_fault_kind(exc) is kind


@pytest.mark.parametrize(
    "exc",
    [
        _sqlite_error(sqlite3.SQLITE_BUSY, "database is locked"),
        _sqlite_error(sqlite3.SQLITE_CONSTRAINT),
        # Message text alone never classifies: a lock whose message mentions
        # "disk" is still not a storage fault.
        sqlite3.OperationalError("database or disk is full"),
        OSError(errno.ENOENT, "No such file or directory"),
        ValueError("disk full"),
    ],
)
def test_non_storage_failures_are_not_faults(exc: BaseException) -> None:
    assert storage_fault_kind(exc) is None


def test_fault_is_found_through_the_cause_chain() -> None:
    outer = RuntimeError("blob publication failed")
    outer.__cause__ = OSError(errno.ENOSPC, "No space left on device")
    assert storage_fault_kind(outer) is StorageFaultKind.CAPACITY
    with pytest.raises(ArchiveStorageFaultError) as raised:
        raise_if_storage_fault(outer)
    assert raised.value.kind is StorageFaultKind.CAPACITY
    assert raised.value.reason == "storage_fault.capacity"
    assert raised.value.__cause__ is outer


def test_archive_side_narrowing_leaves_source_read_errors_to_the_caller() -> None:
    # A failed copy of a source file can report EIO for the *source* read;
    # only capacity and read-only faults are archive-side by construction.
    raise_if_storage_fault(OSError(errno.EIO, "Input/output error"), kinds=ARCHIVE_SIDE_FAULTS)
    with pytest.raises(ArchiveStorageFaultError):
        raise_if_storage_fault(OSError(errno.ENOSPC, "No space left on device"), kinds=ARCHIVE_SIDE_FAULTS)


def test_archive_write_classification_keeps_storage_faults_retryable() -> None:
    """Anti-vacuity: before this classification a full disk was ``parser_defect``,
    non-retryable, so the input backed off into quarantine."""
    disposition = classify_archive_write_exception(_sqlite_error(sqlite3.SQLITE_FULL))
    assert disposition.outcome_code == "transient_error"
    assert disposition.retryable is True
    assert disposition.evidence_ref == "archive_write:storage_fault:capacity"

    defect = classify_archive_write_exception(KeyError("missing"))
    assert defect.outcome_code == "parser_defect"


def test_sqlite_snapshot_export_escapes_only_on_capacity() -> None:
    """The live SQLite export adapter re-raises SQLite errors as ``OSError``
    from the original; a full archive must still be recognized through that
    chain, while read-only (which the source database itself can report)
    stays the file's own failure. Anti-vacuity: without cause-chain
    following the capacity case does not escape."""
    from polylogue.core.storage_faults import CAPACITY_FAULTS
    from polylogue.sources.sqlite_snapshot import sqlite_snapshot_failure_as_oserror

    def exported(code: int) -> OSError:
        try:
            with sqlite_snapshot_failure_as_oserror():
                raise _sqlite_error(code)
        except OSError as exc:
            return exc
        raise AssertionError("adapter did not translate the SQLite error")

    with pytest.raises(ArchiveStorageFaultError):
        raise_if_storage_fault(exported(sqlite3.SQLITE_FULL), kinds=CAPACITY_FAULTS)
    raise_if_storage_fault(exported(sqlite3.SQLITE_READONLY), kinds=CAPACITY_FAULTS)


@pytest.mark.parametrize("kind,number", [(StorageFaultKind.IO, errno.EIO), (StorageFaultKind.CAPACITY, errno.ENOSPC)])
def test_grouped_terminal_storage_failure_is_retryable(kind: StorageFaultKind, number: int) -> None:
    primary = ValueError("synthetic parser work failure")
    cleanup = OSError(number, "synthetic terminal storage failure")
    wrapped = RuntimeError("synthetic native owner failure")
    wrapped.__cause__ = cleanup
    failure = BaseExceptionGroup("mutation and cleanup", [primary, BaseExceptionGroup("native cleanup", [wrapped])])
    assert storage_fault_kind(failure) is kind
    disposition = classify_archive_write_exception(failure)
    assert disposition.retryable and disposition.outcome_code == "transient_error"
    assert disposition.evidence_ref == f"archive_write:storage_fault:{kind.value}"
    with pytest.raises(ArchiveStorageFaultError) as caught:
        raise_if_storage_fault(failure)
    assert caught.value.kind is kind and caught.value.__cause__ is failure


def test_storage_fault_traversal_keeps_deep_siblings_and_terminates_cycles() -> None:
    cycle = RuntimeError("cycle")
    cycle.__cause__ = cycle
    fault: BaseException = OSError(errno.EIO, "synthetic storage failure")
    for _ in range(32):
        wrapper = RuntimeError("nested owner")
        wrapper.__cause__ = fault
        fault = wrapper
    failure = BaseExceptionGroup("terminal siblings", [cycle, fault])
    assert storage_fault_kind(failure) is StorageFaultKind.IO
    assert storage_fault_kind(BaseExceptionGroup("non-storage siblings", [cycle, ValueError("input")])) is None
