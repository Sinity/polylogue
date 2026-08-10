"""Product-layer access to judgment-scheduler receipts.

The daemon and read surfaces share this adapter instead of importing the ops
tier directly.  Keeping the tier boundary here makes the receipt schema an
operations concern while leaving storage-specific representations private to
the adapter.
"""

from __future__ import annotations

import sqlite3

from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ArchiveJudgmentSchedulerReceipt,
    _read_latest_judgment_scheduler_receipt,
)
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    record_judgment_scheduler_receipt as _record_judgment_scheduler_receipt,
)


def record_judgment_scheduler_receipt(
    conn: sqlite3.Connection,
    receipt: ArchiveJudgmentSchedulerReceipt,
) -> None:
    """Persist one validated scheduler receipt through the ops-tier adapter."""

    _record_judgment_scheduler_receipt(conn, receipt)


def read_latest_judgment_scheduler_receipt(
    conn: sqlite3.Connection,
    *,
    operation_id: str | None = None,
) -> ArchiveJudgmentSchedulerReceipt | None:
    """Read the newest typed scheduler receipt, optionally by operation."""

    return _read_latest_judgment_scheduler_receipt(conn, operation_id=operation_id)


__all__ = [
    "ArchiveJudgmentSchedulerReceipt",
    "read_latest_judgment_scheduler_receipt",
    "record_judgment_scheduler_receipt",
]
