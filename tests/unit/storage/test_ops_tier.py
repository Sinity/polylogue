from __future__ import annotations

import re

from polylogue.storage.sqlite.archive_tiers.ops import (
    OPS_DDL,
    OPS_TABLE_DISPOSITIONS,
)


def _declared_tables() -> set[str]:
    return set(re.findall(r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\(", OPS_DDL))


def test_every_canonical_ops_table_has_a_disposition() -> None:
    """The diet cannot delete a table before its owner and restart semantics are recorded.

    Anti-vacuity: removing any entry from ``OPS_TABLE_DISPOSITIONS`` leaves a
    canonical ``CREATE TABLE`` name unmatched and fails this assertion.
    """
    declared = _declared_tables()
    assert declared
    assert declared <= set(OPS_TABLE_DISPOSITIONS)
    for name in declared:
        disposition = OPS_TABLE_DISPOSITIONS[name]
        assert disposition.owner
        assert disposition.grain
        assert disposition.replacement


def test_attempt_tables_are_independently_dispositioned() -> None:
    """Catch-up and scheduler receipts are not lossy folds into ingest_attempts."""
    assert OPS_TABLE_DISPOSITIONS["embedding_catchup_runs"].replacement == "retain independently"
    assert OPS_TABLE_DISPOSITIONS["judgment_scheduler_receipts"].replacement == "retain independently"


def test_mcp_call_log_stays_in_ops_until_audit_models_its_outcome_shape() -> None:
    """A request-admission row is not a replacement for completed call telemetry."""
    for name in ("mcp_call_log", "mcp_call_session_refs"):
        disposition = OPS_TABLE_DISPOSITIONS[name]
        assert disposition.restart_required
        assert disposition.replacement == "retain pending audit migration"
