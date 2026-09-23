from __future__ import annotations

import re

from polylogue.storage.sqlite.archive_tiers.ops import (
    OPS_BENIGN_DDL_CONVERGENCE_PLAN,
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
    assert declared == set(OPS_TABLE_DISPOSITIONS)
    for disposition in OPS_TABLE_DISPOSITIONS.values():
        assert disposition.owner
        assert disposition.grain
        assert disposition.replacement


def test_restart_required_state_is_not_a_retirement_target() -> None:
    """Security, convergence, and admission state remain restart-persistent.

    Anti-vacuity: adding a DROP entry for one of these tables makes the test
    fail even though the SQL shape itself is valid.
    """
    protected = {name for name, disposition in OPS_TABLE_DISPOSITIONS.items() if disposition.restart_required}
    dropped = {
        match.group(1)
        for match in (re.match(r"DROP TABLE IF EXISTS (\w+)$", entry.sql) for entry in OPS_BENIGN_DDL_CONVERGENCE_PLAN)
        if match is not None
    }
    assert not protected & dropped
    assert {
        "secret_scan_status",
        "whole_archive_convergence_pledge",
        "context_injection_ledger",
        "convergence_debt",
    } <= protected


def test_attempt_tables_are_independently_dispositioned() -> None:
    """Catch-up and scheduler receipts are not lossy folds into ingest_attempts."""
    assert OPS_TABLE_DISPOSITIONS["embedding_catchup_runs"].replacement == "retain independently"
    assert OPS_TABLE_DISPOSITIONS["judgment_scheduler_receipts"].replacement == "retain independently"
