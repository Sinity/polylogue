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
    assert declared <= set(OPS_TABLE_DISPOSITIONS)
    for name in declared:
        disposition = OPS_TABLE_DISPOSITIONS[name]
        assert disposition.owner
        assert disposition.grain
        assert disposition.replacement


def test_historical_ops_tables_have_explicit_retirement_dispositions() -> None:
    """A stale archive object is not silently treated as disposable.

    Anti-vacuity: removing one measured historical table from the map leaves
    the inventory incomplete and fails this assertion.  The bootstrap
    convergence plan remains the only authority allowed to retire these
    objects; this test records the decision without dropping live state.
    """
    measured_live = {
        "slo_samples",
        "query_runs",
        "otlp_spans",
        "otlp_telemetry",
        "polylogue_ops_schema_state",
    }
    assert measured_live <= set(OPS_TABLE_DISPOSITIONS)
    assert OPS_TABLE_DISPOSITIONS["polylogue_ops_schema_state"].replacement == "retain"
    assert {OPS_TABLE_DISPOSITIONS[name].replacement for name in measured_live - {"polylogue_ops_schema_state"}} == {
        "retire via convergence"
    }


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


def test_mcp_call_log_stays_in_ops_until_audit_models_its_outcome_shape() -> None:
    """A request-admission row is not a replacement for completed call telemetry."""
    for name in ("mcp_call_log", "mcp_call_session_refs"):
        disposition = OPS_TABLE_DISPOSITIONS[name]
        assert disposition.restart_required
        assert disposition.replacement == "retain pending audit migration"
