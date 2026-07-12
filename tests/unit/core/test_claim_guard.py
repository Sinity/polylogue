"""Unit tests for the claim-guard vocabulary (polylogue-avg).

Covers `derive_claim_guard` as a pure function, and a parity check against
the raw-materialization state classification that used to live only in the
archived devloop-status script
(`.agent/archive/devloop-2026-07/scripts/devloop-status`, frozen evidence —
never resurrected or executed live; see repo `CLAUDE.md`). The parity check
replicates that script's classification logic verbatim as a local reference
implementation (it cannot import or execute the archived shell script) and
asserts the product's `raw_materialization_ready` boolean agrees with it over
a fixed set of representative archive states.
"""

from __future__ import annotations

from polylogue.readiness.claim_guard import ClaimGuard, derive_claim_guard
from polylogue.storage.archive_readiness import raw_materialization_ready


def _base_kwargs() -> dict[str, object]:
    return {
        "archive_schema_ready": True,
        "schema_mismatches": (),
        "missing_tiers": (),
        "raw_materialization_ready": True,
        "raw_materialization_summary": "ready",
        "search_ready": True,
        "search_summary": "ready",
        "active_writer": False,
        "active_writer_summary": "",
    }


def test_fully_ready_archive_claims_all_four_states() -> None:
    guard = derive_claim_guard(**_base_kwargs())  # type: ignore[arg-type]
    assert isinstance(guard, ClaimGuard)
    payload = guard.to_dict()
    assert payload["openable"]["value"] is True
    assert payload["converged"]["value"] is True
    assert payload["search_ready"]["value"] is True
    assert payload["perf_measurable"]["value"] is True
    # Every entry carries a documented signal name for auditability.
    assert all(entry["signal"] for entry in payload.values())


def test_schema_mismatch_blocks_openable_and_converged() -> None:
    kwargs = _base_kwargs()
    kwargs["archive_schema_ready"] = False
    kwargs["schema_mismatches"] = ["index"]
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is False
    assert "index" in str(guard["openable"]["reason"])
    # Converged cannot be claimed once the archive is not openable, even if
    # the raw-materialization signal itself looked converged.
    assert guard["converged"]["value"] is False
    assert "not openable" in str(guard["converged"]["reason"])


def test_missing_tiers_block_openable_with_named_tiers() -> None:
    kwargs = _base_kwargs()
    kwargs["archive_schema_ready"] = False
    kwargs["missing_tiers"] = ["user", "embeddings"]
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is False
    assert "embeddings" in str(guard["openable"]["reason"])
    assert "user" in str(guard["openable"]["reason"])


def test_openable_but_not_converged_reports_raw_materialization_reason() -> None:
    """Bead AC: an openable-but-not-converged archive reports the exact
    raw-materialization reason string, not a generic failure message."""
    kwargs = _base_kwargs()
    kwargs["raw_materialization_ready"] = False
    kwargs["raw_materialization_summary"] = "raw evidence pending materialization"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is True
    assert guard["converged"]["value"] is False
    assert guard["converged"]["reason"] == "raw evidence pending materialization"


def test_search_not_ready_reports_component_summary() -> None:
    kwargs = _base_kwargs()
    kwargs["search_ready"] = False
    kwargs["search_summary"] = "fts index incomplete"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["search_ready"]["value"] is False
    assert guard["search_ready"]["reason"] == "fts index incomplete"


def test_active_writer_blocks_perf_measurable_with_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["active_writer"] = True
    kwargs["active_writer_summary"] = "2 live ingest attempt(s) running"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["perf_measurable"]["value"] is False
    assert guard["perf_measurable"]["reason"] == "2 live ingest attempt(s) running"


def test_active_writer_without_summary_falls_back_to_generic_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["active_writer"] = True
    kwargs["active_writer_summary"] = ""
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["perf_measurable"]["value"] is False
    assert "in flight" in str(guard["perf_measurable"]["reason"])


# ---------------------------------------------------------------------------
# Parity: archived devloop-status raw-materialization classification
# ---------------------------------------------------------------------------


def _devloop_status_raw_materialization_state(totals: dict[str, int]) -> str:
    """Reference re-implementation of the archived script's classification.

    Verbatim port of the Python heredoc embedded in
    `.agent/archive/devloop-2026-07/scripts/devloop-status` (lines ~637-646
    at archive time). The script itself is frozen evidence and is never
    executed live (repo CLAUDE.md); this function exists purely so a test can
    assert the product's `raw_materialization_ready` boolean still agrees
    with the devloop-era classification over the same fixed set of archive
    states.
    """
    if (totals.get("affected_total") or 0) == 0:
        return "ready"
    if (totals.get("blocked") or 0) and not (totals.get("actionable") or 0):
        return "blocked"
    if (
        (totals.get("classified") or 0)
        and not (totals.get("affected_open") or 0)
        and not (totals.get("affected_actionable") or 0)
    ):
        return "ready"
    if (
        not (totals.get("affected_actionable") or 0)
        and not (totals.get("critical") or 0)
        and not (totals.get("warning") or 0)
    ):
        return "degraded"
    return "stale"


_FIXED_ARCHIVE_STATES: tuple[dict[str, int], ...] = (
    # No raw/index join gaps at all.
    {"affected_total": 0},
    # Debt exists but is fully classified/explained — devloop still called
    # this "ready" (raw evidence classified; no materialization debt).
    {"affected_total": 3, "classified": 3, "affected_open": 0, "affected_actionable": 0},
    # Actionable/open debt still pending materialization.
    {"affected_total": 4, "warning": 1, "actionable": 1, "affected_actionable": 4, "affected_open": 4},
    # Blocked debt with nothing actionable.
    {"affected_total": 2, "blocked": 2, "actionable": 0},
    # Unclassified join gaps with no actionable/critical/warning signal.
    {"affected_total": 5, "affected_unchecked": 5, "unchecked": 5},
)


def test_raw_materialization_ready_agrees_with_archived_devloop_classification() -> None:
    """Bead AC: parity between the archived script's old computation and the
    product's `raw_materialization_ready` signal over a fixed set of states.

    devloop-status treated `state == "ready"` as the only state honestly
    claimable as "converged"; every other state (blocked/degraded/stale) was
    NOT converged. `raw_materialization_ready` must agree on this ready/not-
    ready split for every one of the fixed states below.
    """
    for totals in _FIXED_ARCHIVE_STATES:
        devloop_state = _devloop_status_raw_materialization_state(totals)
        payload = {"available": True, **totals}
        product_ready = raw_materialization_ready(payload)
        assert product_ready == (devloop_state == "ready"), (
            f"parity mismatch for {totals}: devloop_state={devloop_state!r} product_ready={product_ready!r}"
        )
