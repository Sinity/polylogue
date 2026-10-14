"""Read completeness comes from the actual Source/Index frame, not a later probe."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.query.evaluator import QueryEvaluationRequest
from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
from polylogue.cli.convergence_feedback import convergence_warning_line
from polylogue.core.enums import Provider
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read, prepare_operation_journals
from polylogue.storage.archive_readiness import ArchiveMaterializationUnavailableError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.query_objects import put_query
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def _admit_raw(root: Path) -> str:
    with write_lease("synthetic.unmaterialized-source", archive_root=root):
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=b'{"id":"neutral","mapping":{},"title":"neutral"}',
                source_path="neutral.json",
                canonical_source_path="neutral.json",
                acquired_at_ms=1,
            )


def test_populated_source_empty_index_read_is_not_authoritative_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_archive_root(tmp_path)
    _admit_raw(tmp_path)

    def no_later_probe(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result feedback opened another archive snapshot")

    monkeypatch.setattr("polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot", no_later_probe)
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation(
            "cli.query", {"params": {"limit": 1}}, archive=pinned.archive, serving_identity="direct"
        )
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "degraded"
    assert outcome["reason"] == "archive_not_converged"
    assert result["items"] == []
    detail = outcome["detail"]
    assert isinstance(detail, dict)
    assessment = detail["archive_materialization"]
    assert isinstance(assessment, dict)
    assert assessment["scope"] == "pinned_archive_raw_state"
    assert assessment["raw_artifact_count"] == 1
    assert assessment["materialized_raw_artifact_count"] == 0
    assert convergence_warning_line(outcome) is not None


def test_empty_source_retains_valid_empty_scope(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation(
            "cli.query", {"params": {"limit": 1}}, archive=pinned.archive, serving_identity="direct"
        )
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "empty"
    assert convergence_warning_line(outcome) is None


def test_exact_canonical_reference_refuses_incomplete_source_frame(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _admit_raw(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as user:
        query = put_query(
            user,
            {"kind": "field", "field": "origin", "op": "=", "values": ["chatgpt-export"]},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
    with open_operation_read(tmp_path) as pinned:
        with pytest.raises(ArchiveMaterializationUnavailableError) as refused:
            ArchiveCanonicalPlanEvaluator(pinned.archive).evaluate(
                QueryEvaluationRequest(query=query, purpose="reference")
            )
        assert refused.value.code == "archive_not_converged"
        assert refused.value.assessment.raw_artifact_count == 1


def test_raw_state_revision_stays_pinned_and_changes_after_same_row_update(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _admit_raw(tmp_path)
    prepare_operation_journals(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        before = pinned.archive.raw_state_read_revision()
        assert before is not None and before.startswith("source:raw-state:")
        with sqlite3.connect(tmp_path / "source.db") as writer:
            writer.execute("UPDATE raw_sessions SET blob_hash=? WHERE raw_id=?", (bytes([7]) * 32, raw_id))
        assert pinned.archive.raw_state_read_revision() == before
    with open_operation_read(tmp_path) as fresh:
        assert fresh.archive.raw_state_read_revision() != before
        # Pruning existing journal rows must not erase this revision authority.
        current = fresh.archive.raw_state_read_revision()
    with sqlite3.connect(tmp_path / "source.db") as writer:
        writer.execute("DELETE FROM raw_existence_changes")
    with open_operation_read(tmp_path) as after_prune:
        assert after_prune.archive.raw_state_read_revision() == current


def test_cached_empty_rows_receive_current_pinned_source_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_archive_root(tmp_path)
    _admit_raw(tmp_path)
    keys: list[Mapping[str, object]] = []
    cached: dict[str, object] = {"items": [], "outcome": {"state": "empty", "reason": "no_rows_in_scope", "detail": {}}}

    def cached_result(_name: str, payload: Mapping[str, object], **_kwargs: Any) -> dict[str, object]:
        keys.append(payload)
        return cached

    monkeypatch.setattr("polylogue.storage.search.cache.get_cached_result", cached_result)
    with open_operation_read(tmp_path) as pinned:
        assert pinned.read_view is not None
        result = execute_read_operation(
            "facets", {"params": {}}, archive=pinned.archive, serving_identity="direct", read_view=pinned.read_view
        )
    outcome = result["outcome"]
    assert isinstance(outcome, dict) and outcome["state"] == "degraded"
    assert cached["outcome"] == {"state": "empty", "reason": "no_rows_in_scope", "detail": {}}
    assert keys
    frame_key = keys[0]["params"]
    assert isinstance(frame_key, dict)
    assert str(frame_key["raw_state_revision"]).startswith("source:raw-state:")


def test_missing_source_does_not_authorize_empty_index_read(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    (tmp_path / "source.db").unlink()
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation(
            "cli.query", {"params": {"limit": 1}}, archive=pinned.archive, serving_identity="direct"
        )
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "degraded"
    assert outcome["reason"] == "archive_materialization_unmeasured"


@pytest.mark.parametrize("mode", ["single", "all"])
def test_mutation_cardinality_refuses_populated_source_empty_index(tmp_path: Path, mode: str) -> None:
    from tests.infra.daemon_operations import running_daemon_operations

    root = tmp_path / "daemon-archive"

    def seed(archive_root: Path) -> None:
        bootstrap_archive_root(archive_root)
        _admit_raw(archive_root)

    with running_daemon_operations(root, seed_archive=seed) as stack:
        envelope = stack.client.operation_to_completion(
            "mutation.session.mark",
            {"selection": {"params": {"list_mode": True}, "mode": mode}, "add_marks": ["star"]},
            archive_root=str(root),
        )
        assert envelope is not None
        assert envelope["outcome"] == "failed", envelope
        assert envelope["error"]["code"] == "selection_not_authoritative", envelope
        assert envelope["accepted_reference"] is None
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            assert list(archive.list_marks()) == []


def test_facets_without_a_prior_row_outcome_still_name_source_gap(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _admit_raw(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation("facets", {"params": {}}, archive=pinned.archive, serving_identity="direct")
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "degraded"
    assert outcome["reason"] == "archive_not_converged"


def test_canonical_receipt_keeps_raw_state_revision_on_its_original_frame(tmp_path: Path) -> None:
    from tests.unit.storage.test_raw_observation_derivation import _admit, _run, _run_raw_law

    def materialize(compute: Any) -> None:
        bootstrap_archive_root(tmp_path)
        _admit(tmp_path, ("neutral",), path="neutral.json")
        assert _run(tmp_path, compute_adapter=compute).done == 1

    _run_raw_law(tmp_path, materialize)
    prepare_operation_journals(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as user:
        query = put_query(
            user,
            {"kind": "field", "field": "origin", "op": "=", "values": ["chatgpt-export"]},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
    with open_operation_read(tmp_path) as pinned:
        evaluator = ArchiveCanonicalPlanEvaluator(pinned.archive)
        before = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))
        with sqlite3.connect(tmp_path / "source.db") as writer:
            writer.execute("UPDATE raw_sessions SET detection_warnings_json='[\"neutral changed note\"]'")
        held = evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))
        assert held.member_refs == before.member_refs
        assert held.receipt.source_generation == before.receipt.source_generation
    with open_operation_read(tmp_path) as fresh:
        changed = ArchiveCanonicalPlanEvaluator(fresh.archive).evaluate(
            QueryEvaluationRequest(query=query, purpose="reference")
        )
        assert changed.member_refs == before.member_refs
        assert changed.receipt.source_generation != before.receipt.source_generation
