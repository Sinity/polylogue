"""Focused real-generation tests for the reindex canary differ.

The production dependency is ``compare_reindex_generations`` reading two
canonical ``index.db`` files.  The anti-vacuity mutation for the core test is
changing the candidate block text: a synthetic summary comparator would stay
green, while the real blocks read model must report the changed row.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from polylogue.maintenance.reindex_canary import (
    CanaryDifferenceReview,
    CanaryDiffReport,
    CanarySelection,
    CanarySelectionError,
    DifferenceClassification,
    DifferenceOperation,
    ExpectedDifference,
    UnclassifiedCanaryDiffError,
    compare_reindex_generations,
    load_canary_report,
    run_reindex_canary,
    select_canary_sessions,
    write_canary_report,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.workload_artifacts import build_seeded_archive, clone_seeded_archive


def _seed_index(
    path: Path,
    *,
    sessions: tuple[str, ...] = ("alpha",),
    block_text: str = "stable transcript",
    profile_materialized_at: str = "first-run",
    profile_message_count: int = 1,
    origins: tuple[str, ...] | None = None,
) -> None:
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as connection:
        session_origins = origins or ("codex-session",) * len(sessions)
        assert len(session_origins) == len(sessions)
        for native_id, origin in zip(sessions, session_origins, strict=True):
            session_id = f"{origin}:{native_id}"
            connection.execute(
                """
                INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (native_id, origin, f"raw-{native_id}", native_id.encode().ljust(32, b"-")),
            )
            connection.execute(
                """
                INSERT INTO messages(session_id, position, role, material_origin, content_hash)
                VALUES (?, 0, 'user', 'human_authored', ?)
                """,
                (session_id, native_id.encode().ljust(32, b"m")),
            )
            connection.execute(
                """
                INSERT INTO blocks(message_id, session_id, position, block_type, text)
                VALUES (?, ?, 0, 'text', ?)
                """,
                (f"{session_id}:0.0", session_id, block_text),
            )
            connection.execute(
                """
                INSERT INTO session_profiles(session_id, materialized_at, message_count, tags_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, profile_materialized_at, profile_message_count, '{"b":2,"a":1}'),
            )
        connection.commit()


def _empty_selection(index_path: Path) -> CanarySelection:
    return CanarySelection(
        index_path=index_path,
        sessions_per_origin=1,
        selected_session_ids=(),
        selected_raw_ids=(),
        sampled_session_ids=(),
        pathology_session_ids=(),
        sample_session_ids=(),
        origin_counts=(),
    )


def _empty_comparison(current: Path, candidate: Path, session_ids: tuple[str, ...]) -> CanaryDiffReport:
    return CanaryDiffReport(
        current_index=current,
        candidate_index=candidate,
        session_ids=session_ids,
        compared_tables=(),
        missing_tables=(),
        missing_columns=(),
        differences=(),
    )


def test_equal_real_generations_ignore_only_materialization_metadata(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, profile_materialized_at="current-build")
    _seed_index(candidate, profile_materialized_at="candidate-build")

    report = compare_reindex_generations(current, candidate)

    assert report.differences == ()
    assert report.unclassified_count == 0
    assert {"sessions", "messages", "blocks", "session_profiles"}.issubset(report.compared_tables)


def test_differ_reports_real_core_and_derived_row_changes(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("alpha", "removed"))
    _seed_index(
        candidate,
        sessions=("alpha", "added"),
        block_text="changed transcript",
        profile_message_count=2,
    )

    report = compare_reindex_generations(current, candidate)

    assert report.unexpected_count > 0
    assert report.unclassified_count == 0
    operations = {(item.table, item.operation) for item in report.differences}
    assert ("blocks", DifferenceOperation.CHANGED) in operations
    assert ("blocks", DifferenceOperation.ADDED) in operations
    assert ("blocks", DifferenceOperation.REMOVED) in operations
    assert any(
        item.table == "session_profiles"
        and item.operation is DifferenceOperation.CHANGED
        and "message_count" in item.changed_columns
        for item in report.differences
    )
    assert all(item.classification is DifferenceClassification.UNEXPECTED for item in report.differences)


def test_missing_tables_and_columns_are_explicit_unexpected_differences(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    with sqlite3.connect(candidate) as connection:
        connection.execute("ALTER TABLE session_profiles DROP COLUMN tags_json")
        connection.execute("DROP TABLE blocks")
        connection.commit()

    report = compare_reindex_generations(current, candidate)

    assert report.missing_tables == ("blocks",)
    assert report.missing_columns == (("session_profiles", ("tags_json",)),)
    schema_differences = [item for item in report.differences if item.identity[0][0] == "__schema__"]
    assert {(item.table, item.operation) for item in schema_differences} == {
        ("blocks", DifferenceOperation.REMOVED),
        ("session_profiles", DifferenceOperation.REMOVED),
    }
    assert report.unexpected_count == len(report.differences)


def test_expected_difference_is_structurally_accounted_for(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="the reviewed materializer change updates this aggregate",
            ),
        ),
    )

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert profile_changes
    assert all(item.classification is DifferenceClassification.EXPECTED for item in profile_changes)
    assert all("polylogue-example" in item.rationale for item in profile_changes)
    assert report.expected_count == len(profile_changes)


def test_expected_difference_cannot_hide_extra_changed_columns(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)
    with sqlite3.connect(candidate) as connection:
        connection.execute("UPDATE session_profiles SET tags_json = ?", ('{"extra":true}',))
        connection.commit()

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="the reviewed materializer change updates this aggregate",
            ),
        ),
    )

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert len(profile_changes) == 1
    assert profile_changes[0].changed_columns == ("tags_json", "message_count")
    assert profile_changes[0].classification is DifferenceClassification.UNEXPECTED


def test_selected_sessions_bound_the_canary_to_a_real_subset(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("kept", "outside"))
    _seed_index(candidate, sessions=("kept", "outside"))
    with sqlite3.connect(candidate) as connection:
        connection.execute(
            "UPDATE blocks SET text = 'outside changed' WHERE session_id = ?",
            ("codex-session:outside",),
        )
        connection.commit()

    report = compare_reindex_generations(current, candidate, session_ids=("codex-session:kept",))

    assert report.session_ids == ("codex-session:kept",)
    assert report.differences == ()


def test_canary_comparison_is_read_only(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    before = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size

    compare_reindex_generations(current, candidate)

    after = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size
    assert after == before


def test_selector_samples_each_origin_and_keeps_explicit_inputs(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(
        index,
        sessions=("codex-a", "codex-pathology", "chat-a", "chat-sample", "claude-a", "claude-pathology"),
        origins=(
            "codex-session",
            "codex-session",
            "chatgpt-export",
            "chatgpt-export",
            "claude-ai-export",
            "claude-ai-export",
        ),
    )

    selection = select_canary_sessions(
        index,
        sessions_per_origin=1,
        pathology_session_ids=("codex-session:codex-pathology", "claude-ai-export:claude-pathology"),
        sample_session_ids=("chatgpt-export:chat-sample",),
    )

    assert selection.origin_counts == (
        ("chatgpt-export", 2),
        ("claude-ai-export", 2),
        ("codex-session", 2),
    )
    assert selection.selected_session_ids == (
        "chatgpt-export:chat-a",
        "chatgpt-export:chat-sample",
        "claude-ai-export:claude-a",
        "claude-ai-export:claude-pathology",
        "codex-session:codex-a",
        "codex-session:codex-pathology",
    )
    assert selection.selected_raw_ids == (
        "raw-chat-a",
        "raw-chat-sample",
        "raw-claude-a",
        "raw-claude-pathology",
        "raw-codex-a",
        "raw-codex-pathology",
    )


def test_selector_refuses_unknown_or_non_replayable_explicit_sessions(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(index)
    with pytest.raises(CanarySelectionError, match="not indexed"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:missing",))
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE sessions SET raw_id = NULL")
        connection.commit()
    with pytest.raises(CanarySelectionError, match="no raw_id"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:alpha",))


def test_run_reindex_canary_automatically_includes_production_pathology_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real canary runner supplements operator IDs from the production manifest."""
    from polylogue.maintenance.pathology_zoo import pathology_zoo_session_ids

    current = tmp_path / "current.db"
    pathology_session_ids = pathology_zoo_session_ids()
    native_ids = tuple(session_id.split(":", 1)[1] for session_id in pathology_session_ids)
    origins = tuple(session_id.split(":", 1)[0] for session_id in pathology_session_ids)
    _seed_index(current, sessions=native_ids, origins=origins)
    captured: dict[str, tuple[str, ...]] = {}

    class Receipt:
        def to_dict(self) -> dict[str, object]:
            return {"status": "replayed"}

    def fake_rebuild(request: object) -> Receipt:
        captured["raw_ids"] = tuple(request.raw_ids)  # type: ignore[attr-defined]
        captured["acceptance_checks"] = tuple(request.candidate_acceptance_checks)  # type: ignore[attr-defined]
        return Receipt()

    def fake_compare(current_index: Path, candidate_index: Path, *, session_ids: tuple[str, ...]) -> CanaryDiffReport:
        captured["session_ids"] = session_ids
        return _empty_comparison(current_index, candidate_index, session_ids)

    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", fake_rebuild)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_canary_candidate", lambda *args, **kwargs: current
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    result = run_reindex_canary(tmp_path, input_index=current, sessions_per_origin=1, no_promote=True)

    assert result.selection.pathology_session_ids == pathology_session_ids
    assert set(pathology_session_ids) <= set(captured["session_ids"])
    assert len(captured["raw_ids"]) == len(pathology_session_ids)
    assert captured["acceptance_checks"] == ("pathology-zoo-invariants",)


def test_run_reindex_canary_rejects_input_index_outside_archive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    external_index = tmp_path / "external" / "index.db"
    external_index.parent.mkdir()
    external_index.touch()
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: tmp_path / "configured-live")
    selector_called = False

    def unexpected_selector(*args: object, **kwargs: object) -> None:
        nonlocal selector_called
        selector_called = True
        raise AssertionError("an outside-root input must be rejected before selection")

    monkeypatch.setattr("polylogue.maintenance.reindex_canary.select_canary_sessions", unexpected_selector)

    with pytest.raises(CanarySelectionError, match="inside or bound to the selected archive root"):
        run_reindex_canary(root, input_index=external_index, no_promote=True)
    assert not selector_called


def test_run_reindex_canary_accepts_split_root_active_pointer_through_real_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "seeded-cache")
    root = clone_seeded_archive(artifact, tmp_path / "archive").root
    external_index_root = tmp_path / "external-index-root"
    external_index_root.mkdir()
    external_index = external_index_root / "index.db"
    shutil.move(root / "index.db", external_index)
    (root / ".index-active-pointer").write_text(str(external_index), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "configured-live"))

    result = run_reindex_canary(root, input_index=external_index, sessions_per_origin=1, no_promote=True)

    receipt = result.rebuild_receipt
    generation = receipt["generation"]
    assert isinstance(generation, dict)
    generation_id = generation["generation_id"]
    owner_id = generation["owner_id"]
    source_snapshot = generation["source_snapshot"]
    candidate_path = Path(str(generation["index_path"]))
    expected_candidate_path = external_index_root / ".index-generations" / str(generation_id) / "index.db"
    assert result.selection.index_path == external_index
    assert result.comparison.current_index.resolve() == external_index.resolve()
    assert result.comparison.candidate_index == candidate_path
    assert candidate_path == expected_candidate_path.resolve()
    assert candidate_path.is_file()
    assert generation["archive_root"] == str(root.resolve())
    assert generation["state"] == "inactive"
    assert isinstance(owner_id, str) and owner_id
    assert isinstance(source_snapshot, str) and source_snapshot
    assert json.loads((candidate_path.parent / "generation.json").read_text(encoding="utf-8")) == generation

    transaction = receipt["transaction"]
    operation = receipt["operation"]
    assert transaction is None
    assert isinstance(operation, dict)
    operation_owner = operation["owner"]
    operation_generation = operation["generation"]
    operation_delta = operation["delta"]
    assert isinstance(operation_owner, dict)
    assert isinstance(operation_generation, dict)
    assert isinstance(operation_delta, dict)
    assert operation_owner["generation_owner_id"] == owner_id
    assert operation_generation == {"generation_id": generation_id, "state": "inactive"}
    assert operation_delta["transaction_source_snapshot"] == source_snapshot
    assert operation_delta["source_snapshot_matches"] is True


def test_run_reindex_canary_does_not_require_zoo_sessions_for_ordinary_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "current.db"
    _seed_index(current)
    selection = CanarySelection(
        index_path=current,
        sessions_per_origin=1,
        selected_session_ids=("codex-session:alpha",),
        selected_raw_ids=("raw-alpha",),
        sampled_session_ids=("codex-session:alpha",),
        pathology_session_ids=(),
        sample_session_ids=(),
        origin_counts=(("codex-session", 1),),
    )
    captured: dict[str, tuple[str, ...]] = {}

    class Receipt:
        def to_dict(self) -> dict[str, object]:
            return {"status": "replayed"}

    def fake_rebuild(request: object) -> Receipt:
        captured["raw_ids"] = tuple(request.raw_ids)  # type: ignore[attr-defined]
        captured["acceptance_checks"] = tuple(request.candidate_acceptance_checks)  # type: ignore[attr-defined]
        return Receipt()

    def fake_compare(current_index: Path, candidate_index: Path, *, session_ids: tuple[str, ...]) -> CanaryDiffReport:
        captured["session_ids"] = session_ids
        return _empty_comparison(current_index, candidate_index, session_ids)

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", fake_rebuild)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_canary_candidate", lambda *args, **kwargs: current
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    result = run_reindex_canary(tmp_path, input_index=current, no_promote=True)

    assert result.selection.pathology_session_ids == ()
    assert captured["raw_ids"] == ("raw-alpha",)
    assert captured["acceptance_checks"] == ("pathology-zoo-invariants",)


def test_run_reindex_canary_compares_its_own_inactive_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "current.db"
    current.touch()
    generation_id = "gen-canary"
    candidate = tmp_path / ".index-generations" / generation_id / "index.db"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    root = tmp_path
    selection = _empty_selection(current)

    class Receipt:
        archive_root = str(root.resolve())
        selected_raw_count = len(selection.selected_raw_ids)
        status = "replayed"
        materialized = True
        generation = {
            "generation_id": generation_id,
            "owner_id": "owner",
            "archive_root": str(root.resolve()),
            "index_path": str(candidate),
            "state": "inactive",
            "source_snapshot": "snapshot",
        }

        def to_dict(self) -> dict[str, object]:
            return {"generation": self.generation}

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", lambda request: Receipt())

    def fake_compare(current_path: Path, candidate_path: Path, *, session_ids: tuple[str, ...]) -> CanaryDiffReport:
        captured.update({"paths": (current_path, candidate_path), "session_ids": session_ids})
        return _empty_comparison(current_path, candidate_path, session_ids)

    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    result = run_reindex_canary(root, input_index=current, no_promote=True)

    assert result.comparison.candidate_index == candidate
    assert captured["paths"] == (current, candidate)


def test_run_reindex_canary_rejects_arbitrary_sqlite_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "current.db"
    current.touch()
    arbitrary = tmp_path / "arbitrary.db"
    arbitrary.touch()
    selection = _empty_selection(current)

    class Receipt:
        archive_root = str(tmp_path.resolve())
        selected_raw_count = 0
        status = "replayed"
        materialized = True
        generation = {
            "generation_id": "gen-canary",
            "owner_id": "owner",
            "archive_root": str(tmp_path.resolve()),
            "index_path": str(arbitrary),
            "state": "inactive",
            "source_snapshot": "snapshot",
        }

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", lambda request: Receipt())

    with pytest.raises(CanarySelectionError, match="outside this archive's generation root"):
        run_reindex_canary(tmp_path, input_index=current, no_promote=True)


def test_run_reindex_canary_refuses_the_configured_live_archive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuild_called = False

    def _unexpected_rebuild(*args: object, **kwargs: object) -> None:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("live archive canary must refuse before rebuild")

    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: tmp_path)
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", _unexpected_rebuild)

    with pytest.raises(CanarySelectionError, match="refuses the configured live archive root"):
        run_reindex_canary(tmp_path, no_promote=True)
    assert not rebuild_called


def test_durable_report_refuses_unclassified_diffs(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)

    with pytest.raises(UnclassifiedCanaryDiffError, match="classification is incomplete"):
        write_canary_report(report_path, selection=selection, comparison=comparison, reviews=())
    assert not report_path.exists()


def test_durable_report_persists_explicit_review_for_every_diff(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )

    durable = write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)
    loaded = load_canary_report(report_path)

    assert durable.unclassified_count == 0
    assert report_path.exists()
    assert loaded["schema_version"] == 1
    comparison_payload = loaded["comparison"]
    assert isinstance(comparison_payload, dict)
    summary = comparison_payload["summary"]
    assert isinstance(summary, dict)
    assert summary["unclassified_count"] == 0
    assert summary["unexpected_count"] == len(reviews)


def test_loading_canary_report_rechecks_exact_review_coverage(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["reviews"] = payload["reviews"][:-1]
    payload["comparison"]["summary"] = {
        "difference_count": 0,
        "expected_count": 0,
        "unexpected_count": 0,
        "unclassified_count": 0,
        "counts_by_table": {},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match="review coverage is incomplete"):
        load_canary_report(report_path)


def test_loading_canary_report_recomputes_tampered_summary(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["comparison"]["summary"] = {
        "difference_count": 0,
        "expected_count": 0,
        "unexpected_count": 0,
        "unclassified_count": 0,
        "counts_by_table": {},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_canary_report(report_path)

    comparison_payload = loaded["comparison"]
    assert isinstance(comparison_payload, dict)
    summary = comparison_payload["summary"]
    assert isinstance(summary, dict)
    assert summary["difference_count"] == len(reviews)
    assert summary["unexpected_count"] == len(reviews)


def test_canary_report_uses_unique_temporary_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    original = tempfile.NamedTemporaryFile
    names: list[str] = []

    def recording_named_temporary_file(*args: Any, **kwargs: Any) -> Any:
        stream = original(*args, **kwargs)
        names.append(stream.name)
        return stream

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", recording_named_temporary_file)
    write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)
    write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)

    assert len(names) == 2
    assert len(set(names)) == 2
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def test_canary_report_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)

    assert not report_path.exists()
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []
