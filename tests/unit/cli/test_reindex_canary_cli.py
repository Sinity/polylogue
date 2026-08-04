from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.maintenance.reindex_canary import (
    CanaryDifferenceReview,
    CanaryDiffReport,
    CanaryRunResult,
    CanarySelection,
    DifferenceClassification,
    DifferenceOperation,
    RowDifference,
    UnclassifiedCanaryDiffError,
    load_canary_report,
    run_reindex_canary,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.workload_artifacts import build_seeded_archive


def _codex_session(native_id: str) -> bytes:
    rows = (
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-08-04T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-user",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-assistant",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "world"}],
            },
        },
    )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed_isolated_canary(root: Path) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("isolated-canary"),
            source_path="isolated-canary.jsonl",
            acquired_at_ms=1,
        )
    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert receipt.status == "replayed"


def _run_result(index_path: Path, *, differences: tuple[object, ...] = ()) -> CanaryRunResult:
    selection = CanarySelection(
        index_path=index_path,
        sessions_per_origin=2,
        selected_session_ids=("codex-session:sample",),
        selected_raw_ids=("raw-sample",),
        sampled_session_ids=("codex-session:sample",),
        pathology_session_ids=("codex-session:pathology",),
        sample_session_ids=("codex-session:sample",),
        origin_counts=(("codex-session", 1),),
    )
    comparison = CanaryDiffReport(
        current_index=index_path,
        candidate_index=index_path.with_name("candidate.db"),
        session_ids=selection.selected_session_ids,
        compared_tables=("sessions",),
        missing_tables=(),
        missing_columns=(),
        differences=differences,  # type: ignore[arg-type]
    )
    return CanaryRunResult(selection=selection, comparison=comparison, rebuild_receipt={"status": "replayed"})


def _nonempty_run_result(index_path: Path) -> CanaryRunResult:
    difference = RowDifference(
        table="session_profiles",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "codex-session:sample"),),
        before={"message_count": 1},
        after={"message_count": 2},
        changed_columns=("message_count",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    result = _run_result(index_path, differences=(difference,))
    return CanaryRunResult(
        selection=result.selection,
        comparison=result.comparison,
        rebuild_receipt={
            "archive_root": str(index_path.parent),
            "selected_raw_count": 1,
            "selected_raw_ids": ["raw-sample"],
            "selected_session_ids": ["codex-session:sample"],
            "status": "replayed",
            "materialized": True,
            "generation": {
                "generation_id": "gen-canary",
                "owner_id": "owner",
                "archive_root": str(index_path.parent),
                "index_path": str(result.comparison.candidate_index),
                "state": "inactive",
                "source_snapshot": "snapshot",
            },
        },
    )


def test_reindex_canary_cli_requires_no_promote(tmp_path: Path) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
        ],
    )

    assert result.exit_code == 2
    assert "requires --no-promote" in result.output


def test_reindex_canary_cli_rejects_input_outside_archive_root_before_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    external_index = tmp_path / "external" / "index.db"
    external_index.parent.mkdir()
    external_index.touch()
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "configured-live"))
    rebuild_called = False

    def unexpected_rebuild(*args: object, **kwargs: object) -> None:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("the CLI must reject an outside-root input before rebuild")

    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", unexpected_rebuild)

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(archive_root),
            "--input",
            str(external_index),
            "--report",
            str(tmp_path / "canary.json"),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "inside or bound to the selected archive root" in result.output
    assert not rebuild_called


def test_reindex_canary_cli_runs_real_no_promote_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = build_seeded_archive(cache_root=tmp_path / "cache").root
    for path in (archive_root, *archive_root.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    index_path = archive_root / "index.db"
    report_path = tmp_path / "reports" / "canary.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "configured-live-root"))

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(archive_root),
            "--input",
            str(index_path),
            "--sample",
            "1000",
            "--report",
            str(report_path),
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1, result.output
    assert "canary report classification is incomplete" in result.output
    assert "refuses the configured live archive root" not in result.output
    assert not report_path.exists()


def test_reindex_canary_cli_refuses_to_write_unclassified_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    run_result = _run_result(index_path, differences=(object(),))
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", lambda *args, **kwargs: run_result)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.write_canary_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(UnclassifiedCanaryDiffError("classification is incomplete")),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1
    assert "classification is incomplete" in result.output


def test_reindex_canary_cli_persists_review_manifest_for_nonempty_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's real report writer accepts an explicit per-difference review."""

    index_path = tmp_path / "index.db"
    index_path.touch()
    report_path = tmp_path / "canary.json"
    review_path = tmp_path / "reviews.json"
    run_result = _nonempty_run_result(index_path)
    difference = run_result.comparison.differences[0]
    assert isinstance(difference, RowDifference)
    review = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="polylogue-review",
        rationale="reviewed materializer change",
    )
    review_path.write_text(json.dumps({"reviews": [review.to_dict()]}), encoding="utf-8")
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", lambda *args, **kwargs: run_result)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda path: {})

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--input",
            str(index_path),
            "--report",
            str(report_path),
            "--review-manifest",
            str(review_path),
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["reviews"] == [review.to_dict()]
    assert persisted["comparison"]["summary"]["expected_count"] == 1


def test_reindex_canary_cli_rejects_manifest_with_wrong_changed_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest coverage is exact down to the changed-column signature."""

    index_path = tmp_path / "index.db"
    index_path.touch()
    report_path = tmp_path / "canary.json"
    review_path = tmp_path / "reviews.json"
    run_result = _nonempty_run_result(index_path)
    difference = run_result.comparison.differences[0]
    assert isinstance(difference, RowDifference)
    review = CanaryDifferenceReview(
        table=difference.table,
        operation=difference.operation,
        identity=difference.identity,
        changed_columns=("different_column",),
        classification=DifferenceClassification.EXPECTED,
        reference="polylogue-review",
        rationale="wrong signature",
    )
    review_path.write_text(json.dumps({"reviews": [review.to_dict()]}), encoding="utf-8")
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", lambda *args, **kwargs: run_result)

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--input",
            str(index_path),
            "--report",
            str(report_path),
            "--review-manifest",
            str(review_path),
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "classification is incomplete" in result.output
    assert not report_path.exists()


def test_reindex_canary_cli_refuses_nonempty_differences_without_review_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.run_reindex_canary",
        lambda *args, **kwargs: _nonempty_run_result(index_path),
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda path: {})

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1
    assert "unreviewed canary report written" in result.output


def test_reindex_canary_cli_persists_unreviewed_real_candidate_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real inactive rebuild records its observed diff before CLI refusal."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / "isolated-canary"
    report_path = tmp_path / "unreviewed.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: live_root)
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE blocks SET text = 'mutated active projection'")

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(report_path),
            "--sample",
            "1",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "unreviewed canary report written" in result.output
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["review_status"] == "unreviewed"
    assert persisted["reviews"] == []
    assert persisted["comparison"]["differences"]
    assert persisted["comparison"]["candidate_index"] != persisted["comparison"]["current_index"]
    assert persisted["rebuild_receipt"]["generation"]["state"] == "inactive"

    persisted["review_status"] = "reviewed"
    persisted["comparison"]["differences"] = []
    persisted["comparison"]["summary"] = {
        "difference_count": 0,
        "expected_count": 0,
        "unexpected_count": 0,
        "unclassified_count": 0,
        "counts_by_table": {},
    }
    report_path.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match="comparison attestation"):
        load_canary_report(report_path)


def test_reindex_canary_cli_rejects_manifest_with_mismatched_changed_columns_from_real_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI review manifests must bind each real difference's column signature."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / "isolated-canary"
    observed_report_path = tmp_path / "unreviewed.json"
    rejected_report_path = tmp_path / "rejected.json"
    review_path = tmp_path / "reviews.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: live_root)
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE blocks SET text = 'mutated active projection'")

    observed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(observed_report_path),
            "--sample",
            "1",
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert observed.exit_code == 1
    observed_report = json.loads(observed_report_path.read_text(encoding="utf-8"))
    differences = observed_report["comparison"]["differences"]
    assert isinstance(differences, list) and differences
    reviews = [
        {
            "table": difference["table"],
            "operation": difference["operation"],
            "identity": difference["identity"],
            "changed_columns": difference["changed_columns"],
            "classification": "expected",
            "reference": "polylogue-review",
            "rationale": "reviewed materializer change",
        }
        for difference in differences
    ]
    reviews[0]["changed_columns"] = ["forged_column"]
    review_path.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")

    rejected = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(rejected_report_path),
            "--review-manifest",
            str(review_path),
            "--sample",
            "1",
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert rejected.exit_code == 1
    assert "classification is incomplete" in rejected.output
    assert not rejected_report_path.exists()


def test_shared_canary_runner_uses_existing_inactive_rebuild_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_index = tmp_path / "index.db"
    candidate_index = tmp_path / ".index-generations" / "gen-test" / "index.db"
    current_index.touch()
    candidate_index.parent.mkdir(parents=True)
    candidate_index.touch()
    selection = _run_result(current_index).selection
    captured: dict[str, object] = {}

    class Receipt:
        archive_root = str(tmp_path.resolve())
        selected_raw_count = len(selection.selected_raw_ids)
        status = "replayed"
        materialized = True
        generation = {
            "generation_id": "gen-test",
            "owner_id": "owner",
            "archive_root": str(tmp_path.resolve()),
            "index_path": str(candidate_index),
            "state": "inactive",
            "source_snapshot": "snapshot",
        }

        def to_dict(self) -> dict[str, object]:
            return {"generation": self.generation}

    def fake_rebuild(request: object) -> Receipt:
        captured["request"] = request
        return Receipt()

    def fake_compare(current: Path, candidate: Path, *, session_ids: tuple[str, ...]) -> CanaryDiffReport:
        captured["compare"] = (current, candidate, session_ids)
        return CanaryDiffReport(
            current_index=current,
            candidate_index=candidate,
            session_ids=session_ids,
            compared_tables=("sessions",),
            missing_tables=(),
            missing_columns=(),
            differences=(),
        )

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", fake_rebuild)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    result = run_reindex_canary(
        tmp_path,
        input_index=current_index,
        sessions_per_origin=2,
        no_promote=True,
    )

    request = captured["request"]
    assert request.raw_ids == selection.selected_raw_ids  # type: ignore[attr-defined]
    assert request.promote is False  # type: ignore[attr-defined]
    assert captured["compare"] == (current_index, candidate_index, selection.selected_session_ids)
    assert result.rebuild_receipt == {
        "generation": Receipt.generation,
        "selected_raw_ids": ["raw-sample"],
        "selected_session_ids": ["codex-session:sample"],
    }
