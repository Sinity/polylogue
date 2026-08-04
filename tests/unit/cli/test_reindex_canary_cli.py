from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.maintenance import _reindex_canary as command_module
from polylogue.maintenance.reindex_canary import (
    CanaryDifferenceReview,
    CanaryDiffReport,
    CanaryRunResult,
    CanarySelection,
    DifferenceClassification,
    DifferenceOperation,
    DurableCanaryReport,
    RowDifference,
    UnclassifiedCanaryDiffError,
    run_reindex_canary,
)


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
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
        ],
    )

    assert result.exit_code == 2
    assert "requires --no-promote" in result.output


def test_reindex_canary_cli_routes_selection_and_report_without_rebuild_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    report_path = tmp_path / "reports" / "canary.json"
    run_result = _run_result(index_path)
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> CanaryRunResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return run_result

    def fake_write(path: Path, **kwargs: object) -> DurableCanaryReport:
        captured["report_path"] = path
        captured["report_kwargs"] = kwargs
        return DurableCanaryReport(
            selection=run_result.selection,
            comparison=run_result.comparison,
            rebuild_receipt=run_result.rebuild_receipt,
            reviews=(),
        )

    monkeypatch.setattr(command_module, "archive_root", lambda: tmp_path)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", fake_run)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.write_canary_report", fake_write)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda path: {})

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--input",
            str(index_path),
            "--sample",
            "2",
            "--pathology-session-id",
            "codex-session:pathology",
            "--sample-session-id",
            "codex-session:sample",
            "--report",
            str(report_path),
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["args"] == (tmp_path,)
    assert captured["kwargs"] == {
        "input_index": index_path,
        "sessions_per_origin": 2,
        "pathology_session_ids": ("codex-session:pathology",),
        "sample_session_ids": ("codex-session:sample",),
        "no_promote": True,
    }
    assert captured["report_path"] == report_path
    report_kwargs = captured["report_kwargs"]
    assert isinstance(report_kwargs, dict)
    assert report_kwargs["reviews"] == ()
    assert json.loads(result.stdout)["selection"]["selected_raw_ids"] == ["raw-sample"]


def test_reindex_canary_cli_refuses_to_write_unclassified_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    run_result = _run_result(index_path, differences=(object(),))
    monkeypatch.setattr(command_module, "archive_root", lambda: tmp_path)
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
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1
    assert "require --review-manifest" in result.output


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
    monkeypatch.setattr(command_module, "archive_root", lambda: tmp_path)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", lambda *args, **kwargs: run_result)

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
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


def test_reindex_canary_cli_refuses_nonempty_differences_without_review_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    monkeypatch.setattr(command_module, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.run_reindex_canary",
        lambda *args, **kwargs: _nonempty_run_result(index_path),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--input",
            str(index_path),
            "--report",
            str(tmp_path / "canary.json"),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1
    assert "require --review-manifest" in result.output


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
    assert result.rebuild_receipt == {"generation": Receipt.generation}
