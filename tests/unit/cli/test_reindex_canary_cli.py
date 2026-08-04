from __future__ import annotations

import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.maintenance.reindex_canary import (
    CanaryDiffReport,
    CanaryRunResult,
    CanarySelection,
    UnclassifiedCanaryDiffError,
    run_reindex_canary,
)
from tests.infra.workload_artifacts import build_seeded_archive


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


def test_shared_canary_runner_uses_existing_inactive_rebuild_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_index = tmp_path / "current.db"
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
