from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.core.enums import Provider
from polylogue.maintenance import reindex_canary as reindex_canary_module
from polylogue.maintenance.rebuild_index import (
    RebuildIndexRequest,
    rebuild_index_from_source_sync,
    rebuild_selection_evidence,
)
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
    load_canary_report,
    run_reindex_canary,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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


def _seed_isolated_canary(root: Path, *, session_names: tuple[str, ...] = ("isolated-canary",)) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for acquired_at_ms, name in enumerate(session_names, start=1):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(name),
                source_path=f"{name}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert receipt.status == "replayed"


def _write_real_unreviewed_canary_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "isolated-canary",
    session_names: tuple[str, ...] = ("isolated-canary",),
    sample: int = 1,
) -> tuple[Path, Path, dict[str, object]]:
    """Exercise the CLI to produce a report bound to one disposable archive."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / name
    report_path = tmp_path / f"{name}.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root, session_names=session_names)
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
            "--archive-root",
            str(canary_root),
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(report_path),
            "--sample",
            str(sample),
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "unreviewed canary report written" in result.output
    return canary_root, report_path, json.loads(report_path.read_text(encoding="utf-8"))


def test_real_report_preserves_receipt_and_independent_source_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary_root, report_path, payload = _write_real_unreviewed_canary_report(tmp_path, monkeypatch)
    receipt = payload["rebuild_receipt"]
    provenance = payload["archive_provenance"]
    assert isinstance(receipt, dict)
    assert isinstance(provenance, dict)
    assert {
        "archive_root",
        "receipt_schema_version",
        "source_evidence_after",
        "selected_raw_count",
        "selection_evidence",
        "status",
        "materialized",
        "generation",
        "transaction",
        "operation",
    } <= receipt.keys()
    generation = receipt["generation"]
    assert isinstance(generation, dict)
    assert {
        "generation_id",
        "owner_id",
        "archive_root",
        "index_path",
        "state",
        "source_snapshot",
    } <= generation.keys()
    assert generation["state"] == "inactive"
    assert provenance["archive_root"] == str(canary_root.resolve())
    assert provenance["candidate_generation"] == generation
    assert provenance["source_snapshot"] == generation["source_snapshot"]
    assert provenance["source_evidence_after"] == receipt["source_evidence_after"]
    candidate_path = Path(str(generation["index_path"]))
    assert json.loads((candidate_path.parent / "rebuild-receipt.json").read_text(encoding="utf-8")) == receipt
    assert report_path.is_file()


def test_cli_rejects_same_count_selected_raw_id_swap_before_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary_root, report_path, payload = _write_real_unreviewed_canary_report(
        tmp_path,
        monkeypatch,
        name="selection-swap",
        session_names=("first", "second", "third"),
        sample=2,
    )
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selected_raw_ids = selection["selected_raw_ids"]
    assert isinstance(selected_raw_ids, list) and len(selected_raw_ids) == 2
    with sqlite3.connect(canary_root / "source.db") as connection:
        source_raw_ids = [str(row[0]) for row in connection.execute("SELECT raw_id FROM raw_sessions")]
    replacement = next(raw_id for raw_id in source_raw_ids if raw_id not in selected_raw_ids)
    selection["selected_raw_ids"] = [selected_raw_ids[0], replacement]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    comparison_called = False

    def fail_if_compared(*args: object, **kwargs: object) -> object:
        nonlocal comparison_called
        comparison_called = True
        raise AssertionError("selection swap reached the SQLite comparator")

    monkeypatch.setattr(reindex_canary_module, "compare_reindex_generations", fail_if_compared)
    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--report",
            str(report_path),
            "--consume-report",
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert consumed.exit_code == 1
    assert "selection does not match the authoritative rebuild receipt" in consumed.output
    assert not comparison_called


def test_cli_rejects_swapped_real_receipt_before_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_root, first_report, first_payload = _write_real_unreviewed_canary_report(
        tmp_path, monkeypatch, name="first-canary"
    )
    _second_root, _second_report, second_payload = _write_real_unreviewed_canary_report(
        tmp_path, monkeypatch, name="second-canary"
    )
    first_payload["rebuild_receipt"] = second_payload["rebuild_receipt"]
    first_report.write_text(json.dumps(first_payload), encoding="utf-8")
    comparison_called = False

    def fail_if_compared(*args: object, **kwargs: object) -> object:
        nonlocal comparison_called
        comparison_called = True
        raise AssertionError("swapped receipt reached the SQLite comparator")

    monkeypatch.setattr(reindex_canary_module, "compare_reindex_generations", fail_if_compared)
    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(first_root),
            "--report",
            str(first_report),
            "--consume-report",
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert consumed.exit_code == 1
    assert "does not identify the compared candidate" in consumed.output
    assert not comparison_called


def test_cli_rejects_real_receipt_identity_forgery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canary_root, report_path, payload = _write_real_unreviewed_canary_report(tmp_path, monkeypatch)
    receipt = payload["rebuild_receipt"]
    provenance = payload["archive_provenance"]
    assert isinstance(receipt, dict)
    assert isinstance(provenance, dict)
    generation = receipt["generation"]
    assert isinstance(generation, dict)

    forged_reports = {
        "root": {"archive_provenance": {**provenance, "archive_root": str(tmp_path / "foreign-root")}},
        "generation": {"generation": {**generation, "generation_id": "gen-forged"}},
        "owner": {"generation": {**generation, "owner_id": "owner-forged"}},
        "candidate-path": {"generation": {**generation, "index_path": str(tmp_path / "foreign.db")}},
        "state": {"generation": {**generation, "state": "active"}},
    }
    for name, changes in forged_reports.items():
        forged = json.loads(json.dumps(payload))
        assert isinstance(forged, dict)
        forged_receipt = forged["rebuild_receipt"]
        forged_provenance = forged["archive_provenance"]
        assert isinstance(forged_receipt, dict)
        assert isinstance(forged_provenance, dict)
        archive_change = changes.get("archive_provenance")
        if isinstance(archive_change, dict):
            forged_provenance.update(archive_change)
        else:
            forged_generation = changes["generation"]
            assert isinstance(forged_generation, dict)
            forged_receipt["generation"] = forged_generation
        forged_path = tmp_path / f"forged-{name}.json"
        forged_path.write_text(json.dumps(forged), encoding="utf-8")
        consumed = CliRunner().invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "reindex-canary",
                "--archive-root",
                str(canary_root),
                "--report",
                str(forged_path),
                "--consume-report",
                "--no-promote",
            ],
            catch_exceptions=False,
        )
        assert consumed.exit_code == 1, name
        assert (
            "archive-owned" in consumed.output
            or "rebuild receipt" in consumed.output
            or "canary report belongs" in consumed.output
            or "candidate generation provenance" in consumed.output
        ), (
            name,
            consumed.output,
        )


def test_cli_consumes_valid_reviewed_real_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canary_root, _observed_path, observed = _write_real_unreviewed_canary_report(
        tmp_path, monkeypatch, name="valid-canary"
    )
    comparison = observed["comparison"]
    assert isinstance(comparison, dict)
    differences = comparison["differences"]
    assert isinstance(differences, list) and differences
    review_path = tmp_path / "valid-reviews.json"
    review_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": difference["table"],
                        "operation": difference["operation"],
                        "identity": difference["identity"],
                        "changed_columns": difference["changed_columns"],
                        "classification": "expected",
                        "reference": "polylogue-review",
                        "rationale": "reviewed real canary difference",
                    }
                    for difference in differences
                ]
            }
        ),
        encoding="utf-8",
    )
    approved_path = tmp_path / "approved.json"
    generated = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(approved_path),
            "--review-manifest",
            str(review_path),
            "--sample",
            "1",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert generated.exit_code == 0, generated.output

    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--report",
            str(approved_path),
            "--consume-report",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert consumed.exit_code == 0, consumed.output
    approved = json.loads(consumed.stdout)
    assert approved["decision"] == "evidence-approved"
    assert approved["promotion_authorized"] is False


def test_cli_consumes_reviewed_report_after_parsed_state_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parser bookkeeping may evolve after replay without changing source evidence."""

    canary_root, _observed_path, observed = _write_real_unreviewed_canary_report(
        tmp_path, monkeypatch, name="parsed-state-canary"
    )
    comparison = observed["comparison"]
    assert isinstance(comparison, dict)
    differences = comparison["differences"]
    assert isinstance(differences, list) and differences
    review_path = tmp_path / "parsed-state-reviews.json"
    review_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": difference["table"],
                        "operation": difference["operation"],
                        "identity": difference["identity"],
                        "changed_columns": difference["changed_columns"],
                        "classification": "expected",
                        "reference": "polylogue-review",
                        "rationale": "reviewed real canary difference",
                    }
                    for difference in differences
                ]
            }
        ),
        encoding="utf-8",
    )
    approved_path = tmp_path / "parsed-state-approved.json"
    generated = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--input",
            str(canary_root / "index.db"),
            "--report",
            str(approved_path),
            "--review-manifest",
            str(review_path),
            "--sample",
            "1",
            "--no-promote",
        ],
        catch_exceptions=False,
    )
    assert generated.exit_code == 0, generated.output

    with sqlite3.connect(canary_root / "source.db") as connection:
        connection.execute("UPDATE raw_sessions SET parsed_at_ms = COALESCE(parsed_at_ms, 0) + 1")

    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--report",
            str(approved_path),
            "--consume-report",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert consumed.exit_code == 0, consumed.output
    approved = json.loads(consumed.stdout)
    assert approved["decision"] == "evidence-approved"
    assert approved["promotion_authorized"] is False


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
    return CanaryRunResult(
        selection=selection,
        comparison=comparison,
        rebuild_receipt={
            "receipt_schema_version": 3,
            "archive_root": str(index_path.parent),
            "selected_raw_count": len(selection.selected_raw_ids),
            "status": "replayed",
            "materialized": True,
            "generation": {
                "generation_id": "gen-sample",
                "owner_id": "owner-sample",
                "archive_root": str(index_path.parent),
                "index_path": str(comparison.candidate_index),
                "state": "inactive",
                "source_snapshot": "snapshot",
            },
            "selection_evidence": rebuild_selection_evidence(
                selection.selected_raw_ids,
                archive_root=index_path.parent,
                generation_id="gen-sample",
                generation_owner_id="owner-sample",
                candidate_index=comparison.candidate_index,
                source_snapshot="snapshot",
            ),
            "source_evidence_after": "0" * 64,
        },
    )


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
            "receipt_schema_version": 3,
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
            "selection_evidence": rebuild_selection_evidence(
                result.selection.selected_raw_ids,
                archive_root=index_path.parent,
                generation_id="gen-canary",
                generation_owner_id="owner",
                candidate_index=result.comparison.candidate_index,
                source_snapshot="snapshot",
            ),
            "source_evidence_after": "0" * 64,
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
    archive_root = tmp_path / "isolated-canary"
    _seed_isolated_canary(archive_root)
    with sqlite3.connect(archive_root / "index.db") as connection:
        connection.execute("UPDATE blocks SET text = 'mutated active projection'")
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
    assert "unreviewed canary report written" in result.output
    assert "refuses the configured live archive root" not in result.output
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["review_status"] == "unreviewed"
    assert payload["rebuild_receipt"]["generation"]["state"] == "inactive"


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
    """The CLI forwards an explicit per-difference review to report writing."""

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
    captured: dict[str, object] = {}

    def fake_write(path: Path, **kwargs: object) -> DurableCanaryReport:
        captured["path"] = path
        captured["reviews"] = kwargs["reviews"]
        return DurableCanaryReport(
            selection=run_result.selection,
            comparison=run_result.comparison,
            rebuild_receipt=run_result.rebuild_receipt,
            reviews=(review,),
            review_status="reviewed",
            comparison_fingerprint="0" * 64,
            archive_provenance={},
        )

    review_path.write_text(json.dumps({"reviews": [review.to_dict()]}), encoding="utf-8")
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", lambda *args, **kwargs: run_result)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.write_canary_report", fake_write)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda path, **kwargs: {})

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
    assert captured == {"path": report_path, "reviews": (review,)}
    assert json.loads(result.stdout)["reviews"] == [review.to_dict()]


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
    run_result = _nonempty_run_result(index_path)

    def fake_write(path: Path, **kwargs: object) -> DurableCanaryReport:
        return DurableCanaryReport(
            selection=run_result.selection,
            comparison=run_result.comparison,
            rebuild_receipt=run_result.rebuild_receipt,
            reviews=(),
            review_status="unreviewed",
            comparison_fingerprint="0" * 64,
            archive_provenance={},
        )

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.run_reindex_canary",
        lambda *args, **kwargs: run_result,
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.write_canary_report", fake_write)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda path, **kwargs: {})

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
            "--archive-root",
            str(canary_root),
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
        load_canary_report(report_path, archive_root=canary_root)


def test_cli_canary_report_red_twin_rejects_arbitrary_copied_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied pair must not become an approved archive just by rewriting JSON."""

    canary_root, report_path, payload = _write_real_unreviewed_canary_report(tmp_path, monkeypatch)
    comparison = payload["comparison"]
    receipt = payload["rebuild_receipt"]
    selection = payload["selection"]
    assert isinstance(comparison, dict)
    assert isinstance(receipt, dict)
    assert isinstance(selection, dict)
    generation = receipt["generation"]
    assert isinstance(generation, dict)
    original_candidate = Path(str(comparison["candidate_index"]))

    copied_root = tmp_path / "copied-archive"
    copied_candidate = copied_root / "unowned-copy" / "index.db"
    copied_current = copied_root / "index.db"
    copied_candidate.parent.mkdir(parents=True)
    shutil.copy2(canary_root / "source.db", copied_root / "source.db")
    shutil.copy2(canary_root / "index.db", copied_current)
    shutil.copy2(original_candidate, copied_candidate)

    comparison["current_index"] = str(copied_current)
    comparison["candidate_index"] = str(copied_candidate)
    selection["index_path"] = str(copied_current)
    receipt["archive_root"] = str(copied_root)
    generation["archive_root"] = str(copied_root)
    generation["index_path"] = str(copied_candidate)
    payload["comparison_fingerprint"] = reindex_canary_module._comparison_fingerprint(
        reindex_canary_module.compare_reindex_generations(
            copied_current,
            copied_candidate,
            session_ids=tuple(selection["selected_session_ids"]),
        )
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--report",
            str(report_path),
            "--consume-report",
            "--no-promote",
        ],
        catch_exceptions=False,
    )
    assert consumed.exit_code == 1
    assert "archive-owned" in consumed.output or "authoritative rebuild receipt" in consumed.output


def test_cli_canary_report_red_twin_rejects_replaced_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the same candidate path after review must invalidate the report."""

    _canary_root, report_path, payload = _write_real_unreviewed_canary_report(tmp_path, monkeypatch)
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    candidate = Path(str(comparison["candidate_index"]))
    replacement = tmp_path / "replacement.db"
    shutil.copy2(candidate, replacement)
    replacement.replace(candidate)

    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(_canary_root),
            "--report",
            str(report_path),
            "--consume-report",
            "--no-promote",
        ],
        catch_exceptions=False,
    )
    assert consumed.exit_code == 1
    assert "candidate index identity" in consumed.output


@pytest.mark.parametrize(
    "drift",
    (
        "active-pointer",
        "candidate-generation",
        "source-byte",
        "source-blob-ref",
        "source-observation",
        "source-snapshot",
    ),
)
def test_cli_canary_report_red_twin_rejects_lifecycle_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    """The report is invalid when live pointer or generation metadata no longer match."""

    canary_root, report_path, payload = _write_real_unreviewed_canary_report(tmp_path, monkeypatch)
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    if drift == "active-pointer":
        alternate = canary_root / "alternate-generation" / "index.db"
        alternate.parent.mkdir()
        shutil.copy2(canary_root / "index.db", alternate)
        (canary_root / ".index-active-pointer").write_text(str(alternate), encoding="utf-8")
    elif drift == "candidate-generation":
        candidate_metadata = Path(str(comparison["candidate_index"])).with_name("generation.json")
        metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
        metadata["owner_id"] = "replaced-owner"
        candidate_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    elif drift == "source-byte":
        with sqlite3.connect(canary_root / "source.db") as connection:
            connection.execute("UPDATE raw_sessions SET blob_hash = zeroblob(length(blob_hash))")
    elif drift == "source-blob-ref":
        with sqlite3.connect(canary_root / "source.db") as connection:
            raw_id = connection.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1").fetchone()[0]
            connection.execute(
                """
                UPDATE blob_refs
                SET acquired_at_ms = acquired_at_ms + 1
                WHERE ref_type = 'raw_payload' AND ref_id = ?
                """,
                (raw_id,),
            )
    elif drift == "source-observation":
        with sqlite3.connect(canary_root / "source.db") as connection:
            raw_id = connection.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1").fetchone()[0]
            connection.execute(
                """
                INSERT INTO raw_capture_observations (raw_id, capture_mode, first_observed_at_ms)
                VALUES (?, 'gemini', 999)
                """,
                (raw_id,),
            )
    else:
        with sqlite3.connect(canary_root / "source.db") as connection:
            connection.execute("UPDATE raw_sessions SET acquired_at_ms = acquired_at_ms + 1")

    consumed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(canary_root),
            "--report",
            str(report_path),
            "--consume-report",
            "--no-promote",
        ],
        catch_exceptions=False,
    )
    assert consumed.exit_code == 1
    assert "archive-owned" in consumed.output


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
            "--archive-root",
            str(canary_root),
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
            "--archive-root",
            str(canary_root),
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
        selection_evidence = rebuild_selection_evidence(
            selection.selected_raw_ids,
            archive_root=tmp_path,
            generation_id="gen-test",
            generation_owner_id="owner",
            candidate_index=candidate_index,
            source_snapshot="snapshot",
        )

        def to_dict(self) -> dict[str, object]:
            return {
                "archive_root": self.archive_root,
                "selected_raw_count": self.selected_raw_count,
                "status": self.status,
                "materialized": self.materialized,
                "generation": self.generation,
                "selection_evidence": self.selection_evidence,
            }

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
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_authoritative_rebuild_receipt", lambda *args, **kwargs: None
    )

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
        "archive_root": Receipt.archive_root,
        "selected_raw_count": Receipt.selected_raw_count,
        "status": Receipt.status,
        "materialized": Receipt.materialized,
        "generation": Receipt.generation,
        "selection_evidence": Receipt.selection_evidence,
    }
