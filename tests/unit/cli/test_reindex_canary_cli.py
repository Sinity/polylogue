from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, cast

import pytest
from click import Command
from click.testing import CliRunner as _ClickCliRunner
from click.testing import Result

import polylogue.cli.commands.maintenance._reindex_canary as reindex_canary_cli
from polylogue.cli.click_app import cli
from polylogue.core.enums import Provider
from polylogue.maintenance import reindex_canary as reindex_canary_module
from polylogue.maintenance.archive_verification import (
    REINDEX_CANARY_ACCEPTANCE_CHECKS,
    REINDEX_CANARY_ACCEPTANCE_PROFILE,
)
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
    run_reindex_canary,
)
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import clone_archive_template, finalize_archive_template
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

_DEFAULT_CANARY_TEMPLATE: Path | None = None


@pytest.fixture(autouse=True)
def _run_cli_canary_tests_through_real_rebuild_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI report tests exercise the real rebuild service while transport is tested separately."""

    def rebuild_for_canary(
        *,
        archive_root: Path,
        raw_ids: tuple[str, ...],
        selected_session_ids: tuple[str, ...],
        index_schema_version: int,
        schema_inference_receipt_path: Path,
        message_owner_scope_backfill_receipt_path: Path | None = None,
    ) -> object:
        return rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=archive_root,
                raw_ids=raw_ids,
                selected_session_ids=selected_session_ids,
                promote=False,
                canary=True,
                schema_inference_receipt_path=schema_inference_receipt_path,
                message_owner_scope_backfill_receipt_path=message_owner_scope_backfill_receipt_path,
            )
        )

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", rebuild_for_canary)

    def consume_for_cli(*, archive_root: Path, report_path: Path) -> dict[str, object]:
        from polylogue.maintenance.reindex_canary import approve_canary_report

        return approve_canary_report(report_path, archive_root=archive_root)

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.consume_daemon_canary_report", consume_for_cli)


def _schema_receipt_path(root: Path) -> Path:
    return root.parent / f"{root.name}-schema-inference-gate-receipt.json"


@pytest.fixture(scope="module", autouse=True)
def _default_canary_template(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    """Build the default real archive once, then clone it for each canary test."""
    global _DEFAULT_CANARY_TEMPLATE
    template = tmp_path_factory.mktemp("reindex-canary-template") / "archive"
    _seed_isolated_canary(template)
    finalize_archive_template(template)
    _DEFAULT_CANARY_TEMPLATE = template
    try:
        yield
    finally:
        _DEFAULT_CANARY_TEMPLATE = None


def _rebase_canary_template(template: Path, root: Path) -> None:
    """Point copied generation metadata and links at this test's private clone."""
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = path.readlink()
        if target.is_absolute() and target.is_relative_to(template):
            target_is_directory = target.is_dir()
            path.unlink()
            path.symlink_to(root / target.relative_to(template), target_is_directory=target_is_directory)
    pointer = root / ".index-active-pointer"
    if pointer.is_file():
        target = Path(pointer.read_text(encoding="utf-8").strip())
        if target.is_absolute() and target.is_relative_to(template):
            pointer.write_text(str(root / target.relative_to(template)), encoding="utf-8")
    for metadata_path in root.glob(".index-generations/gen-*/generation.json"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["archive_root"] = str(root.resolve())
        index_path = Path(str(payload["index_path"]))
        if index_path.is_relative_to(template):
            payload["index_path"] = str(root / index_path.relative_to(template))
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clone_default_canary_template(root: Path) -> bool:
    template = _DEFAULT_CANARY_TEMPLATE
    if template is None:
        return False
    clone_archive_template(template, root)
    _rebase_canary_template(template, root)
    write_valid_rebuild_receipt(root, _schema_receipt_path(root))
    return True


class _CanaryCliRunner(_ClickCliRunner):
    """Supply the fixture's explicit receipt to legacy route invocations."""

    def invoke(
        self,
        cli: Command,
        args: str | Sequence[str] | None = None,
        input: str | bytes | IO[Any] | None = None,
        env: Mapping[str, str | None] | None = None,
        catch_exceptions: bool | None = None,
        color: bool = False,
        **extra: Any,
    ) -> Result:
        command_args = list(args) if args is not None and not isinstance(args, str) else args
        if (
            isinstance(command_args, list)
            and "reindex-canary" in command_args
            and "--consume-report" not in command_args
            and "--schema-inference-receipt" not in command_args
            and "--archive-root" in command_args
        ):
            archive_root = Path(command_args[command_args.index("--archive-root") + 1])
            receipt_path = _schema_receipt_path(archive_root)
            if receipt_path.is_file():
                command_args.extend(("--schema-inference-receipt", str(receipt_path)))
        return super().invoke(
            cli,
            command_args,
            input=input,
            env=env,
            catch_exceptions=catch_exceptions,
            color=color,
            **extra,
        )


CliRunner = _CanaryCliRunner


def test_cli_consumption_dispatches_to_daemon_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command uses the production daemon-client seam for report consumption."""

    report_path = tmp_path / "report.json"
    report_path.touch()
    calls: dict[str, Path] = {}

    def consume(*, archive_root: Path, report_path: Path) -> dict[str, object]:
        calls["archive_root"] = archive_root
        calls["report_path"] = report_path
        return {"review_status": "reviewed"}

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.consume_daemon_canary_report", consume)
    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "reindex-canary",
            "--archive-root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--consume-report",
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert calls == {"archive_root": tmp_path, "report_path": report_path}
    assert json.loads(result.stdout)["decision"] == "evidence-approved"


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


def _seed_isolated_canary(
    root: Path,
    *,
    session_names: tuple[str, ...] = ("isolated-canary",),
    membership_names: tuple[str, ...] = (),
) -> None:
    if session_names == ("isolated-canary",) and not membership_names and _clone_default_canary_template(root):
        return
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for acquired_at_ms, name in enumerate(session_names, start=1):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(name),
                source_path=f"{name}.jsonl",
                acquired_at_ms=acquired_at_ms,
            )
    if membership_names:
        with sqlite3.connect(root / "source.db") as connection:
            for name in membership_names:
                raw_id, blob_hash = connection.execute(
                    "SELECT raw_id, blob_hash FROM raw_sessions WHERE source_path = ?",
                    (f"{name}.jsonl",),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO raw_session_memberships(
                        raw_id, logical_source_key, provider_session_id, source_revision,
                        normalized_content_hash, message_count, predecessor_raw_id,
                        acquisition_generation, revision_authority, decision, decided_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 'quarantined', NULL, NULL)
                    """,
                    (raw_id, f"codex-session:{name}", name, "1", blob_hash, 2),
                )
            connection.commit()
    backfill_historical_revision_evidence(root)
    receipt_path = write_valid_rebuild_receipt(root, _schema_receipt_path(root))
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=receipt_path)
    )
    assert receipt.status == "replayed"
    backfill_historical_revision_evidence(root)
    # The final backfill can change durable source revision evidence. Refresh
    # the fixture receipt only after that mutation so later canary routes see
    # the same source snapshot the production preflight validates.
    write_valid_rebuild_receipt(root, receipt_path)


def _write_real_unreviewed_canary_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "isolated-canary",
    session_names: tuple[str, ...] = ("isolated-canary",),
    sample: int = 1,
) -> tuple[Path, Path, dict[str, object]]:
    """Exercise the CLI to produce a fully classified real canary report."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / name
    report_path = tmp_path / f"{name}.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root, session_names=session_names)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: live_root)
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE sessions SET title_ref = NULL, title_confidence = NULL")

    receipt_path = _schema_receipt_path(canary_root)
    observed_result = run_reindex_canary(
        canary_root,
        input_index=canary_root / "index.db",
        sessions_per_origin=sample,
        sample_session_ids=(),
        no_promote=True,
        schema_inference_receipt_path=receipt_path,
    )
    review_path = tmp_path / f"{name}-reviews.json"
    review_path.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": difference.table,
                        "operation": difference.operation.value,
                        "identity": dict(difference.identity),
                        "changed_columns": list(difference.changed_columns),
                        "classification": "unexpected",
                        "reference": "successor:polylogue-ox2iz",
                        "authority": {"kind": "successor", "id": "polylogue-ox2iz"},
                        "rationale": "reviewed real canary difference",
                    }
                    for difference in observed_result.comparison.differences
                ]
            }
        ),
        encoding="utf-8",
    )

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
            "--review-manifest",
            str(review_path),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["review_status"] == "reviewed"
    return canary_root, report_path, json.loads(report_path.read_text(encoding="utf-8"))


def _write_review_manifest(path: Path, differences: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        **difference,
                        "classification": "unexpected",
                        "reference": "successor:polylogue-ox2iz",
                        "authority": {"kind": "successor", "id": "polylogue-ox2iz"},
                        "rationale": "reviewed real canary difference",
                    }
                    for difference in differences
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_reviewed_real_canary_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    session_names: tuple[str, ...] = ("isolated-canary",),
) -> tuple[Path, Path, dict[str, object]]:
    canary_root, _observed_path, observed = _write_real_unreviewed_canary_report(
        tmp_path, monkeypatch, name=name, session_names=session_names
    )
    comparison = observed["comparison"]
    assert isinstance(comparison, dict)
    differences = comparison["differences"]
    assert isinstance(differences, list) and differences
    review_path = tmp_path / f"{name}-reviews.json"
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
                        "reference": "delta:44",
                        "authority": {"kind": "delta", "id": "44"},
                        "rationale": "reviewed v44 title-reprocess difference",
                    }
                    for difference in differences
                ]
            }
        ),
        encoding="utf-8",
    )
    approved_path = tmp_path / f"{name}-approved.json"
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
    return canary_root, approved_path, json.loads(approved_path.read_text(encoding="utf-8"))


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
    assert (
        "selection evidence cannot be recomputed" in consumed.output
        or "selection does not match the authoritative rebuild receipt" in consumed.output
    )
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
    assert (
        "does not identify the compared candidate" in consumed.output
        or "different configured archive root" in consumed.output
    )
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
                        "reference": "delta:44",
                        "authority": {"kind": "delta", "id": "44"},
                        "rationale": "reviewed v44 title-reprocess difference",
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


def test_cli_rejects_foreign_receipt_root_before_opening_foreign_source_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forged report root fails before selection evidence can read a foreign source tier."""

    canary_root, report_path, _payload = _write_reviewed_real_canary_report(
        tmp_path, monkeypatch, name="foreign-root-canary"
    )
    foreign_root = tmp_path / "foreign-archive"
    foreign_root.mkdir()
    (foreign_root / "source.db").touch()
    forged = json.loads(report_path.read_text(encoding="utf-8"))
    receipt = forged["rebuild_receipt"]
    provenance = forged["archive_provenance"]
    assert isinstance(receipt, dict)
    assert isinstance(provenance, dict)
    generation = receipt["generation"]
    assert isinstance(generation, dict)
    receipt["archive_root"] = str(foreign_root)
    generation["archive_root"] = str(foreign_root)
    provenance["archive_root"] = str(foreign_root)
    forged_path = tmp_path / "foreign-root-forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("foreign source.db was opened"),
    )
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

    assert consumed.exit_code == 1
    assert "different configured archive root" in consumed.output


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
                        "reference": "delta:44",
                        "authority": {"kind": "delta", "id": "44"},
                        "rationale": "reviewed v44 title-reprocess difference",
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


def test_cli_rejects_membership_and_logical_key_expansion_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new durable cohort member invalidates the original replay closure."""

    canary_root = tmp_path / "closure-drift-canary"
    report_path = tmp_path / "closure-drift-canary.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(
        canary_root,
        session_names=("selected", "unselected"),
    )
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: tmp_path / "configured-live")
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE sessions SET title_ref = NULL, title_confidence = NULL")
    observed_result = run_reindex_canary(
        canary_root,
        input_index=canary_root / "index.db",
        sessions_per_origin=1,
        no_promote=True,
        schema_inference_receipt_path=_schema_receipt_path(canary_root),
    )
    review_path = tmp_path / "membership-reviews.json"
    _write_review_manifest(review_path, [difference.to_dict() for difference in observed_result.comparison.differences])
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    for review in review_payload["reviews"]:
        review["classification"] = "expected"
        review["reference"] = "delta:44"
        review["authority"] = {"kind": "delta", "id": "44"}
        review["rationale"] = "reviewed v44 title-reprocess difference"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
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
            str(report_path),
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
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selected_raw_ids = selection["selected_raw_ids"]
    assert isinstance(selected_raw_ids, list) and len(selected_raw_ids) == 1
    with sqlite3.connect(canary_root / "source.db") as connection:
        raw_ids = [str(row[0]) for row in connection.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id")]
        selected_raw_id = str(selected_raw_ids[0])
        unselected_raw_id = next(raw_id for raw_id in raw_ids if raw_id != selected_raw_id)
        selected_blob_hash = connection.execute(
            "SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (selected_raw_id,)
        ).fetchone()[0]
        connection.executemany(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, predecessor_raw_id,
                acquisition_generation, revision_authority, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'quarantined', NULL, NULL)
            """,
            [
                (
                    selected_raw_id,
                    "codex-session:selected",
                    "selected",
                    "1",
                    selected_blob_hash,
                    2,
                    None,
                    0,
                ),
                (
                    unselected_raw_id,
                    "codex-session:selected",
                    "selected",
                    "1",
                    selected_blob_hash,
                    2,
                    None,
                    0,
                ),
            ],
        )
        connection.commit()

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


def test_cli_rejects_tampered_raw_payload_bytes_during_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live raw blob must still hash correctly when a reviewed report is consumed."""

    canary_root, report_path, payload = _write_reviewed_real_canary_report(
        tmp_path, monkeypatch, name="tampered-blob-canary"
    )
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selected_raw_ids = selection["selected_raw_ids"]
    assert isinstance(selected_raw_ids, list) and selected_raw_ids
    with sqlite3.connect(canary_root / "source.db") as connection:
        blob_hash = connection.execute(
            "SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?", (selected_raw_ids[0],)
        ).fetchone()[0]
    blob_path = canary_root / "blob" / str(blob_hash)[:2] / str(blob_hash)[2:]
    blob_path.write_bytes(b"tampered raw payload bytes")

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
    assert "blob bytes failed verification" in consumed.output


def test_cli_rejects_source_mutation_between_approval_checks_and_preserves_active_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approval revalidates source evidence after its initial report check."""

    canary_root, report_path, payload = _write_reviewed_real_canary_report(
        tmp_path, monkeypatch, name="approval-mutation-canary"
    )
    active_before = (canary_root / "index.db").read_bytes()
    real_load = reindex_canary_module.load_canary_report
    calls = 0

    def mutate_after_initial_check(path: Path, *, archive_root: Path | None = None) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = real_load(path, archive_root=archive_root)
        if calls == 1:
            selection = payload["selection"]
            assert isinstance(selection, dict)
            selected_raw_ids = selection["selected_raw_ids"]
            assert isinstance(selected_raw_ids, list) and selected_raw_ids
            with sqlite3.connect(canary_root / "source.db") as connection:
                connection.execute(
                    "UPDATE raw_sessions SET source_index = source_index + 1 WHERE raw_id = ?",
                    (selected_raw_ids[0],),
                )
                connection.commit()
        return result

    monkeypatch.setattr(reindex_canary_module, "load_canary_report", mutate_after_initial_check)
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
    assert calls == 2
    assert (canary_root / "index.db").read_bytes() == active_before


def test_cli_rejects_candidate_promotion_between_approval_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate promoted after the first check cannot receive approval."""

    canary_root, report_path, payload = _write_reviewed_real_canary_report(
        tmp_path, monkeypatch, name="approval-promotion-canary"
    )
    real_load = reindex_canary_module.load_canary_report
    promoted = False

    def promote_after_initial_check(path: Path, *, archive_root: Path | None = None) -> dict[str, object]:
        nonlocal promoted
        result = real_load(path, archive_root=archive_root)
        if not promoted:
            receipt = payload["rebuild_receipt"]
            assert isinstance(receipt, dict)
            generation = receipt["generation"]
            assert isinstance(generation, dict)
            location = ArchiveLocation.resolve(canary_root)
            store = IndexGenerationStore(location)
            store.promote(store.load(str(generation["generation_id"])))
            promoted = True
        return result

    monkeypatch.setattr(reindex_canary_module, "load_canary_report", promote_after_initial_check)
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
    assert promoted is True
    assert (
        "candidate generation" in consumed.output
        or "active pointer" in consumed.output
        or "stale for the current active generation" in consumed.output
    )


def test_cli_consumption_obeys_rebuild_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report consumption cannot race an active rebuild owner."""

    canary_root, report_path, _payload = _write_reviewed_real_canary_report(
        tmp_path, monkeypatch, name="approval-lock-canary"
    )
    with RebuildLease(canary_root):
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
    assert "rebuild lease" in consumed.output or "already held" in consumed.output


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
            "receipt_schema_version": 4,
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
                selected_session_ids=selection.selected_session_ids,
            ),
            "source_evidence_after": "0" * 64,
            "canary_acceptance": _canary_acceptance_attestation(),
        },
    )


def _canary_acceptance_attestation() -> dict[str, object]:
    """Return the daemon-owned acceptance evidence required of canary receipts."""

    return {
        "profile": REINDEX_CANARY_ACCEPTANCE_PROFILE,
        "results": [
            {"name": name, "status": "ok", "summary": "fixture acceptance", "count": 0}
            for name in REINDEX_CANARY_ACCEPTANCE_CHECKS
        ],
    }


def _nonempty_run_result(index_path: Path) -> CanaryRunResult:
    difference = RowDifference(
        table="sessions",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "codex-session:sample"),),
        before={"title_ref": None},
        after={"title_ref": "message:codex-session:sample:user"},
        changed_columns=("title_ref",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    result = _run_result(index_path, differences=(difference,))
    return CanaryRunResult(
        selection=result.selection,
        comparison=result.comparison,
        rebuild_receipt={
            "receipt_schema_version": 4,
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
                selected_session_ids=result.selection.selected_session_ids,
            ),
            "source_evidence_after": "0" * 64,
            "canary_acceptance": _canary_acceptance_attestation(),
        },
    )


def test_reindex_canary_cli_requires_no_promote(tmp_path: Path) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
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
            "--schema-inference-receipt",
            str(receipt_path),
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
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
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
            "--schema-inference-receipt",
            str(receipt_path),
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
    receipt_path = write_valid_rebuild_receipt(archive_root, tmp_path / "schema-inference-gate-receipt.json")
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
            "--schema-inference-receipt",
            str(receipt_path),
            "--no-promote",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1, result.output
    assert "persisted unreviewed" in result.output
    assert "refuses the configured live archive root" not in result.output
    assert json.loads(report_path.read_text(encoding="utf-8"))["review_status"] == "unreviewed"


def test_reindex_canary_cli_forwards_an_explicit_owner_backfill_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canary CLI passes its explicit ownership evidence into the service.

    Anti-vacuity: this invokes the registered command with the production
    option and captures the actual service call. Removing the option or its
    forwarding leaves the canary unable to receive the receipt.
    """

    archive_root = tmp_path / "isolated-canary"
    initialize_active_archive_root(archive_root)
    schema_receipt = tmp_path / "schema-receipt.json"
    owner_receipt = tmp_path / "owner-receipt.json"
    schema_receipt.write_text("{}", encoding="utf-8")
    owner_receipt.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "canary-report.json"
    captured: dict[str, object] = {}

    def capture_run(*args: object, **kwargs: object) -> CanaryRunResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _run_result(archive_root / "index.db")

    class DurableReport:
        unclassified_count = 0
        review_status = "reviewed"

        def to_dict(self) -> dict[str, object]:
            return {"status": "classified"}

    monkeypatch.setattr("polylogue.maintenance.reindex_canary.run_reindex_canary", capture_run)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.write_canary_report", lambda *_args, **_kwargs: DurableReport()
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.load_canary_report", lambda *_args, **_kwargs: {})

    result = CliRunner().invoke(
        reindex_canary_cli.reindex_canary_command,
        [
            "--archive-root",
            str(archive_root),
            "--report",
            str(report_path),
            "--schema-inference-receipt",
            str(schema_receipt),
            "--message-owner-scope-backfill-receipt",
            str(owner_receipt),
            "--no-promote",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"] == {
        "input_index": None,
        "sessions_per_origin": 100,
        "pathology_session_ids": (),
        "sample_session_ids": (),
        "no_promote": True,
        "schema_inference_receipt_path": schema_receipt,
        "message_owner_scope_backfill_receipt_path": owner_receipt,
    }


def test_reindex_canary_cli_refuses_to_write_unclassified_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
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
            "--schema-inference-receipt",
            str(receipt_path),
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
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
    report_path = tmp_path / "canary.json"
    review_path = tmp_path / "reviews.json"
    run_result = _nonempty_run_result(index_path)
    difference = run_result.comparison.differences[0]
    assert isinstance(difference, RowDifference)
    review = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:44",
        rationale="reviewed title-reprocess change",
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
            "--schema-inference-receipt",
            str(receipt_path),
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
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
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
        reference="delta:44",
        rationale="wrong title-reprocess signature",
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
            "--schema-inference-receipt",
            str(receipt_path),
            "--no-promote",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    # Name the offending column rather than a refusal sentence: the wording is
    # the CLI's to change, but a rejection that does not tell the operator
    # which declaration failed to cover the diff is not a usable rejection.
    assert "different_column" in result.output
    assert "sessions" in result.output
    assert not report_path.exists()


def test_reindex_canary_cli_refuses_nonempty_differences_without_review_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.db"
    index_path.touch()
    receipt_path = tmp_path / "schema-inference-gate-receipt.json"
    receipt_path.touch()
    run_result = _nonempty_run_result(index_path)

    def fake_write(path: Path, **kwargs: object) -> DurableCanaryReport:
        raise UnclassifiedCanaryDiffError("classification is incomplete")

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
            "--schema-inference-receipt",
            str(receipt_path),
            "--no-promote",
        ],
    )

    assert result.exit_code == 1
    assert "classification is incomplete" in result.output


def test_reindex_canary_cli_persists_unreviewed_real_candidate_for_later_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real inactive rebuild persists discovery evidence but cannot approve it."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / "isolated-canary"
    report_path = tmp_path / "unreviewed.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: live_root)
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE sessions SET title_ref = NULL, title_confidence = NULL")

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
    assert "persisted unreviewed" in result.output
    assert json.loads(report_path.read_text(encoding="utf-8"))["review_status"] == "unreviewed"


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
    assert (
        "archive-owned" in consumed.output
        or "authoritative rebuild receipt" in consumed.output
        or "different configured archive root" in consumed.output
    )


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
    assert "archive-owned" in consumed.output or "selection" in consumed.output


def test_reindex_canary_cli_rejects_manifest_with_mismatched_changed_columns_from_real_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI review manifests must bind each real difference's column signature."""

    live_root = tmp_path / "configured-live"
    canary_root = tmp_path / "isolated-canary"
    rejected_report_path = tmp_path / "rejected.json"
    review_path = tmp_path / "reviews.json"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(canary_root))
    _seed_isolated_canary(canary_root)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: live_root)
    with sqlite3.connect(canary_root / "index.db") as connection:
        connection.execute("UPDATE sessions SET title_ref = NULL, title_confidence = NULL")

    observed_result = run_reindex_canary(
        canary_root,
        input_index=canary_root / "index.db",
        sessions_per_origin=1,
        no_promote=True,
        schema_inference_receipt_path=_schema_receipt_path(canary_root),
    )
    differences = [difference.to_dict() for difference in observed_result.comparison.differences]
    assert differences
    reviews = [
        {
            "table": difference["table"],
            "operation": difference["operation"],
            "identity": difference["identity"],
            "changed_columns": difference["changed_columns"],
            "classification": "expected",
            "reference": "delta:44",
            "authority": {"kind": "delta", "id": "44"},
            "rationale": "reviewed title-reprocess change",
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
    assert "forged_column" in rejected.output
    assert "sessions" in rejected.output
    assert not rejected_report_path.exists()


def test_shared_canary_runner_uses_existing_inactive_rebuild_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_index = tmp_path / "index.db"
    candidate_index = tmp_path / ".index-generations" / "gen-test" / "index.db"
    current_index.touch()
    owner_receipt = tmp_path / "owner-receipt.json"
    owner_receipt.write_text("{}", encoding="utf-8")
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
            selected_session_ids=selection.selected_session_ids,
        )
        canary_acceptance = _canary_acceptance_attestation()

        def to_dict(self) -> dict[str, object]:
            return {
                "archive_root": self.archive_root,
                "selected_raw_count": self.selected_raw_count,
                "status": self.status,
                "materialized": self.materialized,
                "generation": self.generation,
                "selection_evidence": self.selection_evidence,
                "canary_acceptance": self.canary_acceptance,
            }

    def fake_rebuild(**request: object) -> Receipt:
        captured["request"] = request
        return Receipt()

    def fake_compare(
        current: Path,
        candidate: Path,
        *,
        session_ids: tuple[str, ...],
        **provenance: object,
    ) -> CanaryDiffReport:
        captured["compare"] = (current, candidate, session_ids)
        captured.update(provenance)
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
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", fake_rebuild)
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_authoritative_rebuild_receipt", lambda *args, **kwargs: None
    )

    result = run_reindex_canary(
        tmp_path,
        input_index=current_index,
        schema_inference_receipt_path=tmp_path / "schema-inference-gate-receipt.json",
        message_owner_scope_backfill_receipt_path=owner_receipt,
        sessions_per_origin=2,
        no_promote=True,
    )

    request = cast(dict[str, object], captured["request"])
    assert request["raw_ids"] == selection.selected_raw_ids
    assert request["selected_session_ids"] == selection.selected_session_ids
    assert request["message_owner_scope_backfill_receipt_path"] == owner_receipt
    assert captured["compare"] == (current_index, candidate_index, selection.selected_session_ids)
    assert result.rebuild_receipt == {
        "archive_root": Receipt.archive_root,
        "selected_raw_count": Receipt.selected_raw_count,
        "status": Receipt.status,
        "materialized": Receipt.materialized,
        "generation": Receipt.generation,
        "selection_evidence": Receipt.selection_evidence,
        "canary_acceptance": Receipt.canary_acceptance,
    }
