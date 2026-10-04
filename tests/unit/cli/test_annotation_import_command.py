"""Behavioral proof for the ``polylogue annotations import`` command.

``annotations import`` writes ``user.db``, the archive's one irreplaceable
tier, so it lowers to the declared ``mutation.annotation.import_batch``
operation and the daemon is its sole writer (polylogue-gjwto / polylogue-r29bv
AC3). The write tests here therefore run a real daemon stack rather than an
in-process writer; ``annotations join`` (a read) is untouched and stays direct.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.archive.message.roles import Role
from polylogue.cli import cli
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.live_ingest import write_index_session


def _seed_session(archive_root: Path) -> str:
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="annotation-target",
                title="Annotation target",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
            ),
        )
    assert session_id == "codex-session:annotation-target"
    return session_id


def _import_args(source: Path, *, target_ref: str) -> list[str]:
    return [
        "--plain",
        "annotations",
        "import",
        str(source),
        "--batch-id",
        "cli-batch",
        "--schema-id",
        "seed.activity",
        "--schema-version",
        "2",
        "--target-ref",
        target_ref,
        "--source-result-ref",
        "result-set:cli-evidence",
        "--actor-ref",
        "agent:labeler",
        "--model-ref",
        "agent:model",
        "--prompt-ref",
        "block:prompt:0",
        "--metadata-json",
        json.dumps({"campaign": "cli"}),
    ]


def test_annotations_import_writes_through_the_resident_daemon(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon applies the import and audits the attempt.

    Anti-vacuity: point ``annotations import`` at a second in-process writer
    (the old ``Polylogue.import_annotation_batch`` -> ``import_annotation_batch``
    route this replaced) and this still passes, which is why
    ``test_annotations_import_refuses_without_a_daemon`` below is the
    load-bearing half -- together they say the write happened *and* that it
    could only have happened through the daemon.
    """
    archive_root = cli_workspace["archive_root"]
    session_id = _seed_session(archive_root)
    source = cli_workspace["inbox_dir"] / "labels.jsonl"
    source.write_text(
        json.dumps(
            {"row_key": "r1", "value": {"activity": "debugging", "confidence": 0.9}, "evidence_refs": [session_id]}
        )
        + "\n",
        encoding="utf-8",
    )

    with cli_daemon_archive(archive_root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            _import_args(source, target_ref=f"session:{session_id}"),
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["batch_ref"] == "annotation-batch:cli-batch"
    assert payload["valid_count"] == 1
    assert payload["invalid_count"] == 0

    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone() == (1,)
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM operation_attempts AS attempt "
            "JOIN operation_runs AS run ON run.operation_id = attempt.operation_id "
            "WHERE run.operation_name = ?",
            ("mutate-import-annotation-batch",),
        ).fetchone() == (1,)


def _message_ref(archive_root: Path, session_id: str) -> str:
    with sqlite3.connect(archive_root / "index.db") as conn:
        row = conn.execute("SELECT message_id FROM messages WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
    assert row is not None
    return f"message:{row[0]}"


def test_daemon_import_and_facade_resolve_evidence_refs_through_one_plan(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon's admission decision and the facade's answer are one code path.

    ``annotations import`` admits a ``user.db`` row on whether each evidence
    ref resolves. A resolver copied into the daemon handler could answer a ref
    differently from ``Polylogue.resolve_ref`` with no failure at the point of
    divergence (polylogue-j5u2b).

    Anti-vacuity: give the daemon handler its own resolver instead of
    ``resolve_ref_against_archive`` and the daemon import stops recording its
    evidence refs through ``plan_ref_resolution``, so the first assertion goes
    red; answer the missing message differently on either side and the
    admission/facade comparison goes red.
    """
    from polylogue import Polylogue
    from polylogue.operations import ref_resolution

    archive_root = cli_workspace["archive_root"]
    session_id = _seed_session(archive_root)
    good_anchor = _message_ref(archive_root, session_id)
    bad_anchor = f"message:{session_id}:n:no-such-message"
    source = cli_workspace["inbox_dir"] / "labels.jsonl"
    source.write_text(
        "".join(
            json.dumps({"row_key": key, "value": {"activity": "debugging", "confidence": 0.9}, "evidence_refs": [ref]})
            + "\n"
            for key, ref in (("good", good_anchor), ("bad", bad_anchor))
        ),
        encoding="utf-8",
    )

    planned: list[str] = []
    original_plan = ref_resolution.plan_ref_resolution

    def recording_plan(
        ref: str, *, archive_root: Path, limit: int = 50, offset: int = 0, continuation: str | None = None
    ) -> ref_resolution.RefResolutionPlan:
        planned.append(ref)
        return original_plan(ref, archive_root=archive_root, limit=limit, offset=offset, continuation=continuation)

    monkeypatch.setattr(ref_resolution, "plan_ref_resolution", recording_plan)

    with cli_daemon_archive(archive_root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            _import_args(source, target_ref=f"session:{session_id}"),
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert {good_anchor, bad_anchor} <= set(planned)
    payload = json.loads(result.output)
    assert (payload["valid_count"], payload["invalid_count"]) == (1, 1)

    async def facade_answers() -> tuple[bool, bool]:
        async with Polylogue(archive_root=archive_root) as archive:
            return (
                (await archive.resolve_ref(good_anchor)).resolved,
                (await archive.resolve_ref(bad_anchor)).resolved,
            )

    assert asyncio.run(facade_answers()) == (True, False)


def test_annotations_import_refuses_without_a_daemon(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """No daemon means no write, not a second writer in the CLI process.

    Anti-vacuity: restore the old direct call to
    ``polylogue.annotations.importer.import_annotation_batch`` behind this
    command and the row lands with no daemon at all, so the empty-assertions
    assertion below goes red. That is the whole defect polylogue-gjwto names.
    """
    archive_root = cli_workspace["archive_root"]
    session_id = _seed_session(archive_root)
    source = cli_workspace["inbox_dir"] / "labels.jsonl"
    source.write_text(
        json.dumps(
            {"row_key": "r1", "value": {"activity": "debugging", "confidence": 0.9}, "evidence_refs": [session_id]}
        )
        + "\n",
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        cli,
        _import_args(source, target_ref=f"session:{session_id}"),
    )

    assert result.exit_code != 0, result.output
    assert "daemon" in result.output.lower()
    with sqlite3.connect(archive_root / "user.db") as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'assertions'")
        }
        if tables:
            assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)


def test_cli_annotation_import_rejects_non_object_metadata(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    archive_root = cli_workspace["archive_root"]
    session_id = _seed_session(archive_root)
    source = cli_workspace["inbox_dir"] / "labels.jsonl"
    source.write_text('{"row_key":"r1","value":{},"evidence_refs":[]}\n', encoding="utf-8")

    args = _import_args(source, target_ref=f"session:{session_id}")
    args[args.index("--metadata-json") + 1] = "[]"
    result = cli_runner.invoke(cli, args)

    assert result.exit_code != 0
    assert "must decode to a JSON object" in result.output
