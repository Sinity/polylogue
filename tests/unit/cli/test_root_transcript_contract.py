"""Exact root reads retain the selected projection and canonical read verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import archive_query
from polylogue.cli.click_app import cli
from polylogue.cli.operation_kernel import OperationFailedError
from polylogue.cli.query_output_contracts import RootSessionDocument
from tests.infra.archive_store_double import install_archive_store_double
from tests.infra.cli_transcript_archive import CliTranscriptArchive


@pytest.mark.parametrize("gap_offset", [0, 1])
@pytest.mark.parametrize("output_format", ["json", "yaml", "ndjson", "plaintext"])
def test_exact_root_read_keeps_every_page_lineage_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap_offset: int, output_format: str
) -> None:
    """Dropping a page verdict makes the real product/Click route falsely exit 0."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(archive_query, "_SESSION_READ_WINDOW", 1)
    store = CliTranscriptArchive(gap_offset=gap_offset)
    install_archive_store_double(monkeypatch, store)
    result = CliRunner().invoke(cli, ["--format", output_format, "find", f"id:{store.session_id}"])

    assert result.exit_code == 1, result.output
    assert "lineage_truncated:dangling_branch_point" in result.stderr
    assert store.body_reads == [(1, 0), (1, 1)]
    if output_format == "json":
        payload = RootSessionDocument.model_validate_json(result.stdout)
        assert payload.outcome.state == "degraded"
        assert payload.lineage_complete is False
        assert payload.lineage_truncation_reason == "dangling_branch_point"
        assert len(payload.messages) == 2
    elif output_format == "yaml":
        import yaml

        payload = yaml.safe_load(result.stdout)
        assert payload["outcome"]["state"] == "degraded"
        assert payload["lineage_complete"] is False
    elif output_format == "ndjson":
        assert len([json.loads(line) for line in result.stdout.splitlines()]) == 2


@pytest.mark.parametrize("message_count,expected_state,expected_status", [(0, "empty", 2), (1, "ok", 0)])
def test_exact_root_read_uses_the_product_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message_count: int,
    expected_state: str,
    expected_status: int,
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive(message_count=message_count)
    install_archive_store_double(monkeypatch, store)
    result = CliRunner().invoke(cli, ["--format", "json", "find", f"id:{store.session_id}"])
    assert result.exit_code == expected_status, result.output
    assert json.loads(result.stdout)["outcome"]["state"] == expected_state


@pytest.mark.parametrize("message_count", [0, 2])
def test_root_stream_keeps_lineage_gap_with_and_without_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message_count: int
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive(message_count=message_count, gap_offset=0)
    install_archive_store_double(monkeypatch, store)
    result = CliRunner().invoke(cli, ["--stream", "--format", "ndjson", "find", f"id:{store.session_id}"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stderr)["outcome"]["state"] == "degraded"
    assert len(result.stdout.splitlines()) == message_count


@pytest.mark.parametrize("output_format", ["json", "ndjson"])
@pytest.mark.parametrize("message_count", [0, 2])
def test_gap_shaped_root_document_is_delivered_before_its_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str, message_count: int
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive(gap_offset=0, message_count=message_count)
    install_archive_store_double(monkeypatch, store)
    destination = tmp_path / "transcript"
    result = CliRunner().invoke(
        cli, ["--output", str(destination), "--format", output_format, "find", f"id:{store.session_id}"]
    )
    assert result.exit_code == 1, result.output
    content = destination.read_text()
    if output_format == "json":
        assert json.loads(content)["outcome"]["state"] == "degraded"
    else:
        assert len([json.loads(line) for line in content.splitlines()]) == message_count


def test_failed_root_read_does_not_deliver_a_completed_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive()
    install_archive_store_double(monkeypatch, store)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OperationFailedError("stale_generation", "the selected archive changed")

    monkeypatch.setattr(archive_query, "dispatch_read", refuse)
    destination = tmp_path / "transcript.json"
    result = CliRunner().invoke(
        cli, ["--output", str(destination), "--format", "json", "find", f"id:{store.session_id}"]
    )
    assert result.exit_code == 1, result.output
    assert not destination.exists()


@pytest.mark.parametrize("explicit_view", [False, True])
@pytest.mark.parametrize("selector", ["expression", "flag"])
@pytest.mark.parametrize("missing", [False, True])
def test_summary_all_exact_ids_do_not_hydrate_transcripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_view: bool, selector: str, missing: bool
) -> None:
    """Ignoring list_mode sends an exact ID through session.read and changes shape."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive()
    install_archive_store_double(monkeypatch, store)
    session_id = "codex-session:missing" if missing else store.session_id
    args = ["find", f"id:{session_id}"] if selector == "expression" else ["--id", session_id]
    args.extend(["then", "read"] if selector == "expression" else ["read"])
    args.extend(["--all", "--format", "json"])
    if explicit_view:
        args.extend(["--view", "summary"])
    result = CliRunner().invoke(cli, args)
    expected_status = 2 if missing and selector == "expression" else 0
    assert result.exit_code == expected_status, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "list"
    assert len(payload["items"]) == (0 if missing else 1)
    assert store.body_reads == []
    if not missing:
        assert payload["items"][0]["id"] == store.session_id
        assert "messages" not in payload["items"][0]


def test_summary_all_exact_id_keeps_file_delivery_and_selected_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    store = CliTranscriptArchive()
    install_archive_store_double(monkeypatch, store)
    destination = tmp_path / "summaries.json"
    result = CliRunner().invoke(
        cli,
        [
            "find",
            f"id:{store.session_id}",
            "then",
            "read",
            "--all",
            "--format",
            "json",
            "--fields",
            "id",
            "--to",
            "file",
            "--out",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text())["items"] == [{"id": store.session_id}]
    assert store.body_reads == []
