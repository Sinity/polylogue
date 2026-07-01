from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from polylogue.cli.click_app import cli
from polylogue.sources import import_explain as import_explain_module
from polylogue.sources.import_explain import explain_import_path


def test_explain_import_path_reports_codex_parser_and_counts(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    payload = explain_import_path(target, source_name="codex")

    assert payload.mode == "import-explain"
    assert payload.produced.sessions == 1
    assert payload.produced.messages >= 1
    assert payload.entries[0].detected_provider == "codex"
    assert payload.entries[0].detected_origin == "codex-session"
    assert payload.entries[0].artifact_kind == "session_record_stream"
    assert payload.entries[0].parser_mode == "grouped_records"
    assert payload.entries[0].produced.session_refs


def test_explain_import_path_treats_jsonl_text_json_wrappers_as_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "aggregate.jsonl.txt.json"
    target.write_text(
        "\n".join(
            (
                '{"type":"user","sessionId":"first-session","uuid":"u1","message":{"role":"user","content":"one"}}',
                '{"type":"user","sessionId":"second-session","uuid":"u2","message":{"role":"user","content":"two"}}',
            )
        ),
        encoding="utf-8",
    )

    payload = explain_import_path(target, source_name="claude-code")

    assert payload.produced.sessions == 2
    assert payload.entries[0].detected_origin == "claude-code-session"
    assert payload.entries[0].parser_mode == "grouped_records"
    assert payload.entries[0].produced.session_refs == (
        "session:claude-code:first-session",
        "session:claude-code:second-session",
    )


def test_explain_import_path_reports_malformed_json_as_skip(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")

    payload = explain_import_path(target)

    assert payload.produced.sessions == 0
    assert payload.skipped
    assert payload.skipped[0].reason.startswith("decode failure:")
    assert payload.entries[0].skipped[0].source_path == str(target.resolve())


def test_import_explain_cli_emits_finite_json(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    result = CliRunner().invoke(cli, ["--plain", "import", str(target), "--explain", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "import-explain"
    assert payload["produced"]["sessions"] == 1
    assert payload["entries"][0]["detected_origin"] == "codex-session"
    assert "raw_bytes" not in result.output


def test_import_explain_cli_ndjson_emits_entries(tmp_path: Path) -> None:
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    result = CliRunner().invoke(cli, ["--plain", "import", str(target), "--explain", "--format", "ndjson"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["detected_provider"] == "codex"


def test_import_explain_zip_propagates_member_decode_skip(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/broken.json", "{not json")

    payload = explain_import_path(archive)

    assert payload.produced.sessions == 0
    assert payload.entries[0].skipped
    assert payload.skipped
    skipped_path = payload.skipped[0].source_path
    assert skipped_path is not None
    assert skipped_path.endswith("broken.zip:nested/broken.json")
    assert payload.skipped[0].reason.startswith("decode failure:")


def test_import_explain_zip_rejects_oversized_member_before_read(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    archive = tmp_path / "oversized.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("big.json", b"{}")
    monkeypatch.setattr(import_explain_module, "MAX_UNCOMPRESSED_SIZE", 1)

    payload = explain_import_path(archive)

    assert payload.produced.sessions == 0
    assert payload.skipped
    skipped_path = payload.skipped[0].source_path
    assert skipped_path is not None
    assert skipped_path.endswith("oversized.zip:big.json")
    assert "file size" in payload.skipped[0].reason
