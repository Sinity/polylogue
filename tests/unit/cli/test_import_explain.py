from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.core.enums import Provider
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


def test_explain_import_path_reports_antigravity_trajectory_sqlite(tmp_path: Path) -> None:
    source = tmp_path / "renamed-trajectory.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            INSERT INTO trajectory_meta VALUES ('trajectory-explain', 'cascade-explain');
            INSERT INTO steps VALUES (0, 'message', 'v1', '{"role":"user","text":"hello"}');
            """
        )

    payload = explain_import_path(source)

    assert payload.produced.sessions == 1
    assert payload.produced.messages == 1
    assert payload.entries[0].detected_provider == "antigravity"
    assert payload.entries[0].artifact_kind == "sqlite_trajectory_database"
    assert payload.entries[0].parser_mode == "logical_export"


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


def test_import_explain_zip_recovers_path_classified_json_record_array(tmp_path: Path) -> None:
    """Explain applies decoded-session evidence before a workflow path skip."""
    archive = tmp_path / "workflow-json.zip"
    records = [
        {
            "sessionId": "explain-json-session",
            "parentUuid": None,
            "type": "user",
            "message": {"role": "user", "content": "explain this session"},
            "uuid": "explain-json-user",
            "timestamp": "2025-01-01T00:00:00Z",
        },
        {
            "sessionId": "explain-json-session",
            "parentUuid": "explain-json-user",
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "explained"}]},
            "uuid": "explain-json-assistant",
            "timestamp": "2025-01-01T00:00:01Z",
        },
    ]
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("workflows/explain.json", json.dumps(records))

    payload = explain_import_path(archive, source_name="claude-code")

    assert payload.produced.sessions >= 1
    assert not any(
        row.source_path and row.source_path.endswith("workflow-json.zip:workflows/explain.json")
        for row in payload.skipped
    )


def test_import_explain_zip_preserves_complete_mixed_provider_members(tmp_path: Path) -> None:
    archive = tmp_path / "mixed.zip"
    fixtures = Path(__file__).parents[2] / "fixtures" / "origin-capability"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        chatgpt = json.loads((fixtures / "chatgpt-export.json").read_bytes())
        if isinstance(chatgpt, list):
            chatgpt[0]["padding"] = "x" * (2 * 1024 * 1024)
        else:
            chatgpt["padding"] = "x" * (2 * 1024 * 1024)
        output.writestr("conversations.json", json.dumps(chatgpt))
        output.writestr("minority/claude.json", (fixtures / "claude-ai-export.json").read_bytes())
    payload = explain_import_path(archive)
    assert payload.produced.sessions >= 2
    assert {ref.split(":", 2)[1] for ref in payload.produced.session_refs} >= {"chatgpt", "claude-ai"}
    assert not payload.skipped


def test_import_explain_names_the_container_origin_of_a_claude_ai_export_zip(tmp_path: Path) -> None:
    """polylogue-erf3: the container carried no origin identity while its contents did.

    ``polylogue import --explain`` on a claude.ai GDPR export ZIP reported
    detector=zip.container with detected_origin=unknown-export, even though
    every inner entry lowered to a claude-ai session. The container now takes
    its identity from the same member-dominance rule acquisition uses.

    Anti-vacuity: drop the ``sniff_zip_provider`` block in ``_explain_zip`` and
    ``detected_origin`` falls back to ``unknown-export``.
    """
    archive = tmp_path / "claude-ai-data-batch-0000.zip"
    conversations = json.dumps(
        [
            {
                "uuid": "conv-1",
                "name": "Export conversation",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "m-1",
                        "sender": "human",
                        "text": "hello",
                        "created_at": "2026-01-01T00:00:00Z",
                    },
                    {
                        "uuid": "m-2",
                        "sender": "assistant",
                        "text": "hi",
                        "created_at": "2026-01-01T00:00:01Z",
                    },
                ],
            }
        ]
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("conversations.json", conversations)

    payload = explain_import_path(archive)
    container = payload.entries[0]

    assert container.detector == "zip.container"
    assert container.detected_origin == "claude-ai-export"
    assert container.detected_provider == Provider.CLAUDE_AI.value
