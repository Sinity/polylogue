import json
from pathlib import Path

from polylogue.importers.codex import import_codex_session


def _write_session(tmp_path: Path, name: str, entries: list[dict]) -> Path:
    session_dir = tmp_path / "sessions" / "2025" / "01" / "01"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")
    return path


def test_import_codex_session_extracts_tool_attachment(tmp_path):
    output_lines = "\n".join(f"line {i}" for i in range(120))
    entries = [
        {"type": "session_meta", "payload": {"session_id": "session-test"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "first prompt"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "bash",
                "arguments": {"cmd": "echo hi"},
                "call_id": "call-1",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": output_lines,
            },
        },
    ]
    path = _write_session(tmp_path, "session-test", entries)
    out_dir = tmp_path / "out"

    result = import_codex_session(
        "session-test",
        base_dir=tmp_path / "sessions",
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
    )

    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "Tool call `bash`" in text
    assert "Full content saved to" in text
    assert result.attachments_dir is not None
    attachment_files = list(result.attachments_dir.iterdir())
    assert attachment_files, "expected an attachment file"
    payload = attachment_files[0].read_text(encoding="utf-8")
    assert output_lines in payload
    stats = result.document.stats
    assert stats["chunkCount"] >= 2
    assert stats["modelTurns"] >= 0
    assert stats["userTurns"] >= 1
    assert stats["totalWordsApprox"] >= 1

    # ensure attachment metadata references the saved file
    links = [info for info in result.document.attachments]
    assert links and links[0].name.endswith(".txt")


def test_import_codex_session_removes_empty_attachment_dir(tmp_path):
    entries = [
        {"type": "session_meta", "payload": {"session_id": "simple"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        },
    ]
    _write_session(tmp_path, "simple", entries)
    out_dir = tmp_path / "out"

    result = import_codex_session(
        "simple",
        base_dir=tmp_path / "sessions",
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
    )

    assert result.attachments_dir is None
    assert not (out_dir / "simple_attachments").exists()


def test_import_codex_session_normalises_footnotes(tmp_path):
    entries = [
        {"type": "session_meta", "payload": {"session_id": "foot"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "\\[9\\] Footnote"}],
            },
        },
    ]
    path = _write_session(tmp_path, "session-foot", entries)
    out_dir = tmp_path / "out_foot"

    result = import_codex_session(
        "session-foot",
        base_dir=tmp_path / "sessions",
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
    )

    assert result.markdown_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "[9] Footnote" in text
    assert "\\[9\\]" not in text
