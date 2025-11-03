import json
from pathlib import Path

from polylogue.local_sync import sync_claude_code_sessions, sync_codex_sessions


def _write_codex_session(root: Path, name: str, lines: list[dict]) -> Path:
    session_dir = root / "2025" / "01" / "01"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in lines:
            handle.write(json.dumps(entry))
            handle.write("\n")
    return path


def test_sync_codex_sessions_creates_markdown(tmp_path):
    base_dir = tmp_path / "codex"
    session_path = _write_codex_session(
        base_dir,
        "session-001",
        [
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"text": "hi", "type": "output_text"}]}}
        ],
    )
    out_dir = tmp_path / "out"
    result = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )
    assert len(result.written) == 1
    md_path = result.written[0].markdown_path
    assert md_path.exists()
    assert "hi" in md_path.read_text(encoding="utf-8")

    # second run should skip
    result_skip = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )
    assert result_skip.skipped >= 1

    # prune removes stale outputs
    stale_dir = out_dir / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale = stale_dir / "conversation.md"
    stale.write_text("old", encoding="utf-8")
    result_prune = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=True,
        sessions=[session_path],
    )
    assert result_prune.pruned >= 1
    assert not stale.exists()
    assert not stale_dir.exists()


def _write_claude_code_session(root: Path, name: str, lines: list[str]) -> Path:
    path = root / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sync_claude_code_sessions(tmp_path):
    base_dir = tmp_path / "claude_code"
    session_path = _write_claude_code_session(
        base_dir,
        "workspace/session-1",
        [
            json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "task"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}}),
        ],
    )
    out_dir = tmp_path / "claude_out"
    result = sync_claude_code_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )
    assert result.written
    md_path = result.written[0].markdown_path
    assert md_path.exists()
    body = md_path.read_text(encoding="utf-8")
    assert "task" in body
    assert "answer" in body
