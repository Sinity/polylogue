import json
import pytest
from pathlib import Path

from polylogue.importers.base import ImportResult
from polylogue.local_sync import (
    sync_chatgpt_exports,
    sync_claude_code_sessions,
    sync_claude_exports,
    sync_codex_sessions,
)


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
    assert result_skip.skip_reasons.get("up-to-date", 0) >= 1

    # Touch the source file without changing content; hash freshness should still skip.
    session_path.write_text(session_path.read_text(encoding="utf-8"), encoding="utf-8")
    result_hash_skip = sync_codex_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        sessions=[session_path],
    )
    assert not result_hash_skip.written
    assert result_hash_skip.skipped >= 1
    assert result_hash_skip.skip_reasons.get("up-to-date", 0) >= 1

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
            json.dumps({"type": "user", "sessionId": "sess-1", "message": {"content": [{"type": "text", "text": "task"}]}}),
            json.dumps({"type": "assistant", "sessionId": "sess-1", "message": {"content": [{"type": "text", "text": "answer"}]}}),
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

    result_skip = sync_claude_code_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        sessions=[session_path],
    )
    assert result_skip.skipped >= 1
    assert result_skip.skip_reasons.get("up-to-date", 0) >= 1


def test_sync_claude_code_sessions_merges_agent_logs(tmp_path):
    base_dir = tmp_path / "claude_code"
    session_id = "sess-merge"
    summary_path = _write_claude_code_session(
        base_dir,
        "workspace/summary",
        [
            json.dumps({"type": "summary", "sessionId": session_id, "summary": "short summary"}),
        ],
    )
    _write_claude_code_session(
        base_dir,
        "workspace/agent-1",
        [
            json.dumps({"type": "user", "sessionId": session_id, "message": {"content": [{"type": "text", "text": "alpha"}]}}),
            json.dumps({"type": "assistant", "sessionId": session_id, "message": {"content": [{"type": "text", "text": "beta"}]}}),
        ],
    )
    out_dir = tmp_path / "claude_merge_out"
    result = sync_claude_code_sessions(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        sessions=[summary_path],
    )
    assert result.written
    md_path = result.written[0].markdown_path
    body = md_path.read_text(encoding="utf-8")
    assert "alpha" in body
    assert "beta" in body


def test_sync_claude_code_sessions_skips_summary_only(tmp_path):
    base_dir = tmp_path / "claude_code"
    summary_path = _write_claude_code_session(
        base_dir,
        "workspace/summary-only",
        [
            json.dumps({"type": "summary", "sessionId": "sess-summary", "summary": "only summary"}),
        ],
    )
    result = sync_claude_code_sessions(
        base_dir=base_dir,
        output_dir=tmp_path / "claude_summary_out",
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        sessions=[summary_path],
    )
    assert not result.written
    assert result.skip_reasons.get("summary-only log", 0) >= 1


def test_sync_chatgpt_exports(monkeypatch, tmp_path):
    base_dir = tmp_path / "chatgpt"
    zip_path = base_dir / "bundle-1.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"")
    dir_path = base_dir / "bundle-dir"
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "conversations.json").write_text("{}", encoding="utf-8")

    out_dir = tmp_path / "chatgpt_out"
    calls: list[Path] = []

    def fake_import_chatgpt_export(*, export_path, output_dir, **_kwargs):  # noqa: ANN001
        calls.append(Path(export_path))
        slug = f"chatgpt-{Path(export_path).stem}"
        md_path = output_dir / slug / "conversation.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("body", encoding="utf-8")
        return [
            ImportResult(
                markdown_path=md_path,
                html_path=None,
                attachments_dir=None,
                document=None,
                slug=slug,
            )
        ]

    monkeypatch.setattr("polylogue.local_sync.import_chatgpt_export", fake_import_chatgpt_export)

    result = sync_chatgpt_exports(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )

    assert len(result.written) == 2
    assert {res.slug for res in result.written} == {"chatgpt-bundle-1", "chatgpt-bundle-dir"}
    assert set(calls) == {zip_path, dir_path}

    calls.clear()
    result_skip = sync_chatgpt_exports(
        base_dir=base_dir,
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )
    assert not calls
    assert result_skip.skipped >= 2


def test_sync_chatgpt_exports_normalizes_sessions(monkeypatch, tmp_path):
    base_dir = tmp_path / "chatgpt"
    bundle_dir = base_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    conv = bundle_dir / "conversations.json"
    conv.write_text("{}", encoding="utf-8")

    invoked: list[Path] = []

    def fake_import_chatgpt_export(*, export_path, **_kwargs):  # noqa: ANN001
        invoked.append(Path(export_path))
        slug = "chatgpt-bundle"
        md_path = tmp_path / "out" / slug / "conversation.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("body", encoding="utf-8")
        return [
            ImportResult(
                markdown_path=md_path,
                html_path=None,
                attachments_dir=None,
                document=None,
                slug=slug,
            )
        ]

    monkeypatch.setattr("polylogue.local_sync.import_chatgpt_export", fake_import_chatgpt_export)

    result = sync_chatgpt_exports(
        base_dir=base_dir,
        output_dir=tmp_path / "out",
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
        sessions=[conv],
    )

    assert len(result.written) == 1
    assert invoked == [bundle_dir]


def test_sync_claude_exports_counts_skipped(monkeypatch, tmp_path):
    base_dir = tmp_path / "claude"
    bundle_dir = base_dir / "export"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "conversations.json").write_text("{}", encoding="utf-8")

    def fake_import_claude_export(**_kwargs):  # noqa: ANN001
        md_path = tmp_path / "claude_out" / "slug" / "conversation.md"
        return [
            ImportResult(
                markdown_path=md_path,
                html_path=None,
                attachments_dir=None,
                document=None,
                slug="claude-slug",
                skipped=True,
            )
        ]

    monkeypatch.setattr("polylogue.local_sync.import_claude_export", fake_import_claude_export)

    result = sync_claude_exports(
        base_dir=base_dir,
        output_dir=tmp_path / "claude_out",
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=False,
        prune=False,
    )

    assert not result.written
    assert result.skipped == 1


def test_sync_chatgpt_exports_rejects_invalid_sessions(tmp_path):
    bad_path = tmp_path / "missing.zip"
    with pytest.raises(ValueError, match="Invalid ChatGPT export path"):
        sync_chatgpt_exports(
            base_dir=tmp_path / "chatgpt",
            output_dir=tmp_path / "out",
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=False,
            prune=False,
            sessions=[bad_path],
        )


def test_sync_claude_exports_rejects_invalid_sessions(tmp_path):
    bad_path = tmp_path / "missing.zip"
    with pytest.raises(ValueError, match="Invalid Claude export path"):
        sync_claude_exports(
            base_dir=tmp_path / "claude",
            output_dir=tmp_path / "out",
            collapse_threshold=10,
            html=False,
            html_theme="light",
            force=False,
            prune=False,
            sessions=[bad_path],
        )
