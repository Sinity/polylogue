from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.frontmatter_canonical import canonicalize_markdown
from polylogue.importers.chatgpt import import_chatgpt_export
from polylogue.importers.claude_ai import import_claude_export
from polylogue.importers.claude_code import import_claude_code_session
from polylogue.importers.codex import import_codex_session
from polylogue.renderers.db_renderer import DatabaseRenderer


FIXED_NOW = "2000-01-01T00:00:00Z"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel(path: Path) -> Path:
    root = _repo_root()
    return path.relative_to(root)


def _render_slug(db_path: Path, output_dir: Path, slug: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT provider, conversation_id FROM conversations WHERE slug = ?",
            (slug,),
        ).fetchone()
        assert row, f"slug not found in db: {slug}"
        provider = row["provider"]
        conversation_id = row["conversation_id"]
    finally:
        conn.close()
    renderer = DatabaseRenderer(db_path)
    renderer.render_conversation(provider, conversation_id, output_dir)


def _assert_golden(markdown_path: Path, golden_path: Path) -> None:
    assert markdown_path.exists()
    assert golden_path.exists()
    repo_root = _repo_root()
    actual = canonicalize_markdown(markdown_path.read_text(encoding="utf-8"), repo_root=repo_root, scrub_paths=True)
    expected = canonicalize_markdown(golden_path.read_text(encoding="utf-8"), repo_root=repo_root, scrub_paths=True)
    assert actual == expected


def test_golden_chatgpt_basic(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    results = import_chatgpt_export(
        _rel(fixtures / "chatgpt"),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results
    slug = results[0].slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "chatgpt-basic.md")

def test_golden_chatgpt_tool(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    results = import_chatgpt_export(
        _rel(fixtures / "chatgpt_tool"),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results
    slug = results[0].slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "chatgpt-tool.md")


def test_golden_claude_basic(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    results = import_claude_export(
        _rel(fixtures / "claude"),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results
    slug = results[0].slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "claude-basic.md")

def test_golden_claude_tool(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    results = import_claude_export(
        _rel(fixtures / "claude_tool"),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    assert results
    slug = results[0].slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "claude-tool.md")


def test_golden_codex_basic(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    result = import_codex_session(
        str(_rel(fixtures / "codex" / "codex-golden.jsonl")),
        base_dir=_repo_root(),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        force=True,
    )
    slug = result.slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "codex-basic.md")


def test_golden_claude_code_basic(tmp_path, state_env, monkeypatch):
    monkeypatch.setenv("POLYLOGUE_FIXED_NOW", FIXED_NOW)
    fixtures = _repo_root() / "tests" / "fixtures" / "golden"
    golden_dir = _repo_root() / "tests" / "golden"
    out_dir = tmp_path / "out"

    result = import_claude_code_session(
        str(_rel(fixtures / "claude_code" / "claude-code-golden.jsonl")),
        base_dir=Path("."),
        output_dir=out_dir,
        collapse_threshold=10,
        html=False,
        html_theme="light",
        force=True,
    )
    slug = result.slug
    db_path = Path(state_env) / "polylogue.db"
    _render_slug(db_path, out_dir, slug)
    _assert_golden(out_dir / slug / "conversation.md", golden_dir / "claude-code-basic.md")
