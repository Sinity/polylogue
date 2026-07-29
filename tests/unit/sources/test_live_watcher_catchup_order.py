"""Regression coverage for catch-up scan ordering (#1616)."""

from __future__ import annotations

from os import stat_result
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polylogue.sources.live.watcher as live_watcher


def _candidate(source: str, path: str) -> live_watcher.CandidateSourceFile:
    fake_stat = cast("stat_result", SimpleNamespace(st_size=0, st_mode=0o100644, st_mtime=0))
    return live_watcher.CandidateSourceFile(
        path=Path(path),
        source_name=source,
        suffix=Path(path).suffix,
        stat=fake_stat,
    )


def test_interleave_by_source_round_robins_families() -> None:
    """Catch-up scans interleave by source so every family progresses early."""
    candidates = [
        _candidate("claude-code", "/home/u/.claude/projects/a/1.jsonl"),
        _candidate("claude-code", "/home/u/.claude/projects/a/2.jsonl"),
        _candidate("claude-code", "/home/u/.claude/projects/a/3.jsonl"),
        _candidate("codex", "/home/u/.codex/sessions/x/1.jsonl"),
        _candidate("codex", "/home/u/.codex/sessions/x/2.jsonl"),
        _candidate("hermes", "/home/u/.hermes/sessions/h.json"),
        _candidate("gemini-cli", "/home/u/.gemini/tmp/g.jsonl"),
    ]

    ordered = live_watcher._interleave_by_source(candidates)

    first_round_sources = {c.source_name for c in ordered[:4]}
    assert first_round_sources == {"claude-code", "codex", "gemini-cli", "hermes"}
    assert sorted(c.path for c in ordered) == sorted(c.path for c in candidates)
    claude_paths = [c.path for c in ordered if c.source_name == "claude-code"]
    assert claude_paths == sorted(claude_paths)


def test_interleave_by_source_empty_input_returns_empty() -> None:
    assert live_watcher._interleave_by_source([]) == []


def test_interleave_by_source_single_source_preserves_alphabetical_order() -> None:
    candidates = [
        _candidate("claude-code", "/p/zz.jsonl"),
        _candidate("claude-code", "/p/aa.jsonl"),
        _candidate("claude-code", "/p/mm.jsonl"),
    ]
    ordered = live_watcher._interleave_by_source(candidates)
    assert [c.path.name for c in ordered] == ["aa.jsonl", "mm.jsonl", "zz.jsonl"]


def test_default_sources_watch_hermes_state_db(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    sources = {source.name: source for source in live_watcher.default_sources()}

    hermes = sources["hermes"]
    assert hermes.root == tmp_path / ".hermes"
    assert hermes.accepts(tmp_path / ".hermes" / "state.db")
    assert hermes.accepts(tmp_path / ".hermes" / "verification_evidence.db")
    assert hermes.accepts(tmp_path / ".hermes" / "sessions" / "snapshot.json")
    assert hermes.accepts(tmp_path / ".hermes" / "observability" / "hermes-atof.jsonl")
    assert sources["inbox"].accepts(tmp_path / "archive" / "inbox" / "state.db")


def test_default_sources_uses_resolved_hermes_root(tmp_path: Path) -> None:
    configured_root = tmp_path / "hermes-profile"

    hermes = next(
        source for source in live_watcher.default_sources(hermes_root=configured_root) if source.name == "hermes"
    )

    assert hermes.root == configured_root
    assert hermes.accepts(configured_root / "observability" / "hermes-atof.jsonl")


def test_default_sources_watch_codex_state_db(monkeypatch: Any, tmp_path: Path) -> None:
    """polylogue-0jf4: a second, narrower WatchSource covers the five live
    ~/.codex SQLite databases without widening the existing "codex" JSONL
    source's root (which would otherwise also walk history.jsonl/config.toml
    /log/ under the shared ~/.codex root)."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    sources = {source.name: source for source in live_watcher.default_sources()}

    codex = sources["codex"]
    assert codex.root == tmp_path / ".codex" / "sessions"

    codex_state_source = sources["codex-state"]
    assert codex_state_source.root == tmp_path / ".codex"
    assert codex_state_source.accepts(tmp_path / ".codex" / "state_5.sqlite")
    assert codex_state_source.accepts(tmp_path / ".codex" / "goals_1.sqlite")
    assert codex_state_source.accepts(tmp_path / ".codex" / "memories_1.sqlite")
    assert codex_state_source.accepts(tmp_path / ".codex" / "logs_2.sqlite")
    assert codex_state_source.accepts(tmp_path / ".codex" / "codex-dev.db")
    # JSONL rollouts and non-sqlite files under the shared root are not
    # picked up by the state-db source -- suffix filtering alone is enough
    # to keep it from ever reasoning about the sessions/ subtree's content.
    assert not codex_state_source.accepts(tmp_path / ".codex" / "history.jsonl")
    assert not codex_state_source.accepts(tmp_path / ".codex" / "config.toml")
