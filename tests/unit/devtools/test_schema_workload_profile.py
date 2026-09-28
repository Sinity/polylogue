"""The workload-profile extractor publishes aggregates only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.schema_workload_profile import _stream_families, default_source_root, main, measure
from polylogue.schemas.synthetic.workload import generate_workload_corpus


def _records(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def test_measured_profile_holds_only_aggregates(tmp_path: Path) -> None:
    """Anti-vacuity: a profile that copies text, ids or paths from its sources fails."""
    corpus = generate_workload_corpus(seed=2, target_sessions=8, origins={"claude-code": 1.0})
    corpus.write(tmp_path)
    source_texts = {
        value
        for path in tmp_path.rglob("*.jsonl")
        for record in _records(path.read_bytes())
        for value in (record.get("sessionId"), record.get("uuid"), record.get("cwd"))
        if isinstance(value, str)
    }
    profile = measure("claude-code", tmp_path / "claude-code" / "projects", sample=50, tail=2, seed=1)
    rendered = json.dumps(profile)
    assert profile["streams"]
    assert not [text for text in source_texts if text in rendered]


def test_workflow_journals_are_not_measured_as_subagent_sessions(tmp_path: Path) -> None:
    """Anti-vacuity: counting every ``subagents/**`` stream folds orchestration journals into subagents."""
    projects = tmp_path / "projects"
    corpus = generate_workload_corpus(seed=3, target_sessions=6, origins={"claude-code": 1.0})
    corpus.write(tmp_path)
    (tmp_path / "claude-code" / "projects").rename(projects)
    session_dir = next(path for path in projects.rglob("subagents"))
    journal = session_dir / "workflows" / "wf_1" / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"type": "workflow_step", "state": "done"}\n' * 50, encoding="utf-8")
    families = _stream_families("claude-code", projects)
    assert journal not in families["subagent"]
    assert families["subagent"]


def test_write_refuses_when_no_streams_were_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: writing an empty measurement replaces a usable committed profile."""
    target = tmp_path / "workload-corpus.json"
    target.write_text("committed", encoding="utf-8")
    monkeypatch.setattr("devtools.schema_workload_profile.workload_profile_path", lambda origin: target)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--origin", "claude-code", "--source", str(empty), "--write"]) == 1
    assert target.read_text(encoding="utf-8") == "committed"


def test_default_roots_come_from_the_source_registry() -> None:
    """Anti-vacuity: a hard-coded root that disagrees with Polylogue's own source registry measures nothing."""
    assert default_source_root("claude-code") == Path("~/.claude/projects").expanduser()
    assert default_source_root("codex") == Path("~/.codex/sessions").expanduser()
