"""The workload-profile extractor publishes aggregates only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from devtools.schema_workload_profile import (
    Weights,
    _count_shares,
    _stream_families,
    default_source_root,
    main,
    measure,
)
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
    # Enough sessions that some carry subagent transcripts (most real ones have none).
    corpus = generate_workload_corpus(seed=3, target_sessions=60, origins={"claude-code": 1.0})
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


def test_non_ascii_share_scans_the_whole_measured_text() -> None:
    """Anti-vacuity: checking only text[:2000] leaves a non-ASCII character
    past the prefix uncounted, systematically underreporting non-ASCII texts
    whenever the measured text runs well beyond 2 KiB (as real transcripts
    do), so the generated workload cannot reproduce the advertised rate."""
    record = {"payload": {"type": "custom_tool_call", "input": "x" * 2500 + "é"}}
    shares: Weights = defaultdict(float)
    _count_shares("codex", "custom_tool_call", record, shares, 1.0)
    assert shares.get("non_ascii_texts") == 1.0


def test_default_roots_come_from_the_source_registry() -> None:
    """Anti-vacuity: a hard-coded root that disagrees with Polylogue's own source registry measures nothing."""
    assert default_source_root("claude-code") == Path("~/.claude/projects").expanduser()
    assert default_source_root("codex") == Path("~/.codex/sessions").expanduser()


def test_write_refuses_zero_sampling_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: ``--sample 0`` measures an empty main stream that would overwrite the committed profile."""
    target = tmp_path / "workload-corpus.json"
    target.write_text("committed", encoding="utf-8")
    monkeypatch.setattr("devtools.schema_workload_profile.workload_profile_path", lambda origin: target)
    generate_workload_corpus(seed=1, target_sessions=3, origins={"claude-code": 1.0}).write(tmp_path / "src")
    source = tmp_path / "src" / "claude-code" / "projects"
    with pytest.raises(SystemExit):
        main(["--origin", "claude-code", "--source", str(source), "--sample", "0", "--write"])
    assert target.read_text(encoding="utf-8") == "committed"


def test_template_measures_keep_field_paths_and_list_lengths() -> None:
    """Anti-vacuity (Codex P1/P2, #5670): one kind-wide string pool and no list
    lengths cannot tell a file path from file content or report a 9-item list."""
    from polylogue.schemas.synthetic.workload import template_measures

    record = {
        "type": "attachment",
        "attachment": {"filePath": "/a/b", "content": "x" * 5000, "files": list("abcdefghi")},
    }
    measures = set(template_measures(record))
    assert ("str", "attachment.filePath", 4) in measures
    assert ("str", "attachment.content", 5000) in measures
    assert ("list", "attachment.files", 9) in measures


def test_a_rare_fifth_template_skeleton_is_retained() -> None:
    """Anti-vacuity (Codex P1, #5670): keep the four commonest skeletons and a
    fifth valid variant can never be generated."""
    from devtools.schema_workload_profile import _Templates

    templates = _Templates(frozenset({"type", "a", "b", "c", "d", "e"}), frozenset())
    for index, key in enumerate("abcde"):
        templates.add("record:probe", {"type": "x", key: "v"}, 10.0 - index)
    skeletons, _strings, _lists = templates.payload()
    assert len(skeletons["record:probe"]) == 5  # type: ignore[arg-type]


def test_every_list_item_contributes_its_string_tail() -> None:
    """Anti-vacuity (Codex P1, #5670): measure only the first four items and a
    fifth item's multi-kilobyte payload vanishes from the profile."""
    from polylogue.schemas.synthetic.workload import template_measures

    record = {"items": [{"content": "x"}] * 4 + [{"content": "y" * 50_000}]}
    assert ("str", "items[].content", 50_000) in set(template_measures(record))


@pytest.mark.parametrize(
    ("output", "envelope", "error"),
    [
        ("Wall time: 0.1 seconds\nProcess completed with exit code 1\nOutput:\n", True, True),
        ("Chunk ID: ab12\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\n", True, False),
        ("build log\nProcess exited with code 2\n", False, False),
    ],
)
def test_exec_envelopes_are_recognized_as_the_parser_recognizes_them(output: str, envelope: bool, error: bool) -> None:
    """Anti-vacuity (Codex P2, #5670): a multiline unanchored regex misses the
    older ``Process completed`` envelope and counts a later
    ``Process exited`` line in command output as one."""
    shares: Weights = defaultdict(float)
    _count_shares("codex", "function_call_output", {"payload": {"output": output}}, shares, 1.0)
    assert shares["exec_envelopes"] == (1.0 if envelope else 0.0)
    assert shares["exec_errors"] == (1.0 if error else 0.0)


def _tool_message(*names: str) -> str:
    blocks = [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {"x": "y"}} for i, name in enumerate(names)]
    return json.dumps({"type": "assistant", "message": {"role": "assistant", "content": blocks}}) + "\n"


def test_families_keep_their_own_tool_mix_and_every_parallel_call(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1/P2, #5670): count one call per message and mix
    families into one accumulator, and a two-call main message records one
    Bash call inside a Bash/Read mixture for both families."""
    root = tmp_path / "projects"
    (root / "p" / "s1" / "subagents").mkdir(parents=True)
    (root / "p" / "s1.jsonl").write_text(_tool_message("Bash", "Bash"), encoding="utf-8")
    (root / "p" / "s1" / "subagents" / "agent-a.jsonl").write_text(_tool_message("Read"), encoding="utf-8")
    profile = measure("claude-code", root, sample=10, tail=0, seed=1)
    streams = profile["streams"]
    assert isinstance(streams, dict)
    assert streams["main"]["tool_names"] == {"Bash": 2.0}
    assert streams["subagent"]["tool_names"] == {"Read": 1.0}
    assert streams["main"]["lengths"]["assistant_tool_use:blocks"] == {"2": 1.0}


def test_turn_context_instructions_are_profiled() -> None:
    """Anti-vacuity (Codex P2, #5670): count only the record kind and no
    instruction presence reaches the profile."""
    shares: Weights = defaultdict(float)
    record = {"type": "turn_context", "payload": {"user_instructions": "be brief"}}
    _count_shares("codex", "turn_context", record, shares, 1.0)
    assert shares["turn_context:user_instructions"] == 1.0
    assert shares["turn_context:developer_instructions"] == 0.0
    assert shares["turn_contexts"] == 1.0
