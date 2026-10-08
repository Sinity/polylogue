"""Distribution-driven synthetic workloads: determinism, parse-through, relations, fidelity."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from polylogue.schemas.synthetic.build_records import _declared_numeric
from polylogue.schemas.synthetic.workload import (
    LAZY_TEXT_THRESHOLD,
    Histogram,
    WorkloadFile,
    WorkloadProfile,
    _codex_session,
    _dumps,
    classify_claude_code_record,
    classify_codex_record,
    concat,
    default_origin_weights,
    embedded,
    generate_workload_corpus,
    load_workload_profile,
    measured_text,
    published_field_names,
    record_skeleton,
    synthetic_text,
    text_measure,
)
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from tests.infra.synthetic_workload_bounds import clip_committed_profiles


@pytest.fixture(autouse=True)
def _clipped_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialized corpora stay within test memory (see ``tests.infra.synthetic_workload_bounds``)."""
    clip_committed_profiles(monkeypatch)


def _records(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def test_same_seed_is_byte_identical_and_seeds_differ() -> None:
    """Anti-vacuity: any unseeded randomness (uuid4, time) makes the two runs differ."""
    first = [(item.relpath, item.data) for item in generate_workload_corpus(seed=3, target_sessions=6).iter_files()]
    again = [(item.relpath, item.data) for item in generate_workload_corpus(seed=3, target_sessions=6).iter_files()]
    other = [(item.relpath, item.data) for item in generate_workload_corpus(seed=4, target_sessions=6).iter_files()]
    assert first == again
    assert first != other


_CONVERSATIONAL_KINDS = frozenset(
    {
        "user_text",
        "user_tool_result",
        "assistant_text",
        "assistant_thinking",
        "assistant_tool_use",
        "user_message",
        "assistant_message",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
    }
)


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_every_generated_stream_parses_through_production_dispatch(tmp_path: Path, origin: str) -> None:
    """Anti-vacuity: a renderer emitting a conversational shape the parser rejects or reads as empty fails here.

    Real sources contain metadata-only transcripts (a lone turn-duration or
    snapshot record), which the parser rightly refuses; the generator
    reproduces them, so only streams carrying conversational records must be
    admitted.
    """
    classify = classify_claude_code_record if origin == "claude-code" else classify_codex_record
    corpus = generate_workload_corpus(seed=11, target_sessions=10, origins={origin: 1.0})
    stats = corpus.write(tmp_path)
    streams = sorted(path for path in tmp_path.rglob("*.jsonl"))
    assert stats.sessions == 10
    admitted_streams = 0
    for path in streams:
        records = _records(path.read_bytes())
        conversational = any(classify(record) in _CONVERSATIONAL_KINDS for record in records)
        sessions = parse_payload(origin, records, str(path), source_path=str(path))
        admitted = admit_parsed_sessions_for_publication(sessions, provider=origin, source_path=str(path))
        assert bool(admitted) == conversational, path.name
        if admitted:
            admitted_streams += 1
            assert sum(len(session.messages) for session in admitted) > 0
    assert admitted_streams >= len(streams) * 0.8


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_every_tool_result_answers_an_earlier_call_in_its_stream(origin: str) -> None:
    """Anti-vacuity: random result ids (the schema generator's old behaviour) break pairing."""
    corpus = generate_workload_corpus(seed=5, target_sessions=12, origins={origin: 1.0})
    results = 0
    for item in corpus.iter_files():
        if item.role == "sidecar":
            continue
        calls: set[str] = set()
        answered: set[str] = set()
        every_call = {
            block["id"]
            for record in _records(item.data)
            if isinstance(message := record.get("message"), dict) and isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        for record in _records(item.data):
            if origin == "claude-code":
                message = record.get("message")
                blocks = message.get("content") if isinstance(message, dict) else None
                for block in blocks if isinstance(blocks, list) else []:
                    if block.get("type") == "tool_use":
                        calls.add(block["id"])
                    elif block.get("type") == "tool_result":
                        # A result answers an earlier call, or is an
                        # inherited, unmatched one naming no call at all;
                        # it never names a call that comes later.
                        if block["tool_use_id"] in calls:
                            answered.add(block["tool_use_id"])
                            results += 1
                        else:
                            assert block["tool_use_id"] not in every_call
            else:
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") in {"function_call", "custom_tool_call"}:
                    calls.add(payload["call_id"])
                elif payload.get("type") in {"function_call_output", "custom_tool_call_output"}:
                    assert payload["call_id"] in calls
                    results += 1
    assert results > 0


@pytest.mark.timeout(300)
@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_record_kind_mix_follows_the_committed_profile(origin: str) -> None:
    """Anti-vacuity: sampling kinds uniformly, or from a fixed role cycle, misses the profile by far more."""
    profile = load_workload_profile(origin)
    expected: dict[str, float] = {}
    for stream in profile.streams.values():
        for source, row in stream.transitions.items():
            expected[source] = expected.get(source, 0.0) + sum(row.values())
    total = sum(expected.values())
    classify = classify_claude_code_record if origin == "claude-code" else classify_codex_record
    observed: Counter[str] = Counter()
    for item in generate_workload_corpus(seed=9, target_bytes=40_000_000, origins={origin: 1.0}).iter_files():
        if item.role in {"transcript", "subagent"}:
            observed.update(classify(record) for record in _records(item.data))
    observed_total = sum(observed.values())
    top = sorted(expected, key=lambda kind: -expected[kind])[:6]
    distance = sum(abs(expected[kind] / total - observed[kind] / observed_total) for kind in top)
    assert distance < 0.25, {kind: (expected[kind] / total, observed[kind] / observed_total) for kind in top}


def test_claude_code_sidecar_references_resolve_to_written_files(tmp_path: Path) -> None:
    """Anti-vacuity: an unresolved ``{projects_root}`` placeholder or a missing sidecar file fails."""
    corpus = generate_workload_corpus(seed=21, target_bytes=80_000_000, origins={"claude-code": 1.0})
    corpus.write(tmp_path)
    references = 0
    for path in tmp_path.rglob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        assert "{projects_root}" not in text
        for record in _records(path.read_bytes()):
            message = record.get("message")
            blocks = message.get("content") if isinstance(message, dict) else None
            for block in blocks if isinstance(blocks, list) else []:
                body = block.get("content")
                if isinstance(body, str) and body.startswith("<persisted-output>"):
                    target = body.split("Full output saved to: ", 1)[1].split("\n", 1)[0]
                    assert Path(target).is_file()
                    references += 1
    assert references > 0


def test_lengths_are_sampled_without_a_cap() -> None:
    """Anti-vacuity: clamping text to a fixed maximum (the old 4 KiB string cap) fails."""
    rng = random.Random(1)
    huge = Histogram((23,), (1.0,))
    length = huge.sample(rng)
    assert length >= 1 << 22
    assert len(synthetic_text(rng, length, non_ascii=True)) == length


def test_skeletons_keep_structure_but_no_values_or_data_keys() -> None:
    """Anti-vacuity: keeping values, path-like keys, or schema-property names leaks source data."""
    record = {
        "type": "attachment",
        "attachment": {
            "type": "structured_output",
            "data": {"private_field": "secret"},
            "files": [{"path": "/home/someone/notes.md"}],
        },
        "snapshot": {"trackedFileBackups": {"src/private.py": {"backupTime": "x"}}},
        "/home/someone/key": 1,
        "tool": {"input_schema": {"properties": {"private_argument": {"type": "string"}}}},
    }
    skeleton = record_skeleton(record)
    assert isinstance(skeleton, dict)
    rendered = json.dumps(skeleton)
    for leaked in ("secret", "private_field", "/home/someone", "src/private.py", "private_argument"):
        assert leaked not in rendered
    assert skeleton["attachment"]["files"] == [{"path": "str"}]


def test_multi_block_assistant_record_classifies_by_its_tool_call() -> None:
    """Anti-vacuity: classifying by the first block alone reads ``[thinking, tool_use]`` as thinking."""
    record = {
        "type": "assistant",
        "message": {"content": [{"type": "thinking", "thinking": "x"}, {"type": "tool_use", "input": {"a": "bc"}}]},
    }
    assert classify_claude_code_record(record) == "assistant_tool_use"
    assert text_measure("claude-code", "assistant_tool_use", record) == len('{"a":"bc"}')


def test_written_stats_count_the_bytes_on_disk(tmp_path: Path) -> None:
    """Anti-vacuity: counting pre-resolution sizes misreports corpora whose sidecar paths grow on write."""
    corpus = generate_workload_corpus(seed=21, target_bytes=80_000_000, origins={"claude-code": 1.0})
    stats = corpus.write(tmp_path)
    on_disk = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert stats.bytes == on_disk
    assert stats.per_origin_bytes == {"claude-code": on_disk}


def test_codex_tool_results_carry_structural_outcomes() -> None:
    """Anti-vacuity: bare prose outputs leave every generated Codex result without an exit code."""
    outcomes: Counter[tuple[bool | None, int | None]] = Counter()
    corpus = generate_workload_corpus(seed=13, target_sessions=20, origins={"codex": 1.0})
    for item in corpus.iter_files():
        for session in parse_payload("codex", _records(item.data), item.relpath, source_path=item.relpath):
            for message in session.messages:
                for block in message.blocks:
                    if str(block.type).endswith("tool_result"):
                        outcomes[(block.is_error, block.exit_code)] += 1
    assert outcomes[(False, 0)] > 0
    assert any(is_error and code for (is_error, code) in outcomes)


def test_skeleton_keeps_only_published_field_names() -> None:
    """Anti-vacuity: without the allowlist an unpublished source field name reaches the tracked profile."""
    record = {"type": "attachment", "attachment": {"type": "x", "customer_codename": "y", "hookEvent": "z"}}
    skeleton = record_skeleton(record, allowed=published_field_names("claude-code"))
    assert isinstance(skeleton, dict)
    assert "customer_codename" not in json.dumps(skeleton)
    assert skeleton["attachment"]["hookEvent"] == "str"


def test_numeric_detection_accepts_type_lists_inside_unions() -> None:
    """Anti-vacuity: appending a branch's type list whole raises TypeError on this ordinary schema."""
    assert _declared_numeric({"anyOf": [{"type": ["number", "null"]}]}) == "number"
    assert _declared_numeric({"anyOf": [{"type": ["string", "null"]}, {"type": "object"}]}) is None


def test_integer_only_timestamp_fields_get_integral_defaults() -> None:
    """Anti-vacuity: treating ``integer`` like ``number`` inserts a float that fails the element's schema."""
    assert _declared_numeric({"type": "integer"}) == "integer"
    assert _declared_numeric({"anyOf": [{"type": "integer"}, {"type": "number"}]}) == "number"


def test_short_texts_sampled_non_ascii_always_carry_one() -> None:
    """Anti-vacuity: slicing the mixed pool alone leaves most short slices pure ASCII."""
    rng = random.Random(3)
    texts = [synthetic_text(rng, 12, non_ascii=True) for _ in range(500)]
    assert all(isinstance(text, str) and not text.isascii() and len(text) == 12 for text in texts)


def test_default_origin_mix_follows_session_populations() -> None:
    """Anti-vacuity: weighting sessions by source bytes over-draws the byte-heavy origin."""
    weights = dict(default_origin_weights())
    expected = {origin: load_workload_profile(origin).main_sessions for origin in weights}
    assert weights == {origin: float(count) for origin, count in expected.items()}
    assert all(count > 0 for count in expected.values())


def test_codex_nested_subagents_keep_their_subagent_parent() -> None:
    """Anti-vacuity: collapsing nested spawns into orphans gives them parents that exist nowhere."""
    measured = load_workload_profile("codex")
    assert measured.nested_descendants.buckets != (0,)
    profile = dataclasses.replace(
        measured,
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
        subagents_per_session=Histogram((2,), (1.0,)),
        nested_descendants=Histogram((2,), (1.0,)),
        nested_spawns=Histogram((1,), (1.0,)),
    )
    files, _ = _codex_session(random.Random(1), profile, index=0)
    parents = {item.session_id: item.parent_session_id for item in files}
    starts = {item.session_id: _records(item.data)[0]["timestamp"] for item in files}
    main = next(item.session_id for item in files if item.parent_session_id is None)
    subagents = {thread for thread, parent in parents.items() if parent is not None}
    assert all(parent == main or parent in subagents for parent in parents.values() if parent is not None)
    assert any(parent in subagents for parent in parents.values())
    # A nested child never starts before its direct parent.
    for thread, parent in parents.items():
        if parent in starts:
            assert str(starts[thread]) >= str(starts[parent])


def test_a_long_measured_nesting_chain_generates_a_finite_tree() -> None:
    """Every subagent spawning one more (a rounded mean of 1.0) still ends.

    Anti-vacuity (Codex P1, #5670): apply the rounded nesting mean to every
    child and this session never finishes generating.
    """
    measured = load_workload_profile("codex")
    profile = dataclasses.replace(
        measured,
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
        subagents_per_session=Histogram((1,), (1.0,)),
        nested_descendants=Histogram((5,), (1.0,)),
        nested_spawns=Histogram((1,), (1.0,)),
    )
    files, _ = _codex_session(random.Random(2), profile, index=0)
    subagents = [item for item in files if item.parent_session_id is not None]
    # One first-level subagent and its 16..31 nested descendants, as a chain.
    assert 17 <= len(subagents) <= 32
    parents = [item.parent_session_id for item in subagents]
    assert len(set(parents)) == len(parents)


def test_nesting_is_measured_as_finite_distributions(tmp_path: Path) -> None:
    """A measured chain is its descendant count and one spawn per node.

    Anti-vacuity (Codex P1, #5670): measure only the rounded nested/all
    ratio and a 40-deep chain is indistinguishable from unbounded nesting.
    """
    import uuid as uuid_module

    from devtools.schema_workload_profile import _fanout

    def rollout(thread: str, parent: str | None) -> Path:
        path = tmp_path / f"rollout-2026-01-01T00-00-00-{thread}.jsonl"
        payload: dict[str, object] = {"id": thread}
        if parent is not None:
            payload["parent_thread_id"] = parent
        path.write_text(json.dumps({"type": "session_meta", "payload": payload}) + "\n", encoding="utf-8")
        return path

    main_id = str(uuid_module.UUID(int=1))
    chain = [str(uuid_module.UUID(int=index + 2)) for index in range(40)]
    main = rollout(main_id, None)
    subagents = [rollout(thread, main_id if index == 0 else chain[index - 1]) for index, thread in enumerate(chain)]

    fanout, descendants, spawns, orphans = _fanout("codex", tmp_path, {"main": [main], "subagent": subagents})

    assert dict(descendants) == {6: 1}  # 39 descendants: bucket [32, 63]
    assert dict(spawns) == {1: 39, 0: 1}
    assert orphans == 0.0
    assert dict(fanout) == {1: 1}


def test_a_sampled_claude_result_with_no_open_call_is_kept_unmatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampled result class is emitted even when no call is open.

    Anti-vacuity (Codex P1, #5670): turn such a result into a tool call and
    the generated stream has no results and a call in their place.
    """
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    profile = dataclasses.replace(
        measured,
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(StreamProfile, "kind_sequence", lambda self, rng, count: ["user_tool_result"] * 2)
    files, stats = _claude_code_session(random.Random(8), profile, index=0)
    records = _records(files[0].data)
    blocks = [block for record in records for block in record["message"]["content"]]  # type: ignore[index]
    assert [block["type"] for block in blocks] == ["tool_result", "tool_result"]
    assert stats.tool_calls == 0


def test_a_claude_tool_call_message_keeps_its_thinking_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thinking and text beside tool calls are rendered at their measured multiplicity.

    Anti-vacuity (Codex P1, #5670): render only the calls of a mixed message,
    or collapse its companions to one presence flag, and the second text
    block production parses from it never appears.
    """
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    profile = dataclasses.replace(
        measured,
        streams={
            **measured.streams,
            "main": dataclasses.replace(
                main,
                lengths={
                    **main.lengths,
                    "assistant_tool_use:thinking_blocks": Histogram((1,), (1.0,)),
                    "assistant_tool_use:text_blocks": Histogram((2,), (1.0,)),
                },
            ),
        },
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(StreamProfile, "kind_sequence", lambda self, rng, count: ["assistant_tool_use"])
    files, _stats = _claude_code_session(random.Random(9), profile, index=0)
    content = _records(files[0].data)[0]["message"]["content"]  # type: ignore[index]
    types = [block["type"] for block in content]
    texts = types.count("text")
    assert types[0] == "thinking" and types[1 : 1 + texts] == ["text"] * texts and texts in {2, 3}
    assert set(types[1 + texts :]) == {"tool_use"}


def test_template_booleans_render_at_their_measured_rate() -> None:
    """A flag that is never true stays false.

    Anti-vacuity (Codex P1, #5670): render every boolean leaf as a fair coin
    and about half of these records claim a prevented continuation.
    """
    from polylogue.schemas.synthetic.workload import template_measures

    assert ("bool", "preventedContinuation", 0) in set(template_measures({"preventedContinuation": False}))
    measured = load_workload_profile("claude-code")
    kind = "record:system:stop_hook_summary"
    profile = dataclasses.replace(
        measured,
        templates={kind: (({"preventedContinuation": "bool", "unmeasured": "bool"}, 1.0),)},
        template_bools={kind: {"preventedContinuation": Histogram((0,), (1.0,))}},
    )
    rng = random.Random(10)
    records = [profile.template_record(rng, kind, {}) for _ in range(200)]
    assert not any(record["preventedContinuation"] for record in records)
    assert not any(record["unmeasured"] for record in records)


def test_deep_template_fields_are_kept_and_measured() -> None:
    """A public field nested past six levels keeps its structure and measures.

    Anti-vacuity (Codex P1, #5670): truncate skeletons at depth six and the
    nested ``start``/``end`` fields render as ``{}``.
    """
    from polylogue.schemas.synthetic.workload import template_measures

    deep: dict[str, object] = {"start": {"line": 3}, "end": {"line": 4}}
    for key in ("range", "diagnostics", "files", "attachment", "wrap", "outer"):
        deep = {key: deep}
    skeleton = record_skeleton(deep)
    node: object = skeleton
    for key in ("outer", "wrap", "attachment", "files", "diagnostics", "range"):
        assert isinstance(node, dict)
        node = node[key]
    assert node == {"end": {"line": "int"}, "start": {"line": "int"}}
    paths = {path for _measure, path, _value in template_measures(deep)}
    assert "outer.wrap.attachment.files.diagnostics.range.start.line" in paths


def test_codex_template_events_name_the_active_turn_and_thread() -> None:
    """A template event's turn carriers name the generated turn.

    Anti-vacuity (Codex P1, #5670): fill only envelope fields and
    ``payload.turn_id`` keeps a generated id no ``turn_context`` names.
    """
    from polylogue.schemas.synthetic.workload import _codex_template

    measured = load_workload_profile("codex")
    kind = "record:event_msg:item_completed"
    skeleton = {
        "payload": {
            "turn_id": "str",
            "thread_id": "str",
            "metadata": {"turn_id": "str"},
            "internal_chat_message_metadata_passthrough": {"turn_id": "str"},
        },
        "timestamp": "str",
        "type": "=event_msg",
    }
    profile = dataclasses.replace(measured, templates={kind: ((skeleton, 1.0),)})
    record = _codex_template(profile, random.Random(11), kind, "2026-01-01T00:00:00.000Z", "turn-1", "thread-1")
    payload = record["payload"]
    assert isinstance(payload, dict)
    assert payload["turn_id"] == "turn-1" and payload["thread_id"] == "thread-1"
    assert payload["metadata"] == {"turn_id": "turn-1"}
    assert payload["internal_chat_message_metadata_passthrough"] == {"turn_id": "turn-1"}


def test_claude_agent_progress_names_its_open_dispatch() -> None:
    """An ``agent_progress`` tick names the open Agent call and the child it spawns.

    Anti-vacuity (#5670, same class as the Codex turn binding): keep the
    template's generated ``parentToolUseID``/``data.agentId`` and production
    folds the tick into a dispatch edge to a transcript that does not exist.
    """
    from polylogue.schemas.synthetic.workload import _claude_code_template

    measured = load_workload_profile("claude-code")
    kind = "record:progress:agent_progress"
    skeleton = {"parentToolUseID": "str", "data": {"agentId": "str", "type": "=agent_progress"}}
    profile = dataclasses.replace(measured, templates={kind: ((skeleton, 1.0),)})
    base = {"uuid": "u", "sessionId": "s", "timestamp": "t", "isSidechain": False, "cwd": "/w"}
    open_call: tuple[str, str, str, dict[str, object]] = ("toolu_agent", "u0", "Agent", {})
    bound = _claude_code_template(profile, random.Random(12), kind, base, [open_call], None, ("p", "q"), ["child-1"])
    assert bound["parentToolUseID"] == "toolu_agent"
    assert bound["data"] == {"agentId": "child-1", "type": "agent_progress"}
    unbound = _claude_code_template(profile, random.Random(12), kind, base, [], None, ("p", "q"), ["child-1"])
    assert "agentId" not in unbound["data"]  # type: ignore[operator]


def test_a_byte_target_counts_resolved_sidecar_references() -> None:
    """Sizes are measured after sidecar references name the real root.

    Anti-vacuity (Codex P2, #5670): count ``{projects_root}`` placeholders and
    a long output root writes more bytes than the target accounted for.
    """
    root = "/" + "long-output-root/" * 40
    corpus = generate_workload_corpus(seed=21, target_sessions=20, origins={"claude-code": 1.0})
    for files, stats in corpus.iter_sessions(projects_root=root):
        assert stats.bytes == sum(item.size for item in files)
        assert not any(b"{projects_root}" in item.data for item in files if item.role != "sidecar")


def test_record_type_values_outside_public_vocabulary_are_not_published() -> None:
    """Anti-vacuity: accepting any identifier-like type value publishes private record kinds."""
    assert classify_claude_code_record({"type": "customer_codename"}) == "record:other"
    assert classify_claude_code_record({"type": "system", "subtype": "customer_codename"}) == "record:system"
    assert classify_claude_code_record({"type": "relocated"}) == "record:relocated"
    assert classify_codex_record({"type": "event_msg", "payload": {"type": "customer_codename"}}) == "record:event_msg"


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_every_profiled_kind_has_a_template(origin: str) -> None:
    """Anti-vacuity: a publication threshold that drops a sampled kind's template renders it as an empty record."""
    profile = load_workload_profile(origin)
    kinds = {
        kind
        for stream in profile.streams.values()
        for row in (stream.start, *stream.transitions.values())
        for kind in row
        if kind.startswith("record:")
    }
    assert kinds
    assert kinds <= set(profile.templates)


def test_generated_relocation_carries_its_cwd() -> None:
    """Anti-vacuity: an empty relocation template yields ``type`` alone, which the parser ignores."""
    profile = load_workload_profile("claude-code")
    record = profile.template_record(random.Random(1), "record:relocated", {"type": "relocated"})
    assert "relocatedCwd" in record


def test_claude_tool_inputs_follow_the_tool() -> None:
    """Anti-vacuity: Bash-shaped ``{command}`` inputs for every tool lose paths, edits and dispatch fields."""
    inputs: dict[str, set[str]] = {}
    for item in generate_workload_corpus(seed=8, target_sessions=30, origins={"claude-code": 1.0}).iter_files():
        if item.role == "sidecar":
            continue
        for record in _records(item.data):
            message = record.get("message")
            blocks = message.get("content") if isinstance(message, dict) else None
            for block in blocks if isinstance(blocks, list) else []:
                if block.get("type") == "tool_use":
                    inputs.setdefault(block["name"], set()).update(block["input"])
    assert inputs["Read"] == {"file_path"}
    assert {"file_path", "old_string", "new_string"} <= inputs["Edit"]
    assert {"command"} <= inputs["Bash"]


def test_non_ascii_and_length_are_measured_on_the_classified_block() -> None:
    """Anti-vacuity: reading only ``text``/``content`` fields skips thinking, tool inputs and persisted results."""
    thinking = {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "é" * 5}]}}
    assert measured_text("claude-code", "assistant_thinking", thinking) == "ééééé"
    call = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "x"}, {"type": "tool_use", "input": {"a": "ż"}}]},
    }
    assert measured_text("claude-code", "assistant_tool_use", call) == '{"a":"ż"}'
    persisted = {
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "content": "<persisted-output>\nOutput too large (22.6KB). Full output"}
            ]
        },
    }
    assert text_measure("claude-code", "user_tool_result", persisted) == int(22.6 * 1024)


def test_write_refuses_a_root_holding_an_earlier_workload(tmp_path: Path) -> None:
    """Anti-vacuity: writing over a previous generation leaves its stale files for the daemon to ingest."""
    generate_workload_corpus(seed=1, target_sessions=2, origins={"codex": 1.0}).write(tmp_path)
    with pytest.raises(FileExistsError):
        generate_workload_corpus(seed=2, target_sessions=2, origins={"codex": 1.0}).write(tmp_path)


def test_claude_subagent_ids_match_the_parsed_session_id(tmp_path: Path) -> None:
    """Anti-vacuity: reporting ``owner:a…`` instead of the parsed ``owner:agent-a…`` breaks keyed joins."""
    corpus = generate_workload_corpus(seed=6, target_sessions=40, origins={"claude-code": 1.0})
    corpus.write(tmp_path)
    checked = 0
    for item in corpus.iter_files(projects_root=str((tmp_path / "claude-code" / "projects").resolve())):
        if item.role != "subagent":
            continue
        path = tmp_path / item.relpath
        # Production acquisition passes the file stem as the fallback id.
        parsed = parse_payload("claude-code", _records(path.read_bytes()), path.stem, source_path=str(path))
        assert item.session_id in {session.provider_session_id for session in parsed}
        checked += 1
    assert checked


def test_multi_gigabyte_texts_stream_without_materializing(tmp_path: Path) -> None:
    """Anti-vacuity: building the text as one string allocates ~1.5 GiB here instead of streaming it."""
    import tracemalloc

    rng = random.Random(4)
    huge = synthetic_text(rng, 1_500_000_000, non_ascii=False)
    item = WorkloadFile("claude-code", "x/big.txt", ((huge, 0),), "sidecar", "s")
    tracemalloc.start()
    size = item.size
    head = next(item.iter_chunks())
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert size == 1_500_000_000
    assert head
    assert peak < 64 * 1024 * 1024


def test_lazy_text_encodes_like_json_at_every_escape_level() -> None:
    """Anti-vacuity: escaping lazy pieces differently from ``json.dumps`` corrupts records around large texts."""
    rng = random.Random(5)
    big = synthetic_text(rng, LAZY_TEXT_THRESHOLD + 12_345, non_ascii=True)
    arguments = concat('{"cmd":"', embedded(big), '"}')
    segments = _dumps({"type": "response_item", "payload": {"arguments": arguments, "text": big}})
    line = b"".join(WorkloadFile("codex", "x", segments, "transcript", "s").iter_chunks())
    record = json.loads(line)
    assert json.loads(record["payload"]["arguments"])["cmd"] == record["payload"]["text"]
    assert len(record["payload"]["text"]) == len(big)


def test_codex_apply_patch_calls_carry_their_touched_path() -> None:
    """Anti-vacuity: ``{"cmd": ...}`` arguments for apply_patch leave the parser no path to recover."""
    paths = 0
    # The measured codex profile is heavy-tailed: this seed's first 20 sessions
    # render ~1.5 GB across 248 files, while the first 8 (~9 MiB) already carry
    # 17 apply_patch calls -- enough to falsify a pathless rendering.
    for item in generate_workload_corpus(seed=4, target_sessions=8, origins={"codex": 1.0}).iter_files():
        for session in parse_payload("codex", _records(item.data), item.relpath, source_path=item.relpath):
            for message in session.messages:
                for block in message.blocks:
                    if block.tool_name == "apply_patch" and block.tool_input:
                        assert block.tool_input.get("path")
                        paths += 1
    assert paths > 0


def test_custom_tool_calls_follow_the_measured_apply_patch_share() -> None:
    """Anti-vacuity: unconditionally naming apply_patch makes every custom_tool_call a
    patch, corrupting the tool-class distribution the committed profile measures
    (150k apply_patch vs 560k other)."""
    names: Counter[str] = Counter()
    for item in generate_workload_corpus(seed=11, target_sessions=60, origins={"codex": 1.0}).iter_files():
        for record in _records(item.data):
            payload = record.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "custom_tool_call":
                names[str(payload.get("name"))] += 1
    assert names["other"] > 0, "the measured 'other' share (560k of 710k) must still appear"
    assert names["apply_patch"] > 0


def test_one_record_codex_sessions_stay_one_record() -> None:
    """Anti-vacuity: a lower bound of two records turns every metadata-only session conversational."""
    measured = load_workload_profile("codex")
    profile = dataclasses.replace(
        measured,
        streams={
            **measured.streams,
            "main": dataclasses.replace(measured.streams["main"], records=Histogram((1,), (1.0,))),
        },
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    files, _ = _codex_session(random.Random(2), profile, index=0)
    assert [len(_records(item.data)) for item in files] == [1]


def test_claude_subagents_start_inside_their_owner_and_bind_fork_context(tmp_path: Path) -> None:
    """Anti-vacuity: offsets before the owner's start, or random fork parents, break lineage and timelines."""
    corpus = generate_workload_corpus(seed=12, target_sessions=60, origins={"claude-code": 1.0})
    files = list(corpus.iter_files())
    starts: dict[str, str] = {}
    for item in files:
        if item.role == "transcript":
            stamps = [str(r["timestamp"]) for r in _records(item.data) if "timestamp" in r]
            if stamps:
                starts[item.session_id] = min(stamps)
    checked = 0
    for item in files:
        if item.role != "subagent" or item.parent_session_id not in starts:
            continue
        records = _records(item.data)
        stamps = [str(r["timestamp"]) for r in records if "timestamp" in r]
        if stamps and item.parent_session_id is not None:
            assert min(stamps) >= starts[item.parent_session_id]
            checked += 1
        for record in records:
            if record.get("type") == "fork-context-ref":
                assert record.get("parentSessionId") == item.parent_session_id
    assert checked > 0


def test_codex_completion_items_keep_their_public_type() -> None:
    """Anti-vacuity: a type-free skeleton fills ``item.type`` with gibberish that no call can claim."""
    profile = load_workload_profile("codex")
    skeletons = [skeleton for skeleton, _ in profile.templates["record:event_msg:item_completed"]]
    assert any(
        isinstance(s, dict) and s["payload"]["item"].get("type") in {"=CommandExecution", "=FileChange"}
        for s in skeletons
    )
    record = profile.template_record(random.Random(1), "record:event_msg:item_completed", {})
    assert not str(record.get("type", "")).startswith("=")


def _scripted_codex(
    monkeypatch: pytest.MonkeyPatch, kinds: list[str], **main_fields: object
) -> list[dict[str, object]]:
    from polylogue.schemas.synthetic.workload import StreamProfile

    measured = load_workload_profile("codex")
    profile = dataclasses.replace(
        measured,
        streams={**measured.streams, "main": dataclasses.replace(measured.streams["main"], **main_fields)},  # type: ignore[arg-type]
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(StreamProfile, "kind_sequence", lambda self, rng, count: list(kinds))
    files, _ = _codex_session(random.Random(3), profile, index=0)
    return _records(files[0].data)


def _payloads(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [payload for record in records if isinstance(payload := record.get("payload"), dict)]


def test_an_unmodelled_function_call_is_not_rendered_as_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): rewrite a sampled ``function_call:other``
    to ``exec_command`` and every unmodelled function becomes a shell command."""
    from polylogue.schemas.synthetic.workload import tool_calls_of

    records = _scripted_codex(
        monkeypatch, ["session_meta", "function_call"] * 5, tool_names={"function_call:other": 1.0}
    )
    calls = [record for record in records if _mapping_type(record) == "function_call"]
    assert len(calls) == 5
    for record in calls:
        assert record["payload"]["name"] != "exec_command"  # type: ignore[index]
        assert tool_calls_of("codex", "function_call", record)[0][0] == "other"


def _mapping_type(record: Mapping[str, object]) -> object:
    payload = record.get("payload")
    return payload.get("type") if isinstance(payload, Mapping) else None


def test_event_and_agent_messages_keep_their_template_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): route every ``*_message`` kind through the
    relational renderer and ``record:event_msg:agent_message`` becomes a
    ``response_item`` message with the fabricated role ``record:event_msg:agent``."""
    records = _scripted_codex(monkeypatch, ["session_meta", "record:event_msg:agent_message", "user_message"])
    event = records[1]
    assert event["type"] == "event_msg"
    assert _mapping_type(event) == "agent_message"
    roles = [payload.get("role") for payload in _payloads(records) if payload.get("type") == "message"]
    assert roles == ["user"]


def test_turn_contexts_carry_their_measured_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5670): a hand-built turn context without
    instruction fields never populates ``instructions_text``."""
    records = _scripted_codex(
        monkeypatch,
        ["session_meta", "turn_context", "turn_context"],
        shares={
            "codex_turn_context_user_instructions_share": 1.0,
            "codex_turn_context_developer_instructions_share": 0.0,
        },
    )
    contexts = [record["payload"] for record in records if record.get("type") == "turn_context"]
    assert len(contexts) == 2
    first, second = contexts
    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["user_instructions"] and first["user_instructions"] == second["user_instructions"]
    assert "developer_instructions" not in first


def test_a_codex_child_declares_its_identity_where_the_parser_reads_it() -> None:
    """Anti-vacuity (Codex P2, #5670): write role and nickname only under
    ``source.subagent.thread_spawn`` and the parser's session_meta read finds neither."""
    from polylogue.schemas.synthetic.workload import _Clock, _codex_stream

    profile = load_workload_profile("codex")
    rng = random.Random(4)
    stream = profile.streams["subagent"]
    clock = _Clock(rng, datetime(2026, 1, 1, tzinfo=timezone.utc), stream.gap_ms)
    generated = _codex_stream(rng, profile, stream, thread_id="t", parent_thread_id="p", clock=clock)
    meta = _records(b"".join(_segment_bytes(generated.segments)))[0]["payload"]
    assert isinstance(meta, dict)
    assert meta["agent_role"] and meta["agent_nickname"]


def _segment_bytes(segments: object) -> list[bytes]:
    return [WorkloadFile("codex", "x.jsonl", segments, "transcript", "x").data]  # type: ignore[arg-type]


def test_a_parallel_claude_tool_message_carries_its_measured_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): render one ``tool_use`` block per message
    and parallel calls never reach generated call/result concurrency."""
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    profile = dataclasses.replace(
        measured,
        streams={
            **measured.streams,
            "main": dataclasses.replace(
                main, lengths={**main.lengths, "assistant_tool_use:blocks": Histogram((2,), (1.0,))}
            ),
        },
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(StreamProfile, "kind_sequence", lambda self, rng, count: ["assistant_tool_use"])
    files, stats = _claude_code_session(random.Random(5), profile, index=0)
    message = _records(files[0].data)[0]["message"]
    assert isinstance(message, dict)
    blocks = [block for block in message["content"] if block["type"] == "tool_use"]
    assert len(blocks) >= 2
    assert stats.tool_calls == len(blocks)


def test_each_stream_family_renders_its_own_tool_mix() -> None:
    """Anti-vacuity (Codex P2, #5670): one all-family tool mix renders main
    sessions and subagents from the same mixture."""
    from polylogue.schemas.synthetic.workload import _draw_tool

    profile = load_workload_profile("claude-code")
    main = dataclasses.replace(profile.streams["main"], tool_names={"Bash": 1.0})
    subagent = dataclasses.replace(profile.streams["subagent"], tool_names={"Read": 1.0})
    rng = random.Random(6)
    assert {_draw_tool(rng, profile.for_stream(main), "", ("Grep",)) for _ in range(20)} == {"Bash"}
    assert {_draw_tool(rng, profile.for_stream(subagent), "", ("Grep",)) for _ in range(20)} == {"Read"}


def test_a_result_with_no_open_call_of_its_class_emits_that_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): pop the first open call of any class and
    a sampled function result answers the pending custom call as a
    ``custom_tool_call_output``."""
    records = _scripted_codex(monkeypatch, ["session_meta", "custom_tool_call", "function_call_output"])
    types = [payload.get("type") for record in records if isinstance(payload := record.get("payload"), dict)]
    assert "custom_tool_call_output" not in types
    assert types.count("function_call") == 1


def test_later_session_meta_draws_render_a_replayed_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5670): drop later ``session_meta`` draws and the
    stream never carries the second distinct header continuation logic reads."""
    records = _scripted_codex(monkeypatch, ["session_meta", "user_message", "session_meta"])
    metas = [record["payload"] for record in records if record.get("type") == "session_meta"]
    assert len(metas) == 2
    opening, replayed = metas
    assert isinstance(opening, dict) and isinstance(replayed, dict)
    assert replayed["id"] != opening["id"]
    assert str(replayed["timestamp"]) < str(opening["timestamp"])
    assert replayed["cwd"] == opening["cwd"]


def test_codex_token_subsets_stay_within_their_totals() -> None:
    """Anti-vacuity (Codex P1, #5670): independent draws put cached input above
    input (and reasoning above output) in a large share of token events."""
    checked = 0
    for item in generate_workload_corpus(seed=21, target_sessions=40, origins={"codex": 1.0}).iter_files():
        for record in _records(item.data):
            payload = record.get("payload")
            if not (isinstance(payload, dict) and payload.get("type") == "token_count"):
                continue
            last = payload["info"]["last_token_usage"]
            assert last["cached_input_tokens"] <= last["input_tokens"]
            assert last["reasoning_output_tokens"] <= last["output_tokens"]
            checked += 1
    assert checked > 0


def test_claude_results_carry_tool_specific_evidence(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5670): a generic shell-shaped ``toolUseResult``
    for every call yields no parsed file edits and no Agent result naming a
    generated child transcript."""
    corpus = generate_workload_corpus(seed=12, target_sessions=60, origins={"claude-code": 1.0})
    files = list(corpus.iter_files())
    children = {item.relpath.rsplit("agent-", 1)[1].removesuffix(".jsonl") for item in files if item.role == "subagent"}
    bound = file_edits = 0
    for item in files:
        if item.role != "transcript":
            continue
        records = _records(item.data)
        for record in records:
            result = record.get("toolUseResult")
            if isinstance(result, dict) and result.get("agentId") in children:
                bound += 1
        for session in parse_payload("claude-code", records, item.relpath, source_path=item.relpath):
            for message in session.messages:
                file_edits += sum(1 for block in message.blocks if block.file_edit is not None)
    assert bound > 0
    assert file_edits > 0


def _profile_with_template(
    skeleton: object,
    *,
    template_lists: Mapping[str, Mapping[str, Histogram]] | None = None,
    template_strings: Mapping[str, Mapping[str, Histogram]] | None = None,
) -> WorkloadProfile:
    measured = load_workload_profile("claude-code")
    return dataclasses.replace(
        measured,
        templates={"record:probe": ((skeleton, 1.0),)},
        template_lists=template_lists or {},
        template_strings=template_strings or {},
    )


def test_template_fills_name_only_envelope_fields() -> None:
    """Anti-vacuity (Codex P2, #5670): a key-only recursive fill overwrites the
    nested ``attachment.content.type`` discriminator with the envelope type."""
    profile = _profile_with_template({"type": "str", "attachment": {"content": {"type": "=text"}}})
    record = profile.template_record(random.Random(1), "record:probe", {"type": "attachment"})
    assert record["type"] == "attachment"
    attachment = record["attachment"]
    assert isinstance(attachment, dict)
    assert attachment["content"]["type"] == "text"


def test_template_lists_and_strings_follow_their_own_field_measures() -> None:
    """Anti-vacuity (Codex P1/P2, #5670): regenerate every list with one to three
    items and draw every string from one kind-wide pool, and a measured
    16-31 item list and a 2 KiB-plus content field beside a short path vanish."""
    profile = _profile_with_template(
        {"files": ["str"], "filePath": "str", "content": "str"},
        template_lists={"record:probe": {"files": Histogram((5,), (1.0,))}},
        template_strings={
            "record:probe": {
                "files[]": Histogram((3,), (1.0,)),
                "filePath": Histogram((3,), (1.0,)),
                "content": Histogram((12,), (1.0,)),
            }
        },
    )
    record = profile.template_record(random.Random(2), "record:probe", {})
    files = record["files"]
    assert isinstance(files, list)
    assert 16 <= len(files) <= 31
    assert len(str(record["filePath"])) <= 7
    assert len(str(record["content"])) >= 2048


def test_an_unpaired_codex_result_is_emitted_after_its_synthesized_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): replace the unpaired result with its call
    and the sampled result is never emitted."""
    records = _scripted_codex(monkeypatch, ["session_meta", "custom_tool_call", "function_call_output"])
    types = [_mapping_type(record) for record in records]
    assert types.count("function_call") == 1
    assert types.count("function_call_output") == 1
    assert types.index("function_call") < types.index("function_call_output")


def test_non_exec_codex_tools_get_no_exec_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5670): answer every function call at the broad
    envelope rate and ``update_plan`` results carry ``Process exited`` lines."""
    measured = load_workload_profile("codex").streams["main"]
    records = _scripted_codex(
        monkeypatch,
        ["session_meta", *["function_call", "function_call_output"] * 6],
        tool_names={"function_call:update_plan": 1.0},
        shares={**measured.shares, "codex_exec_envelope_share": 1.0, "codex_exec_envelope_share:update_plan": 0.0},
    )
    outputs = [
        str(payload.get("output")) for payload in _payloads(records) if payload.get("type") == "function_call_output"
    ]
    assert outputs and not [output for output in outputs if "Process exited" in output]


def test_codex_reasoning_carries_its_summary_at_the_measured_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): always emit an empty summary and
    production never materializes reasoning text."""
    measured = load_workload_profile("codex").streams["main"]
    records = _scripted_codex(
        monkeypatch, ["session_meta", "reasoning"], shares={**measured.shares, "codex_reasoning_summary_share": 1.0}
    )
    (reasoning,) = [record for record in records if _mapping_type(record) == "reasoning"]
    assert reasoning["payload"]["summary"]  # type: ignore[index]


def test_parallel_claude_results_share_one_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5670): render one result per message and the
    second parallel call stays unanswered."""
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    lengths = {
        **main.lengths,
        "assistant_tool_use:blocks": Histogram((2,), (1.0,)),
        "user_tool_result:blocks": Histogram((2,), (1.0,)),
    }
    profile = dataclasses.replace(
        measured,
        streams={**measured.streams, "main": dataclasses.replace(main, lengths=lengths)},
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(
        StreamProfile, "kind_sequence", lambda self, rng, count: ["assistant_tool_use", "user_tool_result"]
    )
    files, _stats = _claude_code_session(random.Random(7), profile, index=0)
    call, result = _records(files[0].data)[:2]
    called = [block["id"] for block in call["message"]["content"] if block["type"] == "tool_use"]  # type: ignore[index]
    answered = [block["tool_use_id"] for block in result["message"]["content"] if block["type"] == "tool_result"]  # type: ignore[index]
    # The first blocks answer the parallel calls; any beyond them are
    # unmatched results of calls that precede the transcript.
    assert len(called) >= 2 and len(answered) >= 2
    matched = [tool_use_id for tool_use_id in answered if tool_use_id in called]
    assert matched == called[: len(matched)] and len(matched) == min(len(called), len(answered))


def test_session_bytes_are_merged_in_bounded_segments() -> None:
    """Anti-vacuity (Codex P1, #5670): join every line of a session into one
    buffer and a record-count tail allocates a second transcript-sized copy."""
    from polylogue.schemas.synthetic import workload

    lines = [(b"x" * 1024,) for _ in range(3000)]
    merged = workload._lines(lines)

    assert max(len(segment) for segment in merged if isinstance(segment, bytes)) < 2 * workload._MERGED_SEGMENT_BYTES
    assert len(merged) > 1


def test_a_partly_answered_result_message_keeps_its_sampled_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Result blocks beyond the open calls are emitted unmatched, not dropped.

    Anti-vacuity (Codex P1, #5670): take ``min(open calls, sampled blocks)``
    and a two-block sample with one open call renders one block.
    """
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    lengths = {
        **main.lengths,
        "assistant_tool_use:blocks": Histogram((1,), (1.0,)),
        "user_tool_result:blocks": Histogram((2,), (1.0,)),
        "assistant_tool_use:thinking_blocks": Histogram((0,), (1.0,)),
        "assistant_tool_use:text_blocks": Histogram((0,), (1.0,)),
    }
    profile = dataclasses.replace(
        measured,
        streams={**measured.streams, "main": dataclasses.replace(main, lengths=lengths, tool_names={"Bash": 1.0})},
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(
        StreamProfile, "kind_sequence", lambda self, rng, count: ["assistant_tool_use", "user_tool_result"]
    )
    files, _stats = _claude_code_session(random.Random(10), profile, index=0)
    call, result = _records(files[0].data)[:2]
    called = [block["id"] for block in call["message"]["content"] if block["type"] == "tool_use"]  # type: ignore[index]
    answered = [block["tool_use_id"] for block in result["message"]["content"]]  # type: ignore[index]

    assert len(called) == 1 and len(answered) in {2, 3}
    assert answered[0] == called[0] and not set(answered[1:]) & set(called)


def test_an_envelope_bearing_result_is_answered_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Edit result never shares its record, so its edit is not applied to another block.

    Anti-vacuity (Codex P1, #5670): group Edit and Bash results into one record
    under the first call's ``toolUseResult`` and production attaches the edit
    to the Bash result too.
    """
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    lengths = {
        **main.lengths,
        "assistant_tool_use:blocks": Histogram((2,), (1.0,)),
        "user_tool_result:blocks": Histogram((2,), (1.0,)),
        "assistant_tool_use:thinking_blocks": Histogram((0,), (1.0,)),
        "assistant_tool_use:text_blocks": Histogram((0,), (1.0,)),
    }
    profile = dataclasses.replace(
        measured,
        streams={**measured.streams, "main": dataclasses.replace(main, lengths=lengths, tool_names={"Edit": 1.0})},
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(
        StreamProfile, "kind_sequence", lambda self, rng, count: ["assistant_tool_use", "user_tool_result"]
    )
    files, _stats = _claude_code_session(random.Random(11), profile, index=0)
    result = _records(files[0].data)[1]

    assert len(result["message"]["content"]) == 1  # type: ignore[index]
    assert "oldString" in result["toolUseResult"]  # type: ignore[operator]


def test_a_spawned_agent_sidecar_names_its_spawning_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """``agent-*.meta.json`` carries the Agent call id production joins on.

    Anti-vacuity (Codex P1, #5670): write display metadata only and the
    source-tier sidecar dispatch edge has no join key.
    """
    from polylogue.schemas.synthetic.workload import StreamProfile, _claude_code_session

    measured = load_workload_profile("claude-code")
    main = measured.streams["main"]
    lengths = {
        **main.lengths,
        "assistant_tool_use:blocks": Histogram((1,), (1.0,)),
        "user_tool_result:blocks": Histogram((1,), (1.0,)),
        "assistant_tool_use:thinking_blocks": Histogram((0,), (1.0,)),
        "assistant_tool_use:text_blocks": Histogram((0,), (1.0,)),
    }
    profile = dataclasses.replace(
        measured,
        streams={**measured.streams, "main": dataclasses.replace(main, lengths=lengths, tool_names={"Agent": 1.0})},
        subagents_per_session=Histogram((1,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    real_sequence = StreamProfile.kind_sequence
    main_stream = main

    def sequence(self: StreamProfile, rng: random.Random, count: int) -> list[str]:
        if self.records is main_stream.records:
            return ["assistant_tool_use", "user_tool_result"]
        return real_sequence(self, rng, count)

    monkeypatch.setattr(StreamProfile, "kind_sequence", sequence)
    files, _stats = _claude_code_session(random.Random(12), profile, index=0)
    call = _records(files[0].data)[0]
    call_id = next(block["id"] for block in call["message"]["content"] if block["type"] == "tool_use")  # type: ignore[index]
    meta = next(item for item in files if item.relpath.endswith(".meta.json"))

    assert json.loads(meta.data)["toolUseId"] == call_id


def test_template_floats_render_their_measured_values() -> None:
    """A float leaf keeps its measured magnitude.

    Anti-vacuity (Codex P1, #5670): render floats as uniform 0..100 and a
    cost of about a cent becomes tens of dollars.
    """
    from polylogue.schemas.synthetic.workload import template_measures

    assert ("float", "totalCostUSD", 12) in set(template_measures({"totalCostUSD": 0.012}))
    measured = load_workload_profile("claude-code")
    kind = "record:cost-state"
    profile = dataclasses.replace(
        measured,
        templates={kind: (({"totalCostUSD": "float"}, 1.0),)},
        template_floats={kind: {"totalCostUSD": Histogram((4,), (1.0,))}},
    )
    rng = random.Random(13)
    values = [profile.template_record(rng, kind, {})["totalCostUSD"] for _ in range(100)]

    assert all(isinstance(value, float) and 0.008 <= value <= 0.015 for value in values)


def test_list_skeletons_keep_every_item_shape() -> None:
    """A later list item's optional field survives in the skeleton.

    Anti-vacuity (Codex P1, #5670): keep only the first item's shape and the
    ``range`` of the second diagnostic never reaches generation.
    """
    skeleton = record_skeleton({"items": [{"message": "a"}, {"message": "b", "range": {"line": 1}}]})

    assert skeleton == {"items": [{"message": "str"}, {"message": "str", "range": {"line": "int"}}]}


def test_a_fit_is_exact_when_trimmed_characters_are_escaped() -> None:
    """A body whose characters each measure several units still fits exactly.

    Anti-vacuity: trim once by the overshoot and a trimmed ``\\uXXXX`` escape
    (six units, one character) leaves the field short of its sampled size.
    """
    from polylogue.schemas.synthetic.workload import Plain, _fitted

    def build(body: Plain) -> dict[str, Plain]:
        return {"text": body}

    def escaped_length(built: dict[str, Plain]) -> int:
        return len(json.dumps(built))

    rng = random.Random(3)
    for target in (120, 700, 3_000):
        assert escaped_length(_fitted(rng, target, non_ascii=True, build=build, measure=escaped_length)) == target


def test_codex_arguments_and_claude_inputs_fit_their_sampled_size() -> None:
    """A generated field's whole measured size is the sampled length.

    Anti-vacuity (Codex P2, #5670): pass the sampled length as the body and
    the envelope and escaping make every generated field longer.
    """
    from polylogue.schemas.synthetic.workload import (
        Plain,
        _claude_code_tool_input,
        _codex_arguments,
        _compact_length,
        _fitted,
        _text_length,
    )

    def envelope(build: Callable[[Plain], object], measure: Callable[[Any], int]) -> int:
        # The field with an empty body, drawn from the same randomness the fit replays.
        state = rng.getstate()
        size = measure(build(""))
        rng.setstate(state)
        return size

    rng = random.Random(14)
    # 40 is below both envelopes: the fit is then the envelope alone, never
    # a field longer than an empty body needs.
    for target in (40, 200, 5_000):

        def codex(body: Plain) -> object:
            return _codex_arguments(rng, "exec_command", body)

        def claude(body: Plain) -> object:
            return _claude_code_tool_input(rng, "Bash", "/workspace/x", body)

        least = envelope(codex, _text_length)
        arguments = _fitted(rng, target, non_ascii=False, build=codex, measure=_text_length)
        assert _text_length(arguments) == max(target, least)
        least = envelope(claude, _compact_length)
        tool_input = _fitted(rng, target, non_ascii=False, build=claude, measure=_compact_length)
        assert _compact_length(tool_input) == max(target, least)
