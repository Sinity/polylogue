"""Distribution-driven synthetic workloads: determinism, parse-through, relations, fidelity."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from polylogue.schemas.synthetic.build_records import _declared_numeric
from polylogue.schemas.synthetic.workload import (
    Histogram,
    _codex_session,
    classify_claude_code_record,
    classify_codex_record,
    default_origin_weights,
    generate_workload_corpus,
    load_workload_profile,
    published_field_names,
    record_skeleton,
    synthetic_text,
    text_measure,
)
from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence


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
        admitted = require_positive_conversational_evidence(sessions, provider=origin, source_path=str(path))
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
        for record in _records(item.data):
            if origin == "claude-code":
                message = record.get("message")
                blocks = message.get("content") if isinstance(message, dict) else None
                for block in blocks if isinstance(blocks, list) else []:
                    if block.get("type") == "tool_use":
                        calls.add(block["id"])
                    elif block.get("type") == "tool_result":
                        assert block["tool_use_id"] in calls
                        results += 1
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


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_record_kind_mix_follows_the_committed_profile(origin: str) -> None:
    """Anti-vacuity: sampling kinds uniformly, or from a fixed role cycle, misses the profile by far more."""
    profile = load_workload_profile(origin)
    expected: Counter[str] = Counter()
    for stream in profile.streams.values():
        for source, row in stream.transitions.items():
            expected[source] += sum(row.values())
    total = sum(expected.values())
    classify = classify_claude_code_record if origin == "claude-code" else classify_codex_record
    observed: Counter[str] = Counter()
    for item in generate_workload_corpus(seed=9, target_bytes=40_000_000, origins={origin: 1.0}).iter_files():
        if item.role in {"transcript", "subagent"}:
            observed.update(classify(record) for record in _records(item.data))
    observed_total = sum(observed.values())
    top = [kind for kind, _ in expected.most_common(6)]
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
    assert all(not text.isascii() and len(text) == 12 for text in texts)


def test_default_origin_mix_follows_session_populations() -> None:
    """Anti-vacuity: weighting sessions by source bytes over-draws the byte-heavy origin."""
    weights = dict(default_origin_weights())
    expected = {origin: load_workload_profile(origin).main_sessions for origin in weights}
    assert weights == {origin: float(count) for origin, count in expected.items()}
    assert all(count > 0 for count in expected.values())


def test_codex_nested_subagents_keep_their_subagent_parent() -> None:
    """Anti-vacuity: collapsing nested spawns into orphans gives them parents that exist nowhere."""
    measured = load_workload_profile("codex")
    assert measured.share("nested_subagents_per_subagent") > 0
    profile = dataclasses.replace(
        measured,
        shares={**measured.shares, "nested_subagents_per_subagent": 0.5, "orphan_subagents_per_session": 0.0},
        subagents_per_session=Histogram((2,), (1.0,)),
    )
    files, _ = _codex_session(random.Random(1), profile, index=0)
    parents = {item.session_id: item.parent_session_id for item in files}
    main = next(item.session_id for item in files if item.parent_session_id is None)
    subagents = {thread for thread, parent in parents.items() if parent is not None}
    assert all(parent == main or parent in subagents for parent in parents.values() if parent is not None)
    assert any(parent in subagents for parent in parents.values())
