"""Distribution-driven synthetic workloads: determinism, parse-through, relations, fidelity."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from polylogue.schemas.synthetic.workload import (
    Histogram,
    classify_claude_code_record,
    classify_codex_record,
    generate_workload_corpus,
    load_workload_profile,
    record_skeleton,
    synthetic_text,
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


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_every_generated_stream_parses_through_production_dispatch(tmp_path: Path, origin: str) -> None:
    """Anti-vacuity: a renderer emitting a shape the parser rejects or reads as empty fails here."""
    corpus = generate_workload_corpus(seed=11, target_sessions=10, origins={origin: 1.0})
    stats = corpus.write(tmp_path)
    streams = sorted(path for path in tmp_path.rglob("*.jsonl"))
    assert stats.sessions == 10
    assert streams
    for path in streams:
        sessions = parse_payload(origin, _records(path.read_bytes()), str(path), source_path=str(path))
        admitted = require_positive_conversational_evidence(sessions, provider=origin, source_path=str(path))
        assert admitted, path.name
        assert sum(len(session.messages) for session in admitted) > 0


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
