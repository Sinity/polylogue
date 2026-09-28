"""Distribution-driven synthetic workloads: determinism, parse-through, relations, fidelity."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

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
    assert measured.share("nested_subagents_per_subagent") > 0
    profile = dataclasses.replace(
        measured,
        shares={**measured.shares, "nested_subagents_per_subagent": 0.5, "orphan_subagents_per_session": 0.0},
        subagents_per_session=Histogram((2,), (1.0,)),
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
    for item in generate_workload_corpus(seed=4, target_sessions=20, origins={"codex": 1.0}).iter_files():
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


def _scripted_codex(monkeypatch: pytest.MonkeyPatch, kinds: list[str]) -> list[dict[str, object]]:
    from polylogue.schemas.synthetic.workload import StreamProfile

    measured = load_workload_profile("codex")
    profile = dataclasses.replace(
        measured,
        subagents_per_session=Histogram((0,), (1.0,)),
        shares={**measured.shares, "orphan_subagents_per_session": 0.0},
    )
    monkeypatch.setattr(StreamProfile, "kind_sequence", lambda self, rng, count: list(kinds))
    files, _ = _codex_session(random.Random(3), profile, index=0)
    return _records(files[0].data)


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
