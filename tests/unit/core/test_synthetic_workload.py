"""Workload callers use the measured package session driver."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.record_structure import iter_tool_records, record_kind, tool_identity_operand
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload


@pytest.mark.parametrize("provider", ("chatgpt", "claude-code", "codex", "gemini"))
def test_same_session_driver_is_seeded_and_parses_real_files(
    tmp_path: Path, qualified_schema_registry: SchemaRegistry, provider: str
) -> None:
    selection = select_synthetic_schema(provider, registry_factory=lambda: qualified_schema_registry)

    spec = CorpusSpec.for_provider(provider, count=6, seed=3)

    def collect(seed: int) -> list[bytes]:
        corpus = SyntheticCorpus.from_selection(selection)
        sessions = corpus.iter_sessions(count=6, seed=seed, session_native_ids=spec.session_native_ids)
        try:
            return [b"".join(session.chunks) for session in sessions]
        finally:
            sessions.close()

    first = collect(3)
    assert first == collect(3)
    assert first != collect(4)
    written = SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix="neutral")
    assert [path.read_bytes() for path in written.session_files] == first
    assert written.report.generated_bytes == sum(path.stat().st_size for path in written.files)
    assert written.report.generated_count == len(written.facts) == 6
    for path in written.session_files:
        raw = path.read_bytes()
        payload = [json.loads(line) for line in raw.splitlines()] if path.suffix == ".jsonl" else json.loads(raw)
        parsed = parse_payload(provider, payload, path.stem, source_path=str(path))
        assert parsed and sum(len(session.messages) for session in parsed) > 0


@pytest.mark.parametrize("provider", ("claude-code", "codex"))
def test_session_driver_keeps_original_tool_pairing(
    qualified_tool_schema_registry: SchemaRegistry, provider: str
) -> None:
    selection = select_synthetic_schema(provider, registry_factory=lambda: qualified_tool_schema_registry)
    corpus = SyntheticCorpus.from_selection(selection)
    sessions = corpus.iter_sessions(count=12, seed=5)
    results: Counter[str] = Counter()
    try:
        for session in sessions:
            raw = b"".join(session.chunks)
            parsed = parse_payload(provider, [json.loads(line) for line in raw.splitlines()], "neutral")
            assert parsed
            calls: set[str] = set()
            for record in [json.loads(line) for line in raw.splitlines()]:
                for tool in iter_tool_records(provider, record):
                    operand = tool_identity_operand(tool)
                    if operand is None:
                        continue
                    _kind, event, _field, identity = operand
                    assert isinstance(identity, str)
                    if event == "call":
                        calls.add(identity)
                    else:
                        assert identity in calls
                        results[provider] += 1
    finally:
        sessions.close()
    assert results[provider] > 0


@pytest.mark.parametrize("provider", ("claude-code", "codex"))
def test_session_driver_kind_mix_follows_same_measured_package(
    qualified_tool_schema_registry: SchemaRegistry, provider: str
) -> None:
    selection = select_synthetic_schema(provider, registry_factory=lambda: qualified_tool_schema_registry)
    assert selection.workload_profile is not None
    elements = cast(dict[str, dict[str, object]], selection.workload_profile["elements"])
    structure = cast(dict[str, object], elements[selection.element_kind or "session_record_stream"]["record_structure"])
    expected = Counter(cast(dict[str, int], structure["record_kinds"]))
    assert len(expected) >= 3 and sum(expected.values()) > 0
    corpus = SyntheticCorpus.from_selection(selection)
    sessions = corpus.iter_sessions(count=128, seed=9)
    observed: Counter[str] = Counter()
    try:
        for session in sessions:
            for line in b"".join(session.chunks).splitlines():
                observed[record_kind(provider, json.loads(line))] += 1
    finally:
        sessions.close()
    assert sum(observed.values()) >= 128
    # Same original kind-frequency law, bounded by session count rather than
    # materializing the old forty-megabyte aggregate renderer.
    distance = sum(
        abs(expected[kind] / sum(expected.values()) - observed[kind] / sum(observed.values()))
        for kind in expected.keys() | observed.keys()
    )
    assert distance < 0.25, (provider, expected, observed, distance)


@pytest.mark.parametrize("provider", ("chatgpt", "claude-code", "codex", "gemini"))
def test_written_session_driver_refuses_to_replace_existing_artifact(
    tmp_path: Path, qualified_schema_registry: SchemaRegistry, provider: str
) -> None:
    selection = select_synthetic_schema(provider, registry_factory=lambda: qualified_schema_registry)
    spec = CorpusSpec.for_provider(provider, count=2, seed=3)
    written = SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix="neutral")
    before = {path: path.read_bytes() for path in written.files}
    with pytest.raises(FileExistsError):
        SyntheticCorpus.write_selection_artifacts(selection, spec, tmp_path, prefix="neutral")
    assert {path: path.read_bytes() for path in written.files} == before
