"""Raw evidence must not depend on request partitioning or silently change versions."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from polylogue.operations.raw_sessions.memory import MemoryError, MemoryService
from polylogue.operations.raw_sessions.sessions import (
    OpaqueSessionCursor,
    SessionLogService,
    SourceObservationChangedError,
    StaleContinuationError,
)
from polylogue.operations.session_contracts import RawContent, RawRead, session_operation_contracts
from polylogue.operations.session_reads import raw_operation, session_operation_response
from tests.infra.raw_session_evidence import drain_raw_search, write_raw


@pytest.mark.parametrize("padding", [65_523, 65_524, 65_525, 65_526, 65_527])
@pytest.mark.parametrize(("query", "text"), [("aa", "aaaaa"), ("aba", "abababa"), ("xy", "xyxyxy")])
@pytest.mark.parametrize("limit", [1, 100])
def test_nonoverlap_positions_are_invariant_at_real_block_boundaries(
    tmp_path: Path, padding: int, query: str, text: str, limit: int
) -> None:
    """Removing the carried match end admits overlapping hits at 64 KiB boundaries."""
    payload = (json.dumps({"text": "x" * padding + text}, separators=(",", ":")) + "\n").encode()
    sources, _ = write_raw(tmp_path / "raw", payload)
    expected = [(match.start(), match.end()) for match in re.finditer(re.escape(query.encode()), payload)]
    assert len(expected) >= 2
    previous = None
    for budgets in ((65_535,), (65_536,), (8_388_608,), (65_535, 1, 3, 65_536)):
        rows = drain_raw_search(sources, query, budgets, limit, max_pages=40)
        assert len(rows) == len(expected)
        assert [(row.match_offset, row.match_end) for row in rows] == expected
        observed = [(row.line, row.offset, row.text) for row in rows]
        if previous is not None:
            assert observed == previous
        previous = observed


@pytest.mark.parametrize(("first", "second"), [("hello", "world"), ("héllø", "世界")])
@pytest.mark.parametrize("budget", [1, 2, 3, 7, 64])
@pytest.mark.parametrize("limit", [1, 100])
def test_multiline_evidence_contains_the_exact_utf8_match(
    tmp_path: Path, first: str, second: str, budget: int, limit: int
) -> None:
    """Using the later frontier's line start moves the snippet past its own match."""
    prefix = json.dumps({"text": "prefix"}, separators=(",", ":")) + "\n"
    lines = [json.dumps({"text": value}, ensure_ascii=False, separators=(",", ":")) for value in (first, second)]
    payload = (prefix + "\n".join(lines) + "\n").encode()
    query = first + '"}\n{"text":"' + second
    sources, _ = write_raw(tmp_path / "raw", payload)
    rows = drain_raw_search(sources, query, (budget,), limit, max_pages=len(payload) + 3)
    assert len(rows) == 1
    row = rows[0]
    start = payload.index(query.encode())
    assert row.line == 2
    assert row.match_offset == start
    assert row.match_end == start + len(query.encode())
    assert row.offset == len(prefix.encode())
    assert row.text is not None and query in row.text
    assert payload[row.offset : row.offset + len(row.text.encode())].decode() == row.text


def test_literal_can_span_many_request_blocks_and_end_with_newline(tmp_path: Path) -> None:
    query = "é" * 80 + '"}\n'
    payload = (json.dumps({"text": "é" * 80}, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    sources, _ = write_raw(tmp_path / "raw", payload * 2)
    rows = drain_raw_search(sources, query, (1, 7, 3), 1, max_pages=len(payload) * 2 + 5)
    assert len(rows) == 2
    assert [row.match_offset for row in rows] == [9, len(payload) + 9]
    assert all(row.text is not None and query in row.text for row in rows)


def test_pre_nonoverlap_cursor_is_refused_instead_of_changing_match_semantics(tmp_path: Path) -> None:
    sources, _ = write_raw(tmp_path / "raw", b'{"text":"aaaaa"}\n')
    service = SessionLogService(sources=sources)
    cursor = OpaqueSessionCursor(service.scope, b"test-key", "session-search").encode(
        {"snapshot": "not-used"},
        {"file": 0, "offset": 1, "line": 1, "line_start": 0, "after": 0, "skipped": 0},
        version=2,
    )
    with pytest.raises(StaleContinuationError):
        service.search("codex", "aa", cursor=cursor, cursor_key=b"test-key")


@pytest.mark.parametrize("operation", ["sessions.raw.read", "memory.raw.get"])
@pytest.mark.parametrize("change", ["rewrite", "replace", "truncate", "append"])
def test_bound_byte_pages_refuse_a_different_observation(tmp_path: Path, operation: str, change: str) -> None:
    """Without the expected witness, OLD_A and NEW_B can be concatenated successfully."""
    old = (json.dumps({"text": "OLD_A|OLD_B"}, separators=(",", ":")) + "\n").encode()
    new = (json.dumps({"text": "NEW_A|NEW_B"}, separators=(",", ":")) + "\n").encode()
    sources, path = write_raw(tmp_path / "raw", old)
    request = RawRead.model_validate({"operation": operation, "reference": "codex:sample.jsonl", "max_bytes": 15})
    first = raw_operation(request, sources=sources)
    assert first.consistency == "live" and first.next_offset == 15
    bound = request.model_copy(update={"offset": first.next_offset, "expected_observation": first.observation})
    unchanged = raw_operation(bound, sources=sources)
    assert unchanged.consistency == "bound" and unchanged.observation == first.observation
    assert first.content + unchanged.content == old.decode()
    if change == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(new)
        replacement.replace(path)
    elif change == "truncate":
        path.write_bytes(new[:5])
    elif change == "append":
        with path.open("ab") as handle:
            handle.write(new)
    else:
        path.write_bytes(new)
    response = asyncio.run(session_operation_response(None, bound, raw_sources=sources))
    assert response.model_dump()["outcome"] == "error"
    assert response.model_dump()["code"] == "source_changed"


def test_live_tailing_keeps_appends_available_without_a_bound_witness(tmp_path: Path) -> None:
    sources, path = write_raw(tmp_path / "raw", b'{"text":"first"}\n')
    request = RawRead(reference="codex:sample.jsonl")
    first = raw_operation(request, sources=sources)
    assert first.next_offset is None
    added = b'{"text":"second"}\n'
    with path.open("ab") as handle:
        handle.write(added)
    tail = raw_operation(request.model_copy(update={"offset": first.offset + first.bytes}), sources=sources)
    assert tail.consistency == "live" and tail.content == added.decode()
    assert tail.observation != first.observation and tail.coverage.complete


def test_bound_read_detects_same_size_rewrite_with_restored_mtime(tmp_path: Path) -> None:
    sources, path = write_raw(tmp_path / "raw", b'{"text":"old"}\n')
    os.utime(path, ns=(1, 1))
    first = raw_operation(RawRead(reference="codex:sample.jsonl", max_bytes=4), sources=sources)
    assert first.next_offset is not None
    path.write_bytes(b'{"text":"new"}\n')
    os.utime(path, ns=(1, 1))
    with pytest.raises(SourceObservationChangedError):
        raw_operation(
            RawRead(reference=first.reference, offset=first.next_offset, expected_observation=first.observation),
            sources=sources,
        )


def test_read_witness_is_bound_to_the_source_reference(tmp_path: Path) -> None:
    sources, path = write_raw(tmp_path / "raw", b'{"text":"same inode"}\n')
    other = path.with_name("other.jsonl")
    os.link(path, other)
    first = raw_operation(RawRead(reference="codex:sample.jsonl"), sources=sources)
    with pytest.raises(SourceObservationChangedError):
        raw_operation(RawRead(reference="codex:other.jsonl", expected_observation=first.observation), sources=sources)


def test_read_rejects_a_mutation_inside_its_descriptor_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources, path = write_raw(tmp_path / "raw", b'{"text":"OLD_A|OLD_B"}\n')
    original_fstat = os.fstat
    inode = path.stat().st_ino
    reads = 0

    def mutate_before_final_stat(fd: int) -> os.stat_result:
        nonlocal reads
        info = original_fstat(fd)
        if info.st_ino == inode:
            reads += 1
            if reads == 2:
                path.write_bytes(b'{"text":"NEW_A|NEW_B"}\n')
                return original_fstat(fd)
        return info

    monkeypatch.setattr(os, "fstat", mutate_before_final_stat)
    with pytest.raises(SourceObservationChangedError):
        raw_operation(RawRead(reference="codex:sample.jsonl"), sources=sources)
    assert reads == 2


def test_memory_adapter_preserves_bound_read_witness_and_refusal(tmp_path: Path) -> None:
    sources, path = write_raw(tmp_path / "raw", b'{"text":"unchanged"}\n')
    memory = MemoryService(SessionLogService(sources=sources))
    first = memory.get("codex:sample.jsonl", max_bytes=4)
    second = memory.get("codex:sample.jsonl", offset=first["next_offset"], expected_observation=first["observation"])
    assert second["consistency"] == "bound"
    assert first["content"] + second["content"] == path.read_text()
    path.write_bytes(b'{"text":"changed"}\n')
    with pytest.raises(MemoryError) as error:
        memory.get("codex:sample.jsonl", expected_observation=first["observation"])
    assert error.value.code == "source_changed"


def test_generated_contracts_expose_match_spans_and_observation_binding() -> None:
    contracts = session_operation_contracts()["operations"]
    for name in ("sessions.raw.read", "memory.raw.get"):
        assert "expected_observation" in contracts[name]["request_schema"]["properties"]
        assert {"observation", "consistency"} <= contracts[name]["result_schema"]["properties"].keys()
    assert {"match_offset", "match_end"} <= contracts["sessions.raw.search"]["result_schema"]["$defs"][
        "RawObservation"
    ]["properties"].keys()
    assert {"observation", "consistency"} <= RawContent.model_fields.keys()
