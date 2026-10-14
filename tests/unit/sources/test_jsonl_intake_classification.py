"""Intake excludes only proved non-session streams, independent of record size."""

from __future__ import annotations

import io
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.raw_payload.decode import scan_jsonl_session_artifact
from polylogue.core.enums import Provider
from polylogue.sources.live.batch_support import classify_pre_acquisition

_META = b'{"type":"session_meta","payload":{"id":"bounded","timestamp":"2026-01-01T00:00:00Z"}}\n'
_MESSAGE_START = (
    b'{"type":"response_item","payload":{"type":"message","id":"giant","role":"user",'
    b'"content":[{"type":"input_text","text":"'
)
_MESSAGE_END = b'"}]}}\n'


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "unterminated"])
@pytest.mark.parametrize("scalar_bytes", [4 * 1024 * 1024, 8 * 1024 * 1024], ids=["4MiB", "8MiB"])
def test_giant_jsonl_record_remains_eligible_with_bounded_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool, scalar_bytes: int
) -> None:
    source = tmp_path / "rollout.jsonl"
    with source.open("wb") as output:
        output.write(_META + _MESSAGE_START)
        for _ in range(scalar_bytes // 65536):
            output.write(b"x" * 65536)
        if complete:
            output.write(_MESSAGE_END)
    warm = tmp_path / "warm.jsonl"
    warm.write_bytes(_META + _MESSAGE_START + b"x" + _MESSAGE_END)
    classify_pre_acquisition(warm, fallback_provider=Provider.CODEX, source_only=False, size_bytes=warm.stat().st_size)

    reads: list[int] = []
    original_open = Path.open

    class BoundedReader(io.BufferedReader):
        def read(self, size: int = -1) -> bytes:
            assert 0 <= size <= 1024 * 1024
            reads.append(size)
            return super().read(size)

        def readinto(self, buffer: Any) -> int:
            assert len(buffer) <= 1024 * 1024
            reads.append(len(buffer))
            return super().readinto(buffer)

        def readline(self, size: int = -1) -> bytes:
            raise AssertionError("intake must not read a whole physical record")

    def observed_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == source and args and args[0] == "rb":
            return BoundedReader(io.FileIO(path, "r"))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    tracemalloc.start()
    try:
        decision = classify_pre_acquisition(
            source, fallback_provider=Provider.CODEX, source_only=False, size_bytes=source.stat().st_size
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert reads
    assert decision.excluded_reason is None
    assert decision.detected_provider is Provider.CODEX
    assert decision.detection_crash is None
    assert decision.action is not None
    # A representative Python-heap bound for these scalar inputs, not an RSS
    # or arbitrary-shape claim. Doubling the scalar keeps the same allowance.
    assert peak < 12 * 1024 * 1024
    scan = scan_jsonl_session_artifact(source, provider=Provider.CODEX, source_path=source)
    assert scan.valid_records == (2 if complete else 1)
    assert scan.malformed_records == (0 if complete else 1)
    assert not scan.proved_non_session


@pytest.mark.parametrize("source_only", [False, True], ids=["normal", "source-only"])
@pytest.mark.parametrize(
    ("payload", "excluded"),
    [
        (b"", False),
        (b"{}\n", False),
        (b"not json\n", False),
        (_META, True),
        (_META + b'{"unfinished":', False),
        (_META + _MESSAGE_START + b"x" + _MESSAGE_END + b'{"unfinished":', False),
    ],
    ids=["empty", "unsupported", "malformed", "metadata", "metadata-with-loss", "healthy-prefix-tail"],
)
def test_intake_requires_complete_non_session_proof(
    tmp_path: Path, payload: bytes, excluded: bool, source_only: bool
) -> None:
    source = tmp_path / "rollout.jsonl"
    source.write_bytes(payload)
    decision = classify_pre_acquisition(
        source, fallback_provider=Provider.CODEX, source_only=source_only, size_bytes=len(payload)
    )
    assert (decision.excluded_reason is not None) is excluded
    assert decision.detected_provider is Provider.CODEX
    assert (decision.action is None) is excluded


def test_declared_raw_history_is_retained_without_session_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / ".claude" / "history.jsonl"
    source.parent.mkdir()
    source.write_bytes(b"not a session grammar\n")

    def unexpected_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a raw-only declaration must precede session classification")

    monkeypatch.setattr("polylogue.sources.live.batch_support.scan_jsonl_session_artifact", unexpected_probe)
    decision = classify_pre_acquisition(
        source, fallback_provider=Provider.CLAUDE_CODE, source_only=False, size_bytes=source.stat().st_size
    )
    assert decision.excluded_reason is None
    assert decision.detected_provider is Provider.CLAUDE_CODE
