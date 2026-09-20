"""Untrusted-file probes size their working set, not their input (polylogue-dhkuu).

Two findings, both "a read sized by the input rather than by the reader's own
working set":

* Finding A -- the watcher's incomplete-JSONL-tail probe read the entire
  outstanding tail into one ``bytes``. ``MemoryError`` is not an ``OSError``,
  so the handler never fired, the deferral was never recorded, and every
  catch-up pass re-attempted the identical allocation forever. The
  non-termination is the defect, more than the allocation.
* Finding C -- acquisition re-read the whole stored blob to derive structural
  content identity, immediately after the bounded streaming publication had
  deliberately avoided holding it.
"""

from __future__ import annotations

import io
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.content_identity import (
    CONTENT_IDENTITY_SKIPPED_OVERSIZE,
    bounded_payload_content_identity,
    payload_content_identity,
)
from polylogue.sources.live import watcher as live_watcher


class _ReadRecorder(io.BytesIO):
    """A file object that remembers the size of every ``read`` it served."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_sizes: list[int | None] = []

    def read(self, size: int | None = -1, /) -> bytes:
        self.read_sizes.append(size)
        return super().read(-1 if size is None else size)


# ---------------------------------------------------------------------------
# Finding A -- the watcher tail probe
# ---------------------------------------------------------------------------


def test_tail_scan_never_reads_more_than_one_chunk_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: restore ``handle.read(bytes_to_probe)`` and the single
    recorded read is the whole 40-chunk tail instead of chunk-sized reads."""

    monkeypatch.setattr(live_watcher, "_INCOMPLETE_APPEND_PROBE_CHUNK_BYTES", 64)
    path = tmp_path / "unterminated.jsonl"
    path.write_bytes(b"x" * (64 * 40))

    # Read the bytes BEFORE patching: ``Path.read_bytes`` goes through
    # ``Path.open``, so reading inside the replacement would recurse. The
    # replacement also has to stay transparent for every OTHER path -- the
    # test harness itself opens files through ``Path.open``.
    payload = path.read_bytes()
    recorder: dict[str, _ReadRecorder] = {}
    real_open = Path.open

    def _recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self != path:
            return real_open(self, *args, **kwargs)
        recorder["handle"] = _ReadRecorder(payload)
        return recorder["handle"]

    monkeypatch.setattr(Path, "open", _recording_open)
    found = live_watcher._tail_begins_a_complete_record(path, start_offset=0, scan_bytes=64 * 40)

    assert found is False
    sizes = recorder["handle"].read_sizes
    assert sizes, "the probe performed no read at all"
    assert max(size or 0 for size in sizes) <= 64


def test_tail_scan_finds_a_newline_in_a_later_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete record past the first chunk is still found."""

    monkeypatch.setattr(live_watcher, "_INCOMPLETE_APPEND_PROBE_CHUNK_BYTES", 8)
    path = tmp_path / "late-newline.jsonl"
    path.write_bytes(b"x" * 30 + b"\n" + b"y" * 5)

    assert live_watcher._tail_begins_a_complete_record(path, start_offset=0, scan_bytes=64) is True
    # Still bounded by the caller's allowance: a newline past the window is
    # not visible, which is exactly the bounded probe's declared semantics.
    assert live_watcher._tail_begins_a_complete_record(path, start_offset=0, scan_bytes=16) is False


def test_tail_scan_honours_the_start_offset(tmp_path: Path) -> None:
    path = tmp_path / "offset.jsonl"
    path.write_bytes(b'{"a":1}\n' + b"x" * 20)

    assert live_watcher._tail_begins_a_complete_record(path, start_offset=0, scan_bytes=64) is True
    assert live_watcher._tail_begins_a_complete_record(path, start_offset=8, scan_bytes=64) is False


# ---------------------------------------------------------------------------
# Finding C -- the content-identity re-read
# ---------------------------------------------------------------------------


def test_identity_below_the_ceiling_is_structural_and_reads_the_payload() -> None:
    payload = b'{"b": 1, "a": 2}'
    handle = _ReadRecorder(payload)

    identity, skipped = bounded_payload_content_identity(
        handle, size=len(payload), byte_digest=sha256(payload).hexdigest(), ceiling=1024
    )

    assert skipped is None
    assert identity == payload_content_identity(payload)


def test_identity_above_the_ceiling_reuses_the_byte_digest_and_names_the_skip() -> None:
    """Anti-vacuity: drop the ceiling branch and ``read_sizes`` shows the whole
    payload being loaded again, and ``skipped`` is None so the substitution is
    silent."""

    payload = b'{"a": 1}' + b" " * 4096
    digest = sha256(payload).hexdigest()
    handle = _ReadRecorder(payload)

    identity, skipped = bounded_payload_content_identity(handle, size=len(payload), byte_digest=digest, ceiling=16)

    assert skipped == CONTENT_IDENTITY_SKIPPED_OVERSIZE
    assert identity == digest
    assert handle.read_sizes == [], "an above-ceiling payload must not be read at all"
