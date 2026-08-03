from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.sources.atomic_read import (
    TornReadError,
    read_source_bytes_atomic,
    try_read_source_bytes_atomic,
)


def test_read_source_bytes_atomic_returns_stable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"a":1}\n')

    result = read_source_bytes_atomic(path)

    assert result.payload == b'{"a":1}\n'
    assert result.size == len(b'{"a":1}\n')


def test_read_source_bytes_atomic_detects_torn_read_via_stat_mismatch(tmp_path: Path) -> None:
    """Simulate a mid-rewrite: the file's mtime/size changes between the
    pre-read stat and the post-read stat on every attempt, so every retry
    keeps disagreeing and the call must raise rather than return partial
    bytes -- never a torn blob persisted."""
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"generation-0")

    real_stat = Path.stat
    call_count = {"n": 0}

    def flaky_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        # Every stat call after the very first sees a rewritten file: this
        # forces the pre/post identity comparison to disagree on every pass.
        call_count["n"] += 1
        if call_count["n"] > 1:
            self.write_bytes(f"generation-{call_count['n']}".encode())
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", flaky_stat):
        with pytest.raises(TornReadError) as excinfo:
            read_source_bytes_atomic(path, max_retries=2, retry_delay_s=0.0)

    assert excinfo.value.attempts == 3
    assert excinfo.value.path == str(path)


def test_read_source_bytes_atomic_retries_then_stabilizes(tmp_path: Path) -> None:
    """A file that is mid-rewrite on the first attempt but settles by the
    second retry must return the STABLE final bytes, not the torn ones."""
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"stable-final-content")

    real_read_bytes = Path.read_bytes
    call_count = {"n": 0}

    def flaky_read(self: Path) -> bytes:
        call_count["n"] += 1
        payload = real_read_bytes(self)
        if call_count["n"] == 1:
            # Mutate the file immediately after the read but before the
            # post-read stat, so attempt 1's identity check fails.
            self.write_bytes(self.read_bytes() + b"-appended-mid-read")
        return payload

    with patch.object(Path, "read_bytes", flaky_read):
        result = read_source_bytes_atomic(path, max_retries=3, retry_delay_s=0.0)

    assert result.payload == b"stable-final-content-appended-mid-read"


def test_read_source_bytes_atomic_propagates_final_os_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(FileNotFoundError):
        read_source_bytes_atomic(missing, max_retries=0, retry_delay_s=0.0)


def test_read_source_bytes_atomic_rejects_negative_retries(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_source_bytes_atomic(tmp_path / "x", max_retries=-1)


def test_try_read_source_bytes_atomic_returns_none_on_torn_read(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"generation-0")

    real_stat = Path.stat
    call_count = {"n": 0}

    def flaky_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        call_count["n"] += 1
        if call_count["n"] > 1:
            self.write_bytes(f"generation-{call_count['n']}".encode())
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", flaky_stat):
        result = try_read_source_bytes_atomic(path, max_retries=1, retry_delay_s=0.0)

    assert result is None


def test_try_read_source_bytes_atomic_returns_none_when_missing(tmp_path: Path) -> None:
    assert try_read_source_bytes_atomic(tmp_path / "gone.jsonl", max_retries=0) is None
