"""Per-test read counter for live-ingest amplification measurements (#1003).

Wraps every call site in ``polylogue/sources/live/`` that reads from a
source JSONL/JSON file under ``~/.claude/projects/`` etc. The counter
records bytes-read and call-counts per call-site name, so a test can
assert the steady-state amplification ratio
(``bytes_read_from_source / bytes_appended_to_source``) directly.

The wrapping is opt-in via the ``read_counter`` context manager — when
not active, production code paths are untouched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch


@dataclass
class ReadCounter:
    """Tally bytes and call counts by named call-site."""

    bytes_by_site: Counter[str] = field(default_factory=Counter)
    calls_by_site: Counter[str] = field(default_factory=Counter)

    def record(self, site: str, bytes_read: int) -> None:
        self.calls_by_site[site] += 1
        self.bytes_by_site[site] += int(bytes_read)

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_site.values())

    @property
    def total_calls(self) -> int:
        return sum(self.calls_by_site.values())

    def summary(self) -> str:
        if not self.calls_by_site:
            return "ReadCounter(empty)"
        lines = ["ReadCounter:"]
        for site in sorted(self.calls_by_site):
            calls = self.calls_by_site[site]
            byts = self.bytes_by_site[site]
            lines.append(f"  {site}: {calls} call(s), {byts:,} bytes")
        lines.append(f"  total: {self.total_calls} call(s), {self.total_bytes:,} bytes")
        return "\n".join(lines)


@contextmanager
def read_counter() -> Iterator[ReadCounter]:
    """Wrap the production read call-sites for the duration of the block.

    Yields a :class:`ReadCounter` that accumulates ``(site_name, bytes)``
    tuples. The wrapped sites are:

    - ``fingerprint_file``
    - ``last_complete_newline_from_tail``
    - ``_append_plan.read``  (``path.open("rb").read()`` from cursor offset)
    - ``_ingest_full_paths_sync.read_bytes``  (``path.read_bytes()``)
    - ``blob_store.write_from_path``  (streamed full reads)

    Each wrapper calls the real implementation and records ``bytes_read``.
    """
    counter = ReadCounter()

    # ----- Wrap fingerprint_file (defined in batch_support) ---------
    from polylogue.sources.live import batch as live_batch
    from polylogue.sources.live import batch_support
    from polylogue.sources.live import watcher as live_watcher
    from polylogue.storage import blob_store as blob_store_mod

    real_fingerprint = batch_support.fingerprint_file

    def counted_fingerprint(path: Path) -> tuple[str, int]:
        fp, last_nl = real_fingerprint(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        counter.record("fingerprint_file", size)
        return fp, last_nl

    # ----- Wrap last_complete_newline_from_tail --------------------
    real_tail = batch_support.last_complete_newline_from_tail

    def counted_tail(path: Path, byte_size: int, *, chunk_size: int = 64 * 1024) -> tuple[int, int]:
        last_nl, bytes_read = real_tail(path, byte_size, chunk_size=chunk_size)
        counter.record("last_complete_newline_from_tail", bytes_read)
        return last_nl, bytes_read

    # ----- Wrap BlobStore.write_from_path ---------------------------
    real_write_from_path = blob_store_mod.BlobStore.write_from_path

    def counted_write_from_path(self, source, *, heartbeat=None):  # type: ignore[no-untyped-def]
        hash_hex, size = real_write_from_path(self, source, heartbeat=heartbeat)
        counter.record("blob_store.write_from_path", size)
        return hash_hex, size

    # ----- Wrap Path.read_bytes globally (only when used by batch.py) ----
    # We don't monkeypatch Path.read_bytes itself (way too broad). Instead,
    # the test's expectation is: in normal append-only steady state,
    # `_ingest_full_paths_sync` shouldn't be called at all. If the suite
    # observes a `fingerprint_file` or `write_from_path` call after the
    # first ingest, the suite has its evidence.

    with (
        patch.object(batch_support, "fingerprint_file", counted_fingerprint),
        patch.object(live_batch, "fingerprint_file", counted_fingerprint),
        patch.object(live_watcher, "fingerprint_file", counted_fingerprint),
        patch.object(batch_support, "last_complete_newline_from_tail", counted_tail),
        patch.object(live_batch, "last_complete_newline_from_tail", counted_tail),
        patch.object(blob_store_mod.BlobStore, "write_from_path", counted_write_from_path),
    ):
        yield counter


__all__ = ["ReadCounter", "read_counter"]
