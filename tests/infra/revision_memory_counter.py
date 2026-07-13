"""Phase-level evidence collection for raw-revision replay memory investigations.

The counter deliberately wraps production boundaries rather than recreating
their work in a synthetic benchmark.  RSS/PSS are reported as live-process
observations when Linux exposes them; assertions use the deterministic
payload/session/spill proxies instead, because allocator and page-cache state
make absolute RSS unsuitable for a portable test budget.
"""

from __future__ import annotations

import os
import pickle
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedSession


@dataclass(frozen=True, slots=True)
class MemoryPhase:
    """A point-in-time process and Python-allocation observation."""

    name: str
    rss_bytes: int | None
    pss_bytes: int | None
    traced_current_bytes: int
    traced_peak_bytes: int


@dataclass
class RevisionMemoryCounter:
    """Count production replay boundaries and their retained-data proxies."""

    calls_by_site: Counter[str] = field(default_factory=Counter)
    bytes_by_site: Counter[str] = field(default_factory=Counter)
    parsed_sessions: int = 0
    parsed_messages: int = 0
    parsed_text_bytes: int = 0
    phases: list[MemoryPhase] = field(default_factory=list)

    def record(self, site: str, byte_count: int = 0) -> None:
        self.calls_by_site[site] += 1
        self.bytes_by_site[site] += int(byte_count)

    def record_parsed_sessions(self, sessions: list[ParsedSession]) -> None:
        self.parsed_sessions += len(sessions)
        self.parsed_messages += sum(len(session.messages) for session in sessions)
        self.parsed_text_bytes += sum(
            len(message.text.encode("utf-8"))
            for session in sessions
            for message in session.messages
            if message.text is not None
        )

    def snapshot(self, name: str) -> None:
        current, peak = tracemalloc.get_traced_memory()
        self.phases.append(
            MemoryPhase(
                name=name,
                rss_bytes=_rss_bytes(),
                pss_bytes=_pss_bytes(),
                traced_current_bytes=current,
                traced_peak_bytes=peak,
            )
        )

    def summary(self) -> str:
        lines = [
            "RevisionMemoryCounter:",
            f"  parsed graph: {self.parsed_sessions} sessions, {self.parsed_messages} messages, "
            f"{self.parsed_text_bytes:,} text bytes",
        ]
        for site in sorted(self.calls_by_site):
            lines.append(f"  {site}: {self.calls_by_site[site]} call(s), {self.bytes_by_site[site]:,} proxy bytes")
        for phase in self.phases:
            lines.append(
                "  "
                f"{phase.name}: rss={_format_bytes(phase.rss_bytes)}, pss={_format_bytes(phase.pss_bytes)}, "
                f"traced={phase.traced_current_bytes:,}, traced_peak={phase.traced_peak_bytes:,}"
            )
        return "\n".join(lines)


def _rss_bytes() -> int | None:
    """Return current resident bytes without making Linux mandatory."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _pss_bytes() -> int | None:
    """Return proportional-set-size from Linux smaps_rollup when available."""
    try:
        for line in Path("/proc/self/smaps_rollup").read_text(encoding="ascii").splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (IndexError, OSError, ValueError):
        return None
    return None


def _format_bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value:,}"


@contextmanager
def revision_memory_counter() -> Iterator[RevisionMemoryCounter]:
    """Instrument parse, spill, hash, and raw-replay production boundaries."""
    from polylogue.sources import revision_backfill
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    counter = RevisionMemoryCounter()
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()

    real_parse = revision_backfill._parse_retained_raw
    real_spill_add = revision_backfill._ParsedSessionSpill.add
    # ``archive.py`` imports this helper as a module-private global, so patch
    # by qualified name rather than widening its public export surface solely
    # for test instrumentation.
    real_hash = session_content_hash
    real_replay = ArchiveStore.apply_raw_revision_replay

    def counted_parse(archive: ArchiveStore, raw_id: str) -> tuple[list[ParsedSession], int]:
        sessions, payload_bytes = real_parse(archive, raw_id)
        counter.record("parse_retained_raw", payload_bytes)
        counter.record_parsed_sessions(sessions)
        counter.snapshot("parse_retained_raw")
        return sessions, payload_bytes

    def counted_spill_add(spill: Any, raw_id: str, sessions: list[ParsedSession], *, payload_bytes: int) -> None:
        serialized_bytes = sum(len(pickle.dumps(session, protocol=pickle.HIGHEST_PROTOCOL)) for session in sessions)
        real_spill_add(spill, raw_id, sessions, payload_bytes=payload_bytes)
        counter.record("parsed_session_spill.add", serialized_bytes)
        counter.snapshot("parsed_session_spill.add")

    def counted_hash(session: ParsedSession) -> Any:
        result = real_hash(session)
        text_bytes = sum(len(message.text.encode("utf-8")) for message in session.messages if message.text is not None)
        counter.record("session_content_hash", text_bytes)
        counter.snapshot("session_content_hash")
        return result

    def counted_replay(
        archive: ArchiveStore,
        plan: Any,
        parsed_by_raw_id: dict[str, ParsedSession],
        *,
        acquired_at_ms: int,
    ) -> tuple[str, tuple[str, ...]]:
        counter.snapshot("apply_raw_revision_replay:before")
        result = real_replay(archive, plan, parsed_by_raw_id, acquired_at_ms=acquired_at_ms)
        counter.record(
            "apply_raw_revision_replay",
            sum(
                len(message.text.encode("utf-8"))
                for session in parsed_by_raw_id.values()
                for message in session.messages
                if message.text is not None
            ),
        )
        counter.snapshot("apply_raw_revision_replay:after")
        return result

    try:
        with (
            patch.object(revision_backfill, "_parse_retained_raw", counted_parse),
            patch.object(revision_backfill._ParsedSessionSpill, "add", counted_spill_add),
            patch("polylogue.storage.sqlite.archive_tiers.archive.session_content_hash", counted_hash),
            patch.object(ArchiveStore, "apply_raw_revision_replay", counted_replay),
        ):
            yield counter
    finally:
        if started_tracing:
            tracemalloc.stop()


__all__ = ["MemoryPhase", "RevisionMemoryCounter", "revision_memory_counter"]
