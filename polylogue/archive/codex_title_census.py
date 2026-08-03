"""Corpus-wide Codex UUID-title census (polylogue-ih67 AC#6).

A privacy-safe, bounded report of how many Codex sessions still carry their
native UUID as ``title`` after the assembly-enrichment fix, classifying every
still-unresolved session by structural REASON rather than lumping them all
into an undifferentiated "unresolved" bucket. Reads only structured
``sessions`` columns already present in the derived index tier (counts,
enum-like classification columns, and title-vs-native-id comparison) --
never message text, never file paths. Safe to run against the live archive
read-only and safe to persist/share the resulting JSON.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

CODEX_ORIGIN = "codex-session"


@dataclass(frozen=True, slots=True)
class CodexTitleCensus:
    """Structural counts for one census pass -- no message content, no paths."""

    total_codex_sessions: int
    resolved_count: int
    unresolved_count: int
    resolved_by_title_source: dict[str, int] = field(default_factory=dict)
    unresolved_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def resolved_fraction(self) -> float:
        if self.total_codex_sessions == 0:
            return 1.0
        return self.resolved_count / self.total_codex_sessions

    def to_dict(self) -> dict[str, object]:
        return {
            "total_codex_sessions": self.total_codex_sessions,
            "resolved_count": self.resolved_count,
            "unresolved_count": self.unresolved_count,
            "resolved_fraction": round(self.resolved_fraction, 4),
            "resolved_by_title_source": dict(sorted(self.resolved_by_title_source.items())),
            "unresolved_by_reason": dict(sorted(self.unresolved_by_reason.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CodexTitleCensus:
        def _as_int(value: object) -> int:
            assert isinstance(value, int)
            return value

        def _as_str_int_map(value: object) -> dict[str, int]:
            if value is None:
                return {}
            assert isinstance(value, dict)
            return {str(k): int(cast(int, v)) for k, v in value.items()}

        return cls(
            total_codex_sessions=_as_int(data["total_codex_sessions"]),
            resolved_count=_as_int(data["resolved_count"]),
            unresolved_count=_as_int(data["unresolved_count"]),
            resolved_by_title_source=_as_str_int_map(data.get("resolved_by_title_source")),
            unresolved_by_reason=_as_str_int_map(data.get("unresolved_by_reason")),
        )


def _classify_unresolved_reason(
    *,
    message_count: int,
    authored_user_message_count: int,
    title_source: str | None,
) -> str:
    """Classify why one still-UUID-titled Codex session has no better title.

    Structural only -- never inspects message text. Ordered so each reason
    is mutually exclusive with the ones above it.
    """
    if message_count <= 0:
        return "no_messages_materialized"
    if authored_user_message_count <= 0:
        return "no_human_authored_message"
    if title_source is None:
        # Predates assembly enrichment (or the enrichment step was skipped
        # for this ingest, e.g. no recorded acquisition path) -- a plain
        # reprocess should resolve it once title_ref/title_source wiring has
        # run for this raw record.
        return "not_yet_reprocessed_with_assembly"
    # A human-authored message exists and enrichment ran, yet the title is
    # still the native id: the message-fallback lane's preview synthesis
    # produced nothing usable (e.g. whitespace-only text) or an unexpected
    # code path left title_source stamped without changing title. Distinct
    # from the ordinary "just needs reprocessing" bucket above.
    return "human_authored_present_synthesis_failed"


def compute_codex_title_census(conn: sqlite3.Connection) -> CodexTitleCensus:
    """Compute the census from an index.db connection (read-only safe).

    Ordinary ``sqlite3.connect(..., mode=ro)`` is sufficient; this issues a
    single bounded SELECT over the ``sessions`` table, keyed to the origin
    filter, and aggregates in Python -- no message/block joins.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT native_id, title, title_source, message_count, authored_user_message_count
        FROM sessions
        WHERE origin = ?
        """,
        (CODEX_ORIGIN,),
    ).fetchall()

    total = len(rows)
    resolved_by_source: dict[str, int] = {}
    unresolved_by_reason: dict[str, int] = {}
    resolved_count = 0
    unresolved_count = 0

    for row in rows:
        native_id = row["native_id"]
        title = row["title"]
        title_source = row["title_source"]
        is_unresolved = title is None or title == native_id
        if is_unresolved:
            unresolved_count += 1
            reason = _classify_unresolved_reason(
                message_count=int(row["message_count"] or 0),
                authored_user_message_count=int(row["authored_user_message_count"] or 0),
                title_source=title_source,
            )
            unresolved_by_reason[reason] = unresolved_by_reason.get(reason, 0) + 1
        else:
            resolved_count += 1
            key = title_source or "unknown"
            resolved_by_source[key] = resolved_by_source.get(key, 0) + 1

    return CodexTitleCensus(
        total_codex_sessions=total,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        resolved_by_title_source=resolved_by_source,
        unresolved_by_reason=unresolved_by_reason,
    )


@dataclass(frozen=True, slots=True)
class CodexTitleCensusDelta:
    """Before/after comparison of two census snapshots."""

    before: CodexTitleCensus
    after: CodexTitleCensus

    @property
    def newly_resolved_count(self) -> int:
        return self.after.resolved_count - self.before.resolved_count

    def to_dict(self) -> dict[str, object]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "newly_resolved_count": self.newly_resolved_count,
        }


def compare_censuses(before: CodexTitleCensus, after: CodexTitleCensus) -> CodexTitleCensusDelta:
    return CodexTitleCensusDelta(before=before, after=after)


@dataclass(frozen=True, slots=True)
class CodexHookEventTitleCoverage:
    """Lower-bound simulation of what a live reprocess would resolve via the
    ``codex_thread_title`` hook-event lane alone (bd polylogue-foee AC#3).

    A full reprocess also re-reads the higher-priority live-file lanes
    (provider thread name, ``history.jsonl``, ``state_5.sqlite``), which can
    resolve additional sessions this coverage check does not model -- so
    ``covered_by_hook_event_count`` is a floor, not a prediction of the full
    post-reprocess resolved count. It exists because triggering an actual
    reprocess against the live archive is an operator-authorized mutating
    action outside a read-only census's scope; this reports what the
    already-acquired, durable hook-event evidence alone would fix, without
    running or simulating the rest of the ladder.
    """

    unresolved_count: int
    covered_by_hook_event_count: int

    @property
    def coverage_fraction(self) -> float:
        if self.unresolved_count == 0:
            return 0.0
        return self.covered_by_hook_event_count / self.unresolved_count

    def to_dict(self) -> dict[str, object]:
        return {
            "unresolved_count": self.unresolved_count,
            "covered_by_hook_event_count": self.covered_by_hook_event_count,
            "coverage_fraction": round(self.coverage_fraction, 4),
        }


def compute_codex_hook_event_title_coverage(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
) -> CodexHookEventTitleCoverage:
    """Cross-reference still-unresolved Codex sessions against acquired
    ``codex_thread_title`` hook events (read-only over both ``index.db`` and
    ``source.db`` -- never mutates either).
    """
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.source_write import list_hook_events

    index_conn.row_factory = sqlite3.Row
    rows = index_conn.execute(
        "SELECT native_id, title FROM sessions WHERE origin = ?",
        (CODEX_ORIGIN,),
    ).fetchall()
    unresolved_ids = {row["native_id"] for row in rows if row["title"] is None or row["title"] == row["native_id"]}

    title_by_thread: dict[str, str] = {}
    for event in list_hook_events(source_conn, origin=Origin.CODEX_SESSION):
        if event.event_type != "codex_thread_title":
            continue
        thread_id = event.session_native_id
        title = event.payload.get("title")
        if isinstance(thread_id, str) and thread_id and isinstance(title, str) and title.strip():
            title_by_thread[thread_id] = title.strip()

    covered = unresolved_ids & title_by_thread.keys()
    return CodexHookEventTitleCoverage(
        unresolved_count=len(unresolved_ids),
        covered_by_hook_event_count=len(covered),
    )


__all__ = [
    "CODEX_ORIGIN",
    "CodexHookEventTitleCoverage",
    "CodexTitleCensus",
    "CodexTitleCensusDelta",
    "compare_censuses",
    "compute_codex_hook_event_title_coverage",
    "compute_codex_title_census",
]
