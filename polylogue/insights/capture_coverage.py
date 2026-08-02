"""Capture-completeness coverage: the instrument's own coverage error (polylogue-3uw).

Convergence legibility (the rest of ``insights/``) answers "how converged is
what we ingested". Nothing else in the codebase answers "how much of what
exists did we ingest" -- an instrument that does not know its own coverage
error cannot honestly caveat its findings, and a silent capture regression
(a harness stops firing, a watched root goes unreadable, an extension tab
stops relaying) currently has no number to trip an alert on.

This module correlates *sessions known to have happened* against *sessions
fully archived*, per origin over a window, and materializes the result as a
versioned, content-addressed :class:`CoverageAssessment` -- the ``coverage:
<hash>`` sibling of :class:`~polylogue.insights.measurement.metric.MetricDefinition`'s
``metric:<hash>`` (same :func:`~polylogue.insights.measurement.canon.content_ref`
mechanism, so two independent call sites describing the same coverage
question always resolve to the same ref).

Evidence sources named by the 2026-07-13 authoritative design (three
correlated against the archive):

- ``hook_session_start`` -- ``SessionStart`` hook events in
  ``raw_hook_events`` (source.db) without a matching archived session
  (``sessions.native_id`` in index.db) after a grace window. **Computed**
  today for the two origins the hook harnesses cover
  (``claude-code-session``, ``codex-session``; see
  :mod:`polylogue.hooks`).
- ``watcher_file_inventory`` -- harness-written session files on watched
  roots (:mod:`polylogue.sources.dispatch`) that were never acquired into
  ``raw_sessions`` at all. **Not yet wired**: no filesystem-inventory pass
  correlates watched-root contents against ``raw_sessions`` today.
- ``browser_extension_observation`` -- extension-observed chats that never
  reached the acquisition spool (the browser-capture analogue of a
  capture-gap event, tracked separately as polylogue-3v1). **Not yet
  wired**: no capture-gap event stream exists to correlate against yet.

Honesty contract (AC7): a source this module has not wired evidence for is
reported with ``status="unknown"`` and an explicit reason, never silently
folded into a 100%-covered figure. :attr:`CoverageAssessment.coverage_ratio`
and :attr:`CoverageAssessment.is_frame_complete` only ever consider
``status="computed"`` sources, and an assessment with zero computed sources
is itself unknown (``coverage_ratio`` is ``None``, ``is_frame_complete`` is
``False``) rather than vacuously "fully covered".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from polylogue.insights.measurement.canon import content_ref
from polylogue.storage.table_existence import table_exists

CoverageEvidenceSourceKind = Literal[
    "hook_session_start",
    "watcher_file_inventory",
    "browser_extension_observation",
]

CoverageSourceStatus = Literal["computed", "unknown"]

ResultEnumerationFrame = Literal["exact", "frame_incomplete"]

#: Bump when the correlation method (query shape, grace-window semantics,
#: matching rule) changes in a way that makes an old ref's numbers
#: incomparable to a new ref's numbers under the same origin/window.
METHOD_VERSION = 1

#: Evidence sources this module can already correlate, keyed by the hook
#: harness whose evidence backs them (see ``polylogue.hooks.ORIGIN_BY_HARNESS``).
_ORIGINS_WITH_HOOK_EVIDENCE: frozenset[str] = frozenset({"claude-code-session", "codex-session"})

#: Sources the design names but has not wired evidence for yet. Always
#: reported as ``unknown`` with the reason below -- never silently 100%.
_DECLARED_UNCOMPUTED_SOURCES: tuple[tuple[CoverageEvidenceSourceKind, str], ...] = (
    (
        "watcher_file_inventory",
        "harness-written session files on watched roots are not yet correlated against "
        "acquired raw_sessions rows (polylogue-3uw follow-up)",
    ),
    (
        "browser_extension_observation",
        "browser-extension capture-gap events (polylogue-3v1) are not yet materialized",
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceSourceCoverage:
    """One named evidence source's contribution to a :class:`CoverageAssessment`.

    ``status`` is ``"computed"`` only when this module actually correlated
    real evidence against the archive for this source in this window;
    every other source the design names is ``"unknown"`` with an explicit
    ``reason`` (AC7).
    """

    source: CoverageEvidenceSourceKind
    status: CoverageSourceStatus
    reason: str | None = None
    observed_count: int = 0
    matched_count: int = 0
    missing_native_ids: tuple[str, ...] = ()
    missing_count: int = 0

    def canonical_payload(self) -> dict[str, object]:
        """The identity-relevant subset hashed into the parent ref.

        Counts and refs are evidence, not identity: two assessments over
        the same origin/window/method/generations with different observed
        counts are the *same measurement question* re-run at a different
        moment, so they must resolve to the same ``coverage:<hash>`` (the
        content hash names the *question*, not today's *answer* --
        matching :class:`~polylogue.insights.measurement.metric.MetricDefinition`'s
        own construct/unit/aggregation-only identity).
        """

        return {"source": self.source, "status": self.status}


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    """A versioned, content-addressed capture-completeness measure.

    Binds origin, interval, evidence-source inventory, method, and
    generations, per the 2026-07-13 authoritative corrective contract on
    polylogue-3uw: "Materialize a versioned CoverageRef/object carrying
    expected-signal sources, observed/archived counts, known misses, grace
    window, origin/window, archive/source generations, method version, and
    degraded/unknown reasons."
    """

    origin: str
    since_ms: int
    until_ms: int
    grace_window_ms: int
    method_version: int
    generations: Mapping[str, str]
    sources: tuple[EvidenceSourceCoverage, ...]
    computed_at_ms: int

    def canonical_payload(self) -> dict[str, object]:
        """The exact payload hashed to produce :attr:`ref`."""

        return {
            "origin": self.origin,
            "since_ms": self.since_ms,
            "until_ms": self.until_ms,
            "grace_window_ms": self.grace_window_ms,
            "method_version": self.method_version,
            "generations": dict(self.generations),
            "sources": [source.canonical_payload() for source in self.sources],
        }

    @property
    def ref(self) -> str:
        """The content-addressed ``coverage:<hash>`` identity for this question."""

        return content_ref("coverage", self.canonical_payload())

    @property
    def computed_sources(self) -> tuple[EvidenceSourceCoverage, ...]:
        return tuple(source for source in self.sources if source.status == "computed")

    @property
    def unknown_sources(self) -> tuple[EvidenceSourceCoverage, ...]:
        return tuple(source for source in self.sources if source.status == "unknown")

    @property
    def known_miss_count(self) -> int:
        """Total missing-native-id count across computed sources only."""

        return sum(source.missing_count for source in self.computed_sources)

    @property
    def known_missing_native_ids(self) -> tuple[str, ...]:
        """The drillable known-miss list (AC1): union across computed sources."""

        seen: dict[str, None] = {}
        for source in self.computed_sources:
            for native_id in source.missing_native_ids:
                seen[native_id] = None
        return tuple(seen)

    @property
    def coverage_ratio(self) -> float | None:
        """Matched/observed ratio over computed sources, or ``None`` when unknown.

        ``None`` both when there are no computed sources at all and when
        every computed source observed zero signals -- a ratio of exactly
        0 or 1 is never fabricated from an empty denominator.
        """

        computed = self.computed_sources
        if not computed:
            return None
        observed_total = sum(source.observed_count for source in computed)
        if observed_total == 0:
            return None
        matched_total = sum(source.matched_count for source in computed)
        return matched_total / observed_total

    @property
    def is_frame_complete(self) -> bool:
        """Whether this window's evidence supports an exact-enumeration claim.

        ``False`` whenever any computed source has a known miss, or when
        zero sources were computed at all -- an assessment that cannot
        vouch for anything is unknown, not complete (AC6/AC7).
        """

        computed = self.computed_sources
        if not computed:
            return False
        return self.known_miss_count == 0


def compute_capture_coverage(
    *,
    origin: str,
    since_ms: int,
    until_ms: int,
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    grace_window_ms: int = 15 * 60 * 1000,
    missing_ref_limit: int = 50,
    generations: Mapping[str, str] | None = None,
    now_ms: int | None = None,
) -> CoverageAssessment:
    """Correlate ``SessionStart`` hook evidence for ``origin`` against archived sessions.

    ``source_conn``/``index_conn`` are already-open connections to
    source.db/index.db (this function performs read-only queries and never
    manages connection lifecycle -- the production reference path resolves
    and opens them; devtools/health/tests pass fixtures directly).

    The hook-evidence window's upper bound is clamped to
    ``now_ms - grace_window_ms`` so a session that started moments ago and
    has not finished ingest yet is never counted as a miss.
    """

    current_ms = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    effective_until_ms = min(until_ms, current_ms - grace_window_ms)

    sources: list[EvidenceSourceCoverage] = [
        _compute_hook_session_start_coverage(
            origin=origin,
            since_ms=since_ms,
            until_ms=effective_until_ms,
            source_conn=source_conn,
            index_conn=index_conn,
            missing_ref_limit=missing_ref_limit,
        )
    ]
    for source_kind, reason in _DECLARED_UNCOMPUTED_SOURCES:
        sources.append(EvidenceSourceCoverage(source=source_kind, status="unknown", reason=reason))

    return CoverageAssessment(
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
        grace_window_ms=grace_window_ms,
        method_version=METHOD_VERSION,
        generations=dict(generations or {}),
        sources=tuple(sources),
        computed_at_ms=current_ms,
    )


def _compute_hook_session_start_coverage(
    *,
    origin: str,
    since_ms: int,
    until_ms: int,
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    missing_ref_limit: int,
) -> EvidenceSourceCoverage:
    if origin not in _ORIGINS_WITH_HOOK_EVIDENCE:
        return EvidenceSourceCoverage(
            source="hook_session_start",
            status="unknown",
            reason=f"no hook harness reports SessionStart evidence for origin {origin!r}",
        )
    if not table_exists(source_conn, "raw_hook_events"):
        return EvidenceSourceCoverage(
            source="hook_session_start",
            status="unknown",
            reason="raw_hook_events table is not present in source.db",
        )
    if until_ms <= since_ms:
        return EvidenceSourceCoverage(
            source="hook_session_start",
            status="unknown",
            reason="requested window is entirely inside the grace window; no session has had time to finish ingest yet",
        )

    expected_ids = {
        str(row[0])
        for row in source_conn.execute(
            """
            SELECT DISTINCT session_native_id
            FROM raw_hook_events
            WHERE origin = ? AND event_type = 'SessionStart'
              AND session_native_id IS NOT NULL
              AND observed_at_ms >= ? AND observed_at_ms < ?
            """,
            (origin, since_ms, until_ms),
        ).fetchall()
    }
    archived_ids: set[str] = set()
    if expected_ids and table_exists(index_conn, "sessions"):
        archived_ids = {
            str(row[0])
            for row in index_conn.execute(
                "SELECT native_id FROM sessions WHERE origin = ?",
                (origin,),
            ).fetchall()
        }
    missing = tuple(sorted(expected_ids - archived_ids))
    return EvidenceSourceCoverage(
        source="hook_session_start",
        status="computed",
        observed_count=len(expected_ids),
        matched_count=len(expected_ids & archived_ids),
        missing_native_ids=missing[:missing_ref_limit],
        missing_count=len(missing),
    )


def coverage_citation(assessment: CoverageAssessment) -> dict[str, object]:
    """A small, embeddable citation a sample-frame/result-frame stanza can quote (AC3/AC5).

    Deliberately does not repeat the full known-miss list -- callers that
    need to drill down read ``assessment.known_missing_native_ids``
    directly; a citation stanza only needs the ref plus the headline
    numbers to let a reader decide whether to look further.
    """

    return {
        "coverage_ref": assessment.ref,
        "origin": assessment.origin,
        "since_ms": assessment.since_ms,
        "until_ms": assessment.until_ms,
        "coverage_ratio": assessment.coverage_ratio,
        "known_miss_count": assessment.known_miss_count,
        "frame_complete": assessment.is_frame_complete,
        "unknown_sources": [source.source for source in assessment.unknown_sources],
    }


def apply_coverage_to_enumeration(
    claimed_enumeration: ResultEnumerationFrame,
    assessment: CoverageAssessment,
) -> ResultEnumerationFrame:
    """Downgrade an ``"exact"`` enumeration claim when coverage evidence disagrees (AC6).

    A result surface that would otherwise render an ``"exact"`` count or
    enumeration over ``assessment``'s origin/window must route its claimed
    frame through this before rendering: a known miss (or an assessment
    with zero computed sources) forces ``"frame_incomplete"`` instead of
    silently keeping ``"exact"``.
    """

    if claimed_enumeration == "exact" and not assessment.is_frame_complete:
        return "frame_incomplete"
    return claimed_enumeration


def render_capture_coverage_report(assessments: Sequence[CoverageAssessment]) -> str:
    """Render a per-origin, drillable text report (AC1)."""

    if not assessments:
        return "Capture-completeness coverage: no origins assessed\n"

    lines = ["Capture-completeness coverage", ""]
    for assessment in assessments:
        window = f"[{assessment.since_ms}, {assessment.until_ms})"
        lines.append(f"## {assessment.origin} {window}")
        lines.append(f"ref: {assessment.ref}")
        ratio = assessment.coverage_ratio
        ratio_text = f"{ratio:.1%}" if ratio is not None else "unknown"
        lines.append(f"coverage: {ratio_text} (known misses: {assessment.known_miss_count})")
        for source in assessment.sources:
            if source.status == "computed":
                lines.append(
                    f"  - {source.source}: {source.matched_count}/{source.observed_count} matched, "
                    f"{source.missing_count} missing"
                )
                if source.missing_native_ids:
                    shown = ", ".join(source.missing_native_ids)
                    suffix = " (+more)" if source.missing_count > len(source.missing_native_ids) else ""
                    lines.append(f"    missing refs: {shown}{suffix}")
            else:
                lines.append(f"  - {source.source}: unknown ({source.reason})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CoverageAssessment",
    "CoverageEvidenceSourceKind",
    "CoverageSourceStatus",
    "EvidenceSourceCoverage",
    "METHOD_VERSION",
    "ResultEnumerationFrame",
    "apply_coverage_to_enumeration",
    "compute_capture_coverage",
    "coverage_citation",
    "render_capture_coverage_report",
]
