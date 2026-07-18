"""Hermes verification-ledger coverage over one conversational session (fs1.4).

Part of the Hermes forensics report's "verification coverage" section
(polylogue-fs1.4 Phase 4): per session, what verification commands actually
ran and what their final outcome was, correlated from
``verification_evidence.db`` (imported by ``hermes_verification.py``,
Ref polylogue-wj25) against the conversational session it belongs to.

This is deliberately a pure aggregator over an already-fetched event list --
no I/O, unit-testable in isolation -- mirroring
:mod:`polylogue.insights.postmortem`'s own design note. ``session_events``
(the ``session_events`` SQL table) is intentionally not part of the compact
:class:`~polylogue.archive.session.domain_models.Session` envelope
(``read_archive_session_envelope`` never queries it); the owning read surface
resolves the verification-ledger session id
(:func:`hermes_verification.hermes_verification_session_id_for`) and fetches
its events via
:func:`polylogue.storage.sqlite.queries.session_events.get_session_events`
before calling in.

Honesty contract: a conversational session with no correlated verification
session is not an error -- most non-Hermes sessions, and Hermes sessions that
never ran a verification command, legitimately have none. ``available`` is
``False`` in that case, never fabricated as an empty-but-present ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from polylogue.insights.archive_models import ArchiveInsightModel

if TYPE_CHECKING:
    from polylogue.storage.runtime.archive.records import SessionEventRecord

VerificationEventStatus = Literal["passed", "failed"]


class HermesVerificationEventSummary(ArchiveInsightModel):
    """One structural verification outcome, never inferred from output prose."""

    command: str
    canonical_command: str
    kind: str
    scope: str
    status: VerificationEventStatus
    exit_code: int
    timestamp: str | None = None
    ambiguous_correlation: bool = False


class HermesVerificationCoverage(ArchiveInsightModel):
    """Verification-ledger coverage for one Hermes conversational session."""

    hermes_session_id: str
    available: bool
    events: tuple[HermesVerificationEventSummary, ...] = ()
    final_status: VerificationEventStatus | None = None
    changed_paths: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


def _unavailable(hermes_session_id: str, *, reason: str) -> HermesVerificationCoverage:
    return HermesVerificationCoverage(
        hermes_session_id=hermes_session_id,
        available=False,
        caveats=(reason,),
    )


def correlate_verification_coverage(
    hermes_session_id: str,
    verification_events: Sequence[SessionEventRecord] | None,
) -> HermesVerificationCoverage:
    """Summarize one Hermes session's verification-ledger coverage.

    ``verification_events`` is the already-fetched event list for the
    ``verification:<hermes_session_id>`` observer session (see
    ``hermes_verification.observer_session_provider_id`` /
    ``get_session_events``), or ``None`` when that session does not exist in
    the archive -- the common case for sessions that never ran a
    verification command, or non-Hermes sessions.
    """
    if verification_events is None:
        return _unavailable(hermes_session_id, reason="no verification_evidence.db evidence for this session")

    events: list[HermesVerificationEventSummary] = []
    changed_paths: dict[str, None] = {}
    ambiguous_count = 0
    for session_event in verification_events:
        if session_event.event_type == "hermes_verification_event":
            payload = session_event.payload
            status = payload.get("status")
            exit_code = payload.get("exit_code")
            if status not in ("passed", "failed") or not isinstance(exit_code, int):
                # Structural NULL semantics: a row this parser could not
                # normalize into the typed status/exit_code lanes is never
                # silently coerced into a fabricated outcome.
                continue
            verified_status: VerificationEventStatus = status
            ambiguous = bool(payload.get("ambiguous_correlation"))
            if ambiguous:
                ambiguous_count += 1
            events.append(
                HermesVerificationEventSummary(
                    command=str(payload.get("command", "")),
                    canonical_command=str(payload.get("canonical_command", "")),
                    kind=str(payload.get("kind", "")),
                    scope=str(payload.get("scope", "")),
                    status=verified_status,
                    exit_code=exit_code,
                    timestamp=session_event.timestamp,
                    ambiguous_correlation=ambiguous,
                )
            )
        elif session_event.event_type == "hermes_verification_state":
            raw_changed_paths = session_event.payload.get("changed_paths")
            if isinstance(raw_changed_paths, list):
                for path in raw_changed_paths:
                    if isinstance(path, str):
                        changed_paths[path] = None

    caveats: list[str] = []
    if ambiguous_count:
        caveats.append(
            f"{ambiguous_count} event(s) carry ambiguous_correlation (Hermes recorded session_id='default', "
            "its own fallback for unknown session identity) and are included but not fully trusted."
        )
    if not events:
        caveats.append("verification session exists but has no structurally normalized events")

    return HermesVerificationCoverage(
        hermes_session_id=hermes_session_id,
        available=True,
        events=tuple(events),
        final_status=events[-1].status if events else None,
        changed_paths=tuple(changed_paths),
        caveats=tuple(caveats),
    )


__all__ = [
    "HermesVerificationCoverage",
    "HermesVerificationEventSummary",
    "correlate_verification_coverage",
]
