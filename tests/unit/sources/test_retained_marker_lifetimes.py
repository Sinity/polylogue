"""Retained marker carriers release each marker-only prepared write promptly."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import closing
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base_models import ParsedSession
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import (
    _accepted_marker_request_session_binding,
    _prepared_accepted_marker_sessions,
)
from polylogue.storage import accepted_marker_producer
from polylogue.storage.accepted_marker_producer import PreparedAcceptedMarkerCarrier, prepare_accepted_marker_carrier
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite


class _PreparedArtifact:
    def __init__(self, sessions: tuple[ParsedSession, ...]) -> None:
        self.sessions = sessions

    def iter_sessions(self) -> Generator[ParsedSession, None, None]:
        yield from self.sessions


class _PreparedWrite:
    def __init__(self, tracker: dict[str, int]) -> None:
        self.rows = SimpleNamespace(block_rows=())
        self.tracker = tracker
        self.tracker["live"] += 1
        self.tracker["maximum"] = max(self.tracker["maximum"], self.tracker["live"])

    def close(self) -> None:
        self.tracker["live"] -= 1
        self.tracker["closed"] += 1


class _BorrowedWrite:
    def __init__(self) -> None:
        self.rows = SimpleNamespace(block_rows=())
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _session(native_id: str) -> ParsedSession:
    return ParsedSession(source_name=Provider.CODEX, provider_session_id=native_id, messages=[])


def _carrier(
    *,
    raw_id: str,
    sessions: tuple[ParsedSession, ...],
    writes: Iterable[tuple[str, PreparedSessionWrite, tuple[object, ...]]],
) -> PreparedAcceptedMarkerCarrier:
    bindings = tuple(_accepted_marker_request_session_binding(session) for session in sessions)
    return prepare_accepted_marker_carrier(
        raw_id=raw_id,
        request_facts={
            "blob_hash": "b" * 64,
            "provider": "codex",
            "revision_kind": "full",
            "source_path": "sessions.jsonl",
            "parser_fingerprint": "p" * 64,
            "marker_recipe": "m" * 64,
        },
        request_sessions=lambda: iter(bindings),
        prepared_sessions=writes,
    )


def test_multiple_marker_only_sessions_hold_one_temporary_write_at_a_time() -> None:
    sessions = (_session("one"), _session("two"))
    tracker = {"live": 0, "maximum": 0, "closed": 0}
    artifact = cast(PreparedJsonl, _PreparedArtifact(sessions))

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    prepared = _prepared_accepted_marker_sessions(
        raw_id="raw",
        artifact=artifact,
        selected_session_ids={"codex-session:one", "codex-session:two"},
        prepared_writes={},
        marker_write_factory=make_write,
    )
    with closing(prepared) as marker_writes:
        carrier = _carrier(raw_id="raw", sessions=sessions, writes=marker_writes)
    try:
        assert tracker == {"live": 0, "maximum": 1, "closed": 2}
    finally:
        carrier.close()


@pytest.mark.parametrize("failure", [RuntimeError("carrier failed"), DaemonOperationCancelled("cancelled")])
def test_marker_only_write_closes_when_carrier_consumption_fails_or_cancels(failure: BaseException) -> None:
    sessions = (_session("one"), _session("two"))
    tracker = {"live": 0, "maximum": 0, "closed": 0}
    artifact = cast(PreparedJsonl, _PreparedArtifact(sessions))

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    def fail_candidates(_prepared: PreparedSessionWrite) -> Generator[dict[str, object], None, None]:
        raise failure
        yield {}  # Make this a generator while preserving the injected failure.

    prepared = _prepared_accepted_marker_sessions(
        raw_id="raw",
        artifact=artifact,
        selected_session_ids={"codex-session:one", "codex-session:two"},
        prepared_writes={},
        marker_write_factory=make_write,
    )
    with closing(prepared) as marker_writes:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                accepted_marker_producer, "marker_candidates_for_prepared_write_stream", fail_candidates
            )
            with pytest.raises(type(failure), match=str(failure)):
                _carrier(raw_id="raw", sessions=sessions, writes=marker_writes)
    assert tracker == {"live": 0, "maximum": 1, "closed": 1}


def test_borrowed_ordinary_write_is_not_closed_by_marker_stream() -> None:
    sessions = (_session("one"), _session("two"))
    tracker = {"live": 0, "maximum": 0, "closed": 0}
    ordinary = _BorrowedWrite()
    artifact = cast(PreparedJsonl, _PreparedArtifact(sessions))

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    prepared = _prepared_accepted_marker_sessions(
        raw_id="raw",
        artifact=artifact,
        selected_session_ids={"codex-session:one", "codex-session:two"},
        prepared_writes={("raw", "codex-session:one"): cast(PreparedSessionWrite, ordinary)},
        marker_write_factory=make_write,
    )
    with closing(prepared) as marker_writes:
        carrier = _carrier(raw_id="raw", sessions=sessions, writes=marker_writes)
    carrier.close()
    assert tracker["live"] == 0
    assert tracker["closed"] == 1
    assert not ordinary.closed


def test_identical_duplicate_canonical_session_emits_one_prepared_write() -> None:
    session = _session("one")
    sessions = (session, session.model_copy(deep=True))
    tracker = {"live": 0, "maximum": 0, "closed": 0}
    artifact = cast(PreparedJsonl, _PreparedArtifact(sessions))

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    prepared = _prepared_accepted_marker_sessions(
        raw_id="raw",
        artifact=artifact,
        selected_session_ids={"codex-session:one"},
        prepared_writes={},
        marker_write_factory=make_write,
    )
    with closing(prepared) as marker_writes:
        carrier = _carrier(raw_id="raw", sessions=sessions, writes=marker_writes)
    try:
        assert tracker == {"live": 0, "maximum": 1, "closed": 1}
    finally:
        carrier.close()


def test_conflicting_duplicate_canonical_session_refuses_and_closes_current_write() -> None:
    sessions = (_session("one"), _session("one").model_copy(update={"title": "conflict"}))
    tracker = {"live": 0, "maximum": 0, "closed": 0}
    artifact = cast(PreparedJsonl, _PreparedArtifact(sessions))

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    prepared = _prepared_accepted_marker_sessions(
        raw_id="raw",
        artifact=artifact,
        selected_session_ids={"codex-session:one"},
        prepared_writes={},
        marker_write_factory=make_write,
    )
    with closing(prepared) as marker_writes:
        with pytest.raises(RuntimeError, match="conflicting parsed session"):
            _carrier(raw_id="raw", sessions=sessions, writes=marker_writes)
    assert tracker == {"live": 0, "maximum": 1, "closed": 1}
