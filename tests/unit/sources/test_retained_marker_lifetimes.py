"""Retained marker carriers release each marker-only prepared write promptly."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import closing
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.enums import Provider
from polylogue.pipeline import ids
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


def test_selected_marker_binding_hashes_only_selected_sessions_and_preserves_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = (_session("before"), _session("selected"), _session("after"))
    selected_id = "codex-session:selected"
    borrowed = cast(PreparedSessionWrite, _BorrowedWrite())
    reference = _carrier(raw_id="raw", sessions=sessions, writes=((selected_id, borrowed, ()),))
    try:
        expected_bytes = b"".join(reference.verified_chunks())
        expected_batch = reference.batch
    finally:
        reference.close()
    original_hash = ids.session_content_hash
    hashed_sessions: list[str | None] = []

    def content_hash(session: ParsedSession) -> str:
        hashed_sessions.append(session.provider_session_id)
        return original_hash(session)

    monkeypatch.setattr(ids, "session_content_hash", content_hash)
    tracker = {"live": 0, "maximum": 0, "closed": 0}

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    with closing(
        _prepared_accepted_marker_sessions(
            raw_id="raw",
            artifact=cast(PreparedJsonl, _PreparedArtifact(sessions)),
            selected_session_ids={selected_id},
            prepared_writes={},
            marker_write_factory=make_write,
        )
    ) as writes:
        carrier = _carrier(raw_id="raw", sessions=sessions, writes=writes)
    try:
        assert carrier.batch == expected_batch
        assert b"".join(carrier.verified_chunks()) == expected_bytes
        # The complete request still hashes every cohort member. Only the
        # selected-write pass omits hashes for sessions it cannot emit.
        assert hashed_sessions == ["before", "selected", "after", "selected"], hashed_sessions
        assert tracker == {"live": 0, "maximum": 1, "closed": 1}
    finally:
        carrier.close()


def test_selected_marker_binding_refuses_missing_selected_session() -> None:
    tracker = {"live": 0, "maximum": 0, "closed": 0}

    def make_write(_raw_id: str, _session: ParsedSession) -> PreparedSessionWrite:
        return cast(PreparedSessionWrite, _PreparedWrite(tracker))

    with closing(
        _prepared_accepted_marker_sessions(
            raw_id="raw",
            artifact=cast(PreparedJsonl, _PreparedArtifact((_session("other"),))),
            selected_session_ids={"codex-session:missing"},
            prepared_writes={},
            marker_write_factory=make_write,
        )
    ) as writes:
        with pytest.raises(RuntimeError, match="lost selected parsed session"):
            _carrier(raw_id="raw", sessions=(_session("other"),), writes=writes)
    assert tracker == {"live": 0, "maximum": 0, "closed": 0}


def test_unselected_marker_request_member_still_requires_current_content_hash() -> None:
    sessions = (_session("selected"), _session("other").model_copy(update={"content_hash": "stale"}))
    with pytest.raises(RuntimeError, match="session hash changed during preparation"):
        _carrier(raw_id="raw", sessions=sessions, writes=())


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
