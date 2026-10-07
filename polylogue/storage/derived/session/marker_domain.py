"""Consume immutable source marker inputs into the durable user tier.

Accepted marker inputs are source-owned history. The consumer does not look
at the current index session because a later interpretation may replace that
projection before this durable user effect is delivered. The user tier owns
one applied source-stream position; lowering one contiguous batch and
advancing that position commit in the same transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInput,
    AcceptedMarkerInputRefusedError,
    VerifiedAcceptedMarkerPayload,
    verified_marker_payload_from_blob,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    accepted_marker_delivery_cursor,
    advance_accepted_marker_delivery_cursor,
)

if TYPE_CHECKING:
    from polylogue.markers import MarkerCandidate

__all__ = [
    "SESSION_MARKER_DOMAIN",
    "SESSION_MARKER_RECIPE_VERSION",
    "SessionMarkerDerivation",
    "SessionMarkerReplacement",
    "marker_assertions_present",
]

SESSION_MARKER_DOMAIN = "session_markers"
SESSION_MARKER_RECIPE_VERSION = "2"

_VALID = "valid"
_MISSING = "missing"


@dataclass(frozen=True, slots=True)
class SessionMarkerReplacement:
    """One immutable accepted batch, bound to its source stream position."""

    stream_id: str
    sequence: int
    identity: str
    payload: Iterator[MarkerCandidate]
    #: Earlier child-owned candidates this batch's re-extraction re-owned.
    retired: Iterator[str]
    carrier: VerifiedAcceptedMarkerPayload | None = None

    def close(self) -> None:
        """Close the bounded private spool retained through publication."""
        if self.carrier is not None:
            self.carrier.close()

    @property
    def key(self) -> str:
        return f"{self.stream_id}:{self.sequence}"

    @property
    def input_binding(self) -> str:
        """The sealed accepted payload identity this lowering consumed."""
        return self.identity

    @property
    def empty(self) -> bool:
        """An accepted batch with no markers still advances the source cursor."""
        return self.carrier is None or not self.carrier.candidate_count and not self.carrier.retirement_count


def _key(stream_id: str, sequence: int) -> str:
    return f"{stream_id}:{sequence}"


def _parse_key(value: str) -> tuple[str, int]:
    stream_id, separator, raw_sequence = value.rpartition(":")
    if not separator or not stream_id:
        raise AcceptedMarkerInputRefusedError("marker delivery key is malformed")
    try:
        sequence = int(raw_sequence)
    except ValueError as exc:
        raise AcceptedMarkerInputRefusedError("marker delivery key has an invalid sequence") from exc
    if sequence < 1:
        raise AcceptedMarkerInputRefusedError("marker delivery key has an invalid sequence")
    return stream_id, sequence


def marker_assertions_present(conn: sqlite3.Connection, assertion_ids: Sequence[str]) -> bool:
    """Return assertion set membership for the retained legacy unit contract.

    Delivery no longer uses assertion presence as its completion authority. The
    helper remains for callers that need the stable-ID set query itself.
    """
    unique_ids = tuple(dict.fromkeys(assertion_ids))
    if not unique_ids:
        return True
    placeholders = ",".join("?" * len(unique_ids))
    row = conn.execute(
        f"SELECT COUNT(*) FROM assertions WHERE assertion_id IN ({placeholders})",
        unique_ids,
    ).fetchone()
    return row is not None and int(row[0]) == len(unique_ids)


def _retirements(carrier: VerifiedAcceptedMarkerPayload) -> Iterator[str]:
    """Yield sealed retirements one at a time without collecting the batch."""
    for value in carrier.iter_items("sessions.item.retired_assertions.item"):
        if not isinstance(value, str) or not value:
            raise AcceptedMarkerInputRefusedError("accepted marker carrier has an invalid retirement payload")
        yield value


def _candidate(raw_candidate: object) -> MarkerCandidate:
    """Decode one canonical prepared-write candidate."""
    from polylogue.core.enums import AssertionKind
    from polylogue.markers.models import MarkerCandidate, MarkerMatch, MarkerProvenance

    def string(value: object, *, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} is not text")
        return value

    def integer(value: object, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} is not an integer")
        return value

    def boolean(value: object, *, field: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{field} is not a boolean")
        return value

    try:
        if not isinstance(raw_candidate, dict):
            raise TypeError("candidate is not an object")
        raw_match = raw_candidate["match"]
        raw_provenance = raw_candidate["provenance"]
        if not isinstance(raw_match, dict) or not isinstance(raw_provenance, dict):
            raise TypeError("candidate coordinates are not objects")
        arguments = raw_match["arguments"]
        if not isinstance(arguments, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in arguments.items()
        ):
            raise TypeError("candidate arguments are invalid")
        assertion_kind = raw_candidate.get("assertion_kind")
        return MarkerCandidate(
            MarkerMatch(
                kind=string(raw_match["kind"], field="candidate kind"),
                body=string(raw_match["body"], field="candidate body"),
                arguments=cast(dict[str, str], arguments),
                raw_text=string(raw_match["raw_text"], field="candidate raw text"),
                start=integer(raw_match["start"], field="candidate start"),
                end=integer(raw_match["end"], field="candidate end"),
                inline=boolean(raw_match.get("inline", False), field="candidate inline flag"),
                malformed=boolean(raw_match.get("malformed", False), field="candidate malformed flag"),
            ),
            MarkerProvenance(
                message_id=string(raw_provenance["message_id"], field="candidate message id"),
                block_id=string(raw_provenance["block_id"], field="candidate block id"),
            ),
            None if assertion_kind is None else AssertionKind(string(assertion_kind, field="candidate assertion kind")),
            authority=string(raw_candidate.get("authority", "agent-declared"), field="candidate authority"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has an invalid candidate payload") from exc


def _candidates(carrier: VerifiedAcceptedMarkerPayload) -> Iterator[MarkerCandidate]:
    """Yield sealed prepared-write candidates without reparsing current text."""
    for raw_candidate in carrier.iter_items("sessions.item.candidates.item"):
        yield _candidate(raw_candidate)


class SessionMarkerDerivation:
    """Lower accepted source-marker history in source sequence order."""

    domain = SESSION_MARKER_DOMAIN
    name = domain
    prerequisites: tuple[str, ...] = ()
    recipe_version = SESSION_MARKER_RECIPE_VERSION

    def __init__(
        self,
        source_read_connection: Callable[[], sqlite3.Connection],
        marker_read_connection: Callable[[], sqlite3.Connection],
        marker_write_connection: Callable[[], sqlite3.Connection],
        *,
        page_size: int = 200,
    ) -> None:
        if page_size < 1:
            raise ValueError("marker stream page size must be positive")
        self._source_read_connection = source_read_connection
        self._marker_read_connection = marker_read_connection
        self._marker_write_connection = marker_write_connection
        self._page_size = page_size

    def _cursor(self) -> tuple[str, int] | None:
        conn = self._marker_read_connection()
        try:
            return accepted_marker_delivery_cursor(conn)
        finally:
            conn.close()

    def _accepted_page(self, *, after_sequence: int, limit: int) -> tuple[AcceptedMarkerInput, ...]:
        from polylogue.storage.accepted_marker_inputs import read_accepted_marker_inputs_sync

        conn = self._source_read_connection()
        try:
            return read_accepted_marker_inputs_sync(
                conn, after_sequence=after_sequence, limit=min(limit, self._page_size)
            )
        finally:
            conn.close()

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Page committed source batches without overtaking an unconsumed head."""
        del frame
        applied = self._cursor()
        after_sequence = 0 if applied is None else applied[1]
        # The orchestration protocol commits one source batch with its cursor.
        # Reading only that head avoids observing an unbounded tail we cannot
        # yet atomically acknowledge.
        page = self._accepted_page(after_sequence=after_sequence, limit=min(limit, 1))
        next_sequence = after_sequence + 1
        if not page or page[0].sequence != next_sequence:
            tombstone = self._tombstone_at(next_sequence, stream_id=applied[0] if applied else None)
            if tombstone is None:
                if not page:
                    return (), None
                raise AcceptedMarkerInputRefusedError("accepted marker stream has a missing unconsumed sequence")
            head = _key(tombstone[0], next_sequence)
        else:
            first = page[0]
            if applied is not None and first.stream_id != applied[0]:
                raise AcceptedMarkerInputRefusedError("accepted marker stream changed under durable user cursor")
            head = _key(first.stream_id, first.sequence)
        # The kernel consumes this page before asking for the next one. A
        # successful publication moves the durable cursor and exposes a new
        # head. A held or failed publication must stop this sweep instead of
        # repeating its key forever or delivering a later sequence out of order.
        if head == cursor:
            return (), None
        return (head,), head

    def _tombstone_at(self, sequence: int, *, stream_id: str | None) -> tuple[str, str] | None:
        conn = self._source_read_connection()
        try:
            if stream_id is None:
                row = conn.execute(
                    "SELECT e.stream_id, e.identity FROM excised_marker_inputs e "
                    "JOIN accepted_marker_stream s ON s.singleton = 1 AND s.stream_id = e.stream_id "
                    "WHERE e.state = 'accepted' AND e.accepted_sequence = ?",
                    (sequence,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT stream_id, identity FROM excised_marker_inputs "
                    "WHERE state = 'accepted' AND stream_id = ? AND accepted_sequence = ?",
                    (stream_id, sequence),
                ).fetchone()
            return None if row is None else (str(row[0]), str(row[1]))
        finally:
            conn.close()

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        del frame, cursor, limit
        return (), None

    def quiet(self, frame: object, key: str) -> bool:
        """This source stream has no index-generation-specific quiet state."""
        del frame, key
        return False

    def barrier_sessions(self, frame: object, keys: Sequence[str]) -> Mapping[str, Iterator[str]]:
        """Map each carrier key to every session its accepted batch names.

        A carrier lowers markers for several sessions at once, so the
        publication barrier holds the whole batch when any of them waits.
        """
        del frame
        sessions: dict[str, Iterator[str]] = {}
        for key in dict.fromkeys(keys):
            stream_id, sequence = _parse_key(key)
            page = self._accepted_page(after_sequence=sequence - 1, limit=1)
            if len(page) == 1 and page[0].stream_id == stream_id and page[0].sequence == sequence:
                sessions[key] = self._session_ids(page[0])
        return sessions

    def _session_ids(self, accepted: AcceptedMarkerInput) -> Iterator[str]:
        conn = self._source_read_connection()
        try:
            carrier = verified_marker_payload_from_blob(conn, accepted)
        finally:
            conn.close()
        try:
            yield from carrier.iter_session_ids()
        finally:
            carrier.close()

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        """The user cursor, not current assertions or index rows, is authority."""
        del frame
        applied = self._cursor()
        statuses: dict[str, str] = {}
        for key in keys:
            stream_id, sequence = _parse_key(key)
            statuses[key] = (
                _VALID if applied is not None and applied[0] == stream_id and applied[1] >= sequence else _MISSING
            )
        return statuses

    def prerequisite_keys(self, frame: object, key: str) -> tuple[tuple[str, str], ...]:
        del frame, key
        return ()

    def compute(self, frame: object, key: str) -> SessionMarkerReplacement:
        """Read exactly one retained batch and decode its sealed candidates."""
        del frame
        stream_id, sequence = _parse_key(key)
        page = self._accepted_page(after_sequence=sequence - 1, limit=1)
        if len(page) != 1 or page[0].stream_id != stream_id or page[0].sequence != sequence:
            tombstone = self._tombstone_at(sequence, stream_id=stream_id)
            if tombstone is None:
                raise AcceptedMarkerInputRefusedError("accepted marker batch is absent or no longer contiguous")
            return SessionMarkerReplacement(
                stream_id=stream_id, sequence=sequence, identity=tombstone[1], payload=iter(()), retired=iter(())
            )
        batch = page[0]
        conn = self._source_read_connection()
        try:
            carrier = verified_marker_payload_from_blob(conn, batch)
        finally:
            conn.close()
        try:
            for _candidate_value in _candidates(carrier):
                pass
            for _retirement in _retirements(carrier):
                pass
        except BaseException:
            carrier.close()
            raise
        return SessionMarkerReplacement(
            stream_id=stream_id,
            sequence=sequence,
            identity=batch.batch.identity,
            payload=_candidates(carrier),
            retired=_retirements(carrier),
            carrier=carrier,
        )

    def publish(self, frame: object, replacement: object) -> bool:
        """Commit canonical assertion lowering and the matching cursor together."""
        del frame
        assert isinstance(replacement, SessionMarkerReplacement)
        from polylogue.storage.sqlite.archive_tiers.user_write import _now_ms

        conn: sqlite3.Connection | None = None
        try:
            conn = self._marker_write_connection()
            conn.execute("BEGIN IMMEDIATE")
            applied = accepted_marker_delivery_cursor(conn)
            if applied is not None and applied[0] == replacement.stream_id and applied[1] >= replacement.sequence:
                conn.rollback()
                return True
            expected_prior = 0 if applied is None else applied[1]
            if applied is not None and applied[0] != replacement.stream_id:
                raise AcceptedMarkerInputRefusedError("accepted marker stream changed under durable user cursor")
            if replacement.sequence != expected_prior + 1:
                raise AcceptedMarkerInputRefusedError("accepted marker batch is not the next durable source sequence")
            # Excision can run after derivation read the sealed payload. Its
            # tombstone is authoritative under writer admission and prevents
            # publishing content after the source carrier was erased.
            if self._tombstone_at(replacement.sequence, stream_id=replacement.stream_id) is None:
                from polylogue.markers.lowering import iter_lower_markers, iter_retire_marker_assertions

                for _assertion_id in iter_lower_markers(conn, replacement.payload):
                    pass
                for _assertion_id in iter_retire_marker_assertions(conn, replacement.retired):
                    pass
            advance_accepted_marker_delivery_cursor(
                conn,
                stream_id=replacement.stream_id,
                applied_sequence=replacement.sequence,
                applied_at_ms=_now_ms(),
                expected_prior_sequence=expected_prior,
            )
            conn.commit()
        except BaseException:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            try:
                if conn is not None:
                    conn.close()
            finally:
                replacement.close()
        return True
