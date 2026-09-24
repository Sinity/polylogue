"""Consume immutable source marker inputs into the durable user tier.

Accepted marker inputs are source-owned history. The consumer does not look
at the current index session because a later interpretation may replace that
projection before this durable user effect is delivered. The user tier owns
one applied source-stream position; lowering one contiguous batch and
advancing that position commit in the same transaction.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from polylogue.storage.accepted_marker_inputs import AcceptedMarkerInput, AcceptedMarkerInputRefusedError
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


class _SyncSourceCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> object:
        return self._cursor.fetchone()

    async def fetchall(self) -> list[object]:
        return list(self._cursor.fetchall())


class _SyncSourceConnection:
    """Adapt a profiled synchronous source read to the shared async reader."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _SyncSourceCursor:
        return _SyncSourceCursor(self._conn.execute(sql, parameters))


@dataclass(frozen=True, slots=True)
class SessionMarkerReplacement:
    """One immutable accepted batch, bound to its source stream position."""

    stream_id: str
    sequence: int
    identity: str
    payload: tuple[MarkerCandidate, ...]

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
        return not self.payload


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


def _candidates(batch: AcceptedMarkerInput) -> tuple[MarkerCandidate, ...]:
    """Decode sealed prepared-write candidates without reparsing current text."""
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
        value = json.loads(batch.batch.payload)
        sessions = value["sessions"]
        if not isinstance(sessions, list):
            raise TypeError("sessions are not a list")
        result: list[MarkerCandidate] = []
        for session in sessions:
            if not isinstance(session, dict):
                raise TypeError("session is not an object")
            raw_candidates = session.get("candidates", [])
            if not isinstance(raw_candidates, list):
                raise TypeError("candidates are not a list")
            for raw_candidate in raw_candidates:
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
                result.append(
                    MarkerCandidate(
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
                        None
                        if assertion_kind is None
                        else AssertionKind(string(assertion_kind, field="candidate assertion kind")),
                        authority=string(raw_candidate.get("authority", "agent-declared"), field="candidate authority"),
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptedMarkerInputRefusedError("accepted marker carrier has an invalid candidate payload") from exc
    return tuple(result)


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
        from polylogue.storage.accepted_marker_inputs import read_accepted_marker_inputs

        conn = self._source_read_connection()
        try:
            return asyncio.run(
                read_accepted_marker_inputs(
                    _SyncSourceConnection(conn), after_sequence=after_sequence, limit=min(limit, self._page_size)
                )
            )
        finally:
            conn.close()

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Expose the next source batch only; the durable cursor owns resumption."""
        del frame, cursor
        applied = self._cursor()
        after_sequence = 0 if applied is None else applied[1]
        # The orchestration protocol commits one source batch with its cursor.
        # Reading only that head avoids observing an unbounded tail we cannot
        # yet atomically acknowledge.
        page = self._accepted_page(after_sequence=after_sequence, limit=min(limit, 1))
        if not page:
            return (), None
        first = page[0]
        if applied is not None and first.stream_id != applied[0]:
            raise AcceptedMarkerInputRefusedError("accepted marker stream changed under durable user cursor")
        if first.sequence != after_sequence + 1:
            raise AcceptedMarkerInputRefusedError("accepted marker stream has a missing unconsumed sequence")
        return (_key(first.stream_id, first.sequence),), None

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        del frame, cursor, limit
        return (), None

    def quiet(self, frame: object, key: str) -> bool:
        """This source stream has no index-generation-specific quiet state."""
        del frame, key
        return False

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
            raise AcceptedMarkerInputRefusedError("accepted marker batch is absent or no longer contiguous")
        batch = page[0]
        return SessionMarkerReplacement(
            stream_id=stream_id,
            sequence=sequence,
            identity=batch.batch.identity,
            payload=_candidates(batch),
        )

    def publish(self, frame: object, replacement: object) -> bool:
        """Commit canonical assertion lowering and the matching cursor together."""
        del frame
        assert isinstance(replacement, SessionMarkerReplacement)
        from polylogue.markers import lower_markers
        from polylogue.storage.sqlite.archive_tiers.user_write import _now_ms

        conn = self._marker_write_connection()
        try:
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
            lower_markers(conn, replacement.payload)
            advance_accepted_marker_delivery_cursor(
                conn,
                stream_id=replacement.stream_id,
                applied_sequence=replacement.sequence,
                applied_at_ms=_now_ms(),
                expected_prior_sequence=expected_prior,
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return True
