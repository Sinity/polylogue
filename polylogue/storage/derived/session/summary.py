"""Authoritative session-summary counter derivation.

The thirteen counters stored on ``sessions`` are a materialized projection of
``messages``.  Their partition is one session and the message projection below
is the whole input contract.  The writer deliberately replaces that partition
from ``messages`` unless an admission route can provide a typed proof that its
write set is disjoint.  No current route carries that proof, so an append is
not allowed to turn a parser-side tally into a second authority.

There is no completion row: a present session with no messages has a valid,
all-zero projection.  Inspection recomputes the same projection and compares
it to the stored columns, which makes a damaged historical counter stale.
"""

from __future__ import annotations

import bisect
import hashlib
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.storage.sqlite.deadline import query_deadline
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = [
    "SESSION_SUMMARY_DOMAIN",
    "SESSION_SUMMARY_MEASURES",
    "SESSION_SUMMARY_MESSAGE_PROJECTION",
    "SESSION_SUMMARY_BINDING_RECIPE",
    "SESSION_SUMMARY_RECIPE_VERSION",
    "SessionSummaryDerivation",
    "SessionSummaryInspection",
    "SessionSummaryMeasure",
    "SessionSummaryReplacement",
    "SessionSummaryValues",
    "authoritative_session_summary",
    "inspect_session_summary",
    "refresh_session_summary",
    "session_summary_input_binding",
    "stored_session_summary_binding",
]

SESSION_SUMMARY_DOMAIN = "session_summary"
SESSION_SUMMARY_RECIPE_VERSION = "1"

#: The complete base relation consumed by every measure below. Keeping this
#: beside the typed measures gives each counter one review point for its SQL
#: input and stored-column replacement.
SESSION_SUMMARY_MESSAGE_PROJECTION = (
    "role",
    "material_origin",
    "word_count",
    "has_tool_use",
    "has_thinking",
    "has_paste",
)

_SummarySource = Literal[
    "row",
    "word_count",
    "has_tool_use",
    "has_thinking",
    "has_paste",
]


@dataclass(frozen=True, slots=True)
class SessionSummaryMeasure:
    """One session-column aggregate over :data:`SESSION_SUMMARY_MESSAGE_PROJECTION`."""

    column: str
    source: _SummarySource
    role: str | None = None
    material_origin: str | None = None

    def sql_expression(self, alias: str = "m") -> str:
        """Return the exact aggregate expression for this measure."""
        terms: list[str] = [f"{alias}.message_id IS NOT NULL"]
        if self.role is not None:
            terms.append(f"{alias}.role = {_sql_literal(self.role)}")
        if self.material_origin is not None:
            terms.append(f"{alias}.material_origin = {_sql_literal(self.material_origin)}")
        condition = " AND ".join(terms) if terms else "1"
        value = "1" if self.source == "row" else f"{alias}.{self.source}"
        return f"COALESCE(SUM(CASE WHEN {condition} THEN {value} ELSE 0 END), 0)"


#: This declaration owns every persisted counter.  Do not add a second field
#: list in a writer route: SQL replacement and stored-column comparison both
#: iterate this one tuple.
SESSION_SUMMARY_MEASURES: tuple[SessionSummaryMeasure, ...] = (
    SessionSummaryMeasure("message_count", "row"),
    SessionSummaryMeasure("word_count", "word_count"),
    SessionSummaryMeasure("tool_use_count", "has_tool_use"),
    SessionSummaryMeasure("thinking_count", "has_thinking"),
    SessionSummaryMeasure("paste_count", "has_paste"),
    SessionSummaryMeasure("user_message_count", "row", role="user"),
    SessionSummaryMeasure("authored_user_message_count", "row", material_origin="human_authored"),
    SessionSummaryMeasure("assistant_message_count", "row", role="assistant"),
    SessionSummaryMeasure("system_message_count", "row", role="system"),
    SessionSummaryMeasure("tool_message_count", "row", role="tool"),
    SessionSummaryMeasure("user_word_count", "word_count", role="user"),
    SessionSummaryMeasure("authored_user_word_count", "word_count", material_origin="human_authored"),
    SessionSummaryMeasure("assistant_word_count", "word_count", role="assistant"),
)


@dataclass(frozen=True, slots=True)
class SessionSummaryValues:
    """Ordered values whose column names are defined only by the measures."""

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(SESSION_SUMMARY_MEASURES):
            raise ValueError("session summary values do not match the declared measures")


@dataclass(frozen=True, slots=True)
class SessionSummaryInspection:
    """One binding-bounded verdict on the stored session-counter projection.

    ``stale_sessions`` is measured drift and nothing else: a session whose
    stored counters disagree with the projection its own binding says was
    published.  ``unbound_sessions`` is the opposite kind of fact -- sessions
    this inspection did **not** compare, because they carry no binding at the
    current recipe.  Folding the second number into the first would report
    drift that was never observed; folding it into ``ready`` would certify
    counters from an input set the domain never checked.  Both are wrong in the
    way polylogue-crwl6 exists to prevent, so the two counts stay separate and
    ``unmeasured_reason`` names what was left uninspected.

    ``state`` is ``ready`` only when both counts are zero.  An unbound session
    makes the verdict ``stale`` -- the derivation owes that session a
    publication -- while ``unmeasured_reason`` keeps the claim honest about
    which of the two reasons produced it.
    """

    state: Literal["ready", "stale", "unknown"]
    total_sessions: int = 0
    stale_sessions: int = 0
    unbound_sessions: int = 0
    reason: str | None = None
    unmeasured_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SessionSummaryReplacement:
    """Lease-free session projection prepared for one atomic replacement."""

    key: str
    values: SessionSummaryValues | None
    generation_binding: str | None = None
    empty: bool = False

    @property
    def input_binding(self) -> str:
        """The complete input value projection for this aggregate's output.

        These values are not a generic content address.  They are the exact
        thirteen aggregate inputs this partition publishes, and publication
        recomputes them before writing.  A message-level mutation that leaves
        every one unchanged needs no counter replacement.
        """
        return "" if self.values is None else ":".join(str(value) for value in self.values.values)

    @property
    def payload(self) -> SessionSummaryValues | None:
        return self.values


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _summary_select_sql() -> str:
    # Field ownership is declared above, not inferred from the SQL strings.
    # This makes a measure that names a non-projected message column fail at
    # import time instead of quietly creating a fourth counter definition.
    projection = frozenset(SESSION_SUMMARY_MESSAGE_PROJECTION)
    for measure in SESSION_SUMMARY_MEASURES:
        if measure.source != "row" and measure.source not in projection:
            raise ValueError(f"session summary source {measure.source!r} is outside its message projection")
        if measure.role is not None and "role" not in projection:
            raise ValueError("role-filtered session summary measure lacks role in its message projection")
        if measure.material_origin is not None and "material_origin" not in projection:
            raise ValueError("authored session summary measure lacks material_origin in its message projection")
    aggregates = ",\n       ".join(
        f"{measure.sql_expression()} AS {measure.column}" for measure in SESSION_SUMMARY_MEASURES
    )
    return f"""
SELECT {aggregates}
FROM messages AS m
WHERE m.session_id = ?
"""


_SUMMARY_SELECT_SQL = _summary_select_sql()


#: Bound into every stored binding so that changing *which* counters the
#: domain publishes -- not just how one is computed -- compares unequal on
#: every existing row without touching one. The measure list is the whole
#: output contract, so it is hashed rather than trusted to move
#: ``SESSION_SUMMARY_RECIPE_VERSION`` by hand.
def _summary_binding_recipe() -> str:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(SESSION_SUMMARY_RECIPE_VERSION.encode("utf-8"))
    for measure in SESSION_SUMMARY_MEASURES:
        digest.update(b"\x1e")
        digest.update(
            b"\x1f".join(
                b"" if value is None else str(value).encode("utf-8")
                for value in (measure.column, measure.source, measure.role, measure.material_origin)
            )
        )
    return f"{SESSION_SUMMARY_RECIPE_VERSION}:{digest.hexdigest()}"


SESSION_SUMMARY_BINDING_RECIPE = _summary_binding_recipe()


def session_summary_input_binding(values: SessionSummaryValues) -> str:
    """Return the published counter projection as this domain's binding value.

    Not a generic content address: these are the exact aggregate outputs the
    partition publishes, in the declared measure order, which is what lets a
    counter column overwritten in place compare unequal against its own
    binding without reading a single message row.
    """
    return ":".join(str(value) for value in values.values)


#: The same value expressed over the stored columns, so the comparison happens
#: inside SQLite instead of hydrating every session into Python. The two
#: spellings must agree exactly; ``tests/unit/storage/test_session_summary_derivation.py``
#: pins them against each other.
_SUMMARY_STORED_BINDING_EXPRESSION = " || ':' || ".join(
    f"CAST(s.{measure.column} AS TEXT)" for measure in SESSION_SUMMARY_MEASURES
)

#: One statement, three counts, and not a single ``messages`` row: the whole
#: point of polylogue-crwl6. ``unbound`` names the sessions whose counters this
#: inspection cannot certify; ``drifted`` names the ones it compared and found
#: wrong. Cost is proportional to ``sessions`` and its binding index, never to
#: the archive's message population.
_SUMMARY_BINDING_SQL = f"""
SELECT
    (SELECT COUNT(*) FROM sessions),
    (SELECT COUNT(*)
       FROM sessions AS s
       LEFT JOIN session_summary_bindings AS b ON b.session_id = s.session_id
      WHERE b.session_id IS NULL OR b.recipe_version <> ?),
    (SELECT COUNT(*)
       FROM sessions AS s
       JOIN session_summary_bindings AS b ON b.session_id = s.session_id
      WHERE b.recipe_version = ?
        AND b.input_binding <> ({_SUMMARY_STORED_BINDING_EXPRESSION}))
"""


def inspect_session_summary(
    conn: sqlite3.Connection,
    *,
    deadline_s: float | None = 1.0,
) -> SessionSummaryInspection:
    """Answer counter readiness from this domain's bindings, not from a census.

    The retired form of this function compared every stored counter with its
    message-derived value in one archive-wide ``sessions LEFT JOIN messages
    GROUP BY session_id`` scan, then guillotined that scan on a wall-clock
    deadline and reported ``unknown`` for any archive it could not finish.  Its
    cost was proportional to the message population, which is why it timed out
    on exactly the archives readiness matters for.

    What replaces it is a comparison of two stored facts.  A
    ``session_summary_bindings`` row states the counter projection published
    for that session; the index tier deletes that row from a trigger the moment
    any of the session's ``messages`` rows are written.  So a surviving binding
    already proves the input side, and the only thing left to check is the
    output side -- whether the stored counters still equal the projection the
    binding says was published.  That is an indexed join over ``sessions``.

    Sessions with no current binding are not certified and not accused: they
    are counted in ``unbound_sessions`` and named in ``unmeasured_reason``.
    """
    if deadline_s is not None and deadline_s < 0:
        raise ValueError("session summary inspection deadline must be non-negative")
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    if deadline is not None and time.monotonic() >= deadline:
        return SessionSummaryInspection(state="unknown", reason="session-summary inspection deadline exceeded")

    from polylogue.core.evidence import Measured, Unavailable
    from polylogue.storage.tier_access import capture_sqlite_read

    def compare() -> SessionSummaryInspection:
        with query_deadline(conn, seconds=deadline_s):
            row = conn.execute(
                _SUMMARY_BINDING_SQL,
                (SESSION_SUMMARY_BINDING_RECIPE, SESSION_SUMMARY_BINDING_RECIPE),
            ).fetchone()
        total, unbound, drifted = (int(row[index] or 0) for index in range(3))
        return SessionSummaryInspection(
            state="ready" if not unbound and not drifted else "stale",
            total_sessions=total,
            stale_sessions=drifted,
            unbound_sessions=unbound,
            unmeasured_reason=(
                None
                if not unbound
                else f"{unbound} of {total} sessions carry no session-summary binding at the current recipe"
            ),
        )

    evidence = capture_sqlite_read(compare)
    if isinstance(evidence, Unavailable):
        return SessionSummaryInspection(
            state="unknown",
            reason=f"session-summary inspection unavailable: {evidence.detail or evidence.reason}",
        )
    if not isinstance(evidence, Measured):
        raise AssertionError("summary inspection produced unsupported evidence")
    return evidence.value


def stored_session_summary_binding(conn: sqlite3.Connection, session_id: str) -> str | None:
    """Return the binding value published for ``session_id`` at this recipe."""
    row = conn.execute(
        "SELECT input_binding FROM session_summary_bindings WHERE session_id = ? AND recipe_version = ?",
        (session_id, SESSION_SUMMARY_BINDING_RECIPE),
    ).fetchone()
    return None if row is None else str(row[0])


def _stamp_session_summary_binding(
    conn: sqlite3.Connection,
    session_id: str,
    values: SessionSummaryValues,
) -> None:
    """Record what this session's counters were published from.

    Must run in the same transaction as the counter write and after it: the
    message triggers retire the binding on every ``messages`` write, so a stamp
    taken before the replacement would certify the wrong projection.
    """
    conn.execute(
        "INSERT INTO session_summary_bindings(session_id, input_binding, recipe_version) VALUES (?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET input_binding = excluded.input_binding, "
        "recipe_version = excluded.recipe_version",
        (session_id, session_summary_input_binding(values), SESSION_SUMMARY_BINDING_RECIPE),
    )


def authoritative_session_summary(conn: sqlite3.Connection, session_id: str) -> SessionSummaryValues | None:
    """Recompute one required partition from persisted message rows.

    ``None`` means the required session disappeared.  An existing session with
    no messages instead returns the all-zero, valid-empty projection.
    """
    if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
        return None
    row = conn.execute(_SUMMARY_SELECT_SQL, (session_id,)).fetchone()
    if row is None:  # aggregate queries return one row; retain the guard for unusual connection wrappers.
        return SessionSummaryValues((0,) * len(SESSION_SUMMARY_MEASURES))
    return SessionSummaryValues(tuple(int(row[index] or 0) for index in range(len(SESSION_SUMMARY_MEASURES))))


def _stored_session_summary(conn: sqlite3.Connection, session_id: str) -> SessionSummaryValues | None:
    columns = ", ".join(measure.column for measure in SESSION_SUMMARY_MEASURES)
    row = conn.execute(f"SELECT {columns} FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    return SessionSummaryValues(tuple(int(row[index] or 0) for index in range(len(SESSION_SUMMARY_MEASURES))))


def refresh_session_summary(conn: sqlite3.Connection, session_id: str) -> bool:
    """Atomically replace a session's stored counters and stamp their binding.

    The caller owns transaction/lease policy.  This is the parsed-session
    writer's short, authoritative publication primitive, and it is the only
    place a ``session_summary_bindings`` row is created: the binding states
    what this replacement published, so publishing and stamping cannot be
    separate operations without a window in which the counters are certified
    by a projection nobody computed.
    """
    values = authoritative_session_summary(conn, session_id)
    if values is None:
        return False
    assignments = ", ".join(f"{measure.column} = ?" for measure in SESSION_SUMMARY_MEASURES)
    conn.execute(
        f"UPDATE sessions SET {assignments} WHERE session_id = ?",
        (*values.values, session_id),
    )
    _stamp_session_summary_binding(conn, session_id, values)
    return True


def _connection_generation(conn: sqlite3.Connection) -> str:
    for _sequence, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return str(Path(str(filename)).resolve())
    raise RuntimeError("session summary writer has no main database generation")


class SessionSummaryDerivation:
    """Counter partition adapter for the derivation kernel and daemon composition."""

    domain = SESSION_SUMMARY_DOMAIN
    prerequisites: tuple[str, ...] = ()
    recipe_version = SESSION_SUMMARY_RECIPE_VERSION

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection],
        write_connection: Callable[[], sqlite3.Connection],
        *,
        session_scope: Callable[[object], Sequence[str] | None],
        generation_binding: Callable[[], str] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._session_scope = session_scope
        self._generation_binding = generation_binding

    def _frame_recipe_current(self, frame: object) -> bool:
        versions = getattr(frame, "recipe_versions", {})
        return isinstance(versions, Mapping) and versions.get(self.domain) == self.recipe_version

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        if limit < 1:
            raise ValueError("session summary derivation page limit must be positive")
        scope = self._session_scope(frame)
        if scope is not None:
            keys = tuple(sorted(dict.fromkeys(str(key) for key in scope)))
            start = bisect.bisect(keys, cursor) if cursor is not None else 0
            page = keys[start : start + limit]
            return page, (page[-1] if start + len(page) < len(keys) and page else None)
        conn = self._read_connection()
        try:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id > ? ORDER BY session_id LIMIT ?",
                (cursor or "", limit),
            ).fetchall()
            keys = tuple(str(row[0]) for row in rows)
            return keys, (keys[-1] if len(keys) == limit else None)
        finally:
            conn.close()

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Counters live on their required session row, so they have no excess space."""
        del frame, cursor, limit
        return (), None

    def quiet(self, frame: object, key: str) -> bool:
        del frame, key
        return False

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        del frame, key
        return ()

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        generation = self._generation_binding() if self._generation_binding is not None else None
        if generation is not None and getattr(frame, "source_revision", None) != f"index-generation:{generation}":
            raise RuntimeError("session summary inspection frame names a retired index generation")
        conn = self._read_connection()
        try:
            if generation is not None and _connection_generation(conn) != generation:
                raise RuntimeError("session summary inspection opened another index generation")
            conn.execute("BEGIN")
            statuses: dict[str, str] = {}
            for key in keys:
                authoritative = authoritative_session_summary(conn, key)
                if authoritative is None:
                    statuses[key] = "missing"
                elif not self._frame_recipe_current(frame):
                    # Counter rows carry their complete output relation, not a
                    # second recipe marker. A frame that does not name this
                    # declaration cannot certify those rows or publish one.
                    statuses[key] = "stale"
                elif _stored_session_summary(conn, key) != authoritative:
                    statuses[key] = "stale"
                elif stored_session_summary_binding(conn, key) != session_summary_input_binding(authoritative):
                    # Correct counters with no current binding are still work
                    # this domain owes: readiness reads the binding, so a
                    # partition that cannot say what it was published from
                    # would otherwise stay uncertifiable forever while the
                    # kernel reported it converged.
                    statuses[key] = "stale"
                else:
                    statuses[key] = "valid"
            if self._generation_binding is not None and self._generation_binding() != generation:
                raise RuntimeError("session summary index generation changed during inspection")
            return statuses
        finally:
            conn.close()

    def compute(self, frame: object, key: str) -> SessionSummaryReplacement:
        if not self._frame_recipe_current(frame):
            raise RuntimeError("session summary frame recipe does not match the active declaration")
        generation = self._generation_binding() if self._generation_binding is not None else None
        expected_generation = f"index-generation:{generation}" if generation is not None else None
        if expected_generation is not None and getattr(frame, "source_revision", None) != expected_generation:
            raise RuntimeError("session summary frame names a retired index generation")
        conn = self._read_connection()
        try:
            conn.execute("BEGIN")
            values = authoritative_session_summary(conn, key)
        finally:
            conn.close()
        if generation is not None and self._generation_binding is not None and self._generation_binding() != generation:
            raise RuntimeError("active index generation changed while session summary was prepared")
        return SessionSummaryReplacement(
            key=key,
            values=values,
            generation_binding=generation,
            empty=values is not None and not any(values.values),
        )

    def publish(self, frame: object, replacement: object) -> bool:
        if not isinstance(replacement, SessionSummaryReplacement):
            raise TypeError(f"expected SessionSummaryReplacement, got {type(replacement).__name__}")
        generation = self._generation_binding
        with write_lease(f"derivation.{self.domain}"):
            if not self._frame_recipe_current(frame):
                return False
            if (
                replacement.generation_binding is not None
                and generation is not None
                and generation() != replacement.generation_binding
            ):
                return False
            conn = self._write_connection()
            try:
                if (
                    replacement.generation_binding is not None
                    and _connection_generation(conn) != replacement.generation_binding
                ):
                    return False
                conn.execute("BEGIN IMMEDIATE")
                current = authoritative_session_summary(conn, replacement.key)
                if current != replacement.values:
                    conn.execute("ROLLBACK")
                    return False
                if current is None:
                    conn.execute("ROLLBACK")
                    return False
                published = session_summary_input_binding(current)
                bound = stored_session_summary_binding(conn, replacement.key) == published
                if bound and _stored_session_summary(conn, replacement.key) == current:
                    # Nothing to publish only when the binding is there too: a
                    # retired binding is real work even where the counters
                    # already happen to be right.
                    conn.execute("ROLLBACK")
                    return True
                refresh_session_summary(conn, replacement.key)
                conn.execute("COMMIT")
                return True
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
