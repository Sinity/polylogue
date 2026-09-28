"""The physical limit on one decoded value the archive can store.

The single-object preparation routes (ChatGPT mapping, Gemini CLI, Hermes
snapshot, generic ``messages``, Claude Design, Grok export, bundle members)
stream members, but every stored value ends up as one SQLite TEXT or BLOB
cell, and SQLite refuses any value longer than ``SQLITE_MAX_LENGTH`` bytes
(1,000,000,000 by default). A decoded string whose UTF-8 encoding exceeds
that limit cannot be stored by any route.

Such a value is refused with :class:`ValueBoundRefusedError` as soon as it is
decoded, never truncated or dropped: the refusal reaches
``raw_sessions.parse_error`` like any parser refusal, and source conservation
types it as ``value_bound_refused`` so the gap stays counted. Storing larger
values would need a chunked value representation, which is a separate and
undecided question.
"""

from __future__ import annotations

import sqlite3
from typing import Final

#: SQLite's hard ceiling on ``SQLITE_MAX_LENGTH`` for any build.
_SQLITE_LENGTH_CEILING: Final = 2**31 - 1


def _sqlite_max_length() -> int:
    """Observe the linked library's value-length limit through SQL.

    ``Connection.getlimit`` is missing from supported drivers (the
    ``pysqlite3`` compat build), so the limit is read the way every driver
    exposes it: ``zeroblob(n)`` fails with "string or blob too big" exactly
    when ``n`` exceeds the connection's ``SQLITE_LIMIT_LENGTH``, and
    ``length()`` reads the size without materializing ``n`` bytes.
    """
    connection = sqlite3.connect(":memory:")
    try:
        low, high = 0, _SQLITE_LENGTH_CEILING
        while low < high:
            probe = (low + high + 1) // 2
            try:
                connection.execute("SELECT length(zeroblob(?))", (probe,)).fetchone()
            except sqlite3.DataError:
                high = probe - 1
            else:
                low = probe
        return low
    finally:
        connection.close()


#: SQLite's maximum length, in bytes, of one TEXT or BLOB value, read from the
#: linked library rather than chosen here.
MAX_STORABLE_VALUE_BYTES: Final = _sqlite_max_length()

#: The stable token parse errors, logs and source conservation key on.
VALUE_BOUND_REFUSED: Final = "value_bound_refused"
#: The serialized head of every :class:`ValueBoundRefusedError` message.
#: Classifiers match this head, never the bare token: a source path or other
#: user-controlled text may contain ``value_bound_refused`` on its own.
VALUE_BOUND_REFUSED_HEAD: Final = f"{VALUE_BOUND_REFUSED}: a decoded "


class ValueBoundRefusedError(Exception):
    """One decoded value exceeds what SQLite can store in a single cell.

    Deliberately not a ``ValueError``: decode probes that treat a
    ``ValueError`` as "try the next strategy" must not fall back to a
    whole-document decode of the same unstorable value.
    """

    def __init__(self, kind: str, observed: int, bound: int) -> None:
        super().__init__(
            f"{VALUE_BOUND_REFUSED_HEAD}{kind} of {observed} UTF-8 bytes exceeds SQLite's "
            f"maximum value length of {bound} bytes"
        )
        self.kind = kind
        self.observed = observed
        self.bound = bound


def require_storable_string(value: str, *, kind: str = "string", bound: int | None = None) -> str:
    """Return ``value`` when SQLite can store it; refuse it otherwise."""
    limit = MAX_STORABLE_VALUE_BYTES if bound is None else bound
    # A code point encodes to at most four UTF-8 bytes, so only a string
    # within a factor of four of the limit needs its exact encoded length.
    if len(value) * 4 <= limit:
        return value
    encoded = len(value.encode("utf-8", "surrogatepass"))
    if encoded > limit:
        raise ValueBoundRefusedError(kind, encoded, limit)
    return value


__all__ = [
    "MAX_STORABLE_VALUE_BYTES",
    "VALUE_BOUND_REFUSED",
    "VALUE_BOUND_REFUSED_HEAD",
    "ValueBoundRefusedError",
    "require_storable_string",
]
