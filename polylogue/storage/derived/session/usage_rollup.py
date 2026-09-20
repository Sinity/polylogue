"""Canonical provider/model usage reconciliation as its own derivation.

``session_model_usage`` is the single per-session token and cost authority.
It is *derived*: ``_refresh_provider_usage_rollup`` re-aggregates it from
``messages``, ``session_provider_usage_events`` and ``sessions.reported_cost_usd``.

Until polylogue-bp12n.1 that reconciliation ran inside the session-profile
publisher, which committed it and then checked whether the profile it had
already prepared was still applicable. Publication altered its own premise:
the prepared bundle had read the *old* rollup, so the value check that follows
the refresh necessarily failed, the partition was refused, and a later pass
did the real work. The first computation of every session whose rollup had
moved was doomed by construction.

This module makes the reconciliation an explicit stage that runs *before*
profile preparation, with its own transaction and its own honest result. The
profile derivation names it as a prerequisite key, so the kernel converges it
first in the same pass; a profile is then prepared from a rollup that is
already settled and publishes on the first attempt.

Pricing semantics are unchanged: this calls the same
``_refresh_provider_usage_rollup`` with the same provider reconciliation,
disjoint token lanes and catalog repricing. Nothing here re-sums a provider
total a second way.

**Why a binding row.** The rollup's output rows are totals; they cannot say
which message and provider-event *values* were summed to produce them. That is
the one identity an output relation genuinely cannot carry, which is exactly
when the selected architecture allows a domain-local binding row. The row
holds a value-complete input digest and the recipe the rollup was produced
under -- no timestamp, no attempt counter, no status. Inspection recomputes
both and compares; a damaged or absent row is stale or missing, never valid.
"""

from __future__ import annotations

import bisect
import hashlib
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_RECIPE_VERSION,
    session_input_bindings,
)
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = [
    "SESSION_USAGE_ROLLUP_DOMAIN",
    "SessionUsageRollupDerivation",
    "SessionUsageRollupReplacement",
    "inspect_session_usage_rollups",
    "publish_session_usage_rollup",
    "session_usage_rollup_recipe_version",
    "stamp_session_usage_rollup_binding",
]

SESSION_USAGE_ROLLUP_DOMAIN = "session_usage_rollup"

_VALID = "valid"
_MISSING = "missing"
_STALE = "stale"

#: Bumped when what the reconciliation *means* changes without its declared
#: inputs changing -- a change to ``_refresh_provider_usage_rollup``'s
#: aggregation, reconciliation or provider-cost apportionment. The catalog
#: half of the recipe is derived below rather than declared, because the
#: catalog is data.
_DECLARED_ROLLUP_RECIPE = "1"


@lru_cache(maxsize=1)
def _pricing_catalog_digest() -> str:
    """Digest the in-process pricing catalog the reprice step consults.

    ``_reprice_model_usage_rows`` prices persisted token totals from
    ``polylogue.archive.semantic.pricing.PRICING``, which is loaded data, not
    a declaration in this repository. A rollup bound only to archive values
    would therefore stay VALID across a catalog update that changes every
    ``catalog_cost_usd`` it stores. Binding the catalog's rates into the
    recipe makes that update invalidate exactly the rows it moves.
    """
    from polylogue.archive.semantic.pricing import PRICING

    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(PRICING):
        pricing = PRICING[name]
        digest.update(b"\x1e")
        digest.update(
            b"\x1f".join(
                str(value).encode("utf-8")
                for value in (
                    name,
                    pricing.input_usd_per_1m,
                    pricing.output_usd_per_1m,
                    pricing.cache_read_usd_per_1m,
                    pricing.cache_write_usd_per_1m,
                )
            )
        )
    return digest.hexdigest()


def session_usage_rollup_recipe_version() -> str:
    """The complete recipe identity of one reconciled rollup.

    ``SESSION_INPUT_RECIPE_VERSION`` participates because the binding digest
    below is that projection's digest: if the projection's meaning moves, so
    must every stored binding.
    """
    return f"{_DECLARED_ROLLUP_RECIPE}:{SESSION_INPUT_RECIPE_VERSION}:{_pricing_catalog_digest()}"


_STORED_BINDING_SQL = """
SELECT session_id, input_binding, recipe_version
FROM session_usage_rollup_bindings
WHERE session_id IN ({placeholders})
"""


def _stored_bindings(conn: sqlite3.Connection, session_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    sql = _STORED_BINDING_SQL.format(placeholders=",".join("?" * len(unique)))
    return {str(row[0]): (str(row[1]), str(row[2])) for row in conn.execute(sql, unique).fetchall()}


def _present_sessions(conn: sqlite3.Connection, session_ids: Sequence[str]) -> frozenset[str]:
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return frozenset()
    placeholders = ",".join("?" * len(unique))
    rows = conn.execute(
        f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})",
        unique,
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def inspect_session_usage_rollups(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    recipe_version: str,
) -> Mapping[str, str]:
    """Classify each session's reconciled rollup from its stored binding.

    A session with no binding row is MISSING: it has never been reconciled by
    this derivation, which is also the honest state of a session whose rollup
    rows were written by ingest alone. A binding recorded under another recipe,
    or against input values that have since moved, is STALE.

    This deliberately does not compare the rollup's *rows*. Re-deriving them
    read-only would be a second implementation of the aggregation, and the
    architecture ruling is explicit that a binding must commit to every input
    value rather than to identities -- which is what the digest below is.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    present = _present_sessions(conn, unique)
    stored = _stored_bindings(conn, unique)
    current = session_input_bindings(conn, tuple(key for key in unique if key in present)) if present else {}
    statuses: dict[str, str] = {}
    for key in unique:
        if key not in present:
            statuses[key] = _MISSING
            continue
        binding = stored.get(key)
        if binding is None:
            statuses[key] = _MISSING
        elif binding[1] != recipe_version or binding[0] != current.get(key):
            statuses[key] = _STALE
        else:
            statuses[key] = _VALID
    return statuses


def stamp_session_usage_rollup_binding(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    input_binding: str,
    recipe_version: str,
) -> None:
    """Record which input values and recipe produced the current rollup.

    The caller owns the transaction. The stamp must be written by whatever
    commit reconciled the rows, so a bulk rebuild that refreshes the rollup
    stamps it too and recurring convergence finds no work rather than
    re-reconciling the whole archive after every build.
    """
    conn.execute(
        """
        INSERT INTO session_usage_rollup_bindings (session_id, input_binding, recipe_version)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            input_binding  = excluded.input_binding,
            recipe_version = excluded.recipe_version
        """,
        (session_id, input_binding, recipe_version),
    )


def publish_session_usage_rollup(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    input_binding: str,
    recipe_version: str,
) -> bool:
    """Reconcile one session's rollup and stamp its binding in one transaction.

    The refresh and its binding commit together or not at all. A refusal --
    the inputs moved between preparation and the writer -- rolls the refresh
    back with them, so ``False`` means no effect rather than "the profile was
    refused but usage moved anyway", which is the exact untruth the previous
    publisher told.
    """
    from polylogue.storage.derived.session.rebuild import _refresh_provider_usage_rollup

    conn.execute("BEGIN IMMEDIATE")
    try:
        exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if exists is None:
            # Retirement: the binding row outlived its session. The rollup rows
            # themselves cascade with the session row, so absence is the whole
            # correct output here.
            conn.execute("DELETE FROM session_usage_rollup_bindings WHERE session_id = ?", (session_id,))
            conn.execute("COMMIT")
            return True
        current = session_input_bindings(conn, (session_id,)).get(session_id, "")
        if current != input_binding:
            conn.execute("ROLLBACK")
            return False
        _refresh_provider_usage_rollup(conn, session_id)
        stamp_session_usage_rollup_binding(
            conn,
            session_id,
            input_binding=current,
            recipe_version=recipe_version,
        )
        conn.execute("COMMIT")
        return True
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


@dataclass(frozen=True, slots=True)
class SessionUsageRollupReplacement:
    """The kernel's Replacement shape for one reconciled session rollup."""

    key: str
    input_binding: str
    recipe_version: str
    session_present: bool
    generation_binding: str | None = None

    @property
    def payload(self) -> str:
        return self.input_binding

    @property
    def empty(self) -> bool:
        return not self.session_present


def _connection_generation(conn: sqlite3.Connection) -> str:
    for _sequence, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return str(Path(str(filename)).resolve())
    raise RuntimeError("session usage rollup writer has no main database generation")


class SessionUsageRollupDerivation:
    """Usage reconciliation as a derivation the kernel can order and bound.

    Registered *before* the session-profile derivation, which names this
    domain's key as a concrete prerequisite. The kernel converges a
    prerequisite key earlier in the same pass, so a stale rollup is reconciled
    and the dependent profile is then prepared from the settled values --
    one pass, one publication each, no doomed computation.
    """

    domain = SESSION_USAGE_ROLLUP_DOMAIN
    prerequisites: tuple[str, ...] = ()

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection],
        write_connection: Callable[[], sqlite3.Connection],
        *,
        session_scope: Callable[[object], Sequence[str] | None],
        quiet_key: Callable[[object, str], bool] | None = None,
        generation_binding: Callable[[], str] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._session_scope = session_scope
        self._quiet_key = quiet_key
        self._generation_binding = generation_binding

    @property
    def recipe_version(self) -> str:
        return session_usage_rollup_recipe_version()

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        if limit < 1:
            raise ValueError("session usage rollup page limit must be positive")
        scope = self._session_scope(frame)
        if scope is not None:
            keys = tuple(sorted(dict.fromkeys(str(key) for key in scope)))
            start = bisect.bisect(keys, cursor) if cursor is not None else 0
            page = keys[start : start + limit]
            return page, (page[-1] if start + len(page) < len(keys) and page else None)
        conn = self._read_connection()
        try:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id > COALESCE(?, '') ORDER BY session_id LIMIT ?",
                (cursor, limit),
            ).fetchall()
            keys = tuple(str(row[0]) for row in rows)
            return keys, (keys[-1] if len(keys) == limit else None)
        finally:
            conn.close()

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Binding rows whose session is gone.

        The rollup rows themselves cascade with ``sessions``; the binding row
        is the one member of this partition that a foreign-key-disabled
        maintenance write can strand, so it is the excess space.
        """
        del frame
        conn = self._read_connection()
        try:
            rows = conn.execute(
                """
                SELECT b.session_id
                FROM session_usage_rollup_bindings AS b
                LEFT JOIN sessions AS s ON s.session_id = b.session_id
                WHERE s.session_id IS NULL AND b.session_id > COALESCE(?, '')
                ORDER BY b.session_id
                LIMIT ?
                """,
                (cursor, limit),
            ).fetchall()
            keys = tuple(str(row[0]) for row in rows)
            return keys, (keys[-1] if len(keys) == limit else None)
        finally:
            conn.close()

    def quiet(self, frame: object, key: str) -> bool:
        return False if self._quiet_key is None else self._quiet_key(frame, key)

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        del frame, key
        return ()

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        del frame
        conn = self._read_connection()
        try:
            # Stored binding, session presence and the recomputed projection
            # must come from one commit: mixing snapshots can certify a rollup
            # against inputs that no single generation ever held.
            conn.execute("BEGIN")
            return inspect_session_usage_rollups(conn, keys, recipe_version=self.recipe_version)
        finally:
            conn.close()

    def compute(self, frame: object, key: str) -> SessionUsageRollupReplacement:
        """Read the binding the reconciliation must commit to. No writes."""
        generation = self._generation_binding() if self._generation_binding is not None else None
        expected_generation = f"index-generation:{generation}" if generation is not None else None
        source_revision = getattr(frame, "source_revision", None)
        if (
            expected_generation is not None
            and isinstance(source_revision, str)
            and source_revision.startswith("index-generation:")
            and source_revision != expected_generation
        ):
            raise RuntimeError("session usage rollup frame names a retired index generation")
        conn = self._read_connection()
        try:
            conn.execute("BEGIN")
            present = bool(conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (key,)).fetchone())
            binding = session_input_bindings(conn, (key,)).get(key, "") if present else ""
        finally:
            conn.close()
        if generation is not None and self._generation_binding is not None and self._generation_binding() != generation:
            raise RuntimeError("active index generation changed while the session usage rollup was prepared")
        return SessionUsageRollupReplacement(
            key=key,
            input_binding=binding,
            recipe_version=self.recipe_version,
            session_present=present,
            generation_binding=generation,
        )

    def publish(self, frame: object, replacement: object) -> bool:
        del frame
        if not isinstance(replacement, SessionUsageRollupReplacement):
            raise TypeError(f"expected SessionUsageRollupReplacement, got {type(replacement).__name__}")
        generation_binding = self._generation_binding
        with write_lease(f"derivation.{self.domain}"):
            if (
                replacement.generation_binding is not None
                and generation_binding is not None
                and generation_binding() != replacement.generation_binding
            ):
                return False
            conn = self._write_connection()
            try:
                if (
                    replacement.generation_binding is not None
                    and _connection_generation(conn) != replacement.generation_binding
                ):
                    return False
                return publish_session_usage_rollup(
                    conn,
                    replacement.key,
                    input_binding=replacement.input_binding,
                    recipe_version=replacement.recipe_version,
                )
            finally:
                conn.close()
