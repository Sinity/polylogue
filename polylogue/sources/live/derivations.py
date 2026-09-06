"""Derivation adapters the daemon converges, bound to their storage owners.

These satisfy the derivation kernel's adapter contract structurally: the kernel
type-checks nothing at runtime and this package may not import the daemon ring,
so the seam is the vocabulary (``required`` / ``inspect`` / ``compute`` /
``publish`` / ``prerequisites``) rather than a shared base class.

Each adapter is thin on purpose. Domain SQL, atomic replacement, and the input
binding stay with the tables they own; what lives here is the frame-to-key
mapping and nothing else.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence

from polylogue.storage.derived.session.derivation import (
    SESSION_PROFILE_DOMAIN,
    SESSION_PROFILE_RECIPE_VERSION,
    excess_session_profiles,
    inspect_session_profiles,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = ["SESSION_PROFILE_DOMAIN", "SessionProfileDerivation"]

SessionScope = Callable[[object], Sequence[str]]

#: One partition's publication is a bounded transaction over one session. A
#: hold longer than the storage busy timeout can starve a writer that is not on
#: the daemon's gate, so the budget names that boundary rather than a
#: preference.
_PUBLISH_HOLD_BUDGET_S = 30.0


class SessionProfileDerivation:
    """Session profiles as an ordinary derivation over a message projection.

    ``required`` is the frame's session scope: the batch's sessions during
    incremental convergence, every session at an archive-wide boundary. Both
    reach the same inspection, so a restart that loses every scheduling hint
    reconstructs the identical pending set.
    """

    domain = SESSION_PROFILE_DOMAIN
    prerequisites: tuple[str, ...] = ()
    recipe_version = SESSION_PROFILE_RECIPE_VERSION

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection],
        write_connection: Callable[[], sqlite3.Connection],
        *,
        materializer_version: int,
        session_scope: SessionScope,
        page_size: int = 200,
        quiet_keys: Callable[[object], frozenset[str]] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._materializer_version = materializer_version
        self._session_scope = session_scope
        self._page_size = page_size
        self._quiet_keys = quiet_keys

    def required(self, frame: object) -> Iterable[str]:
        return tuple(dict.fromkeys(self._session_scope(frame)))

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        conn = self._read_connection()
        try:
            return inspect_session_profiles(conn, keys, materializer_version=self._materializer_version)
        finally:
            conn.close()

    def excess_candidates(self, frame: object) -> Iterable[str]:
        conn = self._read_connection()
        try:
            return excess_session_profiles(conn)
        finally:
            conn.close()

    def quiet(self, frame: object, key: str) -> bool:
        return key in self._quiet_keys(frame) if self._quiet_keys is not None else False

    def compute(self, frame: object, key: str) -> object:
        """Read the binding this replacement is computed against, lease-free.

        The profile row build still happens inside ``publish`` through the
        existing writer; moving row construction out of the lease is a separate
        change (daemon-core stage 5) from making staleness value-complete. What
        matters for correctness here is that the binding read outside the lease
        is the one publication revalidates, so a computation that raced an
        ingest is refused rather than published.
        """
        conn = self._read_connection()
        try:
            binding = session_input_bindings(conn, (key,)).get(key, "")
        finally:
            conn.close()
        return _SessionProfileReplacement(domain=self.domain, key=key, input_binding=binding)

    def publish(self, frame: object, replacement: object) -> bool:
        binding = getattr(replacement, "input_binding", "")
        key = getattr(replacement, "key", "")
        with write_lease(f"derivation.{self.domain}", max_hold_seconds=_PUBLISH_HOLD_BUDGET_S):
            conn = self._write_connection()
            try:
                return publish_session_profile(
                    conn,
                    str(key),
                    input_binding=str(binding),
                    page_size=self._page_size,
                )
            finally:
                conn.close()


class _SessionProfileReplacement:
    """The kernel's Replacement shape, built without importing the daemon ring."""

    __slots__ = ("domain", "empty", "input_binding", "key", "payload")

    def __init__(self, *, domain: str, key: str, input_binding: str) -> None:
        self.domain = domain
        self.key = key
        self.input_binding = input_binding
        self.payload = key
        self.empty = False
