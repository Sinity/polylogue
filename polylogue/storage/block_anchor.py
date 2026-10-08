"""Block content-hash citation anchors (svfj) — resolve a stored citation
against the current archive, never guessing.

``blocks.content_hash`` (index.py) hashes a block's canonical EVIDENCE only —
type, text, tool_name, canonical tool_input, semantic/media/language,
is_error, exit_code — deliberately excluding session_id/message_id/position/
tool_id. That is what lets a citation anchor survive fork-position replay,
re-ingest renumbering, and provider tool-id regeneration: the identity
components can shift, but the evidence they point at is still findable by
its hash.

The textual anchor form is ``<session_id>::<message_id>::block@sha256:<hex>``.
Session/message ids are themselves colon-bearing (``codex-session:abc``), so
the outer separator is the double colon, never single.

The resolver returns a TYPED state, never a silent best-guess pick:

- ``ok`` — the hash resolves in the named message at the expected position
  (or no position hint was given).
- ``drifted_position`` — the hash resolves in the named message, but at a
  different position than the caller's hint.
- ``drifted_message`` — the hash is not in the named message, but resolves
  to exactly one other message in the same session.
- ``ambiguous`` — more than one block carries this hash within the resolved
  scope (e.g. the same prompt text repeated N times); candidates are listed,
  never picked for the caller.
- ``hash_mismatch`` — the named message/position exists, but its current
  content_hash differs from the anchor's. A hard fail: never auto-rewrite
  the anchor or guess which content it "really" meant.
- ``missing`` — neither the message nor any block with this hash resolves
  anywhere in the session.
- ``relocated_lineage`` — the hash is present in a composed lineage view
  reached through a concrete ``session_links`` inheritance edge. The edge is
  included in ``detail`` so a caller can explain where the citation moved.
- ``quarantined`` — the anchor's session participates in a quarantined
  topology edge. This is returned before any lineage walk; a broken graph is
  never traversed as if it were trustworthy.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from typing import Literal, cast

from polylogue.archive.topology.edge import topology_status_composes_sql
from polylogue.core.enums import TopologyEdgeStatus
from polylogue.storage.io_phase_metrics import connection_cursor

BlockAnchorState = Literal[
    "ok",
    "drifted_position",
    "drifted_message",
    "relocated_lineage",
    "ambiguous",
    "missing",
    "quarantined",
    "hash_mismatch",
]

_ANCHOR_SEPARATOR = "::"
_BLOCK_PREFIX = "block@sha256:"


@dataclass(frozen=True)
class BlockAnchor:
    """Parsed textual citation anchor."""

    session_id: str
    message_id: str
    content_hash_hex: str

    def to_text(self) -> str:
        return format_block_anchor(self.session_id, self.message_id, self.content_hash_hex)


@dataclass(frozen=True)
class BlockAnchorResolution:
    """Result of resolving a :class:`BlockAnchor` against the current archive."""

    state: BlockAnchorState
    anchor: BlockAnchor
    resolved_message_id: str | None = None
    resolved_position: int | None = None
    candidates: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    detail: str = ""


BeforeAnchorInput = Callable[[str, tuple[str, ...], str, tuple[object, ...]], None]


def _anchor_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rowid_sql: str,
    parameters: tuple[object, ...],
    before_input: BeforeAnchorInput | None,
) -> Generator[sqlite3.Row, None, None]:
    # Canonical predicates/order choose identity once, including LIMIT ties.
    # Every payload read then names that same physical row under the held view.
    projection = ",".join(columns)
    with connection_cursor(conn, rowid_sql, parameters) as identities:
        for (rowid,) in identities:
            if before_input is not None:
                before_input(table, columns, f"SELECT rowid FROM {table} WHERE rowid=?", (rowid,))
            with connection_cursor(conn, f"SELECT {projection} FROM {table} WHERE rowid=?", (rowid,)) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise RuntimeError("anchor input disappeared inside its owned read snapshot")
            yield cast(sqlite3.Row, row)


def _quarantined_edge(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeAnchorInput | None = None
) -> sqlite3.Row | None:
    """Return one quarantined edge incident to ``session_id``, if any."""
    with closing(
        _anchor_rows(
            conn,
            "session_links",
            (
                "src_session_id",
                "resolved_dst_session_id",
                "link_type",
                "inheritance",
                "branch_point_message_id",
                "status",
            ),
            "SELECT rowid FROM session_links WHERE status=? AND (src_session_id=? OR resolved_dst_session_id=?) "
            "ORDER BY src_session_id,resolved_dst_session_id,link_type LIMIT 1",
            (TopologyEdgeStatus.QUARANTINED.value, session_id, session_id),
            before_input,
        )
    ) as rows:
        return next(rows, None)


def _lineage_edges(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeAnchorInput | None = None
) -> list[sqlite3.Row]:
    """Return composing edges incident to one session in preference order."""
    with closing(
        _anchor_rows(
            conn,
            "session_links",
            (
                "src_session_id",
                "resolved_dst_session_id",
                "link_type",
                "inheritance",
                "branch_point_message_id",
                "status",
            ),
            "SELECT rowid FROM session_links WHERE resolved_dst_session_id IS NOT NULL "
            "AND inheritance IN ('prefix-sharing', 'spawned-fresh') "
            f"AND (src_session_id=? OR resolved_dst_session_id=?) AND {topology_status_composes_sql()} "
            "ORDER BY CASE inheritance WHEN 'prefix-sharing' THEN 0 WHEN 'spawned-fresh' THEN 1 ELSE 2 END,"
            "src_session_id,resolved_dst_session_id,link_type",
            (session_id, session_id),
            before_input,
        )
    ) as rows:
        return list(rows)


def _lineage_candidates(
    conn: sqlite3.Connection, session_id: str, before_input: BeforeAnchorInput | None = None
) -> Iterator[tuple[str, sqlite3.Row]]:
    """Visit every composing neighbour once; no cap can establish uniqueness."""

    queue = deque([session_id])
    visited = {session_id}
    while queue:
        current = queue.popleft()
        for edge in _lineage_edges(conn, current, before_input):
            src = str(edge["src_session_id"])
            dst = str(edge["resolved_dst_session_id"])
            neighbour = dst if src == current else src
            if neighbour in visited:
                continue
            visited.add(neighbour)
            yield neighbour, edge
            queue.append(neighbour)


def _hash_matches_in_session(
    conn: sqlite3.Connection, session_id: str, content_hash: bytes, before_input: BeforeAnchorInput | None = None
) -> list[tuple[str, int]]:
    """Blocks carrying ``content_hash`` among one session's own physical rows."""
    with closing(
        _anchor_rows(
            conn,
            "blocks",
            ("message_id", "position"),
            "SELECT b.rowid FROM blocks b JOIN messages m ON m.message_id=b.message_id "
            "WHERE m.session_id=? AND b.content_hash=? ORDER BY b.message_id,b.position",
            (session_id, content_hash),
            before_input,
        )
    ) as rows:
        return [(str(row["message_id"]), int(row["position"])) for row in rows]


def _lineage_detail(edge: sqlite3.Row, resolved_session_id: str) -> str:
    return (
        f"content_hash resolved in composed lineage session {resolved_session_id} via "
        f"inheritance edge {edge['src_session_id']} -> {edge['resolved_dst_session_id']} "
        f"(inheritance={edge['inheritance']}, link_type={edge['link_type']}, "
        f"branch_point_message_id={edge['branch_point_message_id']})"
    )


def _resolve_relocated_lineage(
    conn: sqlite3.Connection,
    anchor: BlockAnchor,
    content_hash: bytes,
    before_input: BeforeAnchorInput | None = None,
) -> BlockAnchorResolution | None:
    """Resolve only a unique physical block across the whole lineage neighbourhood.

    Every composed view in the neighbourhood is built from the own rows of
    sessions in that same neighbourhood, so the union of their own rows is
    exactly the set of blocks any of those reads can expose. Matching own rows
    cites the session whose read physically holds the block, binds two values
    per query however long a transcript is, and sees a block inherited by many
    views once instead of as many candidates.
    """

    matches: dict[tuple[str, int], str] = {}
    for candidate_session_id, edge in _lineage_candidates(conn, anchor.session_id, before_input):
        for match in _hash_matches_in_session(conn, candidate_session_id, content_hash, before_input):
            matches.setdefault(match, _lineage_detail(edge, candidate_session_id))
    if not matches:
        return None
    if len(matches) > 1:
        return BlockAnchorResolution(
            state="ambiguous",
            anchor=anchor,
            candidates=tuple(sorted(matches)),
            detail="multiple physical blocks across the lineage neighbourhood share the anchor's content_hash",
        )
    (message_id, position), detail = next(iter(matches.items()))
    return BlockAnchorResolution(
        state="relocated_lineage",
        anchor=anchor,
        resolved_message_id=message_id,
        resolved_position=position,
        detail=detail,
    )


def format_block_anchor(session_id: str, message_id: str, content_hash_hex: str) -> str:
    """Build the canonical textual anchor form for a block."""

    return f"{session_id}{_ANCHOR_SEPARATOR}{message_id}{_ANCHOR_SEPARATOR}{_BLOCK_PREFIX}{content_hash_hex}"


class InvalidBlockAnchorError(ValueError):
    """Raised when a textual anchor does not parse as ``<session>::<message>::block@sha256:<hex>``."""


def parse_block_anchor(anchor_text: str) -> BlockAnchor:
    """Parse the canonical textual anchor form.

    Raises :class:`InvalidBlockAnchorError` on malformed input rather than
    guessing a partial parse — a citation anchor is meant to be exact.
    """

    parts = anchor_text.split(_ANCHOR_SEPARATOR)
    if len(parts) != 3:
        raise InvalidBlockAnchorError(f"expected <session>::<message>::block@sha256:<hex>, got {anchor_text!r}")
    session_id, message_id, block_part = parts
    if not session_id or not message_id:
        raise InvalidBlockAnchorError(f"empty session_id/message_id in anchor {anchor_text!r}")
    if not block_part.startswith(_BLOCK_PREFIX):
        raise InvalidBlockAnchorError(f"expected {_BLOCK_PREFIX!r} prefix, got {anchor_text!r}")
    content_hash_hex = block_part[len(_BLOCK_PREFIX) :]
    if len(content_hash_hex) != 64 or not all(c in "0123456789abcdef" for c in content_hash_hex):
        raise InvalidBlockAnchorError(f"expected a 64-char lowercase hex sha256 digest, got {anchor_text!r}")
    return BlockAnchor(session_id=session_id, message_id=message_id, content_hash_hex=content_hash_hex)


def resolve_block_anchor(
    conn: sqlite3.Connection,
    anchor: BlockAnchor,
    *,
    position_hint: int | None = None,
    before_input: BeforeAnchorInput | None = None,
) -> BlockAnchorResolution:
    """Resolve a citation anchor against the current archive (read-only).

    ``conn`` must have ``row_factory = sqlite3.Row`` (or plain tuple access
    with matching column order — this function selects by name via
    ``sqlite3.Row``, so a plain-tuple connection will raise).
    """

    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            return resolve_block_anchor(conn, anchor, position_hint=position_hint, before_input=before_input)
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass

    content_hash = bytes.fromhex(anchor.content_hash_hex)

    # A quarantined topology edge is an explicit refusal to trust the graph.
    # Report it before even checking local rows: in particular, do not let a
    # seemingly valid local block mask the fact that lineage traversal is
    # unsafe for this session.
    quarantined = _quarantined_edge(conn, anchor.session_id, before_input)
    if quarantined is not None:
        return BlockAnchorResolution(
            state="quarantined",
            anchor=anchor,
            detail=(
                f"anchor session participates in quarantined topology edge "
                f"{quarantined['src_session_id']} -> {quarantined['resolved_dst_session_id']} "
                f"(inheritance={quarantined['inheritance']}, link_type={quarantined['link_type']}, "
                f"branch_point_message_id={quarantined['branch_point_message_id']})"
            ),
        )

    with closing(
        _anchor_rows(
            conn,
            "messages",
            ("message_id", "session_id"),
            "SELECT rowid FROM messages WHERE message_id=?",
            (anchor.message_id,),
            before_input,
        )
    ) as rows:
        message_row = next(rows, None)

    if message_row is not None and message_row["session_id"] == anchor.session_id:
        with closing(
            _anchor_rows(
                conn,
                "blocks",
                ("position",),
                "SELECT rowid FROM blocks WHERE message_id=? AND content_hash=? ORDER BY position",
                (anchor.message_id, content_hash),
                before_input,
            )
        ) as rows:
            in_message = list(rows)

        if len(in_message) > 1:
            return BlockAnchorResolution(
                state="ambiguous",
                anchor=anchor,
                resolved_message_id=anchor.message_id,
                candidates=tuple((anchor.message_id, int(row["position"])) for row in in_message),
                detail=f"{len(in_message)} blocks in this message share the anchor's content_hash",
            )
        if len(in_message) == 1:
            position = int(in_message[0]["position"])
            state: BlockAnchorState = "ok" if position_hint is None or position_hint == position else "drifted_position"
            return BlockAnchorResolution(
                state=state,
                anchor=anchor,
                resolved_message_id=anchor.message_id,
                resolved_position=position,
            )

        # No block in the named message carries this hash. If the named
        # position still exists but with different content, that is a hard
        # hash_mismatch -- never guess a rewrite.
        if position_hint is not None:
            with closing(
                _anchor_rows(
                    conn,
                    "blocks",
                    ("content_hash",),
                    "SELECT rowid FROM blocks WHERE message_id=? AND position=?",
                    (anchor.message_id, position_hint),
                    before_input,
                )
            ) as rows:
                mismatch_row = next(rows, None)

            if mismatch_row is not None and mismatch_row["content_hash"] != content_hash:
                return BlockAnchorResolution(
                    state="hash_mismatch",
                    anchor=anchor,
                    resolved_message_id=anchor.message_id,
                    resolved_position=position_hint,
                    detail="content_hash at the hinted position differs from the anchor -- never auto-rewritten",
                )

    # Look for the hash elsewhere in the same session (message drift). This
    # also covers an anchored message that no longer exists: a renumbered
    # message whose block survives is drift, not a lineage relocation.
    in_session = _hash_matches_in_session(conn, anchor.session_id, content_hash, before_input)
    if len(in_session) > 1:
        return BlockAnchorResolution(
            state="ambiguous",
            anchor=anchor,
            candidates=tuple(in_session),
            detail=f"{len(in_session)} blocks across the session share the anchor's content_hash",
        )
    if len(in_session) == 1:
        message_id, position = in_session[0]
        return BlockAnchorResolution(
            state="drifted_message",
            anchor=anchor,
            resolved_message_id=message_id,
            resolved_position=position,
        )

    relocated = _resolve_relocated_lineage(conn, anchor, content_hash, before_input)
    if relocated is not None:
        return relocated
    return BlockAnchorResolution(
        state="missing",
        anchor=anchor,
        detail="no block with this content_hash resolves in the named session or its lineage neighborhood",
    )


__all__ = [
    "BlockAnchor",
    "BlockAnchorResolution",
    "BlockAnchorState",
    "InvalidBlockAnchorError",
    "format_block_anchor",
    "parse_block_anchor",
    "resolve_block_anchor",
]
