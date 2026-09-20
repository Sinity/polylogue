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
from dataclasses import dataclass, field
from typing import Literal, cast

from polylogue.archive.topology.edge import topology_status_composes_sql
from polylogue.core.enums import TopologyEdgeStatus

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


_MAX_LINEAGE_SEARCH_NODES = 512


def _quarantined_edge(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    """Return one quarantined edge incident to ``session_id``, if any."""

    return cast(
        sqlite3.Row | None,
        conn.execute(
            """
        SELECT src_session_id, resolved_dst_session_id, link_type, inheritance,
               branch_point_message_id, status
          FROM session_links
         WHERE status = ?
           AND (src_session_id = ? OR resolved_dst_session_id = ?)
         ORDER BY src_session_id, resolved_dst_session_id, link_type
         LIMIT 1
        """,
            (TopologyEdgeStatus.QUARANTINED.value, session_id, session_id),
        ).fetchone(),
    )


def _lineage_edges(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """Return composing edges incident to one session in preference order."""

    return [
        cast(sqlite3.Row, row)
        for row in conn.execute(
            f"""
        SELECT src_session_id, resolved_dst_session_id, link_type, inheritance,
               branch_point_message_id, status
          FROM session_links
         WHERE resolved_dst_session_id IS NOT NULL
           AND (src_session_id = ? OR resolved_dst_session_id = ?)
           AND {topology_status_composes_sql()}
         ORDER BY CASE inheritance
                    WHEN 'prefix-sharing' THEN 0
                    WHEN 'spawned-fresh' THEN 1
                    ELSE 2
                  END,
                  src_session_id, resolved_dst_session_id, link_type
        """,
            (session_id, session_id),
        ).fetchall()
    ]


def _lineage_candidates(conn: sqlite3.Connection, session_id: str) -> list[tuple[str, sqlite3.Row]]:
    """Breadth-first lineage neighbourhood, preferring prefix-sharing edges."""

    candidates: list[tuple[str, sqlite3.Row]] = []
    queue: list[str] = [session_id]
    visited = {session_id}
    while queue and len(visited) < _MAX_LINEAGE_SEARCH_NODES:
        current = queue.pop(0)
        for edge in _lineage_edges(conn, current):
            src = str(edge["src_session_id"])
            dst = str(edge["resolved_dst_session_id"])
            neighbour = dst if src == current else src
            if neighbour in visited:
                continue
            visited.add(neighbour)
            candidates.append((neighbour, edge))
            queue.append(neighbour)
            if len(visited) >= _MAX_LINEAGE_SEARCH_NODES:
                break
    return candidates


def _prefix_edge(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    """Mirror the production composed-read path's immediate parent lookup."""

    return cast(
        sqlite3.Row | None,
        conn.execute(
            f"""
        SELECT src_session_id, resolved_dst_session_id, link_type, inheritance,
               branch_point_message_id, status, branch_point_content_address
          FROM session_links
         WHERE src_session_id = ?
           AND inheritance = 'prefix-sharing'
           AND resolved_dst_session_id IS NOT NULL
           AND branch_point_message_id IS NOT NULL
           AND {topology_status_composes_sql()}
         ORDER BY link_type, dst_origin, dst_native_id
         LIMIT 1
        """,
            (session_id,),
        ).fetchone(),
    )


def _branch_point_witness_matches(conn: sqlite3.Connection, edge: sqlite3.Row) -> bool:
    witness = edge["branch_point_content_address"]
    if witness is None:
        return True
    row = conn.execute(
        "SELECT content_address FROM messages WHERE message_id = ?",
        (edge["branch_point_message_id"],),
    ).fetchone()
    return row is not None and row["content_address"] is not None and bytes(row["content_address"]) == bytes(witness)


def _own_message_ids(conn: sqlite3.Connection, session_id: str) -> list[str]:
    return [
        str(row["message_id"])
        for row in conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position, variant_index",
            (session_id,),
        ).fetchall()
    ]


def _composed_message_ids(
    conn: sqlite3.Connection,
    session_id: str,
    cache: dict[str, list[str]],
    visiting: set[str] | None = None,
) -> list[str]:
    """Return message ids in the same composed order as archive reads."""

    if session_id in cache:
        return cache[session_id]
    active = set() if visiting is None else visiting
    own = _own_message_ids(conn, session_id)
    if session_id in active or len(active) >= _MAX_LINEAGE_SEARCH_NODES:
        cache[session_id] = own
        return own
    edge = _prefix_edge(conn, session_id)
    if edge is None or not _branch_point_witness_matches(conn, edge):
        cache[session_id] = own
        return own
    parent_ids = _composed_message_ids(
        conn,
        str(edge["resolved_dst_session_id"]),
        cache,
        active | {session_id},
    )
    prefix: list[str] = []
    found = False
    for message_id in parent_ids:
        prefix.append(message_id)
        if message_id == str(edge["branch_point_message_id"]):
            found = True
            break
    composed = (prefix if found else []) + own
    cache[session_id] = composed
    return composed


def _hash_matches_in_composed_session(
    conn: sqlite3.Connection,
    session_id: str,
    content_hash: bytes,
    cache: dict[str, list[str]],
) -> list[tuple[str, int]]:
    message_ids = _composed_message_ids(conn, session_id, cache)
    if not message_ids:
        return []
    placeholders = ",".join("?" for _ in message_ids)
    rows = conn.execute(
        f"""
        SELECT message_id, position
          FROM blocks
         WHERE content_hash = ?
           AND message_id IN ({placeholders})
         ORDER BY message_id, position
        """,
        (content_hash, *message_ids),
    ).fetchall()
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
) -> BlockAnchorResolution | None:
    """Resolve a hash in the anchor's composed view or nearby lineage."""

    composed_cache: dict[str, list[str]] = {}
    # The named session may be the child whose physical rows contain only the
    # divergent tail. Search its composed view first and cite its own edge.
    named_matches = _hash_matches_in_composed_session(conn, anchor.session_id, content_hash, composed_cache)
    named_edge = _prefix_edge(conn, anchor.session_id)
    if named_matches and named_edge is not None:
        if len(named_matches) > 1:
            return BlockAnchorResolution(
                state="ambiguous",
                anchor=anchor,
                candidates=tuple(named_matches),
                detail="multiple blocks in the named composed lineage session share the anchor's content_hash",
            )
        message_id, position = named_matches[0]
        return BlockAnchorResolution(
            state="relocated_lineage",
            anchor=anchor,
            resolved_message_id=message_id,
            resolved_position=position,
            detail=_lineage_detail(named_edge, anchor.session_id),
        )

    for candidate_session_id, edge in _lineage_candidates(conn, anchor.session_id):
        matches = _hash_matches_in_composed_session(conn, candidate_session_id, content_hash, composed_cache)
        if not matches:
            continue
        if len(matches) > 1:
            return BlockAnchorResolution(
                state="ambiguous",
                anchor=anchor,
                candidates=tuple(matches),
                detail=(
                    f"multiple blocks in composed lineage session {candidate_session_id} share "
                    "the anchor's content_hash"
                ),
            )
        message_id, position = matches[0]
        return BlockAnchorResolution(
            state="relocated_lineage",
            anchor=anchor,
            resolved_message_id=message_id,
            resolved_position=position,
            detail=_lineage_detail(edge, candidate_session_id),
        )
    return None


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
) -> BlockAnchorResolution:
    """Resolve a citation anchor against the current archive (read-only).

    ``conn`` must have ``row_factory = sqlite3.Row`` (or plain tuple access
    with matching column order — this function selects by name via
    ``sqlite3.Row``, so a plain-tuple connection will raise).
    """

    content_hash = bytes.fromhex(anchor.content_hash_hex)

    # A quarantined topology edge is an explicit refusal to trust the graph.
    # Report it before even checking local rows: in particular, do not let a
    # seemingly valid local block mask the fact that lineage traversal is
    # unsafe for this session.
    quarantined = _quarantined_edge(conn, anchor.session_id)
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

    message_row = conn.execute(
        "SELECT message_id, session_id FROM messages WHERE message_id = ?",
        (anchor.message_id,),
    ).fetchone()

    if message_row is not None and message_row["session_id"] == anchor.session_id:
        in_message = conn.execute(
            "SELECT position FROM blocks WHERE message_id = ? AND content_hash = ? ORDER BY position",
            (anchor.message_id, content_hash),
        ).fetchall()
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
            mismatch_row = conn.execute(
                "SELECT content_hash FROM blocks WHERE message_id = ? AND position = ?",
                (anchor.message_id, position_hint),
            ).fetchone()
            if mismatch_row is not None and mismatch_row["content_hash"] != content_hash:
                return BlockAnchorResolution(
                    state="hash_mismatch",
                    anchor=anchor,
                    resolved_message_id=anchor.message_id,
                    resolved_position=position_hint,
                    detail="content_hash at the hinted position differs from the anchor -- never auto-rewritten",
                )

        # Look for the hash elsewhere in the same session (message drift).
        in_session = conn.execute(
            """
            SELECT b.message_id, b.position
            FROM blocks b
            JOIN messages m ON m.message_id = b.message_id
            WHERE m.session_id = ? AND b.content_hash = ?
            ORDER BY b.message_id, b.position
            """,
            (anchor.session_id, content_hash),
        ).fetchall()
        if len(in_session) > 1:
            return BlockAnchorResolution(
                state="ambiguous",
                anchor=anchor,
                candidates=tuple((str(row["message_id"]), int(row["position"])) for row in in_session),
                detail=f"{len(in_session)} blocks across the session share the anchor's content_hash",
            )
        if len(in_session) == 1:
            return BlockAnchorResolution(
                state="drifted_message",
                anchor=anchor,
                resolved_message_id=str(in_session[0]["message_id"]),
                resolved_position=int(in_session[0]["position"]),
            )

        relocated = _resolve_relocated_lineage(conn, anchor, content_hash)
        if relocated is not None:
            return relocated
        return BlockAnchorResolution(
            state="missing",
            anchor=anchor,
            detail="no block with this content_hash resolves in the named session or its lineage neighborhood",
        )

    relocated = _resolve_relocated_lineage(conn, anchor, content_hash)
    if relocated is not None:
        return relocated
    return BlockAnchorResolution(
        state="missing",
        anchor=anchor,
        detail=("message_id not found in the named session or its lineage neighborhood"),
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
