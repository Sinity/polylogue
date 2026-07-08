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
- ``relocated_lineage`` and ``quarantined`` are reserved states in the type
  but NOT YET PRODUCED by this resolver — they require the lineage-
  composition read path (searching the fork/resume neighborhood, preferring
  prefix-sharing inheritance over spawned-fresh) and the topology-edge
  quarantine model respectively. A message that has moved to a composed
  parent-lineage session currently resolves as ``missing``, not a guess.
  Filed as a follow-up rather than implemented here without that grounding.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Literal

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

        return BlockAnchorResolution(
            state="missing",
            anchor=anchor,
            detail="no block with this content_hash resolves in the named session",
        )

    # The named message_id no longer exists (or belongs to a different
    # session than the anchor claims). Resolving across a fork/resume
    # lineage neighborhood is not yet implemented here (relocated_lineage) --
    # report the honest, conservative state rather than guess.
    return BlockAnchorResolution(
        state="missing",
        anchor=anchor,
        detail=(
            "message_id not found in the named session; lineage-neighborhood search "
            "(relocated_lineage) is not yet implemented, see polylogue-svfj follow-up"
        ),
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
