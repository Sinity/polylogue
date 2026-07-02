"""Streaming markdown export helpers for exact session read views."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TextIO

from polylogue.core.json import JSONDocument, json_document
from polylogue.rendering.block_models import RenderableBlock
from polylogue.rendering.blocks import has_structured_blocks, render_blocks_markdown
from polylogue.rendering.core_markdown import format_message_text
from polylogue.rendering.core_messages import normalize_render_timestamp


def stream_exact_session_markdown(
    archive_root: Path,
    session_ref: str,
    out_path: Path,
    *,
    prose_only: bool,
) -> bool:
    """Write an exact non-lineage session markdown export without buffering.

    Returns ``False`` when the session is absent or requires prefix-sharing
    lineage composition. The caller can then use the established eager path.
    """

    db_path = archive_root / "index.db"
    if not db_path.exists():
        return False

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        session_id = _resolve_session_id(conn, session_ref)
        if session_id is None or _has_prefix_sharing_edge(conn, session_id):
            return False
        session = conn.execute(
            """
            SELECT session_id, native_id, origin, title
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if session is None:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            title = session["title"] or session["native_id"] or session["session_id"]
            fh.write(f"# {title}\n\n")
            fh.write(f"Origin: {session['origin']}\n")
            fh.write(f"Session ID: {session['session_id']}\n\n")
            _write_message_stream(conn, session_id, fh, prose_only=prose_only)
        return True
    finally:
        conn.close()


def _resolve_session_id(conn: sqlite3.Connection, token: str) -> str | None:
    row = conn.execute(
        """
        SELECT session_id
        FROM sessions
        WHERE session_id = ? OR native_id = ?
        """,
        (token, token),
    ).fetchone()
    if row is not None:
        return str(row["session_id"])
    rows = conn.execute(
        """
        SELECT session_id
        FROM sessions
        WHERE session_id LIKE ? OR native_id LIKE ?
        ORDER BY updated_at_ms DESC, session_id
        LIMIT 2
        """,
        (f"{token}%", f"{token}%"),
    ).fetchall()
    return str(rows[0]["session_id"]) if len(rows) == 1 else None


def _has_prefix_sharing_edge(conn: sqlite3.Connection, session_id: str) -> bool:
    if not _table_exists(conn, "session_links"):
        return False
    row = conn.execute(
        """
        SELECT 1
        FROM session_links
        WHERE src_session_id = ?
          AND resolved_dst_session_id IS NOT NULL
          AND inheritance = 'prefix-sharing'
          AND branch_point_message_id IS NOT NULL
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row is not None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (name,),
    ).fetchone()
    return row is not None


def _write_message_stream(
    conn: sqlite3.Connection,
    session_id: str,
    fh: TextIO,
    *,
    prose_only: bool,
) -> None:
    cursor = conn.execute(
        """
        SELECT m.message_id,
               m.role,
               m.occurred_at_ms,
               b.block_id,
               b.block_type,
               b.text,
               b.tool_name,
               b.tool_id,
               b.tool_input,
               b.language,
               b.semantic_type,
               b.tool_result_is_error,
               b.tool_result_exit_code
        FROM messages m
        LEFT JOIN blocks b ON b.message_id = m.message_id
        WHERE m.session_id = ?
        ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id, b.position
        """,
        (session_id,),
    )
    current_id: str | None = None
    current_role = "message"
    current_ts: object = None
    blocks: list[RenderableBlock] = []
    for row in cursor:
        message_id = str(row["message_id"])
        if current_id is not None and message_id != current_id:
            _write_one_message(fh, role=current_role, timestamp=current_ts, blocks=blocks, prose_only=prose_only)
            blocks = []
        current_id = message_id
        current_role = str(row["role"] or "message")
        current_ts = row["occurred_at_ms"]
        if row["block_id"] is not None:
            blocks.append(_row_to_renderable_block(row))
    if current_id is not None:
        _write_one_message(fh, role=current_role, timestamp=current_ts, blocks=blocks, prose_only=prose_only)


def _row_to_renderable_block(row: sqlite3.Row) -> RenderableBlock:
    return RenderableBlock(
        type=str(row["block_type"] or "text"),
        text=row["text"],
        language=row["language"],
        tool_name=row["tool_name"],
        tool_id=row["tool_id"],
        tool_input=_json_mapping(row["tool_input"]),
    )


def _json_mapping(value: str | None) -> JSONDocument | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return json_document(parsed) if isinstance(parsed, dict) else None


def _write_one_message(
    fh: TextIO,
    *,
    role: str,
    timestamp: object,
    blocks: list[RenderableBlock],
    prose_only: bool,
) -> None:
    kept_blocks = [_block for _block in blocks if _keep_block(_block, prose_only=prose_only)]
    if not kept_blocks:
        return
    text = render_blocks_markdown(kept_blocks) if has_structured_blocks(kept_blocks) else _plain_block_text(kept_blocks)
    if not text.strip():
        return
    fh.write(f"## {role}\n")
    if timestamp is not None:
        fh.write(f"_Timestamp: {_timestamp_from_ms(timestamp)}_\n")
    fh.write("\n")
    fh.write(text.rstrip())
    fh.write("\n\n")


def _keep_block(block: RenderableBlock, *, prose_only: bool) -> bool:
    if not prose_only:
        return block.type != "reasoning"
    return block.type == "text"


def _plain_block_text(blocks: list[RenderableBlock]) -> str:
    return "\n\n".join(format_message_text(block.text or "") for block in blocks if block.text)


def _timestamp_from_ms(value: object) -> str | None:
    if isinstance(value, int | float):
        return normalize_render_timestamp(float(value) / 1000.0)
    try:
        return normalize_render_timestamp(float(str(value)) / 1000.0)
    except ValueError:
        return None


__all__ = ["stream_exact_session_markdown"]
