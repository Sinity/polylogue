"""Canonical durable user-overlay reads for daemon operations."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from polylogue.core.user_state_targets import TARGET_MESSAGE
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None else None


def _list(items: list[dict[str, object]]) -> dict[str, object]:
    return {"items": items, "total": len(items)}


def _get(item: dict[str, object] | None) -> dict[str, object]:
    return {"found": item is not None, "item": item}


def _saved_view(row: dict[str, str]) -> dict[str, object]:
    query_json = row["query_json"]
    try:
        query = json.loads(query_json)
    except json.JSONDecodeError:
        query = None
    return {
        "view_id": row["view_id"],
        "name": row["name"],
        "query": query,
        "query_json": query_json,
        "created_at": row["created_at"],
    }


def _recall_pack(row: dict[str, str]) -> dict[str, object]:
    try:
        session_ids = json.loads(row["session_ids_json"])
    except json.JSONDecodeError:
        session_ids = []
    try:
        body = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        body = {}
    return {
        "pack_id": row["pack_id"],
        "label": row["label"],
        "session_ids": session_ids,
        "payload": body,
        "created_at": row["created_at"],
    }


def _workspace(row: dict[str, str]) -> dict[str, object]:
    def decoded(key: str, expected: type, fallback: object) -> object:
        try:
            value = json.loads(row[key])
        except json.JSONDecodeError:
            return fallback
        return value if isinstance(value, expected) else fallback

    return {
        "workspace_id": row["workspace_id"],
        "name": row["name"],
        "mode": row["mode"],
        "open_targets": decoded("open_targets_json", list, []),
        "layout": decoded("layout_json", dict, {}),
        "active_target": decoded("active_target_json", dict, {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _resolve_scope(archive: ArchiveStore, token: str | None, target_id: str | None) -> str | None:
    if token is None or target_id is not None:
        return None
    from polylogue.operations.user_state_resolution import resolve_durable_user_state_session_id

    try:
        resolved = archive.resolve_session_id(token)
    except (KeyError, ValueError):
        resolved = None
    return str(resolved) if resolved else resolve_durable_user_state_session_id(archive.archive_root, token) or token


def _assertions(payload: Mapping[str, object], archive: ArchiveStore) -> dict[str, object]:
    from polylogue.core.enums import AssertionKind, AssertionStatus
    from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_claims
    from polylogue.surfaces.payloads import AssertionClaimListPayload, AssertionClaimPayload

    kinds_input = payload.get("kinds")
    kinds = (
        tuple(AssertionKind.from_string(str(item)) for item in kinds_input) if isinstance(kinds_input, list) else None
    )
    statuses_input = payload.get("statuses", ["active", "candidate"])
    statuses = (
        tuple(AssertionStatus.from_string(str(item)) for item in statuses_input)
        if isinstance(statuses_input, list)
        else None
    )
    raw_limit = payload.get("limit", 20)
    limit = raw_limit if isinstance(raw_limit, int) else int(str(raw_limit))
    context_inject = payload.get("context_inject")
    if not isinstance(context_inject, bool):
        context_inject = None
    conn = open_readonly_connection(archive.user_db_path)
    conn.row_factory = sqlite3.Row
    try:
        if kinds is None:
            claims = list_assertion_claims(
                conn,
                target_ref=_text(payload, "target_ref"),
                scope_ref=_text(payload, "scope_ref"),
                statuses=statuses,
                context_inject=context_inject,
                limit=limit,
            )
        else:
            claims = list_assertion_claims(
                conn,
                kinds=kinds,
                target_ref=_text(payload, "target_ref"),
                scope_ref=_text(payload, "scope_ref"),
                statuses=statuses,
                context_inject=context_inject,
                limit=limit,
            )
    finally:
        conn.close()
    items = tuple(AssertionClaimPayload.from_envelope(claim) for claim in claims)
    return AssertionClaimListPayload(
        items=items,
        total=len(items),
        limit=limit,
        statuses=statuses,
        kinds=kinds,
    ).model_dump(mode="json", exclude_none=True)


def execute_user_overlay_read(name: str, payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Read one overlay from the operation's pinned archive view."""
    if name == "user.assertions.list":
        return _assertions(payload, archive)
    if name in {"user.marks.list", "user.annotations.list"}:
        target_type = _text(payload, "target_type")
        target_id = _text(payload, "target_id")
        message_id = _text(payload, "message_id")
        session_id = _text(payload, "session_id")
        if message_id is not None:
            target_type, target_id = TARGET_MESSAGE, message_id
        scope_session_id = _resolve_scope(archive, session_id, target_id)
        if name == "user.marks.list":
            items = archive.list_marks(
                mark_type=_text(payload, "mark_type"),
                target_type=target_type,
                target_id=target_id,
                session_id=scope_session_id,
            )
        else:
            items = archive.list_annotations(
                target_type=target_type,
                target_id=target_id,
                session_id=scope_session_id,
            )
        return _list(cast(list[dict[str, object]], items))
    if name == "user.annotations.get":
        return _get(cast(dict[str, object] | None, archive.get_annotation(str(payload["id"]))))
    if name == "user.saved_views.list":
        return _list([_saved_view(row) for row in archive.list_views()])
    if name == "user.saved_views.get":
        row = archive.get_view(str(payload["id"]))
        return _get(_saved_view(row) if row is not None else None)
    if name == "user.recall_packs.list":
        return _list([_recall_pack(row) for row in archive.list_recall_packs()])
    if name == "user.recall_packs.get":
        row = archive.get_recall_pack(str(payload["id"]))
        return _get(_recall_pack(row) if row is not None else None)
    if name == "user.workspaces.list":
        return _list([_workspace(row) for row in archive.list_workspaces()])
    if name == "user.workspaces.get":
        row = archive.get_workspace(str(payload["id"]))
        return _get(_workspace(row) if row is not None else None)
    raise ValueError(f"user overlay read is not declared: {name}")
