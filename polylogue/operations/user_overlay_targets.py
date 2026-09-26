"""Resolve stored recall-pack and workspace items against one daemon archive."""

from __future__ import annotations

import asyncio

from polylogue.core.user_state_targets import TARGET_KIND_NAMES, identity_key
from polylogue.operations.daemon_mutations import _resolve_session_target
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _text(item: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value)
    return ""


def normalize_overlay_item(archive: ArchiveStore, item: dict[str, object]) -> dict[str, object]:
    kind = _text(item, "target_type", "type") or "session"
    message_id: str | None
    if kind == "session":
        token = _text(item, "session_id", "target_id", "id")
        try:
            session_id = _resolve_session_target(archive, archive.archive_root, token)
        except ValueError:
            return {
                "target_type": kind,
                "target_id": token,
                "session_id": token or None,
                "status": "missing",
                "disabled_reason": "session_not_found",
            }
        return {
            "target_type": kind,
            "target_id": session_id,
            "session_id": session_id,
            "status": "resolved",
            "identity_key": f"session:{session_id}",
        }
    if kind == "message":
        token = _text(item, "session_id")
        message_id = _text(item, "message_id", "target_id", "id")
        try:
            session_id = _resolve_session_target(archive, archive.archive_root, token)
            present = (
                archive._conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?",
                    (session_id, message_id),
                ).fetchone()
                is not None
            )
            if not present:
                raise ValueError(f"message {message_id!r} is not in session {session_id!r}")
        except ValueError as exc:
            return {
                "target_type": kind,
                "target_id": message_id,
                "session_id": token or None,
                "message_id": message_id or None,
                "status": "missing",
                "disabled_reason": str(exc) or "message_not_found",
            }
        return {
            "target_type": kind,
            "target_id": message_id,
            "session_id": session_id,
            "message_id": message_id,
            "status": "resolved",
            "identity_key": f"message:{session_id}:{message_id}",
        }
    if kind == "annotation":
        annotation_id = _text(item, "annotation_id", "target_id", "id")
        row = archive.get_annotation(annotation_id) if annotation_id else None
        if row is None:
            return {
                "target_type": kind,
                "target_id": annotation_id,
                "annotation_id": annotation_id or None,
                "status": "missing",
                "disabled_reason": "annotation_not_found",
            }
        return {
            "target_type": kind,
            "target_id": row["annotation_id"],
            "annotation_id": row["annotation_id"],
            "session_id": row["session_id"],
            "message_id": row["message_id"] or None,
            "annotated_target_type": row["target_type"],
            "annotated_target_id": row["target_id"],
            "note_text": row["note_text"],
            "status": "resolved",
            "identity_key": f"annotation:{row['annotation_id']}",
        }
    if kind == "mark":
        mark_type = _text(item, "mark_type")
        target_type = _text(item, "mark_target_type", "target_ref_type") or "session"
        target_id = _text(item, "mark_target_id", "target_id", "id")
        session_id = _text(item, "session_id")
        message_id = _text(item, "message_id") or None
        if message_id is not None:
            target_type, target_id = "message", message_id
        if session_id:
            raw_session_id = session_id
            try:
                session_id = _resolve_session_target(archive, archive.archive_root, session_id)
            except ValueError:
                session_id = raw_session_id
        rows = (
            archive.list_marks(
                mark_type=mark_type, target_type=target_type, target_id=target_id or None, session_id=session_id or None
            )
            if mark_type
            else []
        )
        if not rows:
            return {
                "target_type": kind,
                "target_id": f"{target_type}:{target_id}:{mark_type}" if mark_type else target_id,
                "session_id": session_id or None,
                "message_id": message_id,
                "mark_type": mark_type,
                "mark_target_type": target_type,
                "mark_target_id": target_id,
                "status": "missing",
                "disabled_reason": "mark_not_found" if mark_type else "mark_type_missing",
            }
        row = rows[0]
        return {
            "target_type": kind,
            "target_id": f"{row['target_type']}:{row['target_id']}:{row['mark_type']}",
            "session_id": row["session_id"],
            "message_id": row["message_id"] or None,
            "mark_type": row["mark_type"],
            "mark_target_type": row["target_type"],
            "mark_target_id": row["target_id"],
            "status": "resolved",
            "identity_key": f"mark:{row['target_type']}:{row['target_id']}:{row['mark_type']}",
        }
    if kind in TARGET_KIND_NAMES:
        from polylogue.api.user_state_resolver import resolve_insight_target

        token = _text(item, "session_id")
        target_id = _text(item, "target_id", "id")
        message_id = _text(item, "message_id") or None
        if not token:
            return {
                "target_type": kind,
                "target_id": target_id,
                "session_id": None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": "session_id_required",
            }
        try:
            session_id = _resolve_session_target(archive, archive.archive_root, token)
            resolved = asyncio.run(
                resolve_insight_target(
                    archive.archive_root,
                    target_type=kind,
                    target_id=target_id or None,
                    session_id=session_id,
                    message_id=message_id,
                )
            )
        except ValueError as exc:
            return {
                "target_type": kind,
                "target_id": target_id,
                "session_id": token,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": str(exc) or f"{kind}_not_found",
            }
        resolved_id = str(resolved["target_id"])
        resolved_message = resolved.get("message_id")
        return {
            "target_type": kind,
            "target_id": resolved_id,
            "session_id": session_id,
            "message_id": resolved_message,
            "status": "resolved",
            "identity_key": identity_key(
                kind, session_id=session_id, target_id=resolved_id, message_id=resolved_message
            ),
        }
    return {
        "target_type": kind,
        "target_id": _text(item, "target_id", "id"),
        "status": "unsupported",
        "disabled_reason": "unsupported_target_type",
    }


def normalize_recall_pack(
    archive: ArchiveStore, label: str, payload: dict[str, object]
) -> tuple[list[str], dict[str, object]]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not all(isinstance(item, dict) for item in raw_items):
        raise ValueError("recall pack payload must include an items list of objects")
    items = [normalize_overlay_item(archive, item) for item in raw_items]
    session_ids: list[str] = []
    for item in items:
        session_id = item.get("session_id")
        if item.get("status") == "resolved" and isinstance(session_id, str) and session_id not in session_ids:
            session_ids.append(session_id)
    normalized: dict[str, object] = {
        "schema_version": 1,
        "label": label,
        "summary": payload.get("summary") or payload.get("reason") or "",
        "items": items,
        "resolved_count": sum(item.get("status") == "resolved" for item in items),
        "degraded_count": sum(item.get("status") != "resolved" for item in items),
    }
    normalized.update({key: value for key, value in payload.items() if key not in {"items", "summary", "reason"}})
    return session_ids, normalized
