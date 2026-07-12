"""Beads interaction-history parser.

Beads exposes a repository-local, append-only ``interactions.jsonl`` ledger.
Each interaction is evidence of a change to one issue.  Polylogue represents
that ledger as one session per issue, retaining both readable timeline messages
and the original structured event payload.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import cast

from polylogue.archive.artifact_taxonomy.support import looks_like_beads_interaction
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.core.payload_coercion import optional_string

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent


def looks_like(payload: JSONDocument) -> bool:
    """Return whether *payload* is one Beads interaction-history record."""
    return looks_like_beads_interaction(payload)


def parse(
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Parse an interaction ledger into stable, per-issue timeline sessions."""
    del fallback_id
    by_issue: dict[str, list[JSONDocument]] = defaultdict(list)
    for payload in payloads:
        if not isinstance(payload, dict) or not looks_like(payload):
            continue
        issue_id = optional_string(payload.get("issue_id"))
        if issue_id is not None:
            by_issue[issue_id].append(payload)
    return [
        parse_issue_timeline(by_issue[issue_id], issue_id, source_path=source_path) for issue_id in sorted(by_issue)
    ]


def parse_issue_timeline(
    payloads: Iterable[JSONValue],
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> ParsedSession:
    """Parse one issue's interaction records into its evidence timeline."""
    records = [payload for payload in payloads if isinstance(payload, dict) and looks_like(payload)]
    if not records:
        raise ValueError("Beads issue timeline contains no interaction records")

    issue_ids = {optional_string(record.get("issue_id")) for record in records}
    if issue_ids != {fallback_id}:
        raise ValueError("Beads issue timeline must contain exactly one issue_id")

    records.sort(key=_sort_key)
    messages = [_message(record, position) for position, record in enumerate(records)]
    messages[-1] = messages[-1].model_copy(update={"is_active_leaf": True})
    events = [_event(record) for record in records]
    timestamps = [cast(str, record["created_at"]) for record in records]
    repository_root = _repository_root(source_path)
    provider_session_id = _provider_session_id(fallback_id, repository_root)
    return ParsedSession(
        source_name=Provider.BEADS,
        provider_session_id=provider_session_id,
        title=f"Beads issue {fallback_id}",
        created_at=timestamps[0],
        updated_at=timestamps[-1],
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id,
        session_events=events,
        working_directories=[str(repository_root)] if repository_root is not None else [],
    )


def _repository_root(source_path: str | None) -> Path | None:
    """Return the canonical workspace root for a direct ``.beads`` ledger path."""
    if source_path is None:
        return None
    path = Path(source_path)
    if path.name != "interactions.jsonl" or path.parent.name != ".beads":
        return None
    return path.parent.parent.resolve(strict=False)


def _provider_session_id(issue_id: str, repository_root: Path | None) -> str:
    if repository_root is None:
        return issue_id
    workspace_key = sha256(str(repository_root).encode("utf-8")).hexdigest()[:16]
    return f"{issue_id}@workspace-{workspace_key}"


def _sort_key(record: JSONDocument) -> tuple[str, str]:
    return (cast(str, record["created_at"]), cast(str, record["id"]))


def _message(record: JSONDocument, position: int) -> ParsedMessage:
    event_id = cast(str, record["id"])
    timestamp = cast(str, record["created_at"])
    text = _event_text(record)
    return ParsedMessage(
        provider_message_id=event_id,
        role=Role.USER,
        text=text,
        timestamp=timestamp,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        sender_name=optional_string(record.get("actor")),
    )


def _event(record: JSONDocument) -> ParsedSessionEvent:
    event_id = cast(str, record["id"])
    kind = cast(str, record["kind"])
    payload: dict[str, object] = {
        "event_id": event_id,
        "issue_id": cast(str, record["issue_id"]),
        "actor": optional_string(record.get("actor")),
        "extra": cast(dict[str, object], record["extra"]),
    }
    return ParsedSessionEvent(
        event_type=f"beads_{kind}",
        timestamp=cast(str, record["created_at"]),
        payload=payload,
        source_message_provider_id=event_id,
    )


def _event_text(record: JSONDocument) -> str:
    event_id = cast(str, record["id"])
    issue_id = cast(str, record["issue_id"])
    actor = optional_string(record.get("actor")) or "unknown actor"
    kind = cast(str, record["kind"]).replace("_", " ")
    extra = cast(JSONDocument, record["extra"])
    field = optional_string(extra.get("field"))
    if field is not None:
        old_value = json.dumps(extra.get("old_value"), ensure_ascii=False, sort_keys=True)
        new_value = json.dumps(extra.get("new_value"), ensure_ascii=False, sort_keys=True)
        summary = f"{actor} changed {field} from {old_value} to {new_value}"
    else:
        summary = f"{actor} recorded {kind}"
    reason = optional_string(extra.get("reason"))
    suffix = f" Reason: {reason}" if reason else ""
    return f"Beads {issue_id} {event_id}: {summary}.{suffix}"
