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
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.core.payload_coercion import optional_string

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent

_REQUIRED_FIELDS = frozenset({"id", "kind", "created_at", "issue_id", "extra"})


def looks_like(payload: JSONDocument) -> bool:
    """Return whether *payload* is one Beads interaction-history record."""
    if not _REQUIRED_FIELDS.issubset(payload):
        return False
    return (
        isinstance(payload.get("id"), str)
        and isinstance(payload.get("kind"), str)
        and isinstance(payload.get("created_at"), str)
        and isinstance(payload.get("issue_id"), str)
        and isinstance(payload.get("extra"), dict)
    )


def parse(payloads: Iterable[JSONValue], fallback_id: str) -> list[ParsedSession]:
    """Parse an interaction ledger into stable, per-issue timeline sessions."""
    del fallback_id
    by_issue: dict[str, list[JSONDocument]] = defaultdict(list)
    for payload in payloads:
        if not isinstance(payload, dict) or not looks_like(payload):
            continue
        issue_id = optional_string(payload.get("issue_id"))
        if issue_id is not None:
            by_issue[issue_id].append(payload)
    return [parse_issue_timeline(by_issue[issue_id], issue_id) for issue_id in sorted(by_issue)]


def parse_issue_timeline(payloads: Iterable[JSONValue], fallback_id: str) -> ParsedSession:
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
    return ParsedSession(
        source_name=Provider.BEADS,
        provider_session_id=fallback_id,
        title=f"Beads issue {fallback_id}",
        created_at=timestamps[0],
        updated_at=timestamps[-1],
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id,
        session_events=events,
    )


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
