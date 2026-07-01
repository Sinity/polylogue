"""Bounded chronological transcript projection primitives."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import MaterialOrigin
from polylogue.surfaces.payloads import SurfacePayloadModel


class ChronicleMessagePayload(SurfacePayloadModel):
    """One prose message included in a bounded chronicle projection."""

    message_id: str
    role: str
    message_type: str
    occurred_at: datetime | None = None
    text: str


class ChronicleSessionPayload(SurfacePayloadModel):
    """Bounded first/last chronological prose for one session."""

    session_id: str
    title: str | None = None
    origin: str | None = None
    total_matching_messages: int
    included_count: int
    omitted_count: int
    edge_limit: int
    first_messages: tuple[ChronicleMessagePayload, ...] = ()
    last_messages: tuple[ChronicleMessagePayload, ...] = ()
    caveats: tuple[str, ...] = ()


class ChronicleProjectionPayload(SurfacePayloadModel):
    """Machine payload for a bounded multi-session chronicle read."""

    sessions: tuple[ChronicleSessionPayload, ...]
    session_count: int
    edge_limit: int
    body_policy: str = "authored-dialogue"
    caveats: tuple[str, ...] = ()


_AUTHORED_MATERIAL_ORIGINS = frozenset(
    {
        MaterialOrigin.HUMAN_AUTHORED.value,
        MaterialOrigin.ASSISTANT_AUTHORED.value,
    }
)


def _message_attr(message: object, *names: str) -> Any:
    for name in names:
        value = getattr(message, name, None)
        if value is not None:
            return value
    return None


def _enum_value(value: object, default: str) -> str:
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else default


def _timestamp_from_message(message: object) -> datetime | None:
    timestamp = _message_attr(message, "timestamp")
    if timestamp is None:
        sort_key = _message_attr(message, "sort_key")
        if sort_key is None:
            return None
        return datetime.fromtimestamp(float(sort_key), UTC)
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp


def _message_payload(message: object) -> ChronicleMessagePayload | None:
    text = str(_message_attr(message, "text") or "").strip()
    if not text:
        return None
    return ChronicleMessagePayload(
        message_id=str(_message_attr(message, "id", "message_id")),
        role=_enum_value(_message_attr(message, "role"), "unknown"),
        message_type=_enum_value(_message_attr(message, "message_type"), "message"),
        occurred_at=_timestamp_from_message(message),
        text=text,
    )


def _authored_dialogue_material(message: object) -> bool:
    material_origin = _enum_value(_message_attr(message, "material_origin"), "unknown")
    return material_origin in _AUTHORED_MATERIAL_ORIGINS


def _message_payloads(
    messages: Sequence[object], *, edge_limit: int
) -> tuple[tuple[ChronicleMessagePayload, ...], int]:
    payloads: list[ChronicleMessagePayload] = []
    skipped = 0
    for message in messages:
        if not _authored_dialogue_material(message):
            skipped += 1
            continue
        payload = _message_payload(message)
        if payload is None:
            skipped += 1
            continue
        payloads.append(payload)
        if len(payloads) >= edge_limit:
            break
    return tuple(payloads), skipped


def build_chronicle_session_payload(
    summary: SessionSummary,
    *,
    first_messages: Sequence[object],
    last_messages: Sequence[object],
    total_matching_messages: int,
    edge_limit: int,
) -> ChronicleSessionPayload:
    """Build an honest bounded first/last projection for one session."""

    first_payloads, first_skipped = _message_payloads(first_messages, edge_limit=edge_limit)
    first_ids = {payload.message_id for payload in first_payloads}
    last_payload_candidates, last_skipped = _message_payloads(last_messages, edge_limit=edge_limit)
    last_payloads = tuple(payload for payload in last_payload_candidates if payload.message_id not in first_ids)
    included_count = len(first_payloads) + len(last_payloads)
    omitted_count = max(total_matching_messages - included_count, 0)
    caveats: list[str] = []
    if omitted_count:
        caveats.append("middle_messages_omitted")
    if first_skipped or last_skipped:
        caveats.append("non_authored_or_empty_messages_omitted")
    if included_count < min(total_matching_messages, edge_limit * 2):
        caveats.append("empty_text_messages_omitted")
    return ChronicleSessionPayload(
        session_id=str(summary.id),
        title=summary.display_title,
        origin=str(summary.origin),
        total_matching_messages=total_matching_messages,
        included_count=included_count,
        omitted_count=omitted_count,
        edge_limit=edge_limit,
        first_messages=first_payloads,
        last_messages=last_payloads,
        caveats=tuple(caveats),
    )


def build_chronicle_projection_payload(
    sessions: list[ChronicleSessionPayload],
    *,
    edge_limit: int,
) -> ChronicleProjectionPayload:
    """Build the multi-session chronicle envelope."""

    caveats: list[str] = []
    if any(session.omitted_count for session in sessions):
        caveats.append("bounded_first_last_projection")
    if not sessions:
        caveats.append("empty_result_set")
    return ChronicleProjectionPayload(
        sessions=tuple(sessions),
        session_count=len(sessions),
        edge_limit=edge_limit,
        caveats=tuple(caveats),
    )


def _format_message_markdown(message: ChronicleMessagePayload) -> list[str]:
    timestamp = message.occurred_at.isoformat() if message.occurred_at is not None else "timestamp unavailable"
    return [
        f"### {timestamp} - {message.role} / {message.message_type}",
        "",
        message.text,
        "",
        f"`{message.message_id}`",
        "",
    ]


def render_chronicle_markdown(payload: ChronicleProjectionPayload) -> str:
    """Render a chronicle projection as bounded Markdown."""

    lines: list[str] = [
        "# Session Chronicle",
        "",
        f"- Sessions: {payload.session_count}",
        f"- Edge limit: {payload.edge_limit}",
        f"- Body policy: {payload.body_policy}",
    ]
    if payload.caveats:
        lines.append(f"- Caveats: {', '.join(payload.caveats)}")
    for session in payload.sessions:
        title = session.title or session.session_id
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- Session: `{session.session_id}`",
                f"- Origin: {session.origin or 'unknown'}",
                f"- Matching prose messages: {session.total_matching_messages}",
                f"- Included: {session.included_count}",
                f"- Omitted middle messages: {session.omitted_count}",
            ]
        )
        if session.caveats:
            lines.append(f"- Caveats: {', '.join(session.caveats)}")
        lines.extend(["", "### First Messages", ""])
        if session.first_messages:
            for message in session.first_messages:
                lines.extend(_format_message_markdown(message))
        else:
            lines.extend(["_No matching prose in the first edge._", ""])
        lines.extend(["### Last Messages", ""])
        if session.last_messages:
            for message in session.last_messages:
                lines.extend(_format_message_markdown(message))
        else:
            lines.extend(["_No distinct matching prose in the last edge._", ""])
    return "\n".join(lines).rstrip() + "\n"


def chronicle_json_document(payload: ChronicleProjectionPayload) -> dict[str, Any]:
    """Return the stable JSON document for chronicle reads."""

    return {"chronicle": payload.model_dump(mode="json")}


__all__ = [
    "ChronicleMessagePayload",
    "ChronicleProjectionPayload",
    "ChronicleSessionPayload",
    "build_chronicle_projection_payload",
    "build_chronicle_session_payload",
    "chronicle_json_document",
    "render_chronicle_markdown",
]
