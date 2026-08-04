"""Codex Cloud tasks delivered inside the ChatGPT export (``codex.json``).

bd polylogue-2m2e: the 2026-07-29 export ships ``codex.json``, 20 Codex CLOUD
tasks with a ``turns`` structure — a second, independent delivery path for
Codex sessions, entirely distinct from local ``~/.codex/sessions`` rollouts.

Identity does NOT collide with ``codex-session`` records: local Codex CLI
sessions are keyed by a rollout ``session_id`` UUID (``sources/parsers/
codex.py``); these cloud tasks are keyed by ``task_e_<hex>`` ids with turn ids
of the form ``task_e_<hex>~usertrn_e_<hex>`` / ``...~assttrn_e_<hex>`` — a
disjoint namespace with no ULID/UUID overlap. They therefore coalesce with
NOTHING already in the archive; ingesting them adds a new session per task
rather than duplicating an existing one. Sessions stay under
``source_name=Provider.CHATGPT`` (they physically arrive via the ChatGPT
export) tagged with ``INGEST_FLAG`` so the population is identifiable and
distinguishable from real ChatGPT conversations / shared-conversation shells.

Each task carries exactly two turns (measured 2026-07-31: 20/20 tasks): a
user turn (the prompt) and an assistant turn (the final summary), with cloud
run metadata (``branch``, ``external_pull_request_id``, ``pull_request_status``,
``turn_status``) on the assistant turn. There are no timestamps anywhere in
this sidecar — ``created_at``/``updated_at`` are honestly left ``None`` rather
than fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, Role, SessionRefKind, WebConstructType

from .base import (
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedSessionRef,
    ParsedWebConstruct,
    human_authored_override,
)

INGEST_FLAG = "capture:chatgpt-codex-cloud-task"

_TASK_ID_PREFIX = "task_e_"
_TURN_ROLES = frozenset({"user", "assistant"})
_NULLISH_STRINGS = frozenset({"none", "null", ""})


def _clean_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in _NULLISH_STRINGS:
        return None
    return stripped


def looks_like(record: object) -> bool:
    """Structural detector for one ``codex.json`` task record.

    Deliberately tight (id namespace + turns shape) so it cannot accidentally
    swallow a real ChatGPT conversation fragment or an unrelated sibling
    record: real conversation fragments always carry a ``mapping`` dict,
    which this shape never has.
    """
    if not isinstance(record, dict):
        return False
    if "mapping" in record:
        return False
    task_id = record.get("id")
    if not isinstance(task_id, str) or not task_id.startswith(_TASK_ID_PREFIX):
        return False
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        return False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        if not isinstance(turn.get("id"), str) or not turn["id"]:
            return False
        if turn.get("role") not in _TURN_ROLES:
            return False
    return True


def _content_items(turn: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    items = turn.get("input_items") if turn.get("role") == "user" else turn.get("output_items")
    if not isinstance(items, list):
        return []
    out: list[Mapping[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            out.extend(part for part in content if isinstance(part, dict))
    return out


def _turn_text_and_constructs(turn: Mapping[str, object]) -> tuple[str, list[ParsedWebConstruct]]:
    text_parts: list[str] = []
    constructs: list[ParsedWebConstruct] = []
    for rank, part in enumerate(_content_items(turn)):
        content_type = part.get("content_type")
        if content_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif content_type == "repo_file_citation":
            path = part.get("path")
            line_start = part.get("line_range_start")
            line_end = part.get("line_range_end")
            constructs.append(
                ParsedWebConstruct(
                    construct_type=WebConstructType.CONTENT_REFERENCE,
                    provider_key="repo_file_citation",
                    text=path if isinstance(path, str) else None,
                    title=path if isinstance(path, str) else None,
                    rank=rank,
                    start_index=line_start if isinstance(line_start, int) else None,
                    end_index=line_end if isinstance(line_end, int) else None,
                )
            )
    return "\n".join(text_parts), constructs


def _turn_message(turn: Mapping[str, object], *, position: int, parent_id: str | None) -> ParsedMessage | None:
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id:
        return None
    role = Role.normalize(str(turn.get("role")))
    text, constructs = _turn_text_and_constructs(turn)
    blocks: list[ParsedContentBlock] = []
    if constructs:
        blocks.append(ParsedContentBlock(type=BlockType.TEXT, web_constructs=constructs))
    message_type = MessageType.MESSAGE
    return ParsedMessage(
        provider_message_id=turn_id,
        role=role,
        text=text or None,
        blocks=blocks,
        message_type=message_type,
        # codex.json is a ChatGPT-export sidecar, not a local agent transcript:
        # a role=user MESSAGE turn is positive human evidence. Apply the same
        # parser-level override as the main chat-export path so this independent
        # construction route does not depend on ParsedSession's later broad
        # export upgrade (polylogue-gzgyl continuation).
        material_origin=human_authored_override(
            role,
            message_type,
            classify_material_origin(
                role=role,
                message_type=message_type,
                text=text or None,
                block_types=tuple(block.type for block in blocks),
            ),
        ),
        parent_message_provider_id=parent_id,
        position=position,
    )


def _turn_session_events(turn: Mapping[str, object]) -> list[ParsedSessionEvent]:
    if turn.get("role") != "assistant":
        return []
    turn_id = turn.get("id")
    payload: dict[str, object] = {}
    for key in ("branch", "branch_name", "turn_status", "pull_request_status"):
        value = _clean_optional_str(turn.get(key))
        if value is not None:
            payload[key] = value
    if not payload:
        return []
    return [
        ParsedSessionEvent(
            event_type="chatgpt_codex_cloud_turn",
            source_message_provider_id=turn_id if isinstance(turn_id, str) else None,
            payload=payload,
        )
    ]


def _pull_request_ref(turn: Mapping[str, object]) -> ParsedSessionRef | None:
    pr_id = _clean_optional_str(turn.get("external_pull_request_id"))
    if pr_id is None:
        return None
    return ParsedSessionRef(kind=SessionRefKind.PULL_REQUEST.value, url=pr_id)


def parse_codex_task(task: Mapping[str, object], fallback_id: str) -> ParsedSession:
    task_id = task.get("id")
    provider_session_id = task_id if isinstance(task_id, str) and task_id else fallback_id
    title = task.get("title")

    turns = task.get("turns")
    turns_list = [turn for turn in turns if isinstance(turn, dict)] if isinstance(turns, list) else []

    messages: list[ParsedMessage] = []
    session_events: list[ParsedSessionEvent] = []
    session_refs: list[ParsedSessionRef] = []
    parent_id: str | None = None
    for position, turn in enumerate(turns_list):
        message = _turn_message(turn, position=position, parent_id=parent_id)
        if message is not None:
            messages.append(message)
            parent_id = message.provider_message_id
        session_events.extend(_turn_session_events(turn))
        ref = _pull_request_ref(turn)
        if ref is not None:
            session_refs.append(ref)

    branch = next(
        (value for turn in turns_list if (value := _clean_optional_str(turn.get("branch"))) is not None),
        None,
    )

    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=provider_session_id,
        title=title if isinstance(title, str) and title else None,
        messages=messages,
        session_events=session_events,
        session_refs=session_refs,
        git_branch=branch,
        ingest_flags=[INGEST_FLAG],
    )


__all__ = ["INGEST_FLAG", "looks_like", "parse_codex_task"]
