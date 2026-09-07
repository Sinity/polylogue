from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    Provider,
    SessionKind,
    StopReason,
    TitleSource,
    WebConstructType,
)
from polylogue.core.timestamps import parse_timestamp
from polylogue.sources.providers.chatgpt_session_models import ChatGPTNode
from polylogue.sources.tool_result_reasons import unknown_reason

from .base import (
    AdmissionLedger,
    AdmissionRefusalReason,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
    attachment_from_meta,
    human_authored_override,
    parser_admission,
    typed_unknown_block,
)
from .base_support import AttachmentDirection, derive_attachment_provenance
from .chatgpt_sidecars import strip_asset_pointer_scheme

SHARED_CONVERSATION_INDEX_INGEST_FLAG = "capture:chatgpt-shared-index-shell"


@dataclass(frozen=True)
class _GenerationTiming:
    message_provider_id: str
    elapsed_duration_ms: int
    started_at_ms: int | None
    ended_at_ms: int | None
    event_timestamp: str | None
    fidelity: str
    related_message_provider_ids: frozenset[str]
    duplicate_duration_message_provider_ids: frozenset[str]


def _coerce_float(value: object) -> float | None:
    # Exclude bool explicitly (bool is a subclass of int)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed.timestamp()
    return None


def _non_negative_finite_float(value: object) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None or not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _generation_branch_key(mapping: Mapping[str, object], node_id: str) -> str:
    """Return the first assistant-side node below the nearest user ancestor.

    ChatGPT repeats run-wide reasoning metadata across thought, tool, recap,
    and final-answer nodes. Grouping by this branch root deduplicates those
    copies while preserving regenerated alternatives beneath the same user
    message as distinct generations.
    """

    current_id = node_id
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        current = mapping.get(current_id)
        if not isinstance(current, Mapping):
            break
        parent_raw = current.get("parent")
        if not isinstance(parent_raw, str) or not parent_raw:
            break
        parent = mapping.get(parent_raw)
        if not isinstance(parent, Mapping):
            break
        parent_message = parent.get("message")
        parent_author = parent_message.get("author") if isinstance(parent_message, Mapping) else None
        parent_role = parent_author.get("role") if isinstance(parent_author, Mapping) else None
        if parent_role == "user":
            return current_id
        current_id = parent_raw
    return current_id


def _extract_generation_timings(mapping: Mapping[str, object]) -> list[_GenerationTiming]:
    """Select one authoritative lifecycle timing per ChatGPT generation.

    Native conversation payloads commonly copy ``reasoning_start_time`` and
    ``finished_duration_sec`` onto many nodes. A complete ``reasoning_recap``
    is preferred over those partial copies; otherwise the best complete node
    on the same assistant branch owns the timing. Provider-finished duration
    is authoritative, with a valid start/end delta as the derived fallback.

    The final tiebreak among nodes that agree on every other criterion is
    the candidate message id itself, never raw ``mapping`` iteration order
    (bd polylogue-uqwd). Two nodes on one branch can fully tie on source
    rank, reasoning-recap flag, start/end presence, and ``end_turn`` --
    ChatGPT re-exports of the SAME conversation are not guaranteed to
    serialize ``mapping`` keys in the same order every time, so an
    order-derived tiebreak silently anchored the resulting
    ``generation_lifecycle`` event to a different message id per export,
    even though every node's own content was byte-identical. That anchor
    drift then made ``event_base_identity_hash``
    (``pipeline/ids.py``) read the same event as two disjoint identities
    across revisions -- the same "array position is not identity" hazard
    ``session_revision_projection`` already guards against for messages,
    attachments, and events.
    """

    candidates: dict[str, list[tuple[tuple[int, int, int, int, str], _GenerationTiming]]] = {}
    related_message_ids: dict[str, set[str]] = {}
    legacy_duration_by_message_id: dict[str, dict[str, int]] = {}
    for node_id, raw_node in mapping.items():
        if not isinstance(raw_node, Mapping):
            continue
        raw_message = raw_node.get("message")
        if not isinstance(raw_message, Mapping):
            continue
        raw_metadata = raw_message.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            continue

        raw_author = raw_message.get("author")
        raw_role = raw_author.get("role") if isinstance(raw_author, Mapping) else None
        if raw_role not in {"assistant", "tool"}:
            # Duration metadata on a human/system row is message-local evidence,
            # not a ChatGPT generation lifecycle measurement.
            continue

        message_id_raw = raw_message.get("id") or raw_node.get("id") or node_id
        message_id = str(message_id_raw)
        branch_key = _generation_branch_key(mapping, str(node_id))
        native_timing_field_names = (
            "reasoning_start_time",
            "reasoning_end_time",
            "finished_duration_sec",
        )
        has_native_timing_field = any(field_name in raw_metadata for field_name in native_timing_field_names)
        has_legacy_duration_field = "durationMs" in raw_metadata or "duration_ms" in raw_metadata
        if has_native_timing_field or has_legacy_duration_field:
            related_message_ids.setdefault(branch_key, set()).add(message_id)

        start_sec = _non_negative_finite_float(raw_metadata.get("reasoning_start_time"))
        end_sec = _non_negative_finite_float(raw_metadata.get("reasoning_end_time"))
        finished_sec = _non_negative_finite_float(raw_metadata.get("finished_duration_sec"))
        legacy_duration_raw = raw_metadata.get("durationMs")
        if legacy_duration_raw is None:
            legacy_duration_raw = raw_metadata.get("duration_ms")
        legacy_duration_ms = _non_negative_int(legacy_duration_raw)
        if legacy_duration_ms is not None:
            legacy_duration_by_message_id.setdefault(branch_key, {})[message_id] = legacy_duration_ms

        has_valid_native_timing_value = any(value is not None for value in (start_sec, end_sec, finished_sec))
        if finished_sec is not None:
            elapsed_ms = round(finished_sec * 1000)
            fidelity = "exact"
            source_rank = 3
        elif start_sec is not None and end_sec is not None and end_sec >= start_sec:
            elapsed_ms = round((end_sec - start_sec) * 1000)
            fidelity = "derived"
            source_rank = 2
        elif has_valid_native_timing_value and legacy_duration_ms is not None:
            # Legacy duration remains a lifecycle fallback only when the same
            # branch carries structured native lifecycle evidence. A bare
            # durationMs/duration_ms field keeps its established message-local
            # meaning and is not promoted into a synthetic generation event.
            elapsed_ms = legacy_duration_ms
            fidelity = "exact"
            source_rank = 1
        else:
            continue

        content = raw_message.get("content")
        content_type = content.get("content_type") if isinstance(content, Mapping) else None
        timing = _GenerationTiming(
            message_provider_id=message_id,
            elapsed_duration_ms=elapsed_ms,
            started_at_ms=round(start_sec * 1000) if start_sec is not None else None,
            ended_at_ms=round(end_sec * 1000) if end_sec is not None else None,
            event_timestamp=str(end_sec) if end_sec is not None else None,
            fidelity=fidelity,
            related_message_provider_ids=frozenset(),
            duplicate_duration_message_provider_ids=frozenset(),
        )
        score = (
            source_rank,
            int(content_type == "reasoning_recap"),
            int(start_sec is not None and end_sec is not None),
            int(raw_message.get("end_turn") is True),
            message_id,
        )
        candidates.setdefault(branch_key, []).append((score, timing))

    timings: list[_GenerationTiming] = []
    for branch_key, branch_candidates in candidates.items():
        selected = max(branch_candidates, key=lambda item: item[0])[1]
        timings.append(
            replace(
                selected,
                related_message_provider_ids=frozenset(related_message_ids.get(branch_key, ())),
                duplicate_duration_message_provider_ids=frozenset(
                    message_provider_id
                    for message_provider_id, duration_ms in legacy_duration_by_message_id.get(branch_key, {}).items()
                    if duration_ms == selected.elapsed_duration_ms
                ),
            )
        )
    return timings


#: The export's terminal states for a completed tool run. Everything else --
#: including ``in_progress`` -- is a state with no verdict in it.
_CHATGPT_TERMINAL_NODE_STATES: dict[str, bool] = {
    "finished_partial_completion": True,
    "finished_successfully": False,
}
#: Non-terminal states the export is known to emit: the run has not concluded,
#: so no outcome was reported. Distinct from a token outside both sets, which
#: is a state this mapping does not cover.
_CHATGPT_UNCONCLUDED_NODE_STATES = frozenset({"in_progress"})

#: ``metadata.aggregate_result.status``: a code-interpreter run's own verdict.
#: The node's ``status`` describes message delivery, so a run that raised
#: in-kernel still sits on a ``finished_successfully`` node -- measured over
#: 614 captures, all 165 ``failed_with_in_kernel_exception`` and all 7
#: ``cancelled`` runs do. ``cancelled`` is deliberately in neither set: the
#: run concluded, but reports nothing about whether the tool did its work.
_CHATGPT_AGGREGATE_RESULT_STATES: dict[str, bool] = {
    "success": False,
    "failed_with_in_kernel_exception": True,
}


def _author_display_name(author: object) -> str | None:
    """Who the export says sent this message.

    ``author.name`` is the ordinary carrier. ``author.metadata.real_author``
    is the provider naming the tool that actually produced an
    assistant-role turn (``tool:web.run``, ``tool:web.search``); without it
    those turns read as the model's own words.
    """
    if not isinstance(author, Mapping):
        return None
    name = _string_value(author, "name")
    if name is not None:
        return name
    metadata = author.get("metadata")
    return _string_value(metadata, "real_author") if isinstance(metadata, Mapping) else None


def _sibling_ordinals(mapping: Mapping[str, object]) -> dict[str, int]:
    """Ordinal of each node among the siblings naming the same ``parent``.

    ``children`` states sibling order directly and stays authoritative
    wherever the export carries it. The reduced export shape ships
    ``{id, message, parent}`` nodes with no ``children`` array at all, and
    without a parent-edge fallback every regenerated alternative in such an
    export collapses to ``branch_index == 0`` -- indistinguishable from the
    first response. Mapping order is the export's own record order, so
    counting arrivals per parent recovers the sequence ``children`` names.
    """
    ordinals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        parent = node.get("parent")
        parent_key = parent if isinstance(parent, str) and parent else ""
        ordinal = counts.get(parent_key, 0)
        counts[parent_key] = ordinal + 1
        ordinals[node_id] = ordinal
    return ordinals


def _aggregate_result_outcome(aggregate_result: object) -> tuple[bool | None, bool]:
    """Return (is_error, run record present) for a code-interpreter run.

    ``timeout_triggered`` is not a failure flag: it is an integer (60) that
    appears on a successful run as readily as on a failed one, so it is never
    read as error evidence.
    """
    if not isinstance(aggregate_result, Mapping):
        return None, False
    if isinstance(aggregate_result.get("in_kernel_exception"), Mapping):
        return True, True
    if isinstance(aggregate_result.get("system_exception"), Mapping):
        return True, True
    status = aggregate_result.get("status")
    if isinstance(status, str) and status in _CHATGPT_AGGREGATE_RESULT_STATES:
        return _CHATGPT_AGGREGATE_RESULT_STATES[status], True
    return None, True


def _node_status_outcome(node_status: object, aggregate_result: object = None) -> tuple[bool | None, str | None]:
    """Map a tool-result node's structural evidence to (is_error, unknown reason).

    Two independent fields state an outcome. ``metadata.aggregate_result`` is
    the run's own verdict and outranks the node's ``status``, which only says
    the message carrying the run's output was delivered. A run record that
    states no verdict leaves the result honestly unknown rather than letting
    the delivery status stand in for one. Read from the fields, never from the
    result text; ChatGPT exports carry no exit code.
    """
    aggregate_is_error, aggregate_present = _aggregate_result_outcome(aggregate_result)
    if aggregate_is_error is not None:
        return aggregate_is_error, None
    if aggregate_present:
        return None, unknown_reason(is_error=None, outcome_field_present=True)
    if isinstance(node_status, str) and node_status in _CHATGPT_TERMINAL_NODE_STATES:
        return _CHATGPT_TERMINAL_NODE_STATES[node_status], None
    unrecognized = isinstance(node_status, str) and node_status not in _CHATGPT_UNCONCLUDED_NODE_STATES
    return None, unknown_reason(is_error=None, outcome_field_present=unrecognized)


def _string_value(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _int_value(payload: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _iter_mapping_items(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _reference_token(item: Mapping[str, object]) -> str | None:
    """Compose ChatGPT's own citation token for a result, e.g. ``turn0search4``.

    ``ref_id`` is a ``{turn_index, ref_type, ref_index}`` triple, and the token
    it spells is exactly what the inline citation markers in assistant text
    carry -- composing it here is what lets a citation anchor join the result
    it cites. Measured ``ref_type`` values:
    search/view/news/academia/reddit/youtube.
    """
    ref = item.get("ref_id")
    if not isinstance(ref, Mapping):
        return None
    turn_index = _int_value(ref, "turn_index")
    ref_type = _string_value(ref, "ref_type")
    ref_index = _int_value(ref, "ref_index")
    if turn_index is None or ref_type is None or ref_index is None:
        return None
    return f"turn{turn_index}{ref_type}{ref_index}"


def _construct_from_reference(
    item: Mapping[str, object],
    *,
    construct_type: WebConstructType,
    provider_key: str,
    rank: int | None = None,
    group_id: str | None = None,
    group_title: str | None = None,
) -> ParsedWebConstruct:
    # File-search citation rows nest their source identity one or two levels
    # down (item.metadata carries the file name/id/source, item.metadata.extra
    # carries cited_message_id, library_file_id, source_url, ...). Consult the
    # nested layers whenever the top level misses — otherwise a file citation
    # keeps only its answer-side anchor span and loses which document it cites.
    metadata = item.get("metadata")
    nested = metadata if isinstance(metadata, Mapping) else {}
    extra_value = nested.get("extra")
    extra = extra_value if isinstance(extra_value, Mapping) else {}

    def pick(*keys: str) -> str | None:
        return _string_value(item, *keys) or _string_value(nested, *keys) or _string_value(extra, *keys)

    return ParsedWebConstruct(
        construct_type=construct_type,
        provider_key=provider_key,
        title=pick("title", "name", "source_name", "source_label"),
        url=pick("url", "link", "source_url", "cloud_doc_url"),
        text=pick("snippet", "text", "content", "description", "quote"),
        source_id=pick("id", "source_id", "ref_id", "attribution_id", "textdoc_id", "library_file_id")
        or _reference_token(item),
        group_id=group_id,
        group_title=group_title,
        asset_pointer=pick("asset_pointer"),
        mime_type=pick("mime_type", "media_type"),
        rank=rank if rank is not None else _int_value(item, "rank", "index"),
        start_index=_int_value(item, "start_index", "start_idx", "start_ix", "start"),
        end_index=_int_value(item, "end_index", "end_idx", "end_ix", "end"),
    )


def _construct_has_content(construct: ParsedWebConstruct) -> bool:
    return any(
        (
            construct.url,
            construct.title,
            construct.text,
            construct.source_id,
            construct.asset_pointer,
            construct.start_index is not None,
            construct.end_index is not None,
        )
    )


def _constructs_from_content_reference_item(
    item: Mapping[str, object],
    *,
    provider_key: str,
    rank: int,
) -> list[ParsedWebConstruct]:
    """Expand one ``content_references``/``citations`` entry into its constructs.

    Most citation shapes carry their own url/title directly (file-search
    citations nested one or two levels down in ``item.metadata``/
    ``item.metadata.extra``, already handled by ``_construct_from_reference``).
    But ``grouped_webpages`` reference items (polylogue-zocm: measured 60.8%
    of July content_reference URLs) carry NO url of their own -- every URL
    lives one level down, in ``item.items[]`` (primary sources) and
    ``item.fallback_items[]`` (secondary/backup sources). Mirrors the
    ``search_result_groups`` descent below (``results``/``items``/
    ``search_results``/``sources``).
    """
    primary = _construct_from_reference(
        item,
        construct_type=WebConstructType.CONTENT_REFERENCE,
        provider_key=provider_key,
        rank=rank,
    )
    constructs = [primary] if _construct_has_content(primary) else []
    group_title = _string_value(item, "alt", "matched_text", "title")
    group_id = _string_value(item, "id") or f"{provider_key}:{rank}"
    for nested_key in ("items", "fallback_items"):
        for nested_rank, nested_item in enumerate(_iter_mapping_items(item.get(nested_key))):
            constructs.append(
                _construct_from_reference(
                    nested_item,
                    construct_type=WebConstructType.CONTENT_REFERENCE,
                    provider_key=f"{provider_key}.{nested_key}",
                    rank=nested_rank,
                    group_id=group_id,
                    group_title=group_title,
                )
            )
    return constructs


#: ``metadata.finish_details.type`` -> ``messages.stop_reason``. The wire
#: vocabulary is OpenAI's; ``messages.stop_reason`` is Anthropic's
#: :class:`StopReason`, so only exact equivalences are mapped and every other
#: token (``interrupted``, ``skipped``, ``unknown``) leaves the column NULL
#: rather than widening a guess into it.
_CHATGPT_STOP_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
}


def _stop_reason_from_finish_details(finish_details: object) -> str | None:
    """Return the ``messages.stop_reason`` value a node's finish details name."""
    if not isinstance(finish_details, Mapping):
        return None
    mapped = _CHATGPT_STOP_REASONS.get(str(finish_details.get("type")))
    return mapped.value if mapped is not None else None


#: A ChatGPT author whose ``real_author`` names a tool did not write its
#: message: the envelope role is the channel the tool's output was rendered
#: through. Measured values: ``tool:web``, ``tool:web.run``, ``tool:web.search``.
_CHATGPT_TOOL_AUTHOR_PREFIX = "tool:"


def _real_author(author: object) -> str | None:
    """Return ``author.metadata.real_author``, the wire's own authorship note."""
    if not isinstance(author, Mapping):
        return None
    metadata = author.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    return _string_value(metadata, "real_author")


#: A ChatGPT conversation permalink. The captured id is the cited
#: conversation's own native id -- the join key a cross-session edge needs,
#: kept resolvable on the construct so the edge can be built later without
#: reparsing.
_CHATGPT_CONVERSATION_URL_RE = re.compile(r"^https?://chatgpt\.com/c/([0-9a-fA-F-]+)")


def _conversation_context_citation_construct(
    item: Mapping[str, object],
    citation: Mapping[str, object],
    *,
    rank: int,
) -> ParsedWebConstruct:
    """Project one cross-conversation memory retrieval into a construct.

    ChatGPT's personalization layer answers from earlier conversations and
    records each retrieval here: which conversation was read, its title and
    snippet, and the span of the answer the retrieval backs.
    """
    url = _string_value(citation, "url")
    cited_conversation = _CHATGPT_CONVERSATION_URL_RE.match(url) if url else None
    return ParsedWebConstruct(
        construct_type=WebConstructType.CONTENT_REFERENCE,
        provider_key="conversation_context_citation_metadata",
        title=_string_value(citation, "conversation_title", "title"),
        url=url,
        text=_string_value(citation, "snippet", "matched_text"),
        source_id=(
            cited_conversation.group(1)
            if cited_conversation
            else _string_value(citation, "memory_id", "citation_uuid") or _string_value(item, "citation_uuid")
        ),
        group_id=_string_value(item, "citation_uuid"),
        group_title=_string_value(citation, "category") or _string_value(item, "retrieval_origin"),
        status=_string_value(citation, "conversation_context_type"),
        rank=rank,
        start_index=_int_value(citation, "start_idx"),
        end_index=_int_value(citation, "end_idx"),
    )


#: Wire keys that can hold a search group's results. ``entries`` is the shape
#: every measured capture uses (67,867 results over 614 captures); the others
#: are alternate spellings kept so an older export shape still lands.
_SEARCH_RESULT_GROUP_ENTRY_KEYS = ("entries", "results", "items", "search_results", "sources")


def _search_result_group_constructs(container: object, *, provider_key: str) -> list[ParsedWebConstruct]:
    """Project one ``search_result_groups`` list into SEARCH_RESULT constructs.

    A group is ``{type, domain, entries[]}``: ``domain`` is the only label it
    carries, so it is the group title, and each entry carries its own
    title/url/snippet plus the ``ref_id`` triple that names it in the answer's
    inline citations.
    """
    if not isinstance(container, Mapping):
        return []
    constructs: list[ParsedWebConstruct] = []
    for group_rank, group in enumerate(_iter_mapping_items(container.get("search_result_groups"))):
        group_id = _string_value(group, "id", "group_id") or str(group_rank)
        group_title = _string_value(group, "title", "name", "query", "domain")
        candidates: object = []
        for key in _SEARCH_RESULT_GROUP_ENTRY_KEYS:
            value = group.get(key)
            if value:
                candidates = value
                break
        for rank, item in enumerate(_iter_mapping_items(candidates)):
            constructs.append(
                _construct_from_reference(
                    item,
                    construct_type=WebConstructType.SEARCH_RESULT,
                    provider_key=provider_key,
                    rank=rank,
                    group_id=group_id,
                    group_title=group_title,
                )
            )
    return constructs


def _constructs_from_chatgpt_metadata(msg_metadata: object) -> list[ParsedWebConstruct]:
    if not isinstance(msg_metadata, Mapping):
        return []
    constructs: list[ParsedWebConstruct] = []
    for item in _iter_mapping_items(msg_metadata.get("canvas")):
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.CANVAS,
                provider_key="canvas",
                title=_string_value(item, "title", "name"),
                text=_string_value(item, "text", "content"),
                source_id=_string_value(item, "id", "canvas_id", "textdoc_id"),
                status=_string_value(item, "status"),
            )
        )
    for provider_key in ("content_references", "citations", "_cite_metadata"):
        value = msg_metadata.get(provider_key)
        for rank, item in enumerate(_iter_mapping_items(value)):
            constructs.extend(_constructs_from_content_reference_item(item, provider_key=provider_key, rank=rank))
    search_queries = msg_metadata.get("search_queries")
    if isinstance(search_queries, list):
        for rank, item in enumerate(search_queries):
            if isinstance(item, str) and item:
                constructs.append(
                    ParsedWebConstruct(
                        construct_type=WebConstructType.SEARCH_QUERY,
                        provider_key="search_queries",
                        query=item,
                        rank=rank,
                    )
                )
            elif isinstance(item, Mapping):
                constructs.append(
                    ParsedWebConstruct(
                        construct_type=WebConstructType.SEARCH_QUERY,
                        provider_key="search_queries",
                        query=_string_value(item, "query", "text"),
                        title=_string_value(item, "title"),
                        rank=rank,
                    )
                )
    constructs.extend(_search_result_group_constructs(msg_metadata, provider_key="search_result_groups"))
    # The reasoning-trace copy of the same structure: a search a thought step
    # ran, expandable in the UI under the chain of thought. Same group/entry
    # shape, its own provider_key so the two are distinguishable downstream.
    constructs.extend(
        _search_result_group_constructs(
            msg_metadata.get("inline_cot_expandable_content"),
            provider_key="inline_cot_expandable_content.search_result_groups",
        )
    )
    for rank, item in enumerate(_iter_mapping_items(msg_metadata.get("selected_sources"))):
        constructs.append(
            _construct_from_reference(
                item,
                construct_type=WebConstructType.SELECTED_SOURCE,
                provider_key="selected_sources",
                rank=rank,
            )
        )
    for rank, item in enumerate(_iter_mapping_items(msg_metadata.get("image_results"))):
        constructs.append(
            _construct_from_reference(
                item,
                construct_type=WebConstructType.IMAGE_RESULT,
                provider_key="image_results",
                rank=rank,
            )
        )
    async_task_type = _string_value(msg_metadata, "async_task_type")
    async_task_id = _string_value(msg_metadata, "async_task_id")
    async_task_title = _string_value(msg_metadata, "async_task_title")
    if async_task_type or async_task_id or async_task_title:
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.ASYNC_TASK,
                provider_key="async_task",
                title=async_task_title,
                task_id=async_task_id,
                task_type=async_task_type,
            )
        )
    for item in _iter_mapping_items(msg_metadata.get("aggregate_result")):
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.ASYNC_TASK,
                provider_key="aggregate_result",
                title=_string_value(item, "title"),
                # The program the run executed. It exists nowhere else: the
                # calling ``code`` node's text is measured empty or an
                # unrelated tool-call payload on every sampled call, and the
                # result node's text is the run's output, not its input.
                text=_string_value(item, "code"),
                status=_string_value(item, "status"),
                task_id=_string_value(item, "run_id"),
            )
        )
    for rank, item in enumerate(_iter_mapping_items(msg_metadata.get("conversation_context_citation_metadata"))):
        citation = item.get("citation")
        if not isinstance(citation, Mapping):
            continue
        constructs.append(_conversation_context_citation_construct(item, citation, rank=rank))
    return constructs


def _active_path_node_ids(mapping: Mapping[str, object], current_node: str | None) -> list[str]:
    """Return the active ChatGPT path from root to ``current_node``.

    ChatGPT exports preserve regenerated and edited branches in ``mapping`` and
    use ``current_node`` only to identify the leaf the user last saw. The v1
    parser contract keeps every branch and carries the active path explicitly
    instead of using it as a lossy filter (#1743).
    """
    if current_node and current_node in mapping:
        path: list[str] = []
        seen: set[str] = set()
        node_id: str | None = current_node
        while node_id is not None and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            path.append(node_id)
            node = mapping[node_id]
            node_id = node.get("parent") if isinstance(node, dict) else None
        path.reverse()
        return path

    return []


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _asset_pointer_block_metadata(record: Mapping[str, object], pointer: str) -> dict[str, object]:
    """Carry an asset pointer and its intrinsic dimensions on an IMAGE block.

    ``blocks`` has no metadata column; the dict is projected verbatim into a
    ``chatgpt_block_metadata`` session event (``_block_metadata_evidence_events``),
    which is the only place an asset's width/height/byte size survives — the
    attachment row has no dimension columns.
    """
    metadata: dict[str, object] = {"asset_pointer": pointer}
    for key in ("width", "height", "size_bytes"):
        value = _non_negative_int(record.get(key))
        if value is not None:
            metadata[key] = str(value)
    return metadata


def _append_asset_attachment(
    attachments: list[ParsedAttachment],
    record: Mapping[str, object],
    *,
    pointer: str,
    message_provider_id: str,
    attachment_kind: str,
    direction: AttachmentDirection | None,
    producer_ref: str | None,
    dedupe_from: int,
) -> None:
    """Record an asset-pointer record as the attachment its bytes bind to.

    An attachment row is the only acquisition identity an asset has:
    ``assembly_chatgpt.py`` joins acquired export members onto attachments by
    the bare file id, so a pointer that reaches storage as block metadata
    alone leaves its acquired bytes with nothing to bind to.

    ``dedupe_from`` is the index at which this message's own attachments
    start. A user upload is named twice — once by the message's ``metadata``
    attachment row (bare ``file-<id>``) and once by the content part's
    pointer URI (``file-service://file-<id>``) — and both normalize to the
    same id, so the second naming must not mint a second row.
    """
    file_id = strip_asset_pointer_scheme(pointer)
    if not file_id:
        return
    for existing in attachments[dedupe_from:]:
        if strip_asset_pointer_scheme(existing.provider_attachment_id) == file_id:
            return
    attachments.append(
        ParsedAttachment(
            provider_attachment_id=pointer,
            message_provider_id=message_provider_id,
            # Read off the URI: the id space the export's asset members and
            # ``library_files.json`` keys share.
            provider_file_id=file_id,
            size_bytes=_non_negative_int(record.get("size_bytes")),
            attachment_kind=attachment_kind,
            direction=direction,
            producer_ref=producer_ref,
        )
    )


# ChatGPT embeds inline citation anchors in assistant text as private-use
# unicode spans: U+E200 opens, U+E202 separates reference tokens, U+E201
# closes (e.g. "\ue200filecite\ue202turn3file14\ue202L180-L293\ue201").
# The span carries no human-readable text -- the resolvable citation rows
# live in message metadata (`citations`/`content_references`) and are
# preserved as web constructs. The raw markers otherwise leak invisible
# glyphs into search text and rendered transcripts; the untouched original
# remains in the source-tier raw payload.
_CITATION_MARKER_RE = re.compile("\ue200.*?\ue201|[\ue200\ue201\ue202]")
_CITATION_MARKER_SPAN_RE = re.compile("\ue200(.*?)\ue201")


def _strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER_RE.sub("", text)


_SANDBOX_FILE_RE = re.compile(r"sandbox:(/mnt/data/[^\s)\]\"'>]+)")


def _sandbox_file_paths(text: str) -> list[str]:
    """Ordered, deduplicated ``/mnt/data`` paths linked in assistant text.

    Trailing prose punctuation is stripped so ``(sandbox:/mnt/data/kit.zip).``
    yields ``/mnt/data/kit.zip``. Directory links keep their trailing slash in
    the returned path.
    """

    seen: dict[str, None] = {}
    for match in _SANDBOX_FILE_RE.finditer(text):
        path = match.group(1).rstrip(".,;:!?*`")
        if path != "/mnt/data/":
            seen.setdefault(path)
    return list(seen)


def _extract_content_text(content: Mapping[str, object]) -> str:
    """Extract message text from a ChatGPT content block.

    Handles the common ``parts`` array (strings and structured dicts carrying
    ``text``) and falls back to non-``parts`` content shapes — ``code`` and
    ``execution_output`` carry a top-level ``text``, browsing display carries a
    ``result``, ``citable_code_output`` carries ``output_str``,
    ``reasoning_recap`` carries its recap line under ``content``, and
    ``thoughts`` carries an array of ``{summary, content, ...}`` reasoning-step
    entries. Without a fallback the block built for such a node gets an empty
    ``text`` and the message is dropped entirely (#1744).
    """
    parts = content.get("parts")
    if isinstance(parts, list):
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, str) and part:
                text_parts.append(part)
            elif isinstance(part, dict):
                # Extract text from structured parts (e.g. tether_quote dicts)
                t = part.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
                # Skip image_asset_pointer and other non-text dicts
        if text_parts:
            return "\n".join(text_parts)
    # Non-parts content shapes: code / execution_output / system_error /
    # tether_quote carry top-level text; tether_browsing_display carries a
    # result string; citable_code_output carries output_str (polylogue-xofj).
    top_text = content.get("text")
    if isinstance(top_text, str) and top_text:
        return top_text
    result = content.get("result")
    if isinstance(result, str) and result:
        return result
    output_str = content.get("output_str")
    if isinstance(output_str, str) and output_str:
        return output_str
    # ``reasoning_recap`` is the one content type that puts its text under
    # ``content`` (polylogue-3i043); every other shape reserves that key for
    # structured payloads, so a string here is message text.
    recap = content.get("content")
    if isinstance(recap, str) and recap:
        return recap
    thoughts = content.get("thoughts")
    if isinstance(thoughts, list):
        thought_parts: list[str] = []
        for thought in thoughts:
            if not isinstance(thought, dict):
                continue
            step_content = thought.get("content")
            if isinstance(step_content, str) and step_content:
                thought_parts.append(step_content)
                continue
            summary = thought.get("summary")
            if isinstance(summary, str) and summary:
                thought_parts.append(summary)
        if thought_parts:
            return "\n".join(thought_parts)
    return ""


#: Block types that can carry a tool-role node's payload. The first one a
#: tool-role node produced becomes its canonical TOOL_RESULT; a typed-unknown
#: block is excluded so an unrecognized wire shape keeps its own disposition.
_TOOL_RESULT_CARRIER_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.TEXT, BlockType.DOCUMENT, BlockType.CODE, BlockType.TOOL_USE, BlockType.THINKING}
)


def _owning_tool_call_id(mapping: Mapping[str, object], parent_id: str | None) -> str | None:
    """Resolve which node a ``role: tool`` result answers.

    A tool episode is a chain: the calling node, then one or more ``role: tool``
    result nodes. The provider attaches the second and later results to the
    *previous result*, so the direct mapping parent names another answer rather
    than the node the episode hangs off. Skipping the tool-role ancestors
    reaches that node, and every result of one episode names the same owner.
    """
    seen: set[str] = set()
    current = parent_id
    while isinstance(current, str) and current and current not in seen:
        seen.add(current)
        node = mapping.get(current)
        if not isinstance(node, Mapping):
            return current
        message = node.get("message")
        author = message.get("author") if isinstance(message, Mapping) else None
        if not (isinstance(author, Mapping) and author.get("role") == "tool"):
            return current
        parent = node.get("parent")
        if not parent:
            return current
        current = str(parent)
    return current


def _tool_role_result_blocks(
    blocks: list[ParsedContentBlock],
    *,
    tool_id: str | None,
    status: object,
    aggregate_result: object,
    content_type: str,
    text: str,
) -> list[ParsedContentBlock]:
    """Lower a ``author.role == "tool"`` node to exactly one TOOL_RESULT block.

    The role is the export's structural statement that this node IS a tool's
    answer; the content type only says how the answer was carried. Dispatching
    on content type alone left browsing retrieval (``tether_quote``,
    ``tether_browsing_display``, ``sonic_webpage``) as DOCUMENT and the plain
    ``text``/``multimodal_text``/``code`` answers as TEXT/TOOL_USE, so their
    calling ``tool_use`` stayed permanently ``no_result`` and the node's own
    status evidence was unreachable from the actions relation.

    The carrier keeps its text, media type and web constructs; only its type,
    tool linkage and outcome fields change.
    """
    if any(block.type is BlockType.TOOL_RESULT for block in blocks):
        return blocks
    is_error, outcome_unknown = _node_status_outcome(status, aggregate_result)
    metadata: dict[str, object] = {"content_type": content_type}
    for index, block in enumerate(blocks):
        if block.type not in _TOOL_RESULT_CARRIER_TYPES:
            continue
        if block.metadata and block.metadata.get("admission_disposition"):
            continue
        blocks[index] = block.model_copy(
            update={
                "type": BlockType.TOOL_RESULT,
                "tool_id": tool_id,
                # A result has no invocation input; the ``content_type ==
                # "code"`` branch synthesizes one for the call side only.
                "tool_input": None,
                "is_error": is_error,
                "outcome_unknown_reason": outcome_unknown,
                "metadata": {**(block.metadata or {}), **metadata},
            }
        )
        return blocks
    blocks.insert(
        0,
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            text=text or None,
            tool_id=tool_id,
            metadata=metadata,
            is_error=is_error,
            outcome_unknown_reason=outcome_unknown,
        ),
    )
    return blocks


# ChatGPT field disposition register (bd polylogue-vnucj). Every
# ``message``, ``message.metadata``, ``author`` and conversation-level key
# this parser sees is either read into a named destination or excluded here
# with its reason; nothing is left in the unexamined third state. Counts are
# measured over the three acquired exports -- 2025-10 (2,051 conversations /
# 75,480 messages), 2026-04 (2,403 / 109,657) and 2026-07 (2,472 / 72,981).
#
# READ -- message.metadata
#   finish_details          -> messages.stop_reason (_CHATGPT_FINISH_STOP_REASONS)
#   default_model_slug      -> messages.model_name, after model_slug
#   command, args           -> TOOL_USE tool_name / tool_input
#   reasoning_title         -> THINKING block metadata -> chatgpt_block_metadata
#   dalle                   -> IMAGE block metadata -> chatgpt_block_metadata
#   ada_visualizations      -> one attachment per file_id
#   targeted_reply          -> chatgpt_targeted_reply
#   is_visually_hidden_from_conversation -> chatgpt_message_delivery
#   jit_plugin_data         -> chatgpt_jit_plugin_data
# READ -- message
#   weight, channel         -> chatgpt_message_delivery
#   author.metadata.real_author -> messages.sender_name, after author.name
# READ -- conversation
#   default_model_slug      -> sessions.models_used
#   a non-project gizmo token -> chatgpt_custom_gpt
#   the settings keys above -> chatgpt_conversation_settings
#
# EXCLUDED, with reason:
#   metadata.request_id
#       The provider's correlation token for one server-side run
#       (``wfr_019d68e3...``, repeated across every message of that run).
#       It names nothing inside the archive -- no tier carries a ChatGPT
#       request id to join against -- and the grouping it expresses is the
#       conversation tree the archive already stores.
#   metadata.search_display_string
#       One constant render string in every occurrence: "Searching the
#       web..." (1,524 of 1,524 in 2025-10, 1,987 of 1,987 in 2026-04).
#       The queries it labels are already read (``search_queries``) and the
#       results already projected (``search_result_groups``).
#   metadata.initial_text / metadata.finished_text
#       The status line the UI showed before and after a reasoning run:
#       "Reasoning"/"Thinking", then "Reasoned for 4 seconds". The elapsed
#       time is the same measurement ``_extract_generation_timings`` reads
#       into ``duration_ms`` and the ``generation_lifecycle`` event, in
#       prose; the label restates that the block is THINKING.
#   message.update_time
#       Null on 102,886 of 109,657 messages (2026-04). Where it is stated it
#       times an edit whose before-state the export does not carry, and
#       ``messages`` models no per-message revision axis, so storing it would
#       assert a history the archive cannot show.
#   author.metadata.sonicberry_model_id / .source
#       Provider-internal routing labels for the web tool (181 of 109,657 in
#       2026-04, 82 of 75,480 in 2025-10). They name neither content, a
#       source, nor an outcome -- only which internal variant served a call.
#   author.metadata.is_system_initiated_conversation
#       2 occurrences in 109,657. Restates for one message what the
#       conversation's own first turn already shows.
#   conversation.memory_scope
#       An account setting the export stamps onto every conversation:
#       "global_enabled" on 2,394 of 2,421 conversations (2026-04). It names
#       which memory store the account had enabled, not anything this
#       conversation did.
#   conversation.owner
#       Null in every conversation of every acquired export, and provider
#       account identity is out of scope for session content evidence --
#       the same decision ``claude/common.py`` records for claude.ai
#       ``account``.
#
# NOT PRESENT in the 2026-07 export shape, and therefore unreadable from it:
#   node.children; message.status / .recipient / .weight / .channel /
#   .end_turn / .update_time; author.metadata. That export also carries no
#   ``code``, ``execution_output``, ``computer_output``, ``tether_*``,
#   ``system_error`` or ``citable_code_output`` node at all -- 0 of 72,981
#   messages, against 41,697 in 2026-04 -- so it states no tool episode whose
#   outcome could be derived. ``_node_status_outcome`` answers NOT_REPORTED
#   there, which is the whole of what the record supports. ``branch_index``
#   survives through ``_sibling_ordinals``.
def extract_messages_from_mapping(
    mapping: Mapping[str, object],
    current_node: str | None = None,
    *,
    admission: AdmissionLedger | None = None,
    preserve_empty_messages: bool = False,
    default_model_slug: str | None = None,
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    entries: list[tuple[float | None, int, str, ParsedMessage]] = []
    attachments: list[ParsedAttachment] = []
    if admission is not None:
        admission.expect(
            AdmissionUnit.MESSAGE,
            sum(1 for node in mapping.values() if isinstance(node, dict) and isinstance(node.get("message"), dict)),
        )
    message_ordinal = 0
    sibling_ordinals = _sibling_ordinals(mapping)
    active_path_ids = _active_path_node_ids(mapping, current_node)
    active_path_id_set = set(active_path_ids)
    emitted_by_node_id: dict[str, str] = {}
    for idx, node_id in enumerate(mapping.keys(), start=1):
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        current_message_ordinal = message_ordinal
        message_ordinal += 1
        content = msg.get("content")
        if not isinstance(content, dict):
            if admission is not None:
                admission.unknown(
                    AdmissionUnit.MESSAGE,
                    current_message_ordinal,
                    node_id,
                    AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                )
            continue
        parts = content.get("parts") or []
        raw_text = _extract_content_text(content)
        text = _strip_citation_markers(raw_text)
        # Role is required - skip messages without one
        author = msg.get("author")
        raw_role = author.get("role") if isinstance(author, dict) else None
        if not raw_role or not isinstance(raw_role, str):
            if admission is not None:
                admission.refusal(
                    AdmissionUnit.MESSAGE,
                    current_message_ordinal,
                    node_id,
                    AdmissionRefusalReason.INVALID_ROLE,
                )
            continue
        role = Role.normalize(str(raw_role))
        timestamp = msg.get("create_time")
        msg_id = str(msg.get("id") or node.get("id") or "")

        # Extract parent message reference and calculate branch index
        parent_id = node.get("parent")
        parent_message_provider_id = str(parent_id) if parent_id else None
        tool_result_owner_id = (
            _owning_tool_call_id(mapping, parent_message_provider_id)
            if role is Role.TOOL
            else parent_message_provider_id
        )
        branch_index = 0

        # The parent's ``children`` array states sibling order; where the
        # export omits it, the node's arrival ordinal among the siblings
        # naming the same parent carries the same sequence
        # (``_sibling_ordinals``).
        if parent_message_provider_id:
            branch_index = sibling_ordinals.get(node_id, 0)
            parent_node = mapping.get(str(parent_id))
            if isinstance(parent_node, dict):
                children = parent_node.get("children")
                if isinstance(children, list):
                    current_node_id = node.get("id")
                    if current_node_id in children:
                        branch_index = children.index(current_node_id)

        # Where this message's own attachments begin, so an asset named both
        # by a metadata row and by a content part collapses to one row.
        message_attachment_start = len(attachments)

        # Extract attachments from message metadata
        raw_msg_metadata = msg.get("metadata")
        msg_metadata: Mapping[str, object] = raw_msg_metadata if isinstance(raw_msg_metadata, Mapping) else {}
        msg_attachments = msg_metadata.get("attachments") or []
        if isinstance(msg_attachments, list):
            for attach in msg_attachments:
                attachment = attachment_from_meta(attach, str(msg_id), role=role)
                if attachment is not None:
                    attachments.append(attachment)

        # Code-interpreter chart and table deliverables
        # (``{"type": "table", "file_id": "file-...", "title": ...}``).
        # Each is a real exported file with its own id; ``attachments``
        # never lists them, so without this the archive holds the
        # analysis prose and no record that the artifact it describes
        # exists. ``attachment_kind`` keeps them distinguishable from
        # operator uploads.
        raw_visualizations = msg_metadata.get("ada_visualizations")
        if isinstance(raw_visualizations, list):
            for visualization in raw_visualizations:
                if not isinstance(visualization, Mapping):
                    continue
                file_id = _string_value(visualization, "file_id")
                if file_id is None:
                    continue
                attachments.append(
                    ParsedAttachment(
                        provider_attachment_id=file_id,
                        provider_file_id=file_id,
                        message_provider_id=str(msg_id),
                        name=_string_value(visualization, "title"),
                        attachment_kind="ada_visualization",
                        direction="model_output",
                        producer_ref=f"message:{msg_id}",
                    )
                )

        # Assistant-generated downloadable files (#sandbox links). Code
        # Interpreter deliverables surface only as `sandbox:/mnt/data/...`
        # links inside assistant prose; the export/capture carries no bytes
        # and no metadata attachment row for them, and the links expire with
        # the sandbox container. Record each as an unfetchable attachment so
        # the archive knows the file existed, its name, and which message
        # produced it. attachment_kind="sandbox_file" keeps every acquisition
        # path away from it (there is nothing local to fetch).
        if role is Role.ASSISTANT and text:
            for sandbox_path in _sandbox_file_paths(text):
                attachments.append(
                    ParsedAttachment(
                        provider_attachment_id=f"sandbox:{msg_id}:{sandbox_path}",
                        message_provider_id=str(msg_id),
                        name=sandbox_path.rsplit("/", 1)[-1] or None,
                        attachment_kind="sandbox_file",
                        source_url=f"sandbox:{sandbox_path}",
                        direction="model_output",
                        producer_ref=f"message:{msg_id}",
                    )
                )

        model_slug: object = None
        model_effort: str | None = None
        duration_raw: object = None
        stop_reason: str | None = None
        tool_command: str | None = None
        tool_args: object = None
        reasoning_render: dict[str, object] = {}
        dalle_provenance: Mapping[str, object] | None = None

        # Extract message-level metadata from typed fields
        if isinstance(msg_metadata, Mapping):
            # ``default_model_slug`` names the model that answered whenever
            # the provider did not stamp a per-message ``model_slug``; the
            # conversation-level default is the last fallback.
            model_slug = msg_metadata.get("model_slug") or msg_metadata.get("default_model_slug")
            model_effort = _string_value(
                msg_metadata,
                "thinking_effort",
                "reasoning_effort",
                "model_effort",
                "modelEffort",
            )
            duration_raw = msg_metadata.get("durationMs")
            if duration_raw is None:
                duration_raw = msg_metadata.get("duration_ms")
            stop_reason = _stop_reason_from_finish_details(msg_metadata.get("finish_details"))
            tool_command = _string_value(msg_metadata, "command")
            tool_args = msg_metadata.get("args")
            # The title the provider displayed for this reasoning step
            # ("Extracting lines from XML file"). ``blocks`` has no column
            # for it, so it rides the THINKING block's metadata into
            # ``chatgpt_block_metadata``.
            if (reasoning_title := _string_value(msg_metadata, "reasoning_title")) is not None:
                reasoning_render["reasoning_title"] = reasoning_title
            if isinstance(dalle_raw := msg_metadata.get("dalle"), Mapping) and dalle_raw:
                dalle_provenance = dalle_raw
        model_name = str(model_slug) if isinstance(model_slug, str) and model_slug else None
        if model_name is None and role is Role.ASSISTANT:
            model_name = default_model_slug
        duration_ms = _non_negative_int(duration_raw)

        # A non-"all" recipient marks a tool invocation (e.g. the web-search/
        # browsing tool). Computed here (rather than where ParsedMessage is
        # built below) so the content-block builder can use it to recognize
        # a JSON-encoded tool-call payload instead of storing it as raw text.
        recipient_val = msg.get("recipient")
        recipient = (
            recipient_val if isinstance(recipient_val, str) and recipient_val and recipient_val != "all" else None
        )
        # ``metadata.command`` names the tool a turn addressed and
        # ``metadata.args`` carries its parameters. Both are stated
        # independently of the top-level ``recipient``, which the reduced
        # export shape omits entirely -- without them such an export has no
        # tool-call signal left to read.
        tool_target = recipient or tool_command
        tool_call_input: Mapping[str, object] | None = None
        if tool_target is not None and text:
            try:
                parsed_tool_json = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed_tool_json = None
            if isinstance(parsed_tool_json, dict):
                tool_call_input = parsed_tool_json
        if tool_call_input is None and tool_target is not None and tool_args not in (None, [], {}, ""):
            tool_call_input = {"args": tool_args}

        # Build structured content blocks
        content_blocks: list[ParsedContentBlock] = []
        forced_message_type: MessageType | None = None
        content_type = content.get("content_type", "text")
        # ``metadata["language"]`` is the sole input to ``blocks.language``
        # (``storage/sqlite/archive_tiers/write.py:_block_language``). A
        # ``code`` content states its language on every record and reaches a
        # TOOL_USE block through either of the next two branches -- a
        # recipient-addressed call whose code happens to parse as JSON takes
        # the first -- so the language is carried here, once, for both
        # (polylogue-ui3q4).
        tool_use_metadata: dict[str, object] = {"content_type": content_type}
        declared_language = _string_value(content, "language")
        if declared_language:
            tool_use_metadata["language"] = declared_language
        if tool_call_input is not None:
            # Recipient-addressed tool call whose content is a JSON payload
            # (e.g. ChatGPT's web-search tool: {"search_query": [...]}) --
            # a proper TOOL_USE block instead of raw JSON as BlockType.TEXT
            # (#e2yk). The reader already folds tool_use blocks by default.
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name=tool_target,
                    # tool_id = this node's own id, so the mapping-tree child
                    # node that carries the result (parent == this id) can
                    # link back via the same id below (polylogue-ah21: these
                    # were previously always NULL, leaving every ChatGPT
                    # tool_use/tool_result block pair unjoined).
                    tool_id=str(msg_id),
                    tool_input=tool_call_input,
                    metadata=dict(tool_use_metadata),
                )
            )
        elif content_type in ("thoughts", "reasoning_recap"):
            # ChatGPT thinking/reasoning blocks
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=text,
                    metadata={"content_type": content_type, **reasoning_render},
                )
            )
        elif content_type == "code":
            # Code-interpreter input — top-level text, no parts (#1744).
            # bd polylogue-4fm3: this used to emit BlockType.CODE with no
            # tool_id, so every code-interpreter call contributed zero rows
            # to `action_pairs` (which only joins block_type='tool_use') --
            # its paired execution_output below (a real TOOL_RESULT) was
            # left permanently unpaired, producing a measured ~4.6:1
            # tool_result:tool_use skew on browser-captured chatgpt sessions.
            # Classified as TOOL_USE instead, mirroring the recipient-
            # addressed JSON tool-call branch above: tool_name = recipient
            # (e.g. "python", "container.exec") when the provider addressed a
            # tool for this call (polylogue-grub), falling back to
            # "code_interpreter" when it didn't; tool_id = this node's own
            # id, so the execution_output node (whose mapping-tree parent is
            # this call) can join back via the same id below -- the same
            # convention polylogue-ah21 established for the browser-capture
            # typed-blocks path. `text` is kept (not just tool_input) so the
            # raw source keeps rendering as before.
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    text=text,
                    tool_name=tool_target or "code_interpreter",
                    tool_id=str(msg_id),
                    tool_input={"code": text},
                    metadata=dict(tool_use_metadata),
                )
            )
        elif content_type == "execution_output":
            # Code-interpreter output — top-level text, no parts (#1744).
            # tool_id = the calling node's id (mapping-tree `parent`), the
            # same identifier the code-interpreter TOOL_USE node above now
            # stamps onto itself (bd polylogue-4fm3) -- both sides of the
            # pair carry a shared tool_id and the `actions` view can join
            # them.
            #
            # The outcome comes from the run record and the node's status --
            # see ``_node_status_outcome``; this export carries no exit code.
            execution_is_error, execution_unknown_reason = _node_status_outcome(
                msg.get("status"), msg_metadata.get("aggregate_result")
            )
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    text=text,
                    tool_id=tool_result_owner_id,
                    metadata={"content_type": content_type},
                    is_error=execution_is_error,
                    outcome_unknown_reason=execution_unknown_reason,
                )
            )
        elif content_type == "computer_output":
            # Computer-use tool result (April-era browsing/desktop-agent
            # layer, polylogue-xofj: 8,192 measured) -- a screenshot + DOM/
            # browser-state snapshot returned by the computer.do tool loop.
            # tool_id = the calling node's id (mapping-tree `parent`), the
            # same convention execution_output/code use above (polylogue-
            # 4fm3/polylogue-grub) so the actions view can join the pair.
            # is_error reads the same structural evidence execution_output
            # reads.
            computer_is_error, computer_unknown_reason = _node_status_outcome(
                msg.get("status"), msg_metadata.get("aggregate_result")
            )
            state = content.get("state")
            state_url = _string_value(state, "url") if isinstance(state, Mapping) else None
            state_title = _string_value(state, "title") if isinstance(state, Mapping) else None
            summary = " — ".join(part for part in (state_title, state_url) if part)
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    text=summary or None,
                    tool_id=tool_result_owner_id,
                    metadata={"content_type": content_type},
                    is_error=computer_is_error,
                    outcome_unknown_reason=computer_unknown_reason,
                )
            )
            # The screenshot is this tool result's payload -- an
            # `image_asset_pointer` record on every measured computer_output
            # node -- and needs both carriers: the IMAGE block puts it in the
            # content tree (its `asset_pointer` metadata reaches storage as a
            # `chatgpt_block_metadata` event), and the attachment row is the
            # only acquisition identity an asset has. `assembly_chatgpt.py`
            # binds acquired asset bytes to attachments by this id.
            screenshot = content.get("screenshot")
            if isinstance(screenshot, Mapping) and (screenshot_pointer := _string_value(screenshot, "asset_pointer")):
                content_blocks.append(
                    ParsedContentBlock(
                        type=BlockType.IMAGE,
                        metadata=_asset_pointer_block_metadata(screenshot, screenshot_pointer),
                    )
                )
                _append_asset_attachment(
                    attachments,
                    screenshot,
                    pointer=screenshot_pointer,
                    message_provider_id=str(msg_id),
                    attachment_kind="computer_screenshot",
                    direction="model_output",
                    producer_ref=f"message:{msg_id}",
                    dedupe_from=message_attachment_start,
                )
        elif content_type in ("tether_quote", "tether_browsing_display", "sonic_webpage"):
            # Browsing/web-search retrieval (April-era layer, polylogue-xofj):
            # tether_quote (1,178 measured) is a quoted document/file excerpt
            # from the myfiles_browser tool (top-level `text` + `domain`);
            # tether_browsing_display (1,399 measured) is a page-listing from
            # the browser tool (`result` string); sonic_webpage (30 measured)
            # is a single fetched web.search page (`text`/`snippet` +
            # `domain` + `ref_id`). All three are retrieved-source evidence,
            # not free text -- projected as a SEARCH_RESULT web construct
            # (polylogue-zocm: SEARCH_RESULT means "retrieved", distinct from
            # CONTENT_REFERENCE's "cited") carried on a DOCUMENT block,
            # mirroring the audio_transcription/audio_asset_pointer DOCUMENT+
            # web_constructs idiom below. On a ``role: tool`` node this block
            # is the browsing tool's answer, so ``_tool_role_result_blocks``
            # re-types it to TOOL_RESULT and keeps the construct.
            #
            # The record's own ``url`` is its address and its own ``title`` is
            # its name; ``domain`` is only the label to fall back to when the
            # shape carries no title (tether_browsing_display carries
            # neither). The citation path in this file
            # (``_construct_from_reference``) already reads both, so anything
            # narrower makes URL conservation depend on which content type
            # delivered the source.
            domain = _string_value(content, "domain")
            construct_text = text or _string_value(content, "snippet") or None
            source_id = _string_value(content, "tether_id", "ref_id")
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.DOCUMENT,
                    text=text or None,
                    web_constructs=[
                        ParsedWebConstruct(
                            construct_type=WebConstructType.SEARCH_RESULT,
                            provider_key=content_type,
                            title=_string_value(content, "title") or domain,
                            url=_string_value(content, "url"),
                            text=construct_text,
                            source_id=source_id,
                        )
                    ],
                )
            )
        elif content_type == "system_error":
            # Structural tool/browsing failure (April-era layer, polylogue-
            # xofj: 177 measured) -- `content_type` itself IS the provider's
            # error signal, never guessed from prose. tool_id = the calling
            # node's id (mapping-tree `parent`), the execution_output/code
            # convention, so the pair still joins even though the error-
            # report message's own `status` ("finished_successfully" -- the
            # error report itself was delivered fine) says nothing about the
            # underlying failure.
            error_name = _string_value(content, "name")
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    text=text,
                    tool_id=tool_result_owner_id,
                    metadata={"content_type": content_type, **({"error_name": error_name} if error_name else {})},
                    is_error=True,
                )
            )
        elif content_type == "citable_code_output":
            # Connector-sourced retrieval result (April-era layer, polylogue-
            # xofj: 8 measured) -- e.g. api_tool.call_tool reading a Gmail/
            # Drive connector document. Effectively a code/tool result
            # (`output_str`, folded into `_extract_content_text`'s fallback
            # above) plus a citation anchor identifying the connector
            # document it was read from -- "a code result with citation
            # anchors". tool_id/is_error follow the same execution_output
            # convention as the branches above.
            citable_is_error, citable_unknown_reason = _node_status_outcome(
                msg.get("status"), msg_metadata.get("aggregate_result")
            )
            cite_metadata = content.get("metadata")
            cite_constructs: list[ParsedWebConstruct] = []
            if isinstance(cite_metadata, Mapping):
                display_title = _string_value(cite_metadata, "display_title")
                display_url = _string_value(cite_metadata, "display_url")
                connector_id = _string_value(cite_metadata, "connector_id")
                connector_source = _string_value(cite_metadata, "connector_source")
                if display_title or display_url or connector_id:
                    cite_constructs.append(
                        ParsedWebConstruct(
                            construct_type=WebConstructType.CONTENT_REFERENCE,
                            provider_key=content_type,
                            title=display_title,
                            url=display_url,
                            source_id=connector_id,
                            text=connector_source,
                        )
                    )
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    text=text,
                    tool_id=tool_result_owner_id,
                    metadata={"content_type": content_type},
                    is_error=citable_is_error,
                    outcome_unknown_reason=citable_unknown_reason,
                    web_constructs=cite_constructs,
                )
            )
        elif content_type in ("user_editable_context", "model_editable_context"):
            # System-injected conversation context (#runtime evidence): custom
            # instructions / user profile (`user_editable_context`) and the
            # ChatGPT memory payload (`model_set_context`). These carry no
            # `parts`, so without this branch the messages are dropped and the
            # archive loses what context the provider injected. Empty payloads
            # (e.g. memory feature on but no memories) still drop.
            context_fields = (
                ("user_profile", "user_instructions")
                if content_type == "user_editable_context"
                else ("model_set_context",)
            )
            context_texts = [
                value for key in context_fields if isinstance(value := content.get(key), str) and value.strip()
            ]
            if context_texts:
                text = "\n\n".join(context_texts)
                forced_message_type = MessageType.CONTEXT
                content_blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=text,
                        metadata={"content_type": content_type},
                    )
                )
        elif parts:
            for part in parts:
                if isinstance(part, str) and part:
                    content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=_strip_citation_markers(part)))
                elif isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
                    image_pointer = str(part.get("asset_pointer", ""))
                    image_metadata = _asset_pointer_block_metadata(part, image_pointer)
                    # ``metadata.dalle`` states what this image was made from
                    # -- the generation and file it transformed. That edge
                    # exists nowhere else: the derived image's own asset
                    # pointer says nothing about its original.
                    if dalle_provenance is not None:
                        image_metadata["dalle"] = dict(dalle_provenance)
                    content_blocks.append(
                        ParsedContentBlock(
                            type=BlockType.IMAGE,
                            metadata=image_metadata,
                        )
                    )
                    if image_pointer:
                        image_direction, image_producer = derive_attachment_provenance(role, str(msg_id))
                        _append_asset_attachment(
                            attachments,
                            part,
                            pointer=image_pointer,
                            message_provider_id=str(msg_id),
                            attachment_kind="image_asset",
                            direction=image_direction,
                            producer_ref=image_producer,
                            dedupe_from=message_attachment_start,
                        )
                elif isinstance(part, dict) and part.get("content_type") in {
                    "audio_asset_pointer",
                    "audio_transcription",
                    "real_time_user_audio_video_asset_pointer",
                }:
                    part_text = part.get("text")
                    content_type = str(part.get("content_type"))
                    content_blocks.append(
                        ParsedContentBlock(
                            type=BlockType.DOCUMENT,
                            text=part_text if isinstance(part_text, str) and part_text else None,
                            media_type=_string_value(part, "mime_type", "media_type"),
                            web_constructs=[
                                ParsedWebConstruct(
                                    construct_type=(
                                        WebConstructType.AUDIO_TRANSCRIPTION
                                        if content_type == "audio_transcription"
                                        else WebConstructType.AUDIO_ASSET
                                    ),
                                    provider_key=content_type,
                                    text=part_text if isinstance(part_text, str) and part_text else None,
                                    asset_pointer=_string_value(part, "asset_pointer"),
                                    mime_type=_string_value(part, "mime_type", "media_type"),
                                )
                            ],
                        )
                    )
                elif isinstance(part, dict):
                    content_blocks.append(
                        typed_unknown_block(
                            part,
                            wire_type=_string_value(part, "content_type", "type") or str(content_type),
                        )
                    )

        if not content_blocks and content_type not in {
            "text",
            "thoughts",
            "reasoning_recap",
            "code",
            "execution_output",
            "computer_output",
            "tether_quote",
            "tether_browsing_display",
            "sonic_webpage",
            "system_error",
            "citable_code_output",
            "user_editable_context",
            "model_editable_context",
        }:
            content_blocks.append(typed_unknown_block(content, wire_type=str(content_type)))

        if role is Role.TOOL and content_blocks:
            content_blocks = _tool_role_result_blocks(
                content_blocks,
                tool_id=tool_result_owner_id,
                status=msg.get("status"),
                aggregate_result=msg_metadata.get("aggregate_result"),
                # Read the node's own content type: the ``parts`` loop above
                # rebinds ``content_type`` to an audio part's type.
                content_type=str(content.get("content_type", "text")),
                text=text,
            )

        web_constructs = _constructs_from_chatgpt_metadata(msg_metadata)
        # Inline citation anchors are stripped from stored text (invisible
        # glyphs), but their reference tokens — turn/file pointers and line
        # ranges like "filecite turn3file14 L180-L293" — carry source-location
        # detail the metadata rows sometimes lack (line_range is often null).
        # Preserve each span as a construct anchored at its original-text
        # offset, the same coordinate system the citation rows' start_ix/
        # end_ix use.
        web_constructs.extend(
            ParsedWebConstruct(
                construct_type=WebConstructType.CONTENT_REFERENCE,
                provider_key="inline_citation_marker",
                text=" ".join(token for token in marker.group(1).split("\ue202") if token),
                start_index=marker.start(),
                end_index=marker.end(),
            )
            for marker in _CITATION_MARKER_SPAN_RE.finditer(raw_text)
        )
        if web_constructs:
            if not content_blocks:
                content_blocks.append(ParsedContentBlock(type=BlockType.TEXT))
            first_block = content_blocks[0]
            first_block.web_constructs.extend(web_constructs)
        if admission is not None:
            part_offset = admission.next_ordinal(AdmissionUnit.PART)
            admission.expect(AdmissionUnit.PART, len(parts) if isinstance(parts, list) else 0)
            for part_ordinal, part in enumerate(parts if isinstance(parts, list) else []):
                if isinstance(part, str) or (
                    isinstance(part, dict)
                    and part.get("content_type")
                    in {
                        "image_asset_pointer",
                        "audio_asset_pointer",
                        "audio_transcription",
                        "real_time_user_audio_video_asset_pointer",
                    }
                ):
                    admission.materialized(
                        AdmissionUnit.PART,
                        part_offset + part_ordinal,
                        "text" if isinstance(part, str) else str(part.get("content_type")),
                    )
                else:
                    admission.unknown(
                        AdmissionUnit.PART,
                        part_offset + part_ordinal,
                        str(content_type),
                        AdmissionUnknownReason.UNRECOGNIZED_TYPE,
                    )
            admission.expect(AdmissionUnit.BLOCK, len(content_blocks))
            for block in content_blocks:
                block_ordinal = admission.next_ordinal(AdmissionUnit.BLOCK)
                if block.metadata and block.metadata.get("admission_disposition") == "typed_unknown":
                    admission.unknown(
                        AdmissionUnit.BLOCK, block_ordinal, str(block.metadata.get("wire_type") or "unknown")
                    )
                else:
                    admission.materialized(AdmissionUnit.BLOCK, block_ordinal, block.type.value)
        if not text and not content_blocks and not preserve_empty_messages:
            if admission is not None:
                admission.unknown(
                    AdmissionUnit.MESSAGE,
                    current_message_ordinal,
                    node_id,
                    AdmissionUnknownReason.EMPTY_CONTENT,
                )
            continue

        status_val = msg.get("status")
        end_turn_val = msg.get("end_turn")
        user_context_val = msg_metadata.get("user_context_message_data")
        message_type = forced_message_type or classify_text_message_type(text) or MessageType.MESSAGE
        real_author = _real_author(author)
        material_origin = human_authored_override(
            role,
            message_type,
            classify_material_origin(
                role=role,
                message_type=message_type,
                text=text,
                block_types=tuple(block.type for block in content_blocks),
            ),
        )
        # ``real_author`` is the wire stating who actually produced this
        # message's content. A tool author overrides the envelope: 397
        # measured ``role: assistant`` nodes carry ``tool:web``, and reading
        # them as model output is what makes assistant-word accounting
        # over-count.
        if real_author is not None and real_author.startswith(_CHATGPT_TOOL_AUTHOR_PREFIX):
            material_origin = MaterialOrigin.TOOL_RESULT
        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            blocks=content_blocks,
            message_type=message_type,
            material_origin=material_origin,
            parent_message_provider_id=parent_message_provider_id,
            position=idx - 1,
            branch_index=branch_index,
            variant_index=branch_index,
            is_active_path=node_id in active_path_id_set if active_path_ids else None,
            model_name=model_name,
            model_effort=model_effort,
            duration_ms=duration_ms,
            sender_name=_author_display_name(author),
            recipient=recipient,
            delivery_status=status_val if isinstance(status_val, str) and status_val else None,
            end_turn=end_turn_val if isinstance(end_turn_val, bool) else None,
            user_context_text=(
                _string_value(user_context_val, "about_user_message", "text", "content")
                if isinstance(user_context_val, Mapping)
                else None
            ),
            stop_reason=stop_reason,
        )
        emitted_by_node_id[node_id] = parsed.provider_message_id
        entries.append((_coerce_float(timestamp), idx, node_id, parsed))
        if admission is not None:
            admission.materialized(AdmissionUnit.MESSAGE, current_message_ordinal, node_id)
    if any(value is not None for value, _, _, _ in entries):
        # Use explicit None check instead of `or` to handle zero/negative timestamps correctly
        entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0, item[1]))
    messages = [entry[3] for entry in entries]
    emitted_message_ids = {message.provider_message_id for message in messages}
    messages = [
        message.model_copy(
            update={
                "parent_message_provider_id": (
                    emitted_by_node_id.get(
                        message.parent_message_provider_id,
                        message.parent_message_provider_id
                        if message.parent_message_provider_id in emitted_message_ids
                        else None,
                    )
                )
            }
        )
        if message.parent_message_provider_id is not None
        else message
        for message in messages
    ]
    active_leaf_node_id = next(
        (node_id for node_id in reversed(active_path_ids) if node_id in emitted_by_node_id),
        None,
    )
    if active_leaf_node_id is not None:
        active_leaf_position = next(entry[3].position for entry in entries if entry[2] == active_leaf_node_id)
        messages = [
            message.model_copy(update={"is_active_leaf": message.position == active_leaf_position})
            for message in messages
        ]
    return (messages, attachments)


def _mapping_nodes_are_valid(mapping: Mapping[str, object]) -> bool:
    """Pydantic-validate every ``mapping`` entry against the typed node shape.

    Requires the full ``ChatGPTNode`` shape, including a real ``id`` field
    (and, transitively, a real ``message.id`` when a message is present).
    This is the whole-document check's node validator -- see
    ``_mapping_node_shape_is_plausible`` for the lighter fragment-level
    check.
    """
    for node in mapping.values():
        if not isinstance(node, dict):
            return False
        try:
            ChatGPTNode.model_validate(node)
        except ValidationError:
            return False
    return True


def _mapping_node_shape_is_plausible(mapping: Mapping[str, object]) -> bool:
    """Loosely validate mapping-node shape for fragment-level detection.

    A per-record fragment (one JSONL line, or one already-lowered record
    passed without the surrounding document -- see ``looks_like_fragment``)
    routinely omits the ``id``/``message.id`` fields ``ChatGPTNode`` treats
    as mandatory: those fields duplicate identity the caller already knows
    from the mapping key / conversation context, so real per-record
    fragments do not always repeat them. Full ``ChatGPTNode`` Pydantic
    validation is therefore too strict for this tier. This instead checks
    the shape ``ChatGPTNode``/``ChatGPTMessage`` actually constrain
    structurally: every node is a dict, and when a node carries a
    ``message`` it is a dict with an ``author`` dict (the one field every
    real ChatGPT message node has, fragment or not). This still rejects an
    arbitrary dict-with-a-``mapping``-key payload from an unrelated
    provider (empty nodes, non-dict nodes, malformed ``message``/``author``
    shapes), which is what the original bare ``isinstance(mapping, dict)``
    check silently accepted.
    """
    for node in mapping.values():
        if not isinstance(node, dict):
            return False
        message = node.get("message")
        if message is None:
            continue
        if not isinstance(message, dict) or not isinstance(message.get("author"), dict):
            return False
    return True


def looks_like_fragment(payload: object) -> bool:
    """Detect a ChatGPT conversation-mapping *fragment*.

    Individual-record detection (e.g. per-line JSONL sniffing in
    ``sources/emitter.py``, or a single already-lowered record in
    ``dispatch._detect_provider_from_record``/fallback lowering) sees only
    the divergent slice of a conversation a caller chose to hand over one
    record at a time -- it legitimately lacks document-level identity
    fields (``current_node``/``create_time``/``conversation_id``/``id``,
    and even per-node/per-message ``id``) that only exist once a full
    exported document is assembled. This checks the one structural signal
    that *is* present per-record: a non-empty ``mapping`` dict whose nodes
    have a plausible ChatGPT node/message shape (see
    ``_mapping_node_shape_is_plausible``). See ``looks_like`` for the
    stricter whole-document check (polylogue-t0ta).
    """
    if not isinstance(payload, dict):
        return False
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return False
    return _mapping_node_shape_is_plausible(mapping)


def looks_like(payload: object) -> bool:
    """Detect the ChatGPT conversation-export shape (whole document).

    ChatGPT's export format is externally versioned by OpenAI, outside this
    repo's control, so a bare "has a mapping dict-key" check is the loosest,
    highest format-drift-risk detector in dispatch: it silently accepts any
    payload that happens to carry a "mapping" key, including malformed or
    entirely unrelated shapes. This tightens detection to the export's
    stable structural fields -- confirmed present across every real fixture
    and corpus representative in this repo (native/browser-capture/regression
    fixtures, schema catalog representatives) -- plus Pydantic-validated node
    shape for every entry in ``mapping``, mirroring the typed-validation
    pattern already load-bearing for Codex (``codex.looks_like``).

    Use this only where a whole document/list-of-documents is available
    (``dispatch._detect_provider_from_sequence``'s first-record check, and
    direct callers validating an assembled export). For a single record
    that may be an intentionally partial fragment (streamed JSONL lines,
    already-lowered single records), use ``looks_like_fragment`` instead --
    it lacks the document-identity fields this function requires.
    """
    if not isinstance(payload, dict):
        return False
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return False
    if not isinstance(payload.get("current_node"), str):
        return False
    if not isinstance(payload.get("create_time"), (int, float)):
        return False
    if not isinstance(payload.get("conversation_id"), str) and not isinstance(payload.get("id"), str):
        return False
    return _mapping_nodes_are_valid(mapping)


def looks_like_shared_decode(payload: object) -> bool:
    """Detect a ChatGPT shared-page (``chatgpt.com/share/<id>``) stream decode.

    A shared conversation page serves its content through a client React
    Router data stream, not the authenticated export/API ``mapping`` tree
    ``looks_like``/``looks_like_fragment`` require. A local decode of that
    stream (polylogue-4zqh3: the recovery-packet decode of conversation
    ``6a4ac87b-...``, produced by a router-stream parse rather than a DOM
    scrape) flattens the conversation tree into a top-level ``messages``
    list of ``{node_id, parent, children, role, text, ...}`` records with no
    ``mapping`` key anywhere in the document. Before this detector existed,
    such a document matched no provider detector at all: dispatch's generic
    "has a messages list" fallback (``_record_messages``) *would* match it,
    but that fallback also requires a payload-asserted ``id`` field (there
    is none here -- only ``conversation_id``/``shared_conversation_id``), so
    it silently produced zero sessions and the decode never reached the
    archive.
    """
    if not isinstance(payload, dict):
        return False
    if "mapping" in payload:
        return False
    if not isinstance(payload.get("shared_conversation_id"), str) or not payload["shared_conversation_id"]:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict):
        return False
    return isinstance(first.get("node_id"), str) and isinstance(first.get("role"), str)


def _shared_decode_mapping(payload: Mapping[str, object]) -> tuple[dict[str, object], str | None]:
    """Build a ``mapping``-tree-shaped dict from a shared-decode ``messages`` list.

    Reuses ``extract_messages_from_mapping`` (the same node/message-tree
    walk every other ChatGPT shape goes through) instead of duplicating its
    branch-index/attachment/content-type logic for this flattened shape:
    each flat record becomes a native-shaped ``{id, parent, children,
    message: {id, author, create_time, content, metadata, status}}`` node.
    Returns the synthesized mapping plus the leaf node id (the one node
    with no children) to stand in for the native export's ``current_node``,
    since the decode carries no explicit current-node marker.
    """
    messages = payload.get("messages")
    mapping: dict[str, object] = {}
    leaf_node_id: str | None = None
    if not isinstance(messages, list):
        return mapping, leaf_node_id
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        node_id = raw.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        message_id_raw = raw.get("message_id")
        message_id = message_id_raw if isinstance(message_id_raw, str) and message_id_raw else node_id
        role = raw.get("role")
        text = raw.get("text")
        text = text if isinstance(text, str) else ""
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        children_raw = raw.get("children")
        children = [child for child in children_raw if isinstance(child, str)] if isinstance(children_raw, list) else []
        parent = raw.get("parent")
        parent = parent if isinstance(parent, str) else None
        message: dict[str, object] | None = None
        if isinstance(role, str) and role:
            message = {
                "id": message_id,
                "author": {"role": role},
                "create_time": raw.get("create_time"),
                "update_time": raw.get("update_time"),
                "content": {"content_type": "text", "parts": [text] if text else []},
                "metadata": metadata,
                "status": raw.get("status"),
            }
        mapping[node_id] = {"id": node_id, "parent": parent, "children": children, "message": message}
        if not children:
            leaf_node_id = node_id
    return mapping, leaf_node_id


def shared_decode_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the ordinary mapping projection for a validated shared decode.

    Source-side conservation uses the same mapping projection as parsing. A
    caller that needs only the provider document must not reimplement the
    flattened shared-page lowering rules.
    """
    mapping, _leaf_node_id = _shared_decode_mapping(payload)
    return mapping


# polylogue-9x22: ``ParsedContentBlock.metadata`` is never persisted -- the
# ``blocks`` table has no metadata column and the write path only reads a
# ``language`` key back out of it (``storage/sqlite/archive_tiers/write.py:
# _block_language``). ``extract_messages_from_mapping`` above still tags
# TOOL_USE/THINKING/CODE/TOOL_RESULT/context blocks with
# ``metadata={"content_type": ...}`` and IMAGE blocks with
# ``metadata={"asset_pointer": ...}`` as an in-process carrier -- without
# this projection step that disambiguating detail (e.g. "thoughts" vs.
# "reasoning_recap" on a THINKING block) is silently dropped at write time.
# Route it through session_events, same precedent as
# ``claude/common.py``'s ``claude_ai_web_tool_evidence`` and
# ``browser_capture.py``'s ``browser_capture_block_metadata``: one event per
# block carrying non-empty metadata, whole dict verbatim (no fixed key
# vocabulary to prune against here, unlike the Claude AI web-tool case).
def _run_stream_text(aggregate_result: Mapping[str, object]) -> str:
    """Concatenate a code-interpreter run's stdout/stderr stream text.

    Both wire shapes carry it: the slim record under ``messages[]``
    (``message_type: "stream"``) and the full one additionally under
    ``jupyter_messages[]`` (``msg_type: "stream"``), which repeats the same
    bytes. Read the slim list first so the repeat is never concatenated onto
    itself.
    """
    for key, type_key in (("messages", "message_type"), ("jupyter_messages", "msg_type")):
        parts: list[str] = []
        for record in _iter_mapping_items(aggregate_result.get(key)):
            if record.get(type_key) != "stream":
                continue
            content = record.get("content")
            text = _string_value(record, "text") or (
                _string_value(content, "text") if isinstance(content, Mapping) else None
            )
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)
    return ""


def _aggregate_result_events(mapping: Mapping[str, object], emitted_message_ids: set[str]) -> list[ParsedSessionEvent]:
    """Conserve each code-interpreter run's own record.

    The run's output already IS the result node's text, so the stream text is
    stored verbatim only where that duplication does not hold and its length
    is recorded either way -- the same conservation shape Codex's
    ``last_agent_message`` event uses. Everything else here (the run id, its
    clock, the timeout it ran under, the exception class) exists nowhere else
    in the parsed session, and neither does the executed program, which rides
    the ``aggregate_result`` web construct.
    """
    events: list[ParsedSessionEvent] = []
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        metadata = message.get("metadata")
        aggregate_result = metadata.get("aggregate_result") if isinstance(metadata, Mapping) else None
        if not isinstance(aggregate_result, Mapping):
            continue
        message_id = str(message.get("id") or node_id)
        if message_id not in emitted_message_ids:
            continue
        content = message.get("content")
        node_text = _string_value(content, "text") if isinstance(content, Mapping) else None
        stream_text = _run_stream_text(aggregate_result)
        exception = aggregate_result.get("in_kernel_exception")
        exception = exception if isinstance(exception, Mapping) else {}
        system_exception = aggregate_result.get("system_exception")
        payload: dict[str, object] = {"status": _string_value(aggregate_result, "status")}
        for key in ("run_id", "start_time", "end_time", "update_time", "timeout_triggered"):
            value = aggregate_result.get(key)
            if value is not None:
                payload[key] = value
        final_expression_output = aggregate_result.get("final_expression_output")
        if final_expression_output is not None:
            payload["final_expression_output"] = final_expression_output
        if exception:
            payload["in_kernel_exception_name"] = _string_value(exception, "name")
            if exception.get("args") is not None:
                payload["in_kernel_exception_args"] = exception["args"]
        if isinstance(system_exception, Mapping):
            payload["system_exception"] = dict(system_exception)
        if stream_text:
            payload["stream_chars"] = len(stream_text)
            payload["stream_retained_as_message_text"] = stream_text == node_text
            if stream_text != node_text:
                payload["stream_text"] = stream_text
        events.append(
            ParsedSessionEvent(
                event_type="chatgpt_code_interpreter_run",
                timestamp=_string_value(aggregate_result, "end_time", "update_time", "start_time"),
                source_message_provider_id=message_id,
                payload=payload,
            )
        )
    return events


def _message_authorship_events(
    mapping: Mapping[str, object], emitted_message_ids: set[str]
) -> list[ParsedSessionEvent]:
    """Conserve the wire's own statement of what a message is and who wrote it.

    ``channel`` separates a turn's reasoning-adjacent ``commentary`` from the
    ``final`` answer the user was shown -- ChatGPT's analogue of Codex's
    ``phase`` -- and ``author.metadata.real_author`` names the tool that
    actually produced a message rendered through another role's envelope.
    """
    events: list[ParsedSessionEvent] = []
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        channel = _string_value(message, "channel")
        real_author = _real_author(message.get("author"))
        if channel is None and real_author is None:
            continue
        message_id = str(message.get("id") or node_id)
        if message_id not in emitted_message_ids:
            continue
        payload: dict[str, object] = {}
        if channel is not None:
            payload["channel"] = channel
        if real_author is not None:
            payload["real_author"] = real_author
        events.append(
            ParsedSessionEvent(
                event_type="chatgpt_message_authorship",
                timestamp=str(message.get("create_time")) if message.get("create_time") is not None else None,
                source_message_provider_id=message_id,
                payload=payload,
            )
        )
    return events


def _block_metadata_evidence_events(messages: Sequence[ParsedMessage]) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    for message in messages:
        for block_index, block in enumerate(message.blocks):
            if not block.metadata:
                continue
            events.append(
                ParsedSessionEvent(
                    event_type="chatgpt_block_metadata",
                    timestamp=message.timestamp,
                    source_message_provider_id=message.provider_message_id,
                    payload={"block_index": block_index, **dict(block.metadata)},
                )
            )
    return events


def _iter_message_nodes(mapping: Mapping[str, object]) -> list[tuple[str, Mapping[str, object]]]:
    """Every ``(provider_message_id, message)`` pair in mapping order."""
    pairs: list[tuple[str, Mapping[str, object]]] = []
    for node in mapping.values():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        # The same identity ``extract_messages_from_mapping`` mints, so the
        # events these feed bind to the message the parser emitted.
        pairs.append((str(message.get("id") or node.get("id") or ""), message))
    return pairs


def _message_metadata_evidence_events(mapping: Mapping[str, object]) -> list[ParsedSessionEvent]:
    """Message-level ``metadata`` evidence with no column or block to hold it.

    Three separately named facts rather than one metadata bag, so a reader
    filtering on ``event_type`` gets the fact it asked for:

    ``chatgpt_targeted_reply``
        The prose a user quoted when replying to part of an earlier turn.
        Real user words, authored in this conversation and recoverable
        nowhere else in it -- the quoted span is not repeated in the
        message's own text.
    ``chatgpt_message_delivery``
        How the provider delivered this turn: ``weight`` 0 (dropped from the
        model's own context), ``is_visually_hidden_from_conversation`` (never
        shown to the operator), and the ``channel`` it was emitted on
        (``commentary`` for the tool/analysis stream, ``final`` for the
        answer). Absent these a hidden, context-dropped, commentary-channel
        turn reads as conversation the operator saw and the model kept.
    ``chatgpt_jit_plugin_data``
        A just-in-time plugin's call and response payload -- the only
        record of what a plugin was asked and what it answered.
    """
    events: list[ParsedSessionEvent] = []
    for message_id, message in _iter_message_nodes(mapping):
        metadata = message.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        timestamp = message.get("create_time")
        timestamp_text = str(timestamp) if timestamp is not None else None
        targeted_reply = metadata.get("targeted_reply")
        if targeted_reply not in (None, "", [], {}):
            payload: dict[str, object] = {"targeted_reply": targeted_reply}
            if (label := _string_value(metadata, "targeted_reply_label")) is not None:
                payload["targeted_reply_label"] = label
            events.append(
                ParsedSessionEvent(
                    event_type="chatgpt_targeted_reply",
                    timestamp=timestamp_text,
                    source_message_provider_id=message_id,
                    payload=payload,
                )
            )
        delivery: dict[str, object] = {}
        weight = message.get("weight")
        if isinstance(weight, (int, float, Decimal)) and not isinstance(weight, bool) and float(weight) != 1.0:
            delivery["weight"] = float(weight)
        if metadata.get("is_visually_hidden_from_conversation") is True:
            delivery["is_visually_hidden_from_conversation"] = True
        if (channel := _string_value(message, "channel")) is not None:
            delivery["channel"] = channel
        if delivery:
            events.append(
                ParsedSessionEvent(
                    event_type="chatgpt_message_delivery",
                    timestamp=timestamp_text,
                    source_message_provider_id=message_id,
                    payload=delivery,
                )
            )
        jit_plugin_data = metadata.get("jit_plugin_data")
        if isinstance(jit_plugin_data, Mapping) and jit_plugin_data:
            events.append(
                ParsedSessionEvent(
                    event_type="chatgpt_jit_plugin_data",
                    timestamp=timestamp_text,
                    source_message_provider_id=message_id,
                    payload=dict(jit_plugin_data),
                )
            )
    return events


#: Conversation-level keys that state how the operator configured or filed
#: this conversation. Every one is a provider-owned setting with no
#: cross-origin equivalent, so they travel together as one settings event
#: rather than becoming per-provider session columns -- the decision
#: ``sessions.run_settings_json`` recorded and v95 then retired in favour of
#: an event.
_CHATGPT_CONVERSATION_SETTING_KEYS: tuple[str, ...] = (
    "conversation_origin",
    "is_archived",
    "is_starred",
    "is_read_only",
    "is_do_not_remember",
    "is_study_mode",
    "voice",
    "async_status",
    "context_scopes",
    "disabled_tool_ids",
    "plugin_ids",
    "sugar_item_id",
    "gizmo_type",
    "safe_urls",
    "blocked_urls",
    "moderation_results",
)


def _conversation_settings_event(
    payload: Mapping[str, object],
    *,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    """Carry the conversation's own non-empty settings, or nothing."""
    settings = {
        key: value
        for key in _CHATGPT_CONVERSATION_SETTING_KEYS
        if (value := payload.get(key)) not in (None, "", [], {}, False)
    }
    if not settings:
        return None
    return ParsedSessionEvent(
        event_type="chatgpt_conversation_settings",
        timestamp=timestamp,
        payload=settings,
    )


def _custom_gpt_event(
    payload: Mapping[str, object],
    *,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    """Name the custom GPT a conversation ran against, if any.

    ``provider_project_ref`` admits only the ``g-p-`` project token, so a
    bare ``g-<id>`` -- a custom GPT, a different kind of thing from a
    project -- reaches no destination through it. The GPT is what answered
    every turn in the conversation, so its identity is session evidence.
    """
    gizmo_id = _string_value(payload, "conversation_template_id") or _string_value(payload, "gizmo_id")
    if gizmo_id is None or gizmo_id.startswith("g-p-"):
        return None
    event_payload: dict[str, object] = {"gizmo_id": gizmo_id}
    if (gizmo_type := _string_value(payload, "gizmo_type")) is not None:
        event_payload["gizmo_type"] = gizmo_type
    return ParsedSessionEvent(
        event_type="chatgpt_custom_gpt",
        timestamp=timestamp,
        payload=event_payload,
    )


@parser_admission("chatgpt")
def parse(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    mapping = payload.get("mapping") or {}
    if not isinstance(mapping, dict):
        mapping = {}
    derived_current_node: str | None = None
    if not mapping and looks_like_shared_decode(payload):
        # polylogue-4zqh3: a shared-page stream decode has no ``mapping``
        # key at all -- synthesize one from its flat ``messages`` list so
        # the rest of this function (title/timestamps/ingest_flags below)
        # and ``extract_messages_from_mapping`` handle it identically to a
        # native export.
        mapping, derived_current_node = _shared_decode_mapping(payload)
    current_node = payload.get("current_node")
    current_node = current_node if isinstance(current_node, str) else derived_current_node
    admission = AdmissionLedger()
    admission.expect(AdmissionUnit.OUTER_RECORD, 1)
    admission.materialized(AdmissionUnit.OUTER_RECORD, 0, "conversation")
    conversation_model_slug = _string_value(payload, "default_model_slug")
    messages, attachments = extract_messages_from_mapping(
        mapping,
        current_node,
        admission=admission,
        preserve_empty_messages=derived_current_node is not None,
        default_model_slug=conversation_model_slug,
    )
    generation_timings = _extract_generation_timings(mapping)
    emitted_message_ids = {message.provider_message_id for message in messages}
    resolved_generation_timings: list[_GenerationTiming] = []
    for timing in generation_timings:
        if timing.message_provider_id in emitted_message_ids:
            resolved_generation_timings.append(timing)
            continue
        fallback_owner_id = next(
            (
                message.provider_message_id
                for message in reversed(messages)
                if message.provider_message_id in timing.related_message_provider_ids
            ),
            None,
        )
        resolved_generation_timings.append(
            replace(timing, message_provider_id=fallback_owner_id) if fallback_owner_id is not None else timing
        )
    generation_timings = resolved_generation_timings
    timing_by_message_id = {timing.message_provider_id: timing for timing in generation_timings}
    duplicate_duration_message_ids = {
        message_provider_id
        for timing in generation_timings
        for message_provider_id in timing.duplicate_duration_message_provider_ids
    }
    normalized_messages: list[ParsedMessage] = []
    for message in messages:
        resolved_timing = timing_by_message_id.get(message.provider_message_id)
        if resolved_timing is not None:
            normalized_messages.append(message.model_copy(update={"duration_ms": resolved_timing.elapsed_duration_ms}))
        elif message.provider_message_id in duplicate_duration_message_ids:
            normalized_messages.append(message.model_copy(update={"duration_ms": None}))
        else:
            normalized_messages.append(message)
    messages = normalized_messages
    session_events = [
        ParsedSessionEvent(
            event_type="generation_lifecycle",
            timestamp=timing.event_timestamp,
            source_message_provider_id=timing.message_provider_id,
            payload={
                "state": "completed",
                "evidence_source": "provider_native",
                "fidelity": timing.fidelity,
                "duration_semantics": "provider_reported_elapsed",
                "elapsed_duration_ms": timing.elapsed_duration_ms,
                **({"started_at_ms": timing.started_at_ms} if timing.started_at_ms is not None else {}),
                **({"ended_at_ms": timing.ended_at_ms} if timing.ended_at_ms is not None else {}),
            },
        )
        for timing in generation_timings
    ]
    session_events.extend(_block_metadata_evidence_events(messages))
    session_events.extend(_aggregate_result_events(mapping, emitted_message_ids))
    session_events.extend(_message_authorship_events(mapping, emitted_message_ids))
    emitted_provider_ids = {message.provider_message_id for message in messages}
    session_events.extend(
        event
        for event in _message_metadata_evidence_events(mapping)
        if event.source_message_provider_id in emitted_provider_ids
    )
    duration_values = [message.duration_ms for message in messages if message.duration_ms is not None]
    provider_title = payload.get("title") or payload.get("name")
    title = provider_title or fallback_id
    # polylogue-cijx.4 decision 3 / has_real_title (archive_tiers/archive.py):
    # title_source is the sole gate distinguishing a genuine provider title
    # from the bare native-id fallback this parser stores in `title` when
    # ChatGPT's own export carries neither `title` nor `name` -- without it,
    # every ChatGPT session (titled or not) silently degraded to the
    # structural "N msgs" label once #3421 made that gate strict. Only a
    # real payload title counts as ORIGIN evidence; the id fallback is
    # exactly the "worse than the UUID it replaces" case that gate exists
    # to catch.
    title_source = TitleSource.ORIGIN if provider_title else None
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
    ingest_flags: list[str] = []
    if not messages and payload.get("conversation_id") and payload.get("id") and "mapping" not in payload:
        ingest_flags.append(SHARED_CONVERSATION_INDEX_INGEST_FLAG)
    if payload.get("is_temporary") is True:
        ingest_flags.append("capture:temporary-chat")
    session_kind = SessionKind.TEMPORARY if payload.get("is_temporary") is True else SessionKind.STANDARD

    # ChatGPT "project" token (g-p-<id>): present in project-scoped conversations
    # as gizmo_id / conversation_template_id. A bare g-<id> is a custom GPT, not a
    # project, so only the g-p- prefix is treated as a workspace/project ref.
    project_raw = payload.get("conversation_template_id") or payload.get("gizmo_id")
    provider_project_ref = str(project_raw) if isinstance(project_raw, str) and project_raw.startswith("g-p-") else None

    conversation_timestamp = str(payload.get("create_time")) if payload.get("create_time") is not None else None
    if (settings_event := _conversation_settings_event(payload, timestamp=conversation_timestamp)) is not None:
        session_events.append(settings_event)
    if (custom_gpt_event := _custom_gpt_event(payload, timestamp=conversation_timestamp)) is not None:
        session_events.append(custom_gpt_event)

    # The conversation's ``default_model_slug`` names the model it was
    # configured to use; per-message slugs name the models that actually
    # answered. The union is the session's model summary.
    model_names: set[str] = {message.model_name for message in messages if message.model_name}
    if conversation_model_slug:
        model_names.add(conversation_model_slug)
    models_used = sorted(model_names)

    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=str(conv_id or fallback_id),
        title=str(title),
        title_source=title_source,
        session_kind=session_kind,
        provider_project_ref=provider_project_ref,
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
        active_leaf_message_provider_id=next(
            (message.provider_message_id for message in messages if message.is_active_leaf),
            None,
        ),
        attachments=attachments,
        session_events=session_events,
        unit_accounting=admission.close(),
        reported_duration_ms=sum(duration_values) if duration_values else None,
        models_used=models_used,
        ingest_flags=ingest_flags,
    )
