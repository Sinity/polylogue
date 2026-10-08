from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import (
    Callable,
    Container,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from contextlib import closing, nullcontext
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import ValidationError

from polylogue.archive.message.artifacts import (
    classify_material_origin,
    classify_message_type,
)
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
from polylogue.core.types import AttachmentDirection
from polylogue.sources.detection_projection import DetectorProjection
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
from .base_support import derive_attachment_provenance
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


def _coerce_float(value: object) -> float | None:
    """A finite float from a numeric or timestamp value, else ``None``.

    Non-finite values (``"nan"``, ``inf``) are no ordering evidence: NaN
    compares false with everything, so a sort over it depends on the store
    (SQLite keeps NaN as NULL), and message order must not.
    """
    # Exclude bool explicitly (bool is a subclass of int)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value)
        except (ValueError, TypeError):
            pass
        else:
            return result if math.isfinite(result) else None
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed.timestamp()
    return None


def _non_negative_finite_float(value: object) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None or not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _generation_branch_key(
    mapping: Mapping[str, object], node_id: str, memo: MutableMapping[str, str] | None = None
) -> str:
    """Return the first assistant-side node below the nearest user ancestor.

    ChatGPT repeats run-wide reasoning metadata across thought, tool, recap,
    and final-answer nodes. Grouping by this branch root deduplicates those
    copies while preserving regenerated alternatives beneath the same user
    message as distinct generations.

    ``memo`` caches the answer for every node the walk passes through. Each of
    those nodes resolves to the same branch root by definition -- continuing
    the walk from any of them is the same walk -- so the cache changes no
    verdict, it only stops one long assistant chain from being re-walked once
    per node (quadratic in a mapping an export controls). A walk that ended by
    detecting a cycle is not cached: its answer is the node the cycle closed
    on, which is not the answer for the nodes leading into it.
    """

    current_id = node_id
    if memo is not None and current_id in memo:
        return memo[current_id]
    seen: set[str] = set()
    path: list[str] = []
    cycle_detected = True
    result = current_id
    while current_id not in seen:
        seen.add(current_id)
        path.append(current_id)
        if memo is not None and current_id != node_id and current_id in memo:
            cycle_detected = False
            result = memo[current_id]
            path.pop()
            break
        current = mapping.get(current_id)
        if not isinstance(current, Mapping):
            cycle_detected = False
            result = current_id
            break
        parent_raw = current.get("parent")
        if not isinstance(parent_raw, str) or not parent_raw:
            cycle_detected = False
            result = current_id
            break
        parent = mapping.get(parent_raw)
        if not isinstance(parent, Mapping):
            cycle_detected = False
            result = current_id
            break
        parent_message = parent.get("message")
        parent_author = parent_message.get("author") if isinstance(parent_message, Mapping) else None
        parent_role = parent_author.get("role") if isinstance(parent_author, Mapping) else None
        if parent_role == "user":
            cycle_detected = False
            result = current_id
            break
        current_id = parent_raw
    else:
        result = current_id
    if memo is not None and not cycle_detected:
        for visited_id in path:
            memo[visited_id] = result
    return result


class GenerationTimingSelection(Protocol):
    """Where :func:`_extract_generation_timings` selects one timing per generation.

    Implemented over SQLite by ``prepared_message_sink.GenerationTimings``,
    on the spill's scratch database or a private in-memory one, so the
    per-node selection state is never held in process memory.
    """

    branch_memo: MutableMapping[str, str]

    def add_related(self, branch_key: str, message_id: str) -> None: ...

    def set_legacy_duration(self, branch_key: str, message_id: str, duration_ms: int) -> None: ...

    def offer(
        self, branch_key: str, score: tuple[int, int, int, int, str], elapsed_ms: int, timing: Mapping[str, object]
    ) -> None: ...


def _extract_generation_timings(mapping: Mapping[str, object], timings: GenerationTimingSelection) -> None:
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

        message_id = _mapping_message_id(str(node_id), raw_node, raw_message)
        branch_key = _generation_branch_key(mapping, str(node_id), timings.branch_memo)
        native_timing_field_names = (
            "reasoning_start_time",
            "reasoning_end_time",
            "finished_duration_sec",
        )
        has_native_timing_field = any(field_name in raw_metadata for field_name in native_timing_field_names)
        has_legacy_duration_field = "durationMs" in raw_metadata or "duration_ms" in raw_metadata
        if has_native_timing_field or has_legacy_duration_field:
            timings.add_related(branch_key, message_id)

        start_sec = _non_negative_finite_float(raw_metadata.get("reasoning_start_time"))
        end_sec = _non_negative_finite_float(raw_metadata.get("reasoning_end_time"))
        finished_sec = _non_negative_finite_float(raw_metadata.get("finished_duration_sec"))
        legacy_duration_raw = raw_metadata.get("durationMs")
        if legacy_duration_raw is None:
            legacy_duration_raw = raw_metadata.get("duration_ms")
        legacy_duration_ms = _non_negative_int(legacy_duration_raw)
        if legacy_duration_ms is not None:
            timings.set_legacy_duration(branch_key, message_id, legacy_duration_ms)

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
        )
        score = (
            source_rank,
            int(content_type == "reasoning_recap"),
            int(start_sec is not None and end_sec is not None),
            int(raw_message.get("end_turn") is True),
            message_id,
        )
        timings.offer(branch_key, score, timing.elapsed_duration_ms, asdict(timing))


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
    if isinstance(aggregate_result, list):
        if not aggregate_result:
            return None, False
        outcomes = [_aggregate_result_outcome(item)[0] for item in aggregate_result]
        if any(outcome is True for outcome in outcomes):
            return True, True
        if all(outcome is False for outcome in outcomes):
            return False, True
        # An admitted but unrecognized run is not successful delivery.
        return None, True
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
            if math.isfinite(value) and value.is_integer():
                return int(value)
            continue
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
_CHATGPT_CONVERSATION_URL_RE = re.compile(r"^https?://chatgpt\.com/(?:g/g-p-[^/]+/)?c/([0-9a-fA-F-]+)")


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


@dataclass(frozen=True, slots=True)
class _ActivePath:
    """The ChatGPT active path from ``current_node`` up to its root.

    ChatGPT exports preserve regenerated and edited branches in ``mapping`` and
    use ``current_node`` only to identify the leaf the user last saw. The v1
    parser contract keeps every branch and carries the active path explicitly
    instead of using it as a lossy filter (#1743).

    Membership lives in ``members``, a set from the caller's factory (scratch
    on the preparation route), which also stops the walk at a parent cycle.
    The path itself is never materialized: ``leaf_first`` re-walks the parent
    chain for exactly ``length`` steps, the steps the first walk took.
    """

    mapping: Mapping[str, object]
    current_node: str | None
    members: Container[str]
    length: int

    def leaf_first(self) -> Iterator[str]:
        node_id = self.current_node
        for _ in range(self.length):
            assert node_id is not None
            yield node_id
            node = self.mapping[node_id]
            parent = node.get("parent") if isinstance(node, dict) else None
            node_id = parent if isinstance(parent, str) else None


def _active_path(
    mapping: Mapping[str, object],
    current_node: str | None,
    new_set: Callable[[], MutableSet[str]] = set,
) -> _ActivePath:
    members = new_set()
    length = 0
    if current_node and current_node in mapping:
        node_id: object = current_node
        while isinstance(node_id, str) and node_id in mapping and node_id not in members:
            members.add(node_id)
            length += 1
            node = mapping[node_id]
            node_id = node.get("parent") if isinstance(node, dict) else None
    return _ActivePath(mapping, current_node, members, length)


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() and value >= 0 else None
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


class _MessageAttachmentIds:
    """Normalized file id -> index of one message's own attachments.

    Indexed incrementally: each attachment appended to the message since the
    last lookup is read once, so a message naming N attachments and N
    pointers costs O(N) reads rather than a rescan of its suffix per pointer.
    The first attachment with a given id wins, as the linear scan it
    replaces found.
    """

    def __init__(
        self, attachments: MutableSequence[ParsedAttachment], start: int, ids: MutableMapping[str, str]
    ) -> None:
        self.attachments = attachments
        self.ids = ids
        self.indexed = start

    def find(self, file_id: str) -> int | None:
        while self.indexed < len(self.attachments):
            key = strip_asset_pointer_scheme(self.attachments[self.indexed].provider_attachment_id)
            if key and key not in self.ids:
                self.ids[key] = str(self.indexed)
            self.indexed += 1
        found = self.ids.get(file_id)
        return int(found) if found is not None else None


def _append_asset_attachment(
    attachments: MutableSequence[ParsedAttachment],
    record: Mapping[str, object],
    *,
    pointer: str,
    message_provider_id: str,
    attachment_kind: str,
    direction: AttachmentDirection | None,
    producer_ref: str | None,
    dedupe: _MessageAttachmentIds,
) -> None:
    """Record an asset-pointer record as the attachment its bytes bind to.

    An attachment row is the only acquisition identity an asset has:
    ``assembly_chatgpt.py`` joins acquired export members onto attachments by
    the bare file id, so a pointer that reaches storage as block metadata
    alone leaves its acquired bytes with nothing to bind to.

    ``dedupe`` indexes this message's own attachments. A user upload is named twice — once by the message's ``metadata``
    attachment row (bare ``file-<id>``) and once by the content part's
    pointer URI (``file-service://file-<id>``) — and both normalize to the
    same id, so the second naming must not mint a second row.
    """
    file_id = strip_asset_pointer_scheme(pointer)
    if not file_id:
        return
    index = dedupe.find(file_id)
    if index is not None:
        existing = attachments[index]
        # A user upload is commonly named by both a metadata attachment
        # row (the bare ``file-…`` id) and an image/audio pointer part.
        # Keep that one acquisition identity, but do not lose the richer
        # media/provenance facts carried by the pointer part.  In
        # particular, metadata rows predate ``attachment_kind`` and may
        # otherwise remain indistinguishable from an ordinary upload.
        update: dict[str, object] = {}
        if existing.attachment_kind is None:
            update["attachment_kind"] = attachment_kind
        if existing.mime_type is None:
            pointer_mime_type = _string_value(record, "mime_type", "media_type")
            if pointer_mime_type is not None:
                update["mime_type"] = pointer_mime_type
        if existing.size_bytes is None:
            pointer_size = _non_negative_int(record.get("size_bytes"))
            if pointer_size is not None:
                update["size_bytes"] = pointer_size
        if existing.direction is None and direction is not None:
            update["direction"] = direction
        if existing.producer_ref is None and producer_ref is not None:
            update["producer_ref"] = producer_ref
        if update:
            attachments[index] = existing.model_copy(update=update)
        return
    attachments.append(
        ParsedAttachment(
            provider_attachment_id=pointer,
            message_provider_id=message_provider_id,
            name=_string_value(record, "name", "filename", "file_name"),
            mime_type=_string_value(record, "mime_type", "media_type"),
            # Read off the URI: the id space the export's asset members and
            # ``library_files.json`` keys share.
            provider_file_id=file_id,
            size_bytes=_non_negative_int(record.get("size_bytes")),
            attachment_kind=attachment_kind,
            direction=direction,
            producer_ref=producer_ref,
        )
    )


def _iter_asset_pointer_records(
    value: object,
    *,
    parent_record: Mapping[str, object],
) -> Iterator[tuple[str, Mapping[str, object]]]:
    """Yield pointer strings and their closest provider metadata record.

    Ordinary image/audio parts use ``asset_pointer`` directly. Realtime A/V
    parts instead carry ``audio_asset_pointer``,
    ``video_container_asset_pointer`` and ``frames_asset_pointers``; the
    latter two have appeared both as scalar values and nested/list records.
    Keep the traversal deliberately scoped to those pointer-bearing fields so
    neighbouring media metadata (``mime_type``, dimensions, etc.) cannot be
    mistaken for an asset identity.
    """
    if isinstance(value, str):
        if value:
            yield value, parent_record
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_asset_pointer_records(item, parent_record=parent_record)
        return
    if not isinstance(value, Mapping):
        return

    for key in ("asset_pointer", "pointer"):
        pointer = value.get(key)
        if isinstance(pointer, str) and pointer:
            record = dict(parent_record)
            record.update(value)
            yield pointer, record
            return

    # Some realtime exports use a keyed frame map rather than a list. Its
    # values are still pointer records; skip ordinary metadata keys while
    # recursing through the map.
    for key, child in value.items():
        if key in {"mime_type", "media_type", "size_bytes", "width", "height", "name", "filename"}:
            continue
        yield from _iter_asset_pointer_records(child, parent_record=parent_record)


def _chatgpt_media_asset_pointers(
    part: Mapping[str, object],
    content_type: str,
) -> Iterator[tuple[str, Mapping[str, object], str]]:
    """Yield ``(pointer, metadata, pointer_field)`` for an audio/A/V part."""
    fields: tuple[str, ...]
    if content_type == "audio_asset_pointer":
        fields = ("asset_pointer",)
    elif content_type in {"video_asset_pointer", "video_container_asset_pointer"}:
        # A few exports use a standalone video part rather than the realtime
        # wrapper.  Keep both the ordinary and provider-specific spellings so
        # the pointer remains the acquisition identity regardless of which
        # capture path produced the part.
        fields = ("asset_pointer", "video_asset_pointer", "video_container_asset_pointer")
    else:
        # ``asset_pointer`` is retained as a compatibility fallback for
        # captures that used the ordinary image/audio spelling for realtime
        # media. The observed export uses the type-specific fields.
        fields = (
            "audio_asset_pointer",
            "video_container_asset_pointer",
            "frames_asset_pointers",
            "asset_pointer",
        )
    for field in fields:
        for pointer, record in _iter_asset_pointer_records(part.get(field), parent_record=part):
            yield pointer, record, field


def _chatgpt_media_attachment_kind(pointer_field: str, content_type: str | None = None) -> str:
    if pointer_field in {"video_asset_pointer", "video_container_asset_pointer"} or content_type in {
        "video_asset_pointer",
        "video_container_asset_pointer",
    }:
        return "video_asset"
    if pointer_field == "frames_asset_pointers":
        return "video_frame_asset"
    return "audio_asset"


# ChatGPT embeds inline citation anchors in assistant text as private-use
# unicode spans: U+E200 opens, U+E202 separates reference tokens, U+E201
# closes (e.g. "\ue200filecite\ue202turn3file14\ue202L180-L293\ue201").
# The span carries no human-readable text -- the resolvable citation rows
# live in message metadata (`citations`/`content_references`) and are
# preserved as web constructs. The raw markers otherwise leak invisible
# glyphs into search text and rendered transcripts; the untouched original
# remains in the source-tier raw payload.
# The span body excludes both delimiters rather than using a lazy ``.*?``.
# A lazy any-character span over an unterminated opener backtracks once per
# following opener, so assistant text carrying a long run of U+E200 with no
# U+E201 costs quadratic time -- and ChatGPT message text is attacker-authored
# from the archive's point of view. Excluding the delimiters makes each start
# position fail in constant time. On well-formed markers the two agree; on a
# malformed nested run the excluding form keeps the inner text instead of
# swallowing it up to a later closer, which is the better reading of a
# delimiter that never nests.
_CITATION_MARKER_RE = re.compile("\ue200[^\ue200\ue201]*\ue201|[\ue200\ue201\ue202]")
_CITATION_MARKER_SPAN_RE = re.compile("\ue200([^\ue200\ue201]*)\ue201")


def _strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER_RE.sub("", text)


_SANDBOX_FILE_RE = re.compile(r"sandbox:(/mnt/data/[^\s)\]\"'>]+)")


def _sandbox_file_paths(text: str, seen: MutableSet[str] | None = None) -> Iterator[str]:
    """Ordered, distinct ``/mnt/data`` paths linked in assistant text.

    Trailing prose punctuation is stripped so ``(sandbox:/mnt/data/kit.zip).``
    yields ``/mnt/data/kit.zip``. Directory links keep their trailing slash in
    the returned path. Every distinct link becomes an attachment; the prepared
    route keeps them in scratch, so a message linking many files is recorded
    whole rather than capped. ``seen`` holds the distinct paths; the prepared
    route passes a scratch-backed set so a message linking millions of files
    does not hold them all in memory.
    """

    if seen is None:
        seen = set()
    for match in _SANDBOX_FILE_RE.finditer(text):
        path = match.group(1).rstrip(".,;:!?*`")
        if path == "/mnt/data/" or path in seen:
            continue
        seen.add(path)
        yield path


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
            # polylogue-4mbya: a step's ``summary`` is its own short header --
            # the text ChatGPT shows above the step -- not a restatement of
            # ``content``. Keeping only the body dropped 890,288 characters of
            # step headers over one export, so both are carried: the summary as
            # the step's heading line, the content beneath it.
            step_parts = [
                value for value in (thought.get("summary"), thought.get("content")) if isinstance(value, str) and value
            ]
            if len(step_parts) == 2 and step_parts[0] == step_parts[1]:
                del step_parts[1]
            if step_parts:
                thought_parts.append("\n".join(step_parts))
        if thought_parts:
            return "\n".join(thought_parts)
    return ""


#: Block types that can carry a tool-role node's payload. The first one a
#: tool-role node produced becomes its canonical TOOL_RESULT; a typed-unknown
#: block is excluded so an unrecognized wire shape keeps its own disposition.
_TOOL_RESULT_CARRIER_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.TEXT, BlockType.DOCUMENT, BlockType.CODE, BlockType.TOOL_USE, BlockType.THINKING}
)


def _mapping_message_id(node_id: str, node: Mapping[str, object], message: Mapping[str, object]) -> str:
    """Use the provider's message, node or mapping identity, without inventing an ID."""
    return str(message.get("id") or node.get("id") or node_id)


def _owning_tool_call_id(
    mapping: Mapping[str, object], parent_id: str | None, memo: MutableMapping[str, str] | None = None
) -> str | None:
    """Resolve which node a ``role: tool`` result answers.

    A tool episode is a chain: the calling node, then one or more ``role: tool``
    result nodes. The provider attaches the second and later results to the
    *previous result*, so the direct mapping parent names another answer rather
    than the node the episode hangs off. Skipping the tool-role ancestors
    reaches that node, and every result of one episode names the same owner.

    ``memo`` caches the owner of every tool node a walk passes through (each
    of them resolves to the same owner), so a long chain of results is walked
    once rather than once per result. A walk that ends on a parent cycle is
    not cached, as in ``_generation_branch_key``.
    """
    seen: set[str] = set()
    path: list[str] = []
    current = parent_id
    result: str | None = current
    cycle = True
    while isinstance(current, str) and current and current not in seen:
        if memo is not None and current in memo:
            result, cycle = memo[current], False
            break
        seen.add(current)
        node = mapping.get(current)
        if not isinstance(node, Mapping):
            result, cycle = current, False
            break
        message = node.get("message")
        author = message.get("author") if isinstance(message, Mapping) else None
        if not (isinstance(author, Mapping) and author.get("role") == "tool"):
            result, cycle = current, False
            break
        path.append(current)
        parent = node.get("parent")
        if not parent:
            result, cycle = None, False
            break
        current = str(parent)
    else:
        result = None
        cycle = isinstance(current, str) and bool(current)
    if memo is not None and not cycle and result is not None:
        for visited in path:
            memo[visited] = result
    return result


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
#   reasoning_title/titles/status -> chatgpt_message_delivery
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
#
# The machine-readable form of this register is ``CHATGPT_READ_KEYS`` /
# ``CHATGPT_EXCLUDED_KEYS``, declared below next to
# ``_CHATGPT_CONVERSATION_SETTING_KEYS`` because it reuses it.
@runtime_checkable
class DeclaredChildPositions(Protocol):
    """A mapping view that answers declared sibling order without the array.

    The prepared route keeps each node's ``children`` array in scratch rather
    than on the node, so the one question the parser asks of it -- where a
    child sits in its parent's declared order -- is answered by lookup.
    """

    def declared_child_position(self, parent_key: str, child_id: object) -> int | None: ...


def _sibling_ordinal_lookup(mapping: Mapping[str, object]) -> Callable[[str], int]:
    """Answer each node's arrival ordinal among its parent's children.

    A scratch-backed mapping answers from its own index; any other mapping
    computes the ordinals once (``_sibling_ordinals``).
    """
    lookup = getattr(mapping, "sibling_ordinal", None)
    if callable(lookup):
        return cast(Callable[[str], int], lookup)
    ordinals = _sibling_ordinals(mapping)
    return lambda node_id: ordinals.get(node_id, 0)


def _declared_child_position(mapping: Mapping[str, object], parent_key: str, node: Mapping[str, object]) -> int | None:
    """Index of ``node`` in its parent's ``children`` array, when it is listed."""
    if isinstance(mapping, DeclaredChildPositions):
        return mapping.declared_child_position(parent_key, node.get("id"))
    parent_node = mapping.get(parent_key)
    if isinstance(parent_node, dict):
        children = parent_node.get("children")
        if isinstance(children, list):
            current_node_id = node.get("id")
            if current_node_id in children:
                return children.index(current_node_id)
    return None


class MessageEntries(Protocol):
    """Normalized messages before ordering and cross-message resolution.

    The collecting parser keeps them in a list; the prepared route keeps them
    in scratch. Both answer the same questions, so the ordering, parent,
    active-leaf and timing-owner rules below exist once.
    """

    def add(self, timestamp: float | None, idx: int, node_id: str, message: ParsedMessage) -> None: ...

    def ordered(self) -> Iterator[ParsedMessage]:
        """Messages by (timestamp, mapping order), untimestamped last."""
        ...

    def provider_for_node(self, node_id: str) -> str | None: ...

    def position_for_node(self, node_id: str) -> int | None: ...

    def emitted_provider_ids(self) -> Container[str]: ...

    def last_emitted_among(self, provider_ids: frozenset[str]) -> str | None:
        """The provider id among ``provider_ids`` that comes last in :meth:`ordered`."""
        ...


class _ListMessageEntries:
    def __init__(self) -> None:
        self._entries: list[tuple[float | None, int, str, ParsedMessage]] = []
        self._by_node: dict[str, ParsedMessage] = {}
        self._sorted = False

    def add(self, timestamp: float | None, idx: int, node_id: str, message: ParsedMessage) -> None:
        self._entries.append((timestamp, idx, node_id, message))
        self._by_node[node_id] = message
        self._sorted = False

    def _ordered_entries(self) -> list[tuple[float | None, int, str, ParsedMessage]]:
        if not self._sorted:
            if any(value is not None for value, _, _, _ in self._entries):
                # Explicit None check instead of `or` keeps zero/negative timestamps.
                self._entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0, item[1]))
            self._sorted = True
        return self._entries

    def ordered(self) -> Iterator[ParsedMessage]:
        return (entry[3] for entry in self._ordered_entries())

    def provider_for_node(self, node_id: str) -> str | None:
        message = self._by_node.get(node_id)
        return message.provider_message_id if message is not None else None

    def position_for_node(self, node_id: str) -> int | None:
        message = self._by_node.get(node_id)
        return message.position if message is not None else None

    def emitted_provider_ids(self) -> Container[str]:
        return {message.provider_message_id for message in self._by_node.values()}

    def last_emitted_among(self, provider_ids: frozenset[str]) -> str | None:
        return next(
            (
                entry[3].provider_message_id
                for entry in reversed(self._ordered_entries())
                if entry[3].provider_message_id in provider_ids
            ),
            None,
        )


class SessionSpill(Protocol):
    """Where :func:`parse` keeps a session's growing collections.

    Omitted, :func:`parse` collects into lists. The prepared route passes
    scratch-backed sequences so a large mapping never holds its normalized
    messages, attachments or events in memory.
    """

    def entries(self) -> MessageEntries: ...

    def messages(self) -> MutableSequence[ParsedMessage]: ...

    def attachments(self) -> MutableSequence[ParsedAttachment]: ...

    def events(self) -> MutableSequence[ParsedSessionEvent]: ...

    def seen_set(self) -> MutableSet[str]:
        """An empty set for per-message deduplication."""
        ...

    def string_map(self) -> MutableMapping[str, str]:
        """An empty string map for per-node memos and indexes."""
        ...

    def set_attachment_record_origin(self, ordinal: int, raw_position: int) -> None: ...

    def connection(self) -> sqlite3.Connection:
        """The scratch database the session's selection tables live in."""
        ...


def extract_messages_from_mapping(
    mapping: Mapping[str, object],
    current_node: str | None = None,
    *,
    admission: AdmissionLedger | None = None,
    default_model_slug: str | None = None,
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    entries = _ListMessageEntries()
    attachments: list[ParsedAttachment] = []
    active_path = _collect_message_entries(
        mapping,
        current_node,
        entries,
        attachments,
        admission=admission,
        default_model_slug=default_model_slug,
    )
    return list(_resolved_messages(entries, active_path)), attachments


def _resolved_messages(entries: MessageEntries, active_path: _ActivePath) -> Iterator[ParsedMessage]:
    """Final order, parent references resolved to emitted ids, the active leaf marked."""
    emitted_message_ids = entries.emitted_provider_ids()
    active_leaf_position = next(
        (
            position
            for node_id in active_path.leaf_first()
            if (position := entries.position_for_node(node_id)) is not None
        ),
        None,
    )
    for message in entries.ordered():
        parent_id = message.parent_message_provider_id
        if parent_id is not None:
            resolved = entries.provider_for_node(parent_id)
            if resolved is None and parent_id in emitted_message_ids:
                resolved = parent_id
            message = message.model_copy(update={"parent_message_provider_id": resolved})
        # Tool-result owners are mapping-node references until every message
        # has been emitted. Resolve only those references through the same
        # canonical node/message index as parents; TOOL_USE already carries
        # its own emitted message ID. Missing owners remain unlinked.
        if any(block.type is BlockType.TOOL_RESULT and block.tool_id is not None for block in message.blocks):
            message = message.model_copy(
                update={
                    "blocks": [
                        block.model_copy(update={"tool_id": entries.provider_for_node(block.tool_id)})
                        if block.type is BlockType.TOOL_RESULT and block.tool_id is not None
                        else block
                        for block in message.blocks
                    ]
                }
            )
        if active_leaf_position is not None:
            message = message.model_copy(update={"is_active_leaf": message.position == active_leaf_position})
        yield message


def _collect_message_entries(
    mapping: Mapping[str, object],
    current_node: str | None,
    entries: MessageEntries,
    attachments: MutableSequence[ParsedAttachment],
    *,
    admission: AdmissionLedger | None,
    default_model_slug: str | None,
    new_seen_set: Callable[[], MutableSet[str]] = set,
    new_string_map: Callable[[], MutableMapping[str, str]] = dict,
    attachment_origin: Callable[[int, int], None] | None = None,
    attachment_occurrence: Callable[[ParsedAttachment, int], None] | None = None,
) -> _ActivePath:
    """Normalize every message node into ``entries``; return the active path."""
    if admission is not None:
        admission.expect(
            AdmissionUnit.MESSAGE,
            sum(1 for node in mapping.values() if isinstance(node, dict) and isinstance(node.get("message"), dict)),
        )
    message_ordinal = 0
    sibling_ordinal = _sibling_ordinal_lookup(mapping)
    active_path = _active_path(mapping, current_node, new_seen_set)
    tool_owners = new_string_map()
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
        # ``content.parts`` is export content. A non-list value was carried
        # through here and iterated during block construction, raising
        # ``TypeError`` out of the parser and losing the WHOLE bundle rather
        # than this one node. ``_extract_content_text`` already treats a
        # non-list ``parts`` as carrying no text, so matching that isinstance
        # check here loses nothing beyond what the text path already refused.
        raw_parts = content.get("parts")
        parts = raw_parts if isinstance(raw_parts, list) else []
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
        msg_id = _mapping_message_id(str(node_id), node, msg)

        # Extract parent message reference and calculate branch index
        parent_id = node.get("parent")
        parent_message_provider_id = str(parent_id) if parent_id else None
        tool_result_owner_id = (
            _owning_tool_call_id(mapping, parent_message_provider_id, tool_owners)
            if role is Role.TOOL
            else parent_message_provider_id
        )
        branch_index = 0

        # The parent's ``children`` array states sibling order; where the
        # export omits it, the node's arrival ordinal among the siblings
        # naming the same parent carries the same sequence
        # (``_sibling_ordinals``).
        if parent_message_provider_id:
            branch_index = sibling_ordinal(node_id)
            declared_position = _declared_child_position(mapping, str(parent_id), node)
            if declared_position is not None:
                branch_index = declared_position

        # Where this message's own attachments begin, so an asset named both
        # by a metadata row and by a content part collapses to one row.
        attachment_start = len(attachments)
        message_attachment_ids = _MessageAttachmentIds(attachments, attachment_start, new_string_map())

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
            for sandbox_path in _sandbox_file_paths(text, new_seen_set()):
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
        # Text that parsed as the call's JSON payload IS its input; any other
        # text stays on the call block rather than vanishing with it.
        tool_call_text = None if tool_call_input is not None else (text or None)
        if tool_call_input is None and tool_target is not None:
            if tool_args not in (None, [], {}, ""):
                tool_call_input = {"args": tool_args}
            elif tool_command is not None and role is Role.ASSISTANT and content.get("content_type", "text") == "text":
                # ``metadata.command`` states the call outright, so a command
                # with no arguments (``computer.initialize`` with ``args: {}``
                # or none) is still a call: lowering it to TEXT left its
                # result node with nothing to pair with. A bare ``recipient``
                # does not: prose addressed to a tool (an image caption)
                # stays text unless it parses as the call's JSON input.
                tool_call_input = {}

        # Build structured content blocks
        content_blocks: list[ParsedContentBlock] = []
        forced_message_type: MessageType | None = None
        content_type = content.get("content_type", "text")
        if not isinstance(content_type, str):
            content_type = "unknown"
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
                    text=tool_call_text,
                    tool_name=tool_target,
                    # tool_id = this message's emitted id; the mapping-tree child
                    # node that carries the result resolves its owner
                    # to that message id below (polylogue-ah21: these
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
                    metadata={"content_type": content_type},
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
            # "code_interpreter" when it didn't; tool_id = this message's emitted
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
            # The mapping-tree `parent` resolves to the calling message's id, the
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
            # The mapping-tree `parent` resolves to the calling message's id, the
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
                    dedupe=message_attachment_ids,
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
                            dedupe=message_attachment_ids,
                        )
                elif (
                    isinstance(part, dict)
                    and isinstance(part.get("content_type"), str)
                    and part.get("content_type")
                    in {
                        "audio_asset_pointer",
                        "audio_transcription",
                        "real_time_user_audio_video_asset_pointer",
                        "video_asset_pointer",
                        "video_container_asset_pointer",
                    }
                ):
                    part_text = part.get("text")
                    content_type = str(part.get("content_type"))
                    media_mime_type = _string_value(part, "mime_type", "media_type")
                    media_constructs: list[ParsedWebConstruct] = []
                    for pointer, pointer_record, pointer_field in _chatgpt_media_asset_pointers(part, content_type):
                        pointer_mime_type = _string_value(pointer_record, "mime_type", "media_type") or media_mime_type
                        media_constructs.append(
                            ParsedWebConstruct(
                                construct_type=(
                                    WebConstructType.AUDIO_TRANSCRIPTION
                                    if content_type == "audio_transcription"
                                    else WebConstructType.AUDIO_ASSET
                                ),
                                provider_key=content_type,
                                asset_pointer=pointer,
                                mime_type=pointer_mime_type,
                            )
                        )
                        media_direction, media_producer = derive_attachment_provenance(role, str(msg_id))
                        _append_asset_attachment(
                            attachments,
                            pointer_record,
                            pointer=pointer,
                            message_provider_id=str(msg_id),
                            attachment_kind=_chatgpt_media_attachment_kind(pointer_field, content_type),
                            direction=media_direction,
                            producer_ref=media_producer,
                            dedupe=message_attachment_ids,
                        )
                    if not media_constructs:
                        media_constructs.append(
                            ParsedWebConstruct(
                                construct_type=(
                                    WebConstructType.AUDIO_TRANSCRIPTION
                                    if content_type == "audio_transcription"
                                    else WebConstructType.AUDIO_ASSET
                                ),
                                provider_key=content_type,
                                mime_type=media_mime_type,
                                # A media-shaped part without a provider
                                # pointer is still source evidence, but it
                                # cannot identify or claim acquired bytes.
                                # Make that absence explicit instead of
                                # emitting a construct whose only useful
                                # fields are its type and provider spelling.
                                status=(
                                    None
                                    if content_type == "audio_transcription"
                                    and isinstance(part_text, str)
                                    and part_text
                                    else "unavailable"
                                ),
                            )
                        )
                    content_blocks.append(
                        ParsedContentBlock(
                            type=BlockType.DOCUMENT,
                            text=part_text if isinstance(part_text, str) and part_text else None,
                            media_type=media_mime_type,
                            web_constructs=[
                                construct.model_copy(
                                    update={
                                        "text": part_text if isinstance(part_text, str) and part_text else None,
                                    }
                                )
                                for construct in media_constructs
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
            # ``parts`` is normalized to a list at extraction, so the former
            # isinstance guards here are now provably dead.
            admission.expect(AdmissionUnit.PART, len(parts))
            for part_ordinal, part in enumerate(parts):
                if isinstance(part, str) or (
                    isinstance(part, dict)
                    and isinstance(part.get("content_type"), str)
                    and part.get("content_type")
                    in {
                        "image_asset_pointer",
                        "audio_asset_pointer",
                        "audio_transcription",
                        "real_time_user_audio_video_asset_pointer",
                        "video_asset_pointer",
                        "video_container_asset_pointer",
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
        if attachment_origin is not None:
            for attachment_ordinal in range(attachment_start, len(attachments)):
                attachment_origin(attachment_ordinal, idx - 1)
        if attachment_occurrence is not None:
            for attachment_ordinal in range(attachment_start, len(attachments)):
                attachment = attachments[attachment_ordinal]
                attachment_occurrence(attachment, idx - 1)
                attachments[attachment_ordinal] = attachment

        status_val = msg.get("status")
        end_turn_val = msg.get("end_turn")
        user_context_val = msg_metadata.get("user_context_message_data")
        message_type = classify_message_type(
            role=role,
            message_type=forced_message_type or MessageType.MESSAGE,
            text=text,
            block_types=tuple(block.type for block in content_blocks),
        )
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
            is_active_path=node_id in active_path.members if active_path.length else None,
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
        entries.add(_coerce_float(timestamp), idx, node_id, parsed)
        if admission is not None:
            admission.materialized(AdmissionUnit.MESSAGE, current_message_ordinal, node_id)
    return active_path


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
    (the canonical sequence-document detector and direct callers
    validating an assembled export). For a single record
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


def _aggregate_result_events(
    mapping: Mapping[str, object], emitted_message_ids: Container[str]
) -> Iterator[ParsedSessionEvent]:
    """Conserve each code-interpreter run's own record.

    The run's output already IS the result node's text, so the stream text is
    stored verbatim only where that duplication does not hold and its length
    is recorded either way -- the same conservation shape Codex's
    ``last_agent_message`` event uses. Everything else here (the run id, its
    clock, the timeout it ran under, the exception class) exists nowhere else
    in the parsed session, and neither does the executed program, which rides
    the ``aggregate_result`` web construct.
    """
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        metadata = message.get("metadata")
        aggregate_result = metadata.get("aggregate_result") if isinstance(metadata, Mapping) else None
        message_id = _mapping_message_id(str(node_id), node, message)
        if message_id not in emitted_message_ids:
            continue
        content = message.get("content")
        node_text = _string_value(content, "text") if isinstance(content, Mapping) else None
        for run in _iter_mapping_items(aggregate_result):
            stream_text = _run_stream_text(run)
            exception = run.get("in_kernel_exception")
            exception = exception if isinstance(exception, Mapping) else {}
            system_exception = run.get("system_exception")
            payload: dict[str, object] = {"status": _string_value(run, "status")}
            for key in ("run_id", "start_time", "end_time", "update_time", "timeout_triggered", "output", "exit_code"):
                value = run.get(key)
                if value is not None:
                    payload[key] = value
            final_expression_output = run.get("final_expression_output")
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
            yield ParsedSessionEvent(
                event_type="chatgpt_code_interpreter_run",
                timestamp=_string_value(run, "end_time", "update_time", "start_time"),
                source_message_provider_id=message_id,
                payload=payload,
            )


def _message_authorship_events(
    mapping: Mapping[str, object], emitted_message_ids: Container[str]
) -> Iterator[ParsedSessionEvent]:
    """Conserve the wire's own statement of what a message is and who wrote it.

    ``channel`` separates a turn's reasoning-adjacent ``commentary`` from the
    ``final`` answer the user was shown -- ChatGPT's analogue of Codex's
    ``phase`` -- and ``author.metadata.real_author`` names the tool that
    actually produced a message rendered through another role's envelope.
    """
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
        message_id = _mapping_message_id(str(node_id), node, message)
        if message_id not in emitted_message_ids:
            continue
        payload: dict[str, object] = {}
        if channel is not None:
            payload["channel"] = channel
        if real_author is not None:
            payload["real_author"] = real_author
        yield ParsedSessionEvent(
            event_type="chatgpt_message_authorship",
            timestamp=str(message.get("create_time")) if message.get("create_time") is not None else None,
            source_message_provider_id=message_id,
            payload=payload,
        )


def _block_metadata_evidence_events(messages: Iterable[ParsedMessage]) -> Iterator[ParsedSessionEvent]:
    for message in messages:
        for block_index, block in enumerate(message.blocks):
            if not block.metadata:
                continue
            yield ParsedSessionEvent(
                event_type="chatgpt_block_metadata",
                timestamp=message.timestamp,
                source_message_provider_id=message.provider_message_id,
                payload={"block_index": block_index, **dict(block.metadata)},
            )


def _iter_message_nodes(mapping: Mapping[str, object]) -> Iterator[tuple[str, Mapping[str, object]]]:
    """Every ``(provider_message_id, message)`` pair in mapping order."""
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        # The same identity ``extract_messages_from_mapping`` mints, so the
        # events these feed bind to the message the parser emitted.
        yield _mapping_message_id(str(node_id), node, message), message


def _message_metadata_evidence_events(mapping: Mapping[str, object]) -> Iterator[ParsedSessionEvent]:
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
        answer), plus the declared reasoning title, titles and status. Absent these a hidden, context-dropped, commentary-channel
        turn reads as conversation the operator saw and the model kept.
    ``chatgpt_jit_plugin_data``
        A just-in-time plugin's call and response payload -- the only
        record of what a plugin was asked and what it answered.
    """
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
            yield ParsedSessionEvent(
                event_type="chatgpt_targeted_reply",
                timestamp=timestamp_text,
                source_message_provider_id=message_id,
                payload=payload,
            )
        delivery: dict[str, object] = {}
        weight = message.get("weight")
        if isinstance(weight, (int, float, Decimal)) and not isinstance(weight, bool) and float(weight) != 1.0:
            delivery["weight"] = float(weight)
        if metadata.get("is_visually_hidden_from_conversation") is True:
            delivery["is_visually_hidden_from_conversation"] = True
        if (channel := _string_value(message, "channel")) is not None:
            delivery["channel"] = channel
        # Rendering evidence belongs to its native message even when it has
        # no prose or blocks. Do not manufacture a THINKING block for a label.
        for key in ("reasoning_title", "reasoning_titles", "reasoning_status"):
            value = metadata.get(key)
            if key in metadata:
                delivery[key] = value
        if delivery:
            yield ParsedSessionEvent(
                event_type="chatgpt_message_delivery",
                timestamp=timestamp_text,
                source_message_provider_id=message_id,
                payload=delivery,
            )
        jit_plugin_data = metadata.get("jit_plugin_data")
        if isinstance(jit_plugin_data, Mapping) and jit_plugin_data:
            yield ParsedSessionEvent(
                event_type="chatgpt_jit_plugin_data",
                timestamp=timestamp_text,
                source_message_provider_id=message_id,
                payload=dict(jit_plugin_data),
            )


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


#: The register above, as a structure the test suite can read (polylogue-9193q).
#: The prose keeps the rationale; these two mappings are what makes the
#: register load-bearing. ``tests/unit/sources/test_chatgpt_field_ledger.py``
#: walks every key of every committed synthetic ChatGPT export and refuses any
#: key that is in neither, so a new upstream field fails loudly instead of
#: being dropped in silence (the original ~1.9 MB-per-export loss).
#:
#: Scopes are the positions a key can occupy in an export, not types:
#: ``conversation`` is the top-level record, ``node`` a ``mapping`` entry,
#: ``message`` the node's message envelope, and the two ``*.metadata`` scopes
#: the free-form bags hanging off them.
CHATGPT_READ_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "conversation": frozenset(
            {
                "mapping",
                "messages",
                "current_node",
                "title",
                "name",
                "create_time",
                "update_time",
                "id",
                "uuid",
                "conversation_id",
                "shared_conversation_id",
                "is_temporary",
                "conversation_template_id",
                "gizmo_id",
                "default_model_slug",
                *_CHATGPT_CONVERSATION_SETTING_KEYS,
            }
        ),
        "node": frozenset({"id", "message", "parent", "children"}),
        "message": frozenset(
            {
                "id",
                "author",
                "content",
                "create_time",
                "status",
                "recipient",
                "end_turn",
                "weight",
                "channel",
                "metadata",
            }
        ),
        "message.metadata": frozenset(
            {
                "attachments",
                "model_slug",
                "default_model_slug",
                "thinking_effort",
                "reasoning_effort",
                "model_effort",
                "durationMs",
                "duration_ms",
                "reasoning_start_time",
                "reasoning_end_time",
                "finished_duration_sec",
                "user_context_message_data",
                "canvas",
                "content_references",
                "citations",
                "_cite_metadata",
                "conversation_context_citation_metadata",
                "inline_cot_expandable_content",
                "search_queries",
                "search_result_groups",
                "selected_sources",
                "image_results",
                "async_task_type",
                "async_task_id",
                "async_task_title",
                "aggregate_result",
                "finish_details",
                "command",
                "args",
                "reasoning_title",
                "reasoning_titles",
                "reasoning_status",
                "dalle",
                "ada_visualizations",
                "targeted_reply",
                "targeted_reply_label",
                "is_visually_hidden_from_conversation",
                "jit_plugin_data",
                "name",
            }
        ),
        "author": frozenset({"role", "name", "metadata"}),
        "author.metadata": frozenset({"real_author"}),
    }
)

#: Scope -> key -> why this parser deliberately does not read it. The reasons
#: are the register's, compressed to one line; the register above carries the
#: measured occurrence counts they rest on.
CHATGPT_EXCLUDED_KEYS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "conversation": MappingProxyType(
            {
                "memory_scope": (
                    "an account setting stamped onto every conversation; it names which memory store "
                    "the account had enabled, not anything this conversation did"
                ),
                "owner": (
                    "null in every acquired export, and provider account identity is out of scope for "
                    "session content evidence -- the decision claude/common.py records for claude.ai `account`"
                ),
            }
        ),
        "message": MappingProxyType(
            {
                "update_time": (
                    "null on the overwhelming majority of messages; where stated it times an edit whose "
                    "before-state the export does not carry, and `messages` models no per-message revision axis"
                ),
            }
        ),
        "message.metadata": MappingProxyType(
            {
                "request_id": (
                    "the provider's correlation token for one server-side run; no tier carries a ChatGPT "
                    "request id to join against, and the grouping it expresses is the conversation tree"
                ),
                "search_display_string": (
                    "one constant render string in every occurrence; the queries it labels are already read "
                    "as `search_queries` and the results projected from `search_result_groups`"
                ),
                "initial_text": (
                    "the UI status line shown before a reasoning run; its elapsed time is the same "
                    "measurement `_extract_generation_timings` reads, in prose"
                ),
                "finished_text": (
                    "the UI status line shown after a reasoning run; its elapsed time is the same "
                    "measurement `_extract_generation_timings` reads, in prose"
                ),
            }
        ),
        "author.metadata": MappingProxyType(
            {
                "sonicberry_model_id": (
                    "a provider-internal routing label for the web tool; it names neither content, a source, "
                    "nor an outcome -- only which internal variant served a call"
                ),
                "sonicberry_source": (
                    "a provider-internal routing label for the web tool; it names neither content, a source, "
                    "nor an outcome -- only which internal variant served a call"
                ),
                "is_system_initiated_conversation": (
                    "restates for one message what the conversation's own first turn already shows"
                ),
            }
        ),
    }
)

#: The scopes both registers are keyed by, so a walker cannot invent one.
CHATGPT_FIELD_SCOPES: tuple[str, ...] = (
    "conversation",
    "node",
    "message",
    "message.metadata",
    "author",
    "author.metadata",
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
def parse(
    payload: Mapping[str, object],
    fallback_id: str,
    *,
    spill: SessionSpill | None = None,
    attachment_occurrence: Callable[[ParsedAttachment, int], None] | None = None,
) -> ParsedSession:
    mapping = payload.get("mapping") or {}
    if not isinstance(mapping, Mapping):
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
    entries: MessageEntries = spill.entries() if spill is not None else _ListMessageEntries()
    attachments: MutableSequence[ParsedAttachment] = spill.attachments() if spill is not None else []
    active_path = _collect_message_entries(
        mapping,
        current_node,
        entries,
        attachments,
        admission=admission,
        default_model_slug=conversation_model_slug,
        new_seen_set=spill.seen_set if spill is not None else set,
        new_string_map=spill.string_map if spill is not None else dict,
        attachment_origin=spill.set_attachment_record_origin if spill is not None else None,
        attachment_occurrence=attachment_occurrence,
    )
    emitted_message_ids = entries.emitted_provider_ids()
    session_events: MutableSequence[ParsedSessionEvent] = spill.events() if spill is not None else []
    messages: MutableSequence[ParsedMessage] = spill.messages() if spill is not None else []
    message_count = 0
    reported_duration_ms: int | None = None
    model_names: set[str] = set()
    active_leaf_message_provider_id: str | None = None
    # The generation-timing selection lives in the spill's scratch database,
    # or in a private in-memory one when the parse collects into lists.
    with nullcontext(spill.connection()) if spill is not None else closing(sqlite3.connect(":memory:")) as timing_conn:
        from polylogue.sources.prepared_message_sink import GenerationTimings as _Timings

        timings = _Timings(timing_conn)
        _extract_generation_timings(mapping, timings)
        for branch_key, selected in timings.selected():
            timing = _GenerationTiming(**cast("dict[str, Any]", selected))
            if timing.message_provider_id not in emitted_message_ids:
                fallback_owner_id = entries.last_emitted_among(timings.related(branch_key))
                if fallback_owner_id is not None:
                    timing = replace(timing, message_provider_id=fallback_owner_id)
            timings.resolve(timing.message_provider_id, timing.elapsed_duration_ms)
            session_events.append(
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
            )
        # One pass over the final messages writes them and gathers every
        # per-message summary, so neither route holds a second copy.
        for message in _resolved_messages(entries, active_path):
            resolved_duration_ms = timings.resolved_duration_ms(message.provider_message_id)
            if resolved_duration_ms is not None:
                message = message.model_copy(update={"duration_ms": resolved_duration_ms})
            elif timings.repeats_selected_duration(message.provider_message_id):
                message = message.model_copy(update={"duration_ms": None})
            messages.append(message)
            message_count += 1
            session_events.extend(_block_metadata_evidence_events((message,)))
            if message.duration_ms is not None:
                reported_duration_ms = (reported_duration_ms or 0) + message.duration_ms
            if message.model_name:
                model_names.add(message.model_name)
            if active_leaf_message_provider_id is None and message.is_active_leaf:
                active_leaf_message_provider_id = message.provider_message_id
    session_events.extend(_aggregate_result_events(mapping, emitted_message_ids))
    session_events.extend(_message_authorship_events(mapping, emitted_message_ids))
    session_events.extend(
        event
        for event in _message_metadata_evidence_events(mapping)
        if event.source_message_provider_id in emitted_message_ids
    )
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
    if not message_count and payload.get("conversation_id") and payload.get("id") and "mapping" not in payload:
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
    if conversation_model_slug:
        model_names.add(conversation_model_slug)
    models_used = sorted(model_names)

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=str(conv_id or fallback_id),
        title=str(title),
        title_source=title_source,
        session_kind=session_kind,
        provider_project_ref=provider_project_ref,
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=[] if spill is not None else list(messages),
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=[] if spill is not None else list(attachments),
        session_events=[] if spill is not None else list(session_events),
        unit_accounting=admission.close(),
        reported_duration_ms=reported_duration_ms,
        models_used=models_used,
        ingest_flags=ingest_flags,
    )
    if spill is None:
        return session
    # Model validation would copy a scratch-backed sequence into a list, so
    # the spilled collections are attached after construction.
    return session.model_copy(
        update={"messages": messages, "attachments": attachments, "session_events": session_events}
    )


def detection_projection(*, whole_document: bool = False) -> DetectorProjection:
    """Fold every final mapping node using this parser's own shape validators."""
    scalar = DetectorProjection()
    author = DetectorProjection(fields={"role": scalar, "name": scalar, "metadata": scalar})
    content = DetectorProjection(fields=dict.fromkeys(("content_type", "parts", "text", "language"), scalar))
    message = DetectorProjection(
        fields={
            **dict.fromkeys(
                ("id", "create_time", "update_time", "status", "end_turn", "weight", "metadata", "recipient"), scalar
            ),
            "author": author,
            "content": content,
        }
    )
    node = DetectorProjection(
        fields={
            "id": scalar,
            "message": message,
            "parent": scalar,
            "children": DetectorProjection(
                item=scalar, array_fold="all", array_predicate=lambda value: isinstance(value, str)
            ),
        }
    )
    validator = _mapping_nodes_are_valid if whole_document else _mapping_node_shape_is_plausible
    mapping = DetectorProjection(
        item=node,
        mapping_predicate=lambda value: validator({"node": value}),
        mapping_witness={"id": ""} if whole_document else {},
    )
    return DetectorProjection(
        fields={
            **dict.fromkeys(("current_node", "create_time", "conversation_id", "id", "shared_conversation_id"), scalar),
            "mapping": mapping,
            "messages": DetectorProjection(
                item=DetectorProjection(fields={"node_id": scalar, "role": scalar}),
                array_fold="first",
            ),
        }
    )


def whole_document_detection_projection() -> DetectorProjection:
    return detection_projection(whole_document=True)
