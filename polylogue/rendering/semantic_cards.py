"""Pure semantic transcript construction over already-hydrated evidence."""

from __future__ import annotations

import difflib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from polylogue.archive.message.models import Message
from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument
from polylogue.insights.topology import SessionTopology, TopologyEdgeKind
from polylogue.rendering.block_models import RenderableBlock, coerce_renderable_blocks
from polylogue.rendering.semantic_card_models import (
    CardOutcomeState,
    LineageDescriptor,
    PreviewStrategy,
    SemanticCard,
    SemanticCardField,
    SemanticCardKind,
    SemanticCardOutcome,
    SemanticCardPreview,
    SemanticCardRawEvidence,
    SemanticCardSource,
    SemanticTranscript,
    SemanticTranscriptEntry,
    TranscriptProse,
)
from polylogue.rendering.semantic_card_registry import card_kind_for_tool, normalize_provider_family

DEFAULT_PREVIEW_HEAD_LINES = 48
DEFAULT_PREVIEW_TAIL_LINES = 16
DEFAULT_PREVIEW_MAX_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class RenderableMessage:
    """Minimal message projection needed by the pure renderer."""

    id: str
    role: str
    message_type: str
    text: str | None
    provider_family: str
    blocks: tuple[RenderableBlock, ...]
    parent_id: str | None = None
    branch_index: int = 0


@dataclass(frozen=True, slots=True)
class _ResultMatch:
    message: RenderableMessage
    block_index: int
    block: RenderableBlock


def coerce_renderable_message(value: Message | Mapping[str, object] | object) -> RenderableMessage:
    """Normalize a domain message or message-shaped fixture."""

    if isinstance(value, Mapping):
        raw_id = value.get("id", value.get("message_id", ""))
        raw_role = value.get("role", "unknown")
        raw_message_type = value.get("message_type", "message")
        raw_text = value.get("text")
        raw_provider = value.get("provider", value.get("provider_family", "unknown"))
        raw_blocks = value.get("blocks", value.get("content_blocks", ()))
        raw_parent_id = value.get("parent_id")
        raw_branch_index = value.get("branch_index", 0)
    else:
        raw_id = getattr(value, "id", getattr(value, "message_id", ""))
        raw_role = getattr(value, "role", "unknown")
        raw_message_type = getattr(value, "message_type", "message")
        raw_text = getattr(value, "text", None)
        raw_provider = getattr(value, "provider", "unknown")
        raw_blocks = getattr(value, "blocks", getattr(value, "content_blocks", ()))
        raw_parent_id = getattr(value, "parent_id", None)
        raw_branch_index = getattr(value, "branch_index", 0)

    raw_role = _enum_like_value(raw_role)
    raw_message_type = _enum_like_value(raw_message_type)
    raw_provider = _enum_like_value(raw_provider)

    blocks_input: Sequence[object] | None
    if isinstance(raw_blocks, Sequence) and not isinstance(raw_blocks, (str, bytes, bytearray)):
        blocks_input = raw_blocks
    else:
        blocks_input = None

    if isinstance(raw_branch_index, bool):
        branch_index = 0
    elif isinstance(raw_branch_index, (int, str)):
        try:
            branch_index = int(raw_branch_index or 0)
        except ValueError:
            branch_index = 0
    else:
        branch_index = 0

    if isinstance(raw_text, str):
        text = raw_text
    elif isinstance(raw_text, (bytes, bytearray)):
        text = bytes(raw_text).decode("utf-8", errors="replace")
    else:
        text = None

    return RenderableMessage(
        id=str(raw_id),
        role=str(raw_role),
        message_type=str(raw_message_type),
        text=text,
        provider_family=normalize_provider_family(raw_provider),
        blocks=coerce_renderable_blocks(blocks_input),
        parent_id=str(raw_parent_id) if raw_parent_id is not None else None,
        branch_index=branch_index,
    )


def _enum_like_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    candidate = getattr(value, "value", None)
    return candidate if isinstance(candidate, str) else value


def lineage_descriptor_from_topology(topology: SessionTopology, *, session_id: str) -> LineageDescriptor | None:
    """Project an already-read topology into pure lineage-card input."""

    target_edge = next((edge for edge in topology.edges if str(edge.child_id) == session_id), None)
    if target_edge is None:
        if topology.cycle_detected:
            return LineageDescriptor(
                session_id=session_id,
                root_session_id=str(topology.root_id),
                cycle_detected=True,
            )
        return None
    relation = target_edge.kind.value
    return LineageDescriptor(
        session_id=session_id,
        root_session_id=str(topology.root_id),
        parent_session_id=str(target_edge.parent_id) if target_edge.parent_id is not None else None,
        parent_native_id=target_edge.parent_native_id,
        relation=relation,
        resolved=target_edge.resolved,
        cycle_detected=topology.cycle_detected,
    )


def build_semantic_transcript(
    messages: Sequence[Message | Mapping[str, object] | object],
    *,
    session_id: str,
    lineage: LineageDescriptor | None = None,
    provider_family: str | None = None,
) -> SemanticTranscript:
    """Build ordered prose/card entries without archive access.

    Result pairing is exact by ``tool_id`` and FIFO within duplicate IDs. Text
    never influences classification or outcome. A caller may supply the session
    origin as ``provider_family`` when legacy message rows carry ``unknown``.
    """

    normalized: list[RenderableMessage] = []
    session_provider = normalize_provider_family(provider_family) if provider_family else None
    for raw in messages:
        message = coerce_renderable_message(raw)
        if session_provider and message.provider_family in {"", "unknown"}:
            message = replace(message, provider_family=session_provider)
        normalized.append(message)

    result_index: dict[str, deque[_ResultMatch]] = defaultdict(deque)
    for message in normalized:
        for block_index, block in enumerate(message.blocks):
            if block.type == BlockType.TOOL_RESULT.value and block.tool_id:
                result_index[block.tool_id].append(_ResultMatch(message, block_index, block))

    consumed_results: set[tuple[str, int]] = set()
    entries: list[SemanticTranscriptEntry] = []
    if lineage is not None:
        entries.append(SemanticTranscriptEntry(card=_build_lineage_card(lineage)))

    for message in normalized:
        if not message.blocks:
            if message.text:
                entries.append(SemanticTranscriptEntry(prose=_prose_for_message(message, message.text)))
            continue

        emitted_text_block = False
        for block_index, block in enumerate(message.blocks):
            if block.type == BlockType.TOOL_USE.value:
                result = _take_result(result_index, block.tool_id)
                if result is not None:
                    consumed_results.add((result.message.id, result.block_index))
                entries.append(
                    SemanticTranscriptEntry(
                        card=_build_tool_card(
                            session_id=session_id,
                            message=message,
                            block_index=block_index,
                            block=block,
                            result=result,
                        )
                    )
                )
                continue
            if block.type == BlockType.TOOL_RESULT.value:
                if (message.id, block_index) not in consumed_results:
                    entries.append(
                        SemanticTranscriptEntry(
                            card=_build_orphan_result_card(
                                session_id=session_id,
                                message=message,
                                block_index=block_index,
                                block=block,
                            )
                        )
                    )
                continue
            if block.type in {BlockType.IMAGE.value, BlockType.DOCUMENT.value, "file"}:
                entries.append(
                    SemanticTranscriptEntry(
                        card=_build_attachment_card(
                            session_id=session_id,
                            message=message,
                            block_index=block_index,
                            block=block,
                        )
                    )
                )
                continue
            if block.text:
                entries.append(
                    SemanticTranscriptEntry(
                        prose=TranscriptProse(
                            message_id=message.id,
                            role=message.role,
                            message_type=message.message_type,
                            text=block.text,
                            block_id=block.block_id,
                            block_type=block.type,
                        )
                    )
                )
                emitted_text_block = True

        # Preserve legacy ordinary prose only when typed blocks did not already
        # carry it. Aggregate tool-use/result text is not emitted a second time.
        if message.text and not emitted_text_block and message.message_type == "message":
            entries.append(SemanticTranscriptEntry(prose=_prose_for_message(message, message.text)))

    return SemanticTranscript(session_id=session_id, entries=tuple(entries))


def _take_result(index: Mapping[str, deque[_ResultMatch]], tool_id: str | None) -> _ResultMatch | None:
    if not tool_id:
        return None
    queue = index.get(tool_id)
    if not queue:
        return None
    return queue.popleft()


def _prose_for_message(message: RenderableMessage, text: str) -> TranscriptProse:
    return TranscriptProse(
        message_id=message.id,
        role=message.role,
        message_type=message.message_type,
        text=text,
    )


def _build_tool_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
    result: _ResultMatch | None,
) -> SemanticCard:
    result_block = result.block if result is not None else None
    kind = card_kind_for_tool(
        provider_family=message.provider_family,
        tool_name=block.tool_name,
        semantic_type=block.semantic_type,
    )
    source = SemanticCardSource(
        session_id=session_id,
        provider_family=message.provider_family,
        message_id=message.id,
        block_id=block.block_id,
        block_index=block_index,
        tool_name=block.tool_name,
        tool_id=block.tool_id,
        result_message_id=result.message.id if result is not None else None,
        result_block_id=result_block.block_id if result_block is not None else None,
        result_block_index=result.block_index if result is not None else None,
    )
    outcome, outcome_caveats = _outcome_from_result(result_block)
    common_caveats: list[str] = list(outcome_caveats)
    if block.tool_input_raw:
        common_caveats.append("tool input was retained as raw text because it was not valid JSON")
    if result_block is None:
        common_caveats.append("no structurally paired tool result is present")

    if kind is SemanticCardKind.SHELL:
        return _build_shell_card(block, result_block, source, outcome, tuple(common_caveats))
    if kind is SemanticCardKind.FILE_EDIT:
        return _build_file_edit_card(block, result_block, source, outcome, tuple(common_caveats))
    if kind is SemanticCardKind.TASK:
        return _build_task_card(block, result_block, source, outcome, tuple(common_caveats))
    return _build_fallback_card(block, result_block, source, outcome, tuple(common_caveats))


def _build_shell_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
) -> SemanticCard:
    command = _first_scalar(block.tool_input, ("command", "cmd", "script"))
    fields = [SemanticCardField("tool", block.tool_name or "unknown")]
    if command:
        fields.append(SemanticCardField("command", command))
    previews = _result_previews(result, kind="output")
    return SemanticCard(
        kind=SemanticCardKind.SHELL,
        title="Shell command",
        summary=command,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(caveats + _preview_caveats(previews)),
    )


def _build_file_edit_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
) -> SemanticCard:
    path = _first_scalar(block.tool_input, ("file_path", "path", "file", "filename"))
    diff_text = _edit_diff(block.tool_input, path=path)
    previews: list[SemanticCardPreview] = []
    local_caveats = list(caveats)
    if diff_text:
        previews.append(_bounded_preview(diff_text, kind="diff"))
    else:
        local_caveats.append("no exact diff could be constructed from the available tool input")
    if result is not None and result.text:
        previews.append(
            _bounded_preview(
                result.text,
                kind="result",
                encoding_replacements=result.text_encoding_replacements,
            )
        )
    fields = [SemanticCardField("tool", block.tool_name or "unknown")]
    if path:
        fields.append(SemanticCardField("path", path))
    return SemanticCard(
        kind=SemanticCardKind.FILE_EDIT,
        title="File edit",
        summary=path,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=tuple(previews),
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(tuple(local_caveats) + _preview_caveats(tuple(previews))),
    )


def _build_task_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
) -> SemanticCard:
    prompt = _first_scalar(block.tool_input, ("prompt", "task", "instructions", "description"))
    agent_type = _first_scalar(block.tool_input, ("subagent_type", "agent_type", "agent"))
    fields = [SemanticCardField("tool", block.tool_name or "unknown")]
    if agent_type:
        fields.append(SemanticCardField("agent", agent_type))
    if prompt:
        fields.append(SemanticCardField("request", prompt))
    previews = _result_previews(result, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.TASK,
        title="Task / subagent",
        summary=prompt or agent_type,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(caveats + _preview_caveats(previews)),
    )


def _build_fallback_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
) -> SemanticCard:
    input_preview = _input_preview(block)
    previews = (() if input_preview is None else (input_preview,)) + _result_previews(result, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.FALLBACK,
        title=f"Tool evidence · {block.tool_name or 'unknown'}",
        source=source,
        outcome=outcome,
        fields=(SemanticCardField("tool", block.tool_name or "unknown"),),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(
            ("tool classification is unknown; raw evidence is shown without a guessed semantic card type",)
            + caveats
            + _preview_caveats(previews)
        ),
    )


def _build_orphan_result_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
) -> SemanticCard:
    outcome, outcome_caveats = _outcome_from_result(block)
    previews = _result_previews(block, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.FALLBACK,
        title="Unpaired tool result",
        source=SemanticCardSource(
            session_id=session_id,
            provider_family=message.provider_family,
            result_message_id=message.id,
            result_block_id=block.block_id,
            result_block_index=block_index,
            tool_id=block.tool_id,
        ),
        outcome=outcome,
        previews=previews,
        raw_evidence=SemanticCardRawEvidence(result_preview=previews[0] if previews else None),
        caveats=_deduplicate(
            ("no matching tool-use block is present; result semantics were not guessed",)
            + outcome_caveats
            + _preview_caveats(previews)
        ),
    )


def _build_attachment_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
) -> SemanticCard:
    fields: list[SemanticCardField] = []
    if block.name:
        fields.append(SemanticCardField("name", block.name))
    if block.mime_type:
        fields.append(SemanticCardField("media type", block.mime_type))
    if block.url:
        fields.append(SemanticCardField("url", block.url))
    return SemanticCard(
        kind=SemanticCardKind.ATTACHMENT,
        title="Attachment",
        summary=block.name or block.mime_type or block.type,
        source=SemanticCardSource(
            session_id=session_id,
            provider_family=message.provider_family,
            message_id=message.id,
            block_id=block.block_id,
            block_index=block_index,
        ),
        fields=tuple(fields),
        caveats=("attachment bytes are not embedded in the transcript card",),
    )


def _build_lineage_card(lineage: LineageDescriptor) -> SemanticCard:
    fields = [
        SemanticCardField("session", f"session:{lineage.session_id}"),
        SemanticCardField("root", f"session:{lineage.root_session_id}"),
        SemanticCardField("relation", lineage.relation),
    ]
    if lineage.parent_session_id:
        fields.append(SemanticCardField("parent", f"session:{lineage.parent_session_id}"))
    elif lineage.parent_native_id:
        fields.append(SemanticCardField("native parent", lineage.parent_native_id))
    caveats: list[str] = []
    if not lineage.resolved:
        caveats.append("the provider-native parent has not resolved to a stored session")
    if lineage.relation == TopologyEdgeKind.UNKNOWN.value:
        caveats.append("the parent edge is structural but its narrower lineage relation is unknown")
    if lineage.cycle_detected:
        caveats.append("a cycle was detected in the session topology")
    return SemanticCard(
        kind=SemanticCardKind.LINEAGE,
        title=f"Lineage boundary · {lineage.relation}",
        summary=f"session:{lineage.session_id}",
        source=SemanticCardSource(session_id=lineage.session_id),
        fields=tuple(fields),
        caveats=tuple(caveats),
    )


def _outcome_from_result(result: RenderableBlock | None) -> tuple[SemanticCardOutcome, tuple[str, ...]]:
    if result is None:
        return SemanticCardOutcome(CardOutcomeState.UNKNOWN), ()
    is_error = result.tool_result_is_error
    exit_code = result.tool_result_exit_code
    if is_error is True or (exit_code is not None and exit_code != 0):
        caveats: tuple[str, ...] = ()
        if is_error is False and exit_code is not None and exit_code != 0:
            caveats = ("source outcome fields disagree; non-zero exit code is treated as failure",)
        elif is_error is True and exit_code == 0:
            caveats = ("source outcome fields disagree; explicit error flag is treated as failure",)
        return SemanticCardOutcome(CardOutcomeState.FAILED, is_error=is_error, exit_code=exit_code), caveats
    if is_error is False or exit_code == 0:
        return SemanticCardOutcome(CardOutcomeState.SUCCEEDED, is_error=is_error, exit_code=exit_code), ()
    return (
        SemanticCardOutcome(CardOutcomeState.UNKNOWN, is_error=is_error, exit_code=exit_code),
        ("tool result exists but carries no structural success/failure outcome",),
    )


def _first_scalar(document: JSONDocument | None, keys: Iterable[str]) -> str | None:
    if document is None:
        return None
    for key in keys:
        value = document.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _edit_diff(document: JSONDocument | None, *, path: str | None) -> str | None:
    if document is None:
        return None
    explicit = _first_scalar(document, ("patch", "diff"))
    if explicit:
        return explicit
    old = _first_scalar(document, ("old_string", "old_text", "before"))
    new = _first_scalar(document, ("new_string", "new_text", "after", "content"))
    if new is None:
        return None
    label = path or "file"
    lines = difflib.unified_diff(
        (old or "").splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        lineterm="",
    )
    return "\n".join(lines)


def _input_preview(block: RenderableBlock) -> SemanticCardPreview | None:
    if block.tool_input is not None:
        rendered = json.dumps(block.tool_input, indent=2, sort_keys=True, ensure_ascii=False)
        return _bounded_preview(rendered, kind="input", encoding_replacements=rendered.count("\ufffd"))
    if block.tool_input_raw is not None:
        return _bounded_preview(
            block.tool_input_raw,
            kind="raw_input",
            encoding_replacements=block.tool_input_raw.count("\ufffd"),
        )
    return None


def _result_previews(result: RenderableBlock | None, *, kind: str) -> tuple[SemanticCardPreview, ...]:
    if result is None or not result.text:
        return ()
    return (
        _bounded_preview(
            result.text,
            kind=kind,
            encoding_replacements=result.text_encoding_replacements,
        ),
    )


def _raw_evidence(block: RenderableBlock, result: RenderableBlock | None) -> SemanticCardRawEvidence:
    previews = _result_previews(result, kind="raw_result")
    return SemanticCardRawEvidence(
        tool_input=block.tool_input,
        tool_input_raw=block.tool_input_raw,
        result_preview=previews[0] if previews else None,
    )


def _bounded_preview(
    text: str,
    *,
    kind: str,
    encoding_replacements: int = 0,
    head_lines: int = DEFAULT_PREVIEW_HEAD_LINES,
    tail_lines: int = DEFAULT_PREVIEW_TAIL_LINES,
    max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
) -> SemanticCardPreview:
    lines = text.splitlines()
    line_count = len(lines)
    if line_count > head_lines + tail_lines:
        selected = lines[:head_lines] + lines[-tail_lines:]
        rendered = "\n".join(selected)
        if len(rendered) <= max_chars:
            return SemanticCardPreview(
                kind=kind,
                text=rendered,
                line_count=line_count,
                omitted_lines=line_count - len(selected),
                truncated=True,
                strategy=PreviewStrategy.HEAD_TAIL,
                encoding_replacements=encoding_replacements,
            )
    if len(text) > max_chars:
        head_chars = max_chars * 3 // 4
        tail_chars = max_chars - head_chars
        rendered = text[:head_chars] + "\n… [character-bounded preview] …\n" + text[-tail_chars:]
        return SemanticCardPreview(
            kind=kind,
            text=rendered,
            line_count=line_count,
            omitted_characters=len(text) - max_chars,
            truncated=True,
            strategy=PreviewStrategy.CHARACTER_BOUNDED,
            encoding_replacements=encoding_replacements,
        )
    return SemanticCardPreview(
        kind=kind,
        text=text,
        line_count=line_count,
        strategy=PreviewStrategy.FULL,
        encoding_replacements=encoding_replacements,
    )


def _preview_caveats(previews: Sequence[SemanticCardPreview]) -> tuple[str, ...]:
    caveats: list[str] = []
    for preview in previews:
        if preview.omitted_lines:
            caveats.append(f"{preview.omitted_lines} {preview.kind} lines are omitted from the bounded preview")
        if preview.omitted_characters:
            caveats.append(
                f"{preview.omitted_characters} {preview.kind} characters are omitted from the bounded preview"
            )
        if preview.encoding_replacements:
            caveats.append(
                f"{preview.encoding_replacements} invalid UTF-8 sequence(s) were rendered with replacement characters"
            )
    return tuple(caveats)


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "build_semantic_transcript",
    "coerce_renderable_message",
    "lineage_descriptor_from_topology",
    "RenderableMessage",
]
