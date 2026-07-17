"""Markdown presentation for the shared semantic transcript document."""

from __future__ import annotations

import re
from collections.abc import Iterable

from polylogue.rendering.semantic_card_models import (
    CardOutcomeState,
    SemanticCard,
    SemanticCardPreview,
    SemanticNoticeKind,
    SemanticTranscript,
    TranscriptNotice,
    TranscriptProse,
)


def render_semantic_transcript_markdown(transcript: SemanticTranscript) -> str:
    """Render one ordered semantic transcript without reclassification."""

    rendered: list[str] = []
    for entry in transcript.entries:
        if entry.card is not None:
            rendered.append(render_semantic_card_markdown(entry.card))
        elif entry.prose is not None:
            rendered.append(_render_prose(entry.prose))
        elif entry.notice is not None:
            rendered.append(_render_notice(entry.notice))
    if not rendered:
        return ""
    return "\n\n---\n\n".join(rendered).rstrip() + "\n"


def render_semantic_card_markdown(card: SemanticCard) -> str:
    """Render one semantic card to inspectable Markdown."""

    status = _status_label(card)
    heading = f"### {card.title}" + (f" · {status}" if status else "")
    parts = [heading]
    if card.summary and not any(field.value == card.summary for field in card.fields):
        parts.append(_render_inline_or_fenced("summary", card.summary))
    for field in card.fields:
        parts.append(_render_inline_or_fenced(field.label, field.value))

    source_bits = [f"session:{card.source.session_id}", f"provider:{card.source.provider_family}"]
    _append_source_bit(source_bits, "origin", card.source.origin)
    _append_source_bit(source_bits, "message", card.source.message_id)
    _append_source_bit(source_bits, "block", card.source.block_id)
    if card.source.block_id is None and card.source.block_index is not None:
        source_bits.append(f"block-index:{card.source.block_index}")
    _append_source_bit(source_bits, "tool", card.source.tool_name)
    _append_source_bit(source_bits, "tool-id", card.source.tool_id)
    _append_source_bit(source_bits, "attachment", card.source.attachment_id)
    _append_source_bit(source_bits, "material", card.source.material_origin)
    _append_source_bit(source_bits, "occurred-at", card.source.occurred_at)
    if card.source.duration_ms is not None:
        source_bits.append(f"duration-ms:{card.source.duration_ms}")
    _append_source_bit(source_bits, "parent-message", card.source.parent_message_id)
    if card.source.variant_index is not None:
        source_bits.append(f"variant:{card.source.variant_index}")
    _append_bool_source_bit(source_bits, "active-path", card.source.is_active_path)
    _append_bool_source_bit(source_bits, "active-leaf", card.source.is_active_leaf)
    _append_bool_source_bit(source_bits, "inherited-prefix", card.source.inherited_prefix)
    _append_source_bit(source_bits, "result-message", card.source.result_message_id)
    _append_source_bit(source_bits, "result-block", card.source.result_block_id)
    if card.source.result_block_id is None and card.source.result_block_index is not None:
        source_bits.append(f"result-block-index:{card.source.result_block_index}")
    if card.source.result_duration_ms is not None:
        source_bits.append(f"result-duration-ms:{card.source.result_duration_ms}")
    _append_source_bit(source_bits, "result-material", card.source.result_material_origin)
    _append_bool_source_bit(source_bits, "result-inherited-prefix", card.source.result_inherited_prefix)
    parts.append("- evidence: " + ", ".join(_inline_code(value) for value in source_bits))

    if card.outcome is not None:
        bits = [card.outcome.state.value]
        if card.outcome.is_error is not None:
            bits.append(f"is_error={str(card.outcome.is_error).lower()}")
        if card.outcome.exit_code is not None:
            bits.append(f"exit_code={card.outcome.exit_code}")
        parts.append("- structural outcome: " + ", ".join(_inline_code(value) for value in bits))

    parts.extend(_render_preview(preview) for preview in card.previews)
    parts.extend(f"> Caveat: {caveat}" for caveat in card.caveats)
    return "\n\n".join(parts)


def render_semantic_cards_markdown(cards: Iterable[SemanticCard]) -> str:
    """Render cards without prose for compatibility fallback surfaces."""

    return "\n\n---\n\n".join(render_semantic_card_markdown(card) for card in cards)


def _render_prose(prose: TranscriptProse) -> str:
    label = f"**{prose.role} · {prose.message_type}**"
    if prose.block_type:
        label += f" · {_inline_code(prose.block_type)}"
    refs = [f"message:{prose.message_id}", f"provider:{prose.provider_family}"]
    _append_source_bit(refs, "origin", prose.origin)
    _append_source_bit(refs, "material", prose.material_origin)
    _append_source_bit(refs, "block", prose.block_id)
    if prose.block_id is None and prose.block_index is not None:
        refs.append(f"block-index:{prose.block_index}")
    _append_source_bit(refs, "occurred-at", prose.occurred_at)
    if prose.duration_ms is not None:
        refs.append(f"duration-ms:{prose.duration_ms}")
    _append_source_bit(refs, "parent-message", prose.parent_message_id)
    if prose.variant_index is not None:
        refs.append(f"variant:{prose.variant_index}")
    _append_bool_source_bit(refs, "active-path", prose.is_active_path)
    _append_bool_source_bit(refs, "active-leaf", prose.is_active_leaf)
    _append_bool_source_bit(refs, "inherited-prefix", prose.inherited_prefix)
    rendered_refs = ", ".join(_inline_code(ref) for ref in refs)

    if prose.block_type == "code":
        language = prose.language or "text"
        body = _fenced(prose.text, language=language)
    else:
        body = prose.text
    return f"{label} · {rendered_refs}\n\n{body}"


def _render_notice(notice: TranscriptNotice) -> str:
    if notice.kind is SemanticNoticeKind.EMPTY_THINKING:
        noun = "typed block" if notice.count == 1 else "typed blocks"
        heading = f"**thinking absent · {notice.count} {noun}**"
    else:  # pragma: no cover - closed enum keeps this defensive branch honest
        heading = f"**{notice.kind.value} · {notice.count} sources**"
    refs: list[str] = []
    for source in notice.sources:
        coordinate = source.block_id or f"index:{source.block_index}"
        ref = f"message:{source.message_id}/block:{coordinate}/type:{source.block_type}"
        if source.inherited_prefix is not None:
            ref += f"/inherited:{str(source.inherited_prefix).lower()}"
        refs.append(ref)
    return heading + "\n\n- evidence: " + ", ".join(_inline_code(ref) for ref in refs)


def _append_source_bit(bits: list[str], label: str, value: str | None) -> None:
    if value is not None:
        bits.append(f"{label}:{value}")


def _append_bool_source_bit(bits: list[str], label: str, value: bool | None) -> None:
    if value is not None:
        bits.append(f"{label}:{str(value).lower()}")


def _status_label(card: SemanticCard) -> str | None:
    if card.outcome is None:
        return None
    return {
        CardOutcomeState.SUCCEEDED: "succeeded",
        CardOutcomeState.FAILED: "FAILED",
        CardOutcomeState.UNKNOWN: "outcome unknown",
    }[card.outcome.state]


def _render_inline_or_fenced(label: str, value: str) -> str:
    if "\n" in value or len(value) > 180:
        language = "sh" if label == "command" else "text"
        return f"- {label}:\n\n{_fenced(value, language=language)}"
    return f"- {label}: {_inline_code(value)}"


def _render_preview(preview: SemanticCardPreview) -> str:
    language = {
        "diff": "diff",
        "input": "json",
        "raw_input": "text",
        "output": "text",
        "content": "text",
        "matches": "text",
        "result": "text",
        "raw_result": "text",
    }.get(preview.kind, "text")
    notes: list[str] = []
    if preview.omitted_lines:
        notes.append(f"{preview.omitted_lines} lines omitted")
    if preview.omitted_characters:
        notes.append(f"{preview.omitted_characters} characters omitted")
    if preview.encoding_replacements:
        notes.append(f"{preview.encoding_replacements} encoding replacements")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return f"**{preview.kind}{suffix}**\n\n{_fenced(preview.text, language=language)}"


def _fenced(text: str, *, language: str) -> str:
    fence = "`" * max(3, _longest_backtick_run(text) + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _longest_backtick_run(text: str) -> int:
    return max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)


def _inline_code(value: str) -> str:
    """Render a Markdown code span without corrupting values containing ticks."""

    fence = "`" * max(1, _longest_backtick_run(value) + 1)
    padding = " " if "`" in value or value.startswith(" ") or value.endswith(" ") else ""
    return f"{fence}{padding}{value}{padding}{fence}"


__all__ = [
    "render_semantic_card_markdown",
    "render_semantic_cards_markdown",
    "render_semantic_transcript_markdown",
]
