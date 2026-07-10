"""Markdown rendering for pure semantic transcript cards."""

from __future__ import annotations

import re
from collections.abc import Iterable

from polylogue.rendering.semantic_card_models import (
    CardOutcomeState,
    SemanticCard,
    SemanticCardPreview,
    SemanticTranscript,
    TranscriptProse,
)


def render_semantic_transcript_markdown(transcript: SemanticTranscript) -> str:
    """Render an ordered semantic transcript without archive access."""

    rendered: list[str] = []
    for entry in transcript.entries:
        if entry.card is not None:
            rendered.append(render_semantic_card_markdown(entry.card))
        elif entry.prose is not None:
            rendered.append(_render_prose(entry.prose))
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

    source_bits = [f"session:{card.source.session_id}"]
    if card.source.message_id:
        source_bits.append(f"message:{card.source.message_id}")
    if card.source.block_id:
        source_bits.append(f"block:{card.source.block_id}")
    elif card.source.block_index is not None:
        source_bits.append(f"block-index:{card.source.block_index}")
    if card.source.result_message_id:
        source_bits.append(f"result-message:{card.source.result_message_id}")
    if card.source.result_block_id:
        source_bits.append(f"result-block:{card.source.result_block_id}")
    elif card.source.result_block_index is not None:
        source_bits.append(f"result-block-index:{card.source.result_block_index}")
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
    """Render cards without prose for server-side fallback surfaces."""

    return "\n\n---\n\n".join(render_semantic_card_markdown(card) for card in cards)


def _render_prose(prose: TranscriptProse) -> str:
    label = f"**{prose.role} · {prose.message_type}**"
    refs = [f"message:{prose.message_id}"]
    if prose.block_id:
        refs.append(f"block:{prose.block_id}")
    if prose.block_type:
        label += f" · {_inline_code(prose.block_type)}"
    rendered_refs = ", ".join(_inline_code(ref) for ref in refs)
    return f"{label} · {rendered_refs}\n\n{prose.text}"


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
    # CommonMark requires padding when content starts or ends with a backtick,
    # and padding is harmless for other embedded-tick values.
    padding = " " if "`" in value or value.startswith(" ") or value.endswith(" ") else ""
    return f"{fence}{padding}{value}{padding}{fence}"


__all__ = [
    "render_semantic_card_markdown",
    "render_semantic_cards_markdown",
    "render_semantic_transcript_markdown",
]
