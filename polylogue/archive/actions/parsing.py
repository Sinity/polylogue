"""Tool-call parsing helpers for semantic actions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from polylogue.archive.viewport.viewports import ToolCall, ToolCategory, classify_tool
from polylogue.core.enums import Origin
from polylogue.core.json import JSONDocument, json_document

#: Structural pass/fail verdict for a tool_result, or "unknown" when the
#: origin never populated the keystone columns for this result at all.
ToolResultOutcome = Literal["ok", "failed", "unknown"]


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


def _extract_first_string(mapping: Mapping[str, object], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = _clean_str(mapping.get(field))
        if value is not None:
            return value
    return None


def _normalized_mapping(value: object) -> JSONDocument:
    if isinstance(value, Mapping):
        return json_document(dict(value))
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return json_document(dict(parsed))
    return {}


def _tool_category_from_semantic(value: object) -> ToolCategory | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ToolCategory(value)
    except ValueError:
        return None


def _block_outcome_int(block: Mapping[str, object], key: str, *, legacy_key: str) -> int | None:
    """Read a structural outcome int, preferring the hydrated storage-column
    name (e.g. ``tool_result_is_error``) and falling back to the bare
    parse-time field name (``is_error``) that :class:`ParsedContentBlock`
    and some pre-hydration block dicts use -- the same dual-key convention
    ``rendering/block_models.py`` already applies when reading blocks that
    may not have passed through storage hydration.
    """
    value = block.get(key, block.get(legacy_key))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def tool_result_outcome(is_error: int | None, exit_code: int | None) -> ToolResultOutcome:
    """Structural pass/fail from the provider-reported tool_result columns.

    ``exit_code`` is authoritative when present (0 = ok, anything else =
    failed); otherwise the boolean ``is_error`` flag decides. Both ``None``
    means the origin never populated these keystone columns (index schema
    v16: ``blocks.tool_result_is_error`` / ``tool_result_exit_code``) for
    this result -- returns ``"unknown"``, never guessed from prose.

    Coverage is origin-gated, not universal (polylogue-9e5.3 column-honesty
    audit, ``.agent/scratch/research/2026-07-09-substrate-honesty-audit.md``):
    ``tool_result_is_error`` is well-populated only for claude-code-session
    (44.8%) and claude-ai-export (100% of a small volume); it is 0% for
    chatgpt-export, hermes-session, and aistudio-drive. ``tool_result_exit_code``
    is *only ever* populated for codex-session, and just 14.2% of even that.
    Callers must not treat an "unknown" result as a negative claim -- for
    most origins it is the expected, honest outcome, not a gap.

    This is the single canonical implementation of the exit_code/is_error
    precedence rule; :func:`polylogue.insights.transforms._tool_status`
    delegates to it to avoid two independently-maintained copies drifting.
    """

    if exit_code is not None:
        return "ok" if exit_code == 0 else "failed"
    if is_error is not None:
        return "failed" if is_error else "ok"
    return "unknown"


def tool_result_block_outcome(block: Mapping[str, object]) -> ToolResultOutcome:
    """Structural pass/fail for one ``tool_result``-typed block dict.

    Reads both the hydrated storage-column keys (``tool_result_is_error``/
    ``tool_result_exit_code``) and the bare parse-time keys (``is_error``/
    ``exit_code``) a block dict may carry, then applies
    :func:`tool_result_outcome`'s exit_code/is_error precedence. Used both
    by :func:`build_tool_calls_from_content_blocks` (per-message pairing)
    and by session-wide tool_id/result scans (e.g.
    ``archive/session/runtime.py::_terminal_state``) that pair a tool_use in
    one message with its tool_result in a later message -- the common shape
    for Claude/Codex-style transcripts, which per-message pairing alone
    misses.
    """
    is_error = _block_outcome_int(block, "tool_result_is_error", legacy_key="is_error")
    exit_code = _block_outcome_int(block, "tool_result_exit_code", legacy_key="exit_code")
    return tool_result_outcome(is_error, exit_code)


def build_tool_calls_from_content_blocks(
    *,
    origin: Origin | str | None,
    content_blocks: Sequence[Mapping[str, object]],
) -> tuple[ToolCall, ...]:
    """Normalize canonical ToolCall viewports from content blocks."""
    tool_result_outputs: dict[str, str] = {}
    tool_result_outcomes: dict[str, ToolResultOutcome] = {}
    tool_use_blocks: list[Mapping[str, object]] = []
    for block in content_blocks:
        block_type = str(block.get("type"))
        if block_type == "tool_result":
            tool_id = block.get("tool_id")
            text = block.get("text")
            if isinstance(tool_id, str) and tool_id and isinstance(text, str) and text:
                tool_result_outputs.setdefault(tool_id, text)
            if isinstance(tool_id, str) and tool_id:
                tool_result_outcomes.setdefault(tool_id, tool_result_block_outcome(block))
            continue
        if block_type != "tool_use":
            continue
        tool_use_blocks.append(block)

    if not tool_use_blocks:
        return ()

    normalized_origin = (
        origin if isinstance(origin, Origin) else Origin.from_string(origin) if origin is not None else None
    )
    calls: list[ToolCall] = []
    for block in tool_use_blocks:
        name = block.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        tool_id = block.get("tool_id")
        normalized_input = _normalized_mapping(block.get("tool_input"))
        semantic_category = _tool_category_from_semantic(block.get("semantic_type"))
        classified_category = classify_tool(name, normalized_input)
        if semantic_category is None or semantic_category is ToolCategory.OTHER:
            category = classified_category
        else:
            category = semantic_category
        raw = {
            "block_id": block.get("block_id"),
            "block_index": block.get("block_index"),
            "message_id": block.get("message_id"),
            "session_id": block.get("session_id"),
            "type": block.get("type"),
            "tool_name": name,
            "tool_id": tool_id,
            "tool_input": normalized_input,
            "media_type": block.get("media_type"),
            "metadata": _normalized_mapping(block.get("metadata")),
            "semantic_type": block.get("semantic_type"),
            "text": block.get("text"),
        }
        outcome_tool_id = tool_id if isinstance(tool_id, str) and tool_id else None
        outcome = tool_result_outcomes.get(outcome_tool_id, "unknown") if outcome_tool_id else "unknown"
        success = {"ok": True, "failed": False, "unknown": None}[outcome]
        calls.append(
            ToolCall(
                name=name,
                id=outcome_tool_id,
                input=normalized_input,
                output=tool_result_outputs.get(tool_id) if isinstance(tool_id, str) else None,
                success=success,
                category=category,
                origin=normalized_origin,
                raw=raw,
            )
        )
    return tuple(calls)


__all__ = [
    "ToolResultOutcome",
    "_clean_str",
    "_extract_first_string",
    "build_tool_calls_from_content_blocks",
    "tool_result_block_outcome",
    "tool_result_outcome",
]
