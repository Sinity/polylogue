"""Tool-call parsing helpers for semantic actions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from polylogue.archive.viewport.viewports import ToolCall, ToolCategory, ToolResult, classify_tool
from polylogue.core.enums import ActionResultState, Origin, ToolOutcome
from polylogue.core.json import JSONDocument, json_document

#: Structural pass/fail verdict for a tool_result, or "unknown" for a
#: pre-admission block that has not passed through the archive writer.
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

    This is the canonical implementation of the exit_code/is_error precedence
    rule. Hydrated block consumers additionally preserve the canonical verdict
    through :func:`tool_result_block_outcome`.
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
    canonical = block.get("tool_outcome")
    if canonical == ToolOutcome.OK.value:
        return "ok"
    if canonical == ToolOutcome.ERROR.value:
        return "failed"
    if canonical == ToolOutcome.NO_RESULT.value:
        return "unknown"
    if canonical == ToolOutcome.UNKNOWN.value:
        return "unknown"
    is_error = _block_outcome_int(block, "tool_result_is_error", legacy_key="is_error")
    exit_code = _block_outcome_int(block, "tool_result_exit_code", legacy_key="exit_code")
    return tool_result_outcome(is_error, exit_code)


def build_tool_calls_from_content_blocks(
    *,
    origin: Origin | str | None,
    content_blocks: Sequence[Mapping[str, object]],
    results_by_use_block_id: Mapping[int, Sequence[ToolResult]] | None = None,
    result_state_by_use_block_id: Mapping[int, ActionResultState] | None = None,
) -> tuple[ToolCall, ...]:
    """Normalize canonical ToolCall viewports from content blocks.

    Per-message callers retain the one-result FIFO rule. Archive callers pass
    the canonical context-filtered result association and aggregate state.
    """
    tool_result_blocks: dict[str, list[Mapping[str, object]]] = {}
    tool_use_blocks: list[Mapping[str, object]] = []
    for block in content_blocks:
        block_type = str(block.get("type"))
        if block_type == "tool_result":
            tool_id = block.get("tool_id")
            if isinstance(tool_id, str) and tool_id:
                tool_result_blocks.setdefault(tool_id, []).append(block)
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
    use_ranks: dict[str, int] = {}
    for block in tool_use_blocks:
        tool_id = block.get("tool_id")
        outcome_tool_id = tool_id if isinstance(tool_id, str) and tool_id else None
        canonical_results = results_by_use_block_id is not None and id(block) in results_by_use_block_id
        results: tuple[ToolResult, ...] = ()
        result_blocks: Sequence[Mapping[str, object]] = ()
        if canonical_results and results_by_use_block_id is not None:
            results = tuple(results_by_use_block_id[id(block)])
        elif outcome_tool_id is not None:
            rank = use_ranks.get(outcome_tool_id, 0)
            use_ranks[outcome_tool_id] = rank + 1
            pending = tool_result_blocks.get(outcome_tool_id, ())
            result_blocks = pending[rank : rank + 1]
        else:
            result_blocks = ()
        name = block.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
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
        if not canonical_results:
            results = tuple(
                ToolResult(
                    block_id=str(result.get("block_id")) if result.get("block_id") is not None else None,
                    output_text=text if isinstance(text := result.get("text"), str) else None,
                    is_error=_block_outcome_int(result, "tool_result_is_error", legacy_key="is_error"),
                    exit_code=_block_outcome_int(result, "tool_result_exit_code", legacy_key="exit_code"),
                    outcome={"ok": ToolOutcome.OK, "failed": ToolOutcome.ERROR, "unknown": ToolOutcome.UNKNOWN}[
                        tool_result_block_outcome(result)
                    ],
                    outcome_unknown_reason=(
                        str(result["tool_result_outcome_unknown_reason"])
                        if result.get("tool_result_outcome_unknown_reason") is not None
                        else None
                    ),
                )
                for result in result_blocks
            )
        outcome = (
            "failed"
            if any(result.outcome == ToolOutcome.ERROR for result in results)
            else "unknown"
            if not results or any(result.outcome == "unknown" for result in results)
            else "ok"
        )
        result_state = (
            ActionResultState.NO_RESULT
            if not results
            else ActionResultState.OUTCOME_ERROR
            if outcome == "failed"
            else ActionResultState.OUTCOME_UNKNOWN
            if outcome == "unknown"
            else ActionResultState.OUTCOME_SUCCESS
        )
        if result_state_by_use_block_id is not None and id(block) in result_state_by_use_block_id:
            result_state = result_state_by_use_block_id[id(block)]
            outcome = {
                ActionResultState.NO_RESULT: "unknown",
                ActionResultState.OUTCOME_UNKNOWN: "unknown",
                ActionResultState.OUTCOME_SUCCESS: "ok",
                ActionResultState.OUTCOME_ERROR: "failed",
            }[result_state]
        success = {"ok": True, "failed": False, "unknown": None}[outcome]
        result_texts = [result.output_text for result in results]
        output = (
            "\n".join(text for text in result_texts if text is not None)
            if any(text is not None for text in result_texts)
            else None
        )
        calls.append(
            ToolCall(
                name=name,
                id=outcome_tool_id,
                input=normalized_input,
                output=output,
                success=success,
                result_state=result_state,
                results=results,
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
