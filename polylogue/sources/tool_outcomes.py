"""Resolve canonical tool outcomes from parser evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from polylogue.core.enums import BlockType, Origin, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.origin_specs import tool_outcome_unknown_reasons_for_origin
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSessionEvent


def derive_tool_outcomes(
    messages: list[ParsedMessage], events: Sequence[ParsedSessionEvent], *, origin: Origin
) -> list[ParsedMessage]:
    """Resolve tool outcomes from each origin's structured parser evidence.

    ``is_error``, exit codes and ``outcome_unknown_reason`` are all
    parser-normalized evidence; this function adds no meaning of its own and
    never invents a reason an origin's parser did not derive. Claude Code
    additionally carries outcome fields in its record-level execution event.
    A result without any such evidence is a parser defect and refuses the
    write. A tool-use without a paired result is a recorded interruption and
    receives the distinct, known ``no_result`` outcome.
    """
    sidecar: dict[str, ToolOutcome] = {}
    sidecar_exit_codes: dict[str, int] = {}
    for event in events:
        if origin is not Origin.CLAUDE_CODE_SESSION or event.event_type != "claude_tool_execution_result":
            continue
        tool_id = event.payload.get("tool_use_id")
        if not isinstance(tool_id, str):
            continue
        payload = event.payload
        raw_error = payload.get("is_error", payload.get("isError"))
        raw_exit = payload.get("exit_code", payload.get("exitCode"))
        event_candidates: list[ToolOutcome] = []
        if isinstance(raw_error, bool):
            event_candidates.append(ToolOutcome.ERROR if raw_error else ToolOutcome.OK)
        if isinstance(raw_exit, int) and not isinstance(raw_exit, bool):
            event_candidates.append(ToolOutcome.ERROR if raw_exit else ToolOutcome.OK)
            sidecar_exit_codes[tool_id] = raw_exit
        if not event_candidates:
            continue
        if len(set(event_candidates)) > 1:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        candidate = event_candidates[0]
        previous = sidecar.get(tool_id)
        if previous is not None and previous is not candidate:
            raise ValueError(
                f"tool outcome derivation refused for origin {origin.value!r}: "
                f"conflicting execution evidence for tool_id={tool_id!r}"
            )
        sidecar[tool_id] = candidate

    result_outcomes: dict[str, list[ToolOutcome]] = defaultdict(list)
    for message in messages:
        for block in message.blocks:
            if block.type is not BlockType.TOOL_RESULT or not block.tool_id:
                continue
            result_candidates = _result_candidates(
                block, sidecar, sidecar_exit_codes, unknown_reason=block.outcome_unknown_reason
            )
            distinct = set(result_candidates)
            if any(candidate is ToolOutcome.NO_RESULT for candidate in distinct) or len(distinct) > 1:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"conflicting result evidence for tool_id={block.tool_id!r}"
                )
            outcome = next(iter(distinct), None)
            if outcome is None:
                raise ValueError(
                    f"tool outcome derivation refused for origin {origin.value!r}: "
                    f"unsupported tool_result block shape tool_id={block.tool_id!r}"
                )
            result_outcomes[block.tool_id].append(outcome)

    use_ranks: dict[str, int] = defaultdict(int)
    updated: list[ParsedMessage] = []
    for message in messages:
        blocks: list[ParsedContentBlock] = []
        for block in message.blocks:
            if block.type is BlockType.TOOL_RESULT:
                unknown_reason = (
                    None if _sidecar_resolves_not_reported(block, sidecar) else block.outcome_unknown_reason
                )
                _require_declared_reason(unknown_reason, origin=origin, tool_id=block.tool_id)
                resolved_candidates = _result_candidates(
                    block, sidecar, sidecar_exit_codes, unknown_reason=unknown_reason
                )
                if any(candidate is ToolOutcome.NO_RESULT for candidate in resolved_candidates):
                    raise ValueError(
                        f"tool outcome derivation refused for origin {origin.value!r}: "
                        f"tool_result cannot be no_result for tool_id={block.tool_id!r}"
                    )
                distinct = set(resolved_candidates)
                if len(distinct) > 1:
                    raise ValueError(
                        f"tool outcome derivation refused for origin {origin.value!r}: "
                        f"conflicting result evidence for tool_id={block.tool_id!r}"
                    )
                outcome = next(iter(distinct), None)
                if outcome is None:
                    raise ValueError(
                        f"tool outcome derivation refused for origin {origin.value!r}: "
                        f"unsupported tool_result block shape tool_id={block.tool_id!r}"
                    )
                if outcome is ToolOutcome.UNKNOWN and unknown_reason is None:
                    raise ValueError(
                        f"tool outcome derivation refused for origin {origin.value!r}: "
                        f"unknown outcome without a structural reason for tool_id={block.tool_id!r}"
                    )
                if outcome is not ToolOutcome.UNKNOWN and unknown_reason is not None:
                    raise ValueError(
                        f"tool outcome derivation refused for origin {origin.value!r}: "
                        f"known outcome carries unknown reason for tool_id={block.tool_id!r}"
                    )
                exit_code = block.exit_code
                if exit_code is None and block.tool_id in sidecar_exit_codes:
                    exit_code = sidecar_exit_codes[block.tool_id]
                is_error = None if outcome is ToolOutcome.UNKNOWN else outcome is ToolOutcome.ERROR
                blocks.append(
                    block.model_copy(
                        update={
                            "tool_outcome": outcome,
                            "is_error": is_error,
                            "exit_code": exit_code,
                            "outcome_unknown_reason": unknown_reason if outcome is ToolOutcome.UNKNOWN else None,
                        }
                    )
                )
            elif block.type is BlockType.TOOL_USE:
                if block.tool_id and use_ranks[block.tool_id] < len(result_outcomes.get(block.tool_id, ())):
                    outcome = result_outcomes[block.tool_id][use_ranks[block.tool_id]]
                    use_ranks[block.tool_id] += 1
                elif block.tool_id and block.tool_id in sidecar:
                    outcome = sidecar[block.tool_id]
                else:
                    outcome = ToolOutcome.NO_RESULT
                blocks.append(block.model_copy(update={"tool_outcome": outcome}))
            else:
                blocks.append(block)
        updated.append(message.model_copy(update={"blocks": blocks}))
    return updated


def _result_candidates(
    block: ParsedContentBlock,
    sidecar: dict[str, ToolOutcome],
    sidecar_exit_codes: dict[str, int],
    *,
    unknown_reason: str | None,
) -> list[ToolOutcome]:
    candidates: list[ToolOutcome] = []
    if block.tool_outcome is not None:
        candidates.append(block.tool_outcome)
    if isinstance(block.is_error, bool):
        candidates.append(ToolOutcome.ERROR if block.is_error else ToolOutcome.OK)
    if isinstance(block.exit_code, int) and not isinstance(block.exit_code, bool):
        candidates.append(ToolOutcome.ERROR if block.exit_code else ToolOutcome.OK)
    if block.tool_id in sidecar:
        candidates.append(sidecar[block.tool_id])
    sidecar_exit = sidecar_exit_codes.get(block.tool_id or "")
    if sidecar_exit is not None and not isinstance(block.exit_code, int):
        candidates.append(ToolOutcome.ERROR if sidecar_exit else ToolOutcome.OK)
    if unknown_reason is not None and not any(
        candidate in (ToolOutcome.OK, ToolOutcome.ERROR) for candidate in candidates
    ):
        candidates.append(ToolOutcome.UNKNOWN)
    return candidates


def _sidecar_resolves_not_reported(block: ParsedContentBlock, sidecar: dict[str, ToolOutcome]) -> bool:
    """Return whether matching Claude evidence supersedes an absent inline verdict.

    ``content[].tool_result`` is lowered before Claude Code's record-level
    ``toolUseResult`` sidecar is appended as a session event. That ordering
    legitimately leaves a parser-derived ``NOT_REPORTED`` reason on a block
    whose same-call sidecar later proves a known outcome. Only that narrow
    shape is stale: explicit block verdicts and stronger unknown reasons stay
    authoritative and are handled by the existing conflict checks.
    """
    return (
        block.tool_id is not None
        and block.tool_id in sidecar
        and block.tool_outcome is None
        and block.is_error is None
        and block.exit_code is None
        and block.outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value
    )


def _require_declared_reason(reason: str | None, *, origin: Origin, tool_id: str | None) -> None:
    """Refuse a reason the origin's parsers are not declared to derive.

    A reason with no owner is one nothing in this origin's record structures
    can account for -- the shape this contract exists to keep out of the
    archive. The refusal is per session and clears once the origin declares
    the construct its parser now reads.
    """
    if reason is None:
        return
    declared = tool_outcome_unknown_reasons_for_origin(origin)
    if ToolResultUnknownReason(reason) not in declared:
        raise ValueError(
            f"tool outcome derivation refused for origin {origin.value!r}: "
            f"undeclared unknown reason {reason!r} for tool_id={tool_id!r}"
        )
