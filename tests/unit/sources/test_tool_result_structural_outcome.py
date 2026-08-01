"""Pins the structural-only outcome invariant for ``blocks.tool_result_is_error``.

CLAUDE.md / docs/internals.md state this as a hard rule: ``is_error`` is read
from the provider's structural ``tool_result`` segment field and is NEVER
regex-guessed from the result's prose text. A mutation audit of
``content_blocks_from_segments`` (polylogue/sources/parsers/base_support.py)
found the existing coverage (``test_tool_result_missing_is_error_gets_not_reported_reason``
in test_claude_code_unread_wire_fields.py) only exercises innocuous text
("hi") that happens not to contain the word "error" -- a prose-sniffing
regression that keyed off substrings like "error"/"fail" in the result text
would still pass that test. This file adds the missing case: prose that reads
as failure-shaped but carries no structural ``is_error`` field must still
resolve to ``is_error=None`` / ``outcome_unknown_reason=NOT_REPORTED``, and
prose that reads as success-shaped alongside a structural ``is_error=true``
must still resolve to ``is_error=True`` -- the text must never override or
substitute for the structural signal in either direction.
"""

from __future__ import annotations

from polylogue.core.enums import BlockType, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, content_blocks_from_segments


def _tool_result_block(content: list[object]) -> ParsedContentBlock:
    blocks = content_blocks_from_segments(content)
    result_blocks = [b for b in blocks if b.type is BlockType.TOOL_RESULT]
    assert len(result_blocks) == 1
    return result_blocks[0]


def test_failure_shaped_prose_without_structural_field_stays_unknown() -> None:
    """Result text screaming failure, but no structural ``is_error`` key, must
    NOT be regex-guessed to ``is_error=True`` -- it stays a structurally
    unknown NOT_REPORTED outcome, exactly like innocuous text would."""
    block = _tool_result_block(
        [
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": "Error: command failed with exit code 1, fatal error occurred",
            }
        ]
    )
    assert block.is_error is None
    assert block.outcome_unknown_reason == ToolResultUnknownReason.NOT_REPORTED.value


def test_success_shaped_prose_with_structural_error_field_is_still_an_error() -> None:
    """Result text reading as success, but a structural ``is_error: true``, must
    resolve to ``is_error=True`` -- prose never overrides the structural
    signal in the other direction either."""
    block = _tool_result_block(
        [
            {
                "type": "tool_result",
                "tool_use_id": "tool-2",
                "content": "Done! Success, completed with no errors, all good.",
                "is_error": True,
            }
        ]
    )
    assert block.is_error is True
    assert block.outcome_unknown_reason is None


def test_structural_false_is_trusted_even_with_failure_shaped_prose() -> None:
    """A structural ``is_error: false`` is trusted verbatim even when the
    prose itself uses failure-adjacent words (e.g. reporting on a caught
    exception it successfully handled)."""
    block = _tool_result_block(
        [
            {
                "type": "tool_result",
                "tool_use_id": "tool-3",
                "content": "Handled 3 errors gracefully; no failures remain.",
                "is_error": False,
            }
        ]
    )
    assert block.is_error is False
    assert block.outcome_unknown_reason is None
