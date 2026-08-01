from __future__ import annotations

import pytest

from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.types import (
    MessageType,
    message_type_sql_values,
    normalize_message_types,
    validate_message_type_filter,
)
from polylogue.core.enums import BlockType, MaterialOrigin, Role


def test_message_type_normalization_accepts_enums_strings_lists_and_unknowns() -> None:
    assert MessageType.normalize(MessageType.SUMMARY) is MessageType.SUMMARY
    assert MessageType.normalize("tool-use") is MessageType.TOOL_USE
    assert MessageType.normalize(" tool_result ") is MessageType.TOOL_RESULT
    assert MessageType.normalize("thinking") is MessageType.THINKING
    assert MessageType.normalize("context") is MessageType.CONTEXT
    assert MessageType.normalize("protocol") is MessageType.PROTOCOL
    assert MessageType.normalize(None) is MessageType.MESSAGE
    assert MessageType.normalize("") is MessageType.MESSAGE
    assert MessageType.normalize("unknown") is MessageType.MESSAGE

    assert normalize_message_types(None) == ()
    assert normalize_message_types("summary") == (MessageType.SUMMARY,)
    assert normalize_message_types([MessageType.TOOL_USE, "tool-result"]) == (
        MessageType.TOOL_USE,
        MessageType.TOOL_RESULT,
    )
    assert message_type_sql_values(["summary", "thinking", "protocol"]) == ("summary", "thinking", "protocol")


def test_message_type_filter_validation_rejects_unknown_user_input() -> None:
    assert validate_message_type_filter("tool-use") is MessageType.TOOL_USE
    assert validate_message_type_filter("message") is MessageType.MESSAGE

    with pytest.raises(ValueError, match="Unknown message type"):
        validate_message_type_filter("summmary")


def test_plain_user_message_does_not_imply_human_authorship() -> None:
    """Absence of runtime markers is not positive evidence of human authorship."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="Please inspect the failing tests.",
        )
        is MaterialOrigin.UNKNOWN
    )


# --- classify_material_origin: direct branch coverage (polylogue-nqx2) ---
#
# classify_material_origin has 8 top-level `if` branches (plus the UNKNOWN
# fall-through covered above). Every branch below is exercised directly, with
# a minimal input constructed to hit exactly that branch's condition, and each
# was mutation-verified against the corresponding line in
# polylogue/archive/message/artifacts.py before landing (see PR body / bead
# notes for the exact mutation + failure evidence per case).


def test_tool_role_yields_tool_result_regardless_of_message_type() -> None:
    """Branch 1a: `role is Role.TOOL` alone routes to TOOL_RESULT."""
    assert (
        classify_material_origin(
            role=Role.TOOL,
            message_type=MessageType.MESSAGE,
            text="some tool payload",
        )
        is MaterialOrigin.TOOL_RESULT
    )


def test_tool_result_message_type_yields_tool_result_regardless_of_role() -> None:
    """Branch 1b: `message_type is TOOL_RESULT` alone routes to TOOL_RESULT."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.TOOL_RESULT,
            text="some tool payload",
        )
        is MaterialOrigin.TOOL_RESULT
    )


def test_all_tool_result_blocks_yield_tool_result_even_with_mismatched_message_type() -> None:
    """Branch 2 (the polylogue-nqx2 gap): block_types are all TOOL_RESULT but
    message_type disagrees (parser-inconsistency case). This is the exact
    scenario `classify_block_message_type` should have normalized away, but
    when it hasn't, classify_material_origin must still recover TOOL_RESULT
    rather than silently falling through to UNKNOWN.
    """
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="",
            block_types=(BlockType.TOOL_RESULT, BlockType.TOOL_RESULT),
        )
        is MaterialOrigin.TOOL_RESULT
    )


def test_generated_analysis_pack_marker_is_classified_directly() -> None:
    """Branch 3: text starting with a generated-analysis-pack marker."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="Generate a retrospective for: session abc123",
        )
        is MaterialOrigin.GENERATED_ANALYSIS_PACK
    )


def test_generated_context_pack_marker_is_classified_directly() -> None:
    """Branch 4: text starting with a generated-context-pack marker."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="Generate all artifacts for the change described below.",
        )
        is MaterialOrigin.GENERATED_CONTEXT_PACK
    )


def test_summary_message_type_is_generated_context_pack() -> None:
    """Branch 5: message_type SUMMARY (independent of text shape)."""
    assert (
        classify_material_origin(
            role=Role.ASSISTANT,
            message_type=MessageType.SUMMARY,
            text="This session covered ...",
        )
        is MaterialOrigin.GENERATED_CONTEXT_PACK
    )


def test_context_message_type_is_runtime_context() -> None:
    """Branch 6: message_type CONTEXT (independent of text shape)."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.CONTEXT,
            text="<environment_context>...</environment_context>",
        )
        is MaterialOrigin.RUNTIME_CONTEXT
    )


def test_protocol_with_operator_marker_is_operator_command() -> None:
    """Branch 7a: message_type PROTOCOL with an operator-command marker in text."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.PROTOCOL,
            text="<command-name>fix-tests</command-name>",
        )
        is MaterialOrigin.OPERATOR_COMMAND
    )


def test_protocol_without_operator_marker_is_runtime_protocol() -> None:
    """Branch 7b: message_type PROTOCOL with no operator-command marker in text."""
    assert (
        classify_material_origin(
            role=Role.USER,
            message_type=MessageType.PROTOCOL,
            text="<task-notification>build finished</task-notification>",
        )
        is MaterialOrigin.RUNTIME_PROTOCOL
    )


def test_assistant_role_with_message_type_is_assistant_authored() -> None:
    """Branch 8: assistant role + MESSAGE/TOOL_USE type, with no earlier branch
    matching, is positive evidence of model authorship."""
    assert (
        classify_material_origin(
            role=Role.ASSISTANT,
            message_type=MessageType.MESSAGE,
            text="Here is the fix.",
        )
        is MaterialOrigin.ASSISTANT_AUTHORED
    )


def test_assistant_role_with_tool_use_type_is_assistant_authored() -> None:
    """Branch 8 (TOOL_USE side of the `in (MESSAGE, TOOL_USE)` check)."""
    assert (
        classify_material_origin(
            role=Role.ASSISTANT,
            message_type=MessageType.TOOL_USE,
            text="",
            block_types=(BlockType.TOOL_USE,),
        )
        is MaterialOrigin.ASSISTANT_AUTHORED
    )
