from __future__ import annotations

import pytest

from polylogue.archive.message.types import (
    MessageType,
    message_type_sql_values,
    normalize_message_types,
    validate_message_type_filter,
)


def test_message_type_normalization_accepts_enums_strings_lists_and_unknowns() -> None:
    assert MessageType.normalize(MessageType.SUMMARY) is MessageType.SUMMARY
    assert MessageType.normalize("tool-use") is MessageType.TOOL_USE
    assert MessageType.normalize(" tool_result ") is MessageType.TOOL_RESULT
    assert MessageType.normalize("thinking") is MessageType.THINKING
    assert MessageType.normalize(None) is MessageType.MESSAGE
    assert MessageType.normalize("") is MessageType.MESSAGE
    assert MessageType.normalize("unknown") is MessageType.MESSAGE

    assert normalize_message_types(None) == ()
    assert normalize_message_types("summary") == (MessageType.SUMMARY,)
    assert normalize_message_types([MessageType.TOOL_USE, "tool-result"]) == (
        MessageType.TOOL_USE,
        MessageType.TOOL_RESULT,
    )
    assert message_type_sql_values(["summary", "thinking"]) == ("summary", "thinking")


def test_message_type_filter_validation_rejects_unknown_user_input() -> None:
    assert validate_message_type_filter("tool-use") is MessageType.TOOL_USE
    assert validate_message_type_filter("message") is MessageType.MESSAGE

    with pytest.raises(ValueError, match="Unknown message type"):
        validate_message_type_filter("summmary")
