"""polylogue-tztk L13: a parse-skip log must not reproduce payload content.

The 2026-07-31 leak-surfaces audit found ``polylogue/sources/parsers/codex.py``
interpolating a Pydantic ``ValidationError`` into a DEBUG log line. Pydantic
v2's ``__str__`` embeds ``input_value`` -- i.e. the raw captured record -- so
``polylogue --verbose`` wrote conversation content into the operator's
scrollback. It was the only such call site in the tree.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.sources.parsers.codex import CodexRecord, _redacted_validation_errors

_CANARY = "SECRETVALUE_LEAK_CANARY"


def _rejected_record_error() -> ValidationError:
    with pytest.raises(ValidationError) as excinfo:
        CodexRecord.model_validate({"type": "message", "timestamp": {"nested": _CANARY}})
    return excinfo.value


class TestRedactedValidationErrors:
    def test_the_rendered_summary_carries_no_payload_content(self) -> None:
        """Anti-vacuity: revert the call site to ``%s`` on the exception (or
        make this helper return ``str(exc)``) and the canary reappears --
        ``test_the_unredacted_exception_would_have_leaked_it`` below proves
        the input really is present in the unredacted form, so this pair
        cannot both pass vacuously."""
        summary = _redacted_validation_errors(_rejected_record_error())

        assert _CANARY not in summary

    def test_the_unredacted_exception_would_have_leaked_it(self) -> None:
        assert _CANARY in str(_rejected_record_error())

    def test_the_summary_still_names_the_field_and_rule(self) -> None:
        summary = _redacted_validation_errors(_rejected_record_error())

        assert "timestamp" in summary
        assert "_type" in summary
