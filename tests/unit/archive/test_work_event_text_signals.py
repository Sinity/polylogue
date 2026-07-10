"""Word-boundary regression tests for the work-event text-signal classifier (#b0b.1).

`_text_signal_from_lowered_text` matches keyword tables against user text to
produce a heuristic `WorkEventHeuristicLabel`. Before this fix it used naive
`pattern in text` substring checks, which false-positived on unrelated words
that happen to contain a pattern as a substring.
"""

from __future__ import annotations

import pytest

from polylogue.archive.session.extraction import WorkEventHeuristicLabel, _text_signal_from_lowered_text

# (text, must NOT match) -- substring false positives the naive matcher hit.
FALSE_POSITIVE_CASES: list[tuple[str, str]] = [
    ("please check the prefix path", "fix"),
    ("this is the latest version", "test"),
    ("here is a brief explanation", "plan"),
    ("update the metadata", "data"),
    ("i respect that decision", "spec"),
    ("remove the old file", "move"),
    ("reconfigured the daemon on startup", "config"),
    ("contest results are in", "test"),
    ("validate the input first", "data"),
    ("inspect the object closely", "spec"),
    ("we should build an airplane", "plan"),
]

# (text, expected label) -- genuine signals that must still match.
GENUINE_SIGNAL_CASES: list[tuple[str, WorkEventHeuristicLabel]] = [
    ("can you fix the bug", WorkEventHeuristicLabel.DEBUGGING),
    ("let's plan this out first", WorkEventHeuristicLabel.PLANNING),
    ("run pytest against this module", WorkEventHeuristicLabel.TESTING),
    ("looks good, nit: rename this", WorkEventHeuristicLabel.REVIEW),
    ("let's refactor this function", WorkEventHeuristicLabel.REFACTORING),
    ("please document this behavior", WorkEventHeuristicLabel.DOCUMENTATION),
    ("edit the config file", WorkEventHeuristicLabel.CONFIGURATION),
    ("run this sql query", WorkEventHeuristicLabel.DATA_ANALYSIS),
    ("we hit a stack trace here", WorkEventHeuristicLabel.DEBUGGING),
    ("should we use postgres instead", WorkEventHeuristicLabel.PLANNING),
]


@pytest.mark.parametrize("text,collided_pattern", FALSE_POSITIVE_CASES)
def test_substring_false_positives_no_longer_match(text: str, collided_pattern: str) -> None:
    result = _text_signal_from_lowered_text(text.lower())
    assert result is None, (
        f"{text!r} incorrectly matched a signal via the {collided_pattern!r} substring collision: {result}"
    )


@pytest.mark.parametrize("text,expected_label", GENUINE_SIGNAL_CASES)
def test_genuine_word_and_phrase_signals_still_match(text: str, expected_label: WorkEventHeuristicLabel) -> None:
    result = _text_signal_from_lowered_text(text.lower())
    assert result is not None, f"{text!r} should have matched {expected_label} but matched nothing"
    _, label, _ = result
    assert label == expected_label
