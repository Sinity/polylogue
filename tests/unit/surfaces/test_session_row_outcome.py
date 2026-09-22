"""The session-row outcome projection speaks the canonical terminal vocabulary."""

from __future__ import annotations

from polylogue.core.enums import TERMINAL_STATE_VALUES
from polylogue.surfaces.query_rows import session_row


def test_every_canonical_terminal_state_survives_the_row_projection() -> None:
    """A structurally decided outcome reaches CLI/query/select rows intact.

    `SessionProfile.terminal_state` is decided in the closed `TerminalState`
    vocabulary (`tool_left`, `error_left`, `question_left`, `refused`,
    `truncated`, `unknown`). This projection used to admit its own private
    set -- `{completed, failed, abandoned, unknown}` -- which is disjoint from
    the canonical one except for `unknown`, so every informative outcome was
    rewritten to `unknown` in list/search/select rows and in the published
    `terminal_state` of every session-list row.

    Anti-vacuity: restoring
    `OUTCOME_VALUES = frozenset({"completed", "failed", "abandoned", "unknown"})`
    makes each non-`unknown` case below project `unknown`.
    """
    for state in sorted(TERMINAL_STATE_VALUES):
        assert session_row({"id": "s", "terminal_state": state}).outcome == state


def test_a_value_outside_the_vocabulary_is_still_unknown() -> None:
    """The opposite direction: membership is checked, not echoed through."""
    for state in ("completed", "agent_hanging", "", None):
        assert session_row({"id": "s", "terminal_state": state}).outcome == "unknown"
