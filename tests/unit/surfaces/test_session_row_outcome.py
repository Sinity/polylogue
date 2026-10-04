"""The session-row outcome projection speaks the canonical terminal vocabulary."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    ("role", "stop_reason", "expected"),
    [
        ("user", None, "question_left"),
        ("assistant", "refusal", "refused"),
        ("assistant", "max_tokens", "truncated"),
        ("assistant", None, "unknown"),
    ],
)
def test_hydrated_rows_use_narrow_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    stop_reason: str | None,
    expected: str,
) -> None:
    from polylogue.archive.session import runtime, session_profile
    from polylogue.core.enums import Provider
    from polylogue.surfaces.payloads import session_list_envelope_from_domain, session_summary_envelope_from_domain
    from tests.infra.builders import make_conv, make_msg

    session = make_conv(
        id="session-terminal",
        origin=Provider.CLAUDE_CODE,
        messages=[
            make_msg(
                id="message-terminal",
                role=role,
                origin=Provider.CLAUDE_CODE,
                text="Neutral text.",
                stop_reason=stop_reason,
            ),
        ],
    )
    profile = session_profile.build_session_profile(session)
    assert profile.terminal_state == expected

    def unrelated_analysis(*args: object, **kwargs: object) -> object:
        raise AssertionError("hydrated row rebuilt unrelated profile analysis")

    monkeypatch.setattr(session_profile, "build_session_profile", unrelated_analysis)
    monkeypatch.setattr(runtime, "build_session_analysis", unrelated_analysis)
    monkeypatch.setattr(runtime, "build_session_semantic_facts", unrelated_analysis)
    assert runtime.build_session_terminal_state(session) == (
        profile.terminal_state,
        profile.terminal_state_confidence,
        profile.terminal_state_evidence,
        profile.terminal_state_method,
    )
    assert session_list_envelope_from_domain(session).terminal_state == expected
    assert session_summary_envelope_from_domain(session).terminal_state == expected
