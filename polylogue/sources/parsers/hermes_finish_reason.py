"""Shared interpretation of the Hermes ``finish_reason`` wire field.

Hermes reports the provider's turn-terminal signal on both of its
conversational artifact families -- the ``state.db`` message rows
(``hermes_state.py``) and the ``sessions/*.json`` snapshots
(``local_agent.parse_hermes``). Both families mint the same archive session
identity (``hermes_identity``), so a message arriving through either one must
land the same ``messages.end_turn`` and ``messages.stop_reason``.

The wire vocabulary is OpenAI's ``finish_reason`` enumeration plus Hermes's
own additions (``verification_required``). ``messages.stop_reason`` is
constrained to Anthropic's :class:`StopReason`; only exact equivalences are
mapped and every other token leaves the column NULL rather than widening a
guess into it.
"""

from __future__ import annotations

from polylogue.core.enums import StopReason

__all__ = ["end_turn_from_finish_reason", "stop_reason_from_finish_reason"]

#: ``finish_reason`` tokens that name the same fact as a ``StopReason``
#: member. ``content_filter`` is absent on purpose -- a provider-side content
#: filter is not the model's own ``refusal`` -- as is Hermes's
#: ``verification_required``, which has no Anthropic equivalent.
_STOP_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
}


def end_turn_from_finish_reason(finish_reason: object) -> bool | None:
    """Return whether this turn ended, or ``None`` when the wire is silent.

    Absence is not a terminal turn: tool-result and user rows carry no
    ``finish_reason`` at all, so deriving ``True`` from a missing value would
    assert a signal the provider never reported.
    """
    if not isinstance(finish_reason, str) or not finish_reason:
        return None
    return finish_reason != "tool_calls"


def stop_reason_from_finish_reason(finish_reason: object) -> str | None:
    """Return the ``messages.stop_reason`` value for a Hermes finish reason."""
    if not isinstance(finish_reason, str):
        return None
    mapped = _STOP_REASONS.get(finish_reason)
    return mapped.value if mapped is not None else None
