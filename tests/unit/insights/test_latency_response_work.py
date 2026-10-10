"""Response timing retains pairing semantics without quadratic tail copies."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import FrameType
from typing import TYPE_CHECKING, SupportsIndex, overload

import pytest

from polylogue.archive.message.models import Message
from polylogue.archive.semantic.timing import _message_response_latencies, compute_session_latency_profile
from polylogue.core.enums import MaterialOrigin
from tests.infra.builders import make_msg

if TYPE_CHECKING:
    from _typeshed import TraceFunction

STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _message(index: int, role: str, offset: int | None, *, human: bool = False) -> Message:
    return make_msg(
        id=f"m-{index}",
        role=role,
        timestamp=None if offset is None else STAMP + timedelta(milliseconds=offset),
        material_origin=MaterialOrigin.HUMAN_AUTHORED if human else MaterialOrigin.RUNTIME_PROTOCOL,
    )


def test_response_pairing_skips_undated_and_ineligible_messages_without_sorting() -> None:
    messages = [
        _message(0, "user", 0, human=True),
        _message(1, "system", 10),
        _message(2, "user", 20),
        _message(3, "assistant", None),
        _message(4, "assistant", 30),
        _message(5, "user", 40, human=True),
        _message(6, "user", None, human=True),
        _message(7, "system", 50),
        _message(8, "assistant", -10),
        _message(9, "user", 2_000_000, human=True),
    ]
    assert _message_response_latencies(messages) == ([30, 0], [10])
    facts = compute_session_latency_profile(messages, [])
    assert facts.median_agent_response_ms == 15
    assert facts.median_user_response_ms == 10


@pytest.mark.parametrize("count", [256, 1024])
def test_response_pairing_does_not_copy_message_tails(count: int) -> None:
    """Count actual copied references, without asserting wall-clock timing.

    The trace swaps only local message lists for an equivalent list subclass.
    Python's writable frame-locals proxy makes the existing slice operation
    observable; message values, iteration and pairing remain unchanged.
    """
    copied = [0]

    class ObservedMessages(list[Message]):
        @overload
        def __getitem__(self, key: SupportsIndex) -> Message: ...
        @overload
        def __getitem__(self, key: slice) -> list[Message]: ...
        def __getitem__(self, key: SupportsIndex | slice) -> Message | list[Message]:
            value = super().__getitem__(key)
            if isinstance(key, slice):
                assert isinstance(value, list)
                copied[0] += len(value)
            return value

    messages = [_message(i, "user" if i % 2 == 0 else "assistant", i, human=i % 2 == 0) for i in range(count)]
    # Negative control proves the observer sees precisely the copied values.
    observed = ObservedMessages(messages)
    assert len(observed[1:]) == copied[0] == count - 1
    copied[0] = 0

    def observe(frame: FrameType, event: str, argument: object) -> TraceFunction:
        if frame.f_code is _message_response_latencies.__code__ and event == "line":
            for name, value in tuple(frame.f_locals.items()):
                if type(value) is list and value and isinstance(value[0], Message):
                    frame.f_locals[name] = ObservedMessages(value)
        return observe

    previous_trace = sys.gettrace()
    try:
        sys.settrace(observe)
        agent, user = _message_response_latencies(messages)
    finally:
        sys.settrace(previous_trace)
    assert agent == [1] * (count // 2)
    assert user == [1] * (count // 2 - 1)
    assert copied[0] == 0
