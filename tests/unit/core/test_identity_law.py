from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.core.identity_law import block_id, message_id, message_local_id, session_id

_TOKEN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=32,
).filter(lambda value: bool(value.strip()))
_POSITION = st.integers(min_value=0, max_value=100_000)


@given(origin=_TOKEN, native_id=_TOKEN)
def test_session_id_is_origin_native_id(origin: str, native_id: str) -> None:
    observed = session_id(origin, native_id)
    assert observed == f"{origin.strip()}:{native_id.strip()}"


@given(parent=_TOKEN, native_id=_TOKEN, position=_POSITION, variant_index=_POSITION)
def test_native_message_id_ignores_position_fallback(
    parent: str,
    native_id: str,
    position: int,
    variant_index: int,
) -> None:
    sid = session_id("codex", parent)
    assert message_id(sid, native_id, position=position, variant_index=variant_index) == f"{sid}:{native_id.strip()}"


@given(parent=_TOKEN, position=_POSITION, left_variant=_POSITION, right_variant=_POSITION)
def test_no_native_message_id_uses_variant_index_for_collision_avoidance(
    parent: str,
    position: int,
    left_variant: int,
    right_variant: int,
) -> None:
    sid = session_id("codex", parent)
    left = message_id(sid, None, position=position, variant_index=left_variant)
    right = message_id(sid, None, position=position, variant_index=right_variant)
    assert left == f"{sid}:{position}.{left_variant}"
    assert right == f"{sid}:{position}.{right_variant}"
    assert (left == right) is (left_variant == right_variant)


@given(parent=_TOKEN, message_native_id=_TOKEN, block_position=_POSITION)
def test_block_id_appends_block_position(parent: str, message_native_id: str, block_position: int) -> None:
    mid = message_id(session_id("chatgpt", parent), message_native_id, position=0)
    assert block_id(mid, position=block_position) == f"{mid}:{block_position}"


def test_native_ids_are_opaque_and_may_contain_colons() -> None:
    sid = session_id("antigravity-session", "cascade:with:colon")
    mid = message_id(sid, "cascade:0:planner_response", position=0)

    assert sid == "antigravity-session:cascade:with:colon"
    assert mid == "antigravity-session:cascade:with:colon:cascade:0:planner_response"


@pytest.mark.parametrize(
    "fn,args,kwargs",
    [
        (session_id, ("", "native"), {}),
        (session_id, ("bad:origin", "native"), {}),
        (session_id, ("origin", ""), {}),
        (message_local_id, (None,), {"position": -1}),
        (message_local_id, (None,), {"position": 0, "variant_index": -1}),
        (block_id, ("",), {"position": 0}),
        (block_id, ("message",), {"position": -1}),
    ],
)
def test_identity_law_rejects_invalid_inputs(fn: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    callable_fn = cast(Callable[..., object], fn)
    with pytest.raises(ValueError):
        callable_fn(*args, **kwargs)
