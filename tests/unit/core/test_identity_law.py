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


@given(parent=_TOKEN, native_id=_TOKEN, content_identity=_TOKEN, occurrence=_POSITION)
def test_native_message_id_ignores_the_content_fallback(
    parent: str,
    native_id: str,
    content_identity: str,
    occurrence: int,
) -> None:
    sid = session_id("codex", parent)
    observed = message_id(sid, native_id, content_identity=content_identity, content_occurrence=occurrence)
    assert observed == f"{sid}:n:{native_id.strip()}"


@given(parent=_TOKEN, content_identity=_TOKEN, left=_POSITION, right=_POSITION)
def test_idless_message_id_is_its_content_identity_plus_occurrence(
    parent: str,
    content_identity: str,
    left: int,
    right: int,
) -> None:
    """The fallback carries no ordinal: only the digest and the occurrence.

    Anti-vacuity: reintroduce ``position`` into ``message_local_id``'s
    fallback and the constructed id no longer equals this expectation.
    """
    sid = session_id("codex", parent)
    left_id = message_id(sid, None, content_identity=content_identity, content_occurrence=left)
    right_id = message_id(sid, None, content_identity=content_identity, content_occurrence=right)
    assert left_id == f"{sid}:c:{content_identity}.{left}"
    assert right_id == f"{sid}:c:{content_identity}.{right}"
    assert (left_id == right_id) is (left == right)


@given(parent=_TOKEN, message_native_id=_TOKEN, block_position=_POSITION)
def test_block_id_appends_block_position(parent: str, message_native_id: str, block_position: int) -> None:
    mid = message_id(session_id("chatgpt", parent), message_native_id)
    assert block_id(mid, position=block_position) == f"{mid}:{block_position}"


def test_native_ids_are_opaque_and_may_contain_colons() -> None:
    sid = session_id("antigravity-session", "cascade:with:colon")
    mid = message_id(sid, "cascade:0:planner_response")

    assert sid == "antigravity-session:cascade:with:colon"
    assert mid == "antigravity-session:cascade:with:colon:n:cascade:0:planner_response"


def test_native_and_content_message_ids_are_disjoint() -> None:
    """A provider id spelled like a fallback must not collide with one."""
    sid = session_id("codex", "collision")
    digest = "b1d7189e2d0ccae512b833b4d6cb77da"

    assert message_id(sid, f"c:{digest}.0") != message_id(sid, None, content_identity=digest)


@pytest.mark.parametrize(
    "fn,args,kwargs",
    [
        (session_id, ("", "native"), {}),
        (session_id, ("bad:origin", "native"), {}),
        (session_id, ("origin", ""), {}),
        (message_local_id, (None,), {}),
        (message_local_id, (None,), {"content_identity": "   "}),
        (message_local_id, (None,), {"content_identity": "abc", "content_occurrence": -1}),
        (block_id, ("",), {"position": 0}),
        (block_id, ("message",), {"position": -1}),
    ],
)
def test_identity_law_rejects_invalid_inputs(fn: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    callable_fn = cast(Callable[..., object], fn)
    with pytest.raises(ValueError):
        callable_fn(*args, **kwargs)
