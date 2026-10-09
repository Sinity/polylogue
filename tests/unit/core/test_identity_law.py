from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polylogue.core.identity_law import block_id, message_id, message_local_id, session_id, split_message_local_id

_TOKEN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
    min_size=1,
    max_size=32,
).filter(lambda value: bool(value.strip()))
_POSITION = st.integers(min_value=0, max_value=100_000)
_BLOCK_CONTENT_IDENTITY = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)


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


@given(parent=_TOKEN, message_native_id=_TOKEN, identity=_BLOCK_CONTENT_IDENTITY, left=_POSITION, right=_POSITION)
def test_block_id_uses_content_identity_and_identical_content_occurrence(
    parent: str, message_native_id: str, identity: str, left: int, right: int
) -> None:
    mid = message_id(session_id("chatgpt", parent), message_native_id)
    left_id = block_id(mid, content_identity=identity, content_occurrence=left)
    right_id = block_id(mid, content_identity=identity, content_occurrence=right)
    assert left_id == f"{mid}:b:{identity}:{left}"
    assert right_id == f"{mid}:b:{identity}:{right}"
    assert (left_id == right_id) is (left == right)


def test_block_identity_rejects_a_transcript_position() -> None:
    with pytest.raises(TypeError):
        cast(Callable[..., str], block_id)("message", position=0)


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
        (block_id, ("",), {"content_identity": "a" * 64}),
        (block_id, ("message",), {"content_identity": "a" * 64, "content_occurrence": -1}),
        (block_id, ("message",), {"content_identity": "a" * 63}),
        (block_id, ("message",), {"content_identity": "A" * 64}),
        (block_id, ("message",), {"content_identity": "g" * 64}),
        (block_id, ("message",), {"content_identity": None}),
        (block_id, ("message",), {"content_identity": "a" * 64, "content_occurrence": True}),
        (block_id, ("message",), {"content_identity": "a" * 64, "content_occurrence": 1.5}),
        (block_id, ("message",), {"content_identity": "a" * 64, "content_occurrence": None}),
        (block_id, ("message",), {"content_identity": "a" * 64, "content_occurrence": "1"}),
    ],
)
def test_identity_law_rejects_invalid_inputs(fn: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    callable_fn = cast(Callable[..., object], fn)
    with pytest.raises(ValueError):
        callable_fn(*args, **kwargs)


@pytest.mark.parametrize("native_id", [" a ", "   ", "e\u0301", "é"])
def test_message_native_identity_preserves_exact_nonempty_text(native_id: str) -> None:
    assert message_local_id(native_id) == f"n:{native_id}"
    assert message_id("session", native_id) == f"session:n:{native_id}"


@pytest.mark.parametrize("native_id", [1, False, [], {}])
def test_message_native_identity_refuses_nontext(native_id: object) -> None:
    with pytest.raises(ValueError):
        message_local_id(cast(str, native_id), content_identity="abc")


def test_block_identity_preserves_opaque_parent_message_id() -> None:
    assert block_id("session:n: a ", content_identity="a" * 64) == f"session:n: a :b:{'a' * 64}:0"


@pytest.mark.parametrize("native_id", ["abc:n:tail", "abc:c:digest.0", "x:s:eda080", " a ", "   "])
def test_message_identity_inverse_uses_the_declared_parent_boundary(native_id: str) -> None:
    sid = "origin:session:n:part:c:tail"
    mid = message_id(sid, native_id)
    assert split_message_local_id(mid, parent_session_id=sid) == (native_id, None, 0)
    with pytest.raises(ValueError):
        split_message_local_id(mid, parent_session_id="other-session")
