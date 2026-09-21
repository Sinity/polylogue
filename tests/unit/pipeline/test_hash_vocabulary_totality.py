"""The declared hash vocabulary must be total over what parsers actually emit.

polylogue-m706z: ``ParsedContentBlock.metadata`` is typed ``dict[str, object]``
and a set is the natural Python representation of an unordered member list, so
ordinary parser output could hand ``message_content_identity`` a payload the
digest encoder refuses -- ``TypeError: Object of type set is not JSON
serializable``. ``content_identity`` is the content-derived half of the message
identity fallback (``messages.message_id`` uses ``c:`` when a provider supplies
no native id), so a payload shape that raises instead of hashing makes that
identity unavailable for exactly the messages that depend on it.

Anti-vacuity for each test is named on the test. A payload of JSON-native types
alone proves nothing here: it hashed before the fix and hashes after.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType
from polylogue.core.hashing import hash_payload
from polylogue.pipeline.ids import (
    UnhashablePayloadValueError,
    _normalize_nested_for_hash,
    message_content_identity,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage


def _message(metadata: dict[str, object]) -> ParsedMessage:
    """An id-less message -- the case that *needs* the content-derived fallback."""
    return ParsedMessage(
        provider_message_id="",
        role=Role.normalize("assistant"),
        text="hello",
        timestamp="2024-01-01T00:00:00Z",
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello", metadata=metadata)],
    )


def test_block_metadata_set_hashes_instead_of_raising() -> None:
    """The reported defect, at the production entry point.

    Anti-vacuity: with the ``set``/``frozenset`` case removed from
    ``_normalize_nested_for_hash`` this raises ``TypeError: Object of type set
    is not JSON serializable`` from the digest encoder.
    """
    identity = message_content_identity(_message({"unordered": {"z", "a"}}))
    assert len(identity) == 32
    assert all(character in "0123456789abcdef" for character in identity)


def test_member_order_does_not_change_content_identity() -> None:
    """Two constructions of the same unordered collection are the same message.

    Anti-vacuity: a fix that lowered a set with ``list(value)`` instead of
    sorting would leave this green only by luck of one process's iteration
    order -- which is why the cross-process test below exists too.
    """
    forward = message_content_identity(_message({"unordered": {"a", "m", "z"}}))
    reverse = message_content_identity(_message({"unordered": {"z", "m", "a"}}))
    frozen = message_content_identity(_message({"unordered": frozenset(("m", "z", "a"))}))
    assert forward == reverse == frozen


_CROSS_PROCESS_PROGRAM = textwrap.dedent(
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.pipeline.ids import message_content_identity
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage

    members = {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"}
    block = ParsedContentBlock(type=BlockType.TEXT, text="hello", metadata={"tags": members})
    message = ParsedMessage(
        provider_message_id="",
        role=Role.normalize("assistant"),
        text="hello",
        timestamp="2024-01-01T00:00:00Z",
        blocks=[block],
    )
    print(message_content_identity(message))
    """
)


def test_content_identity_is_stable_across_hash_seeds() -> None:
    """A durable identity may not depend on one process's set iteration order.

    ``content_identity`` is persisted in ``messages.content_identity`` and is
    what a durable ``user.db`` assertion resolves through. Python's string hash
    is randomized per process, so an unsorted lowering of a set would give the
    same message two different identities on two runs of the same import.

    Anti-vacuity: replacing the sort with ``list(value)`` turns this red (the
    two seeds below were chosen because they produce different iteration orders
    for this member set); removing the case entirely turns it red by raising.
    """
    digests = set()
    for seed in ("1", "2"):
        completed = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_PROGRAM],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, completed.stderr
        digests.add(completed.stdout.strip())
    assert len(digests) == 1, f"content_identity varied with PYTHONHASHSEED: {digests}"


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        (Decimal("1.5"), 1.5),
        (datetime(2024, 1, 1, 12, 30), "2024-01-01T12:30:00"),
        (date(2024, 1, 1), "2024-01-01"),
        (time(12, 30), "12:30:00"),
        (b"\xde\xad", "dead"),
        (bytearray(b"\xde\xad"), "dead"),
    ],
)
def test_declared_non_json_types_lower_to_their_canonical_form(value: object, canonical: object) -> None:
    """Sets were not the only gap: the encoder admits no non-JSON-native value.

    Anti-vacuity: each of these raises from the digest encoder without its
    case in ``_normalize_nested_for_hash``.
    """
    assert _normalize_nested_for_hash({"k": value}) == {"k": canonical}
    assert hash_payload(_normalize_nested_for_hash({"k": value}))


def test_enum_lowers_to_its_value() -> None:
    """Anti-vacuity: a non-``str`` Enum raises from the encoder without this case."""

    class Kind(Enum):
        FIRST = "first"

    assert _normalize_nested_for_hash({"k": Kind.FIRST}) == {"k": "first"}


def test_a_value_outside_the_vocabulary_is_refused_by_name() -> None:
    """The next gap must arrive as a vocabulary refusal, not a backend message.

    Stringifying an unknown value instead would be worse than refusing:
    ``str(object())`` embeds a memory address, which would make a
    content-derived identity vary per process while looking healthy.

    Anti-vacuity: before the fix this raised the JSON encoder's
    ``TypeError: Object of type object is not JSON serializable``, which names
    neither the hash vocabulary nor the field that carried the value.
    """
    with pytest.raises(UnhashablePayloadValueError) as caught:
        _normalize_nested_for_hash({"outer": [object()]})
    message = str(caught.value)
    assert "object" in message
    assert "payload.outer[]" in message
    assert issubclass(UnhashablePayloadValueError, TypeError)


@pytest.mark.parametrize(
    "payload",
    [
        {"a": 1, "b": "x", "c": None, "d": "", "e": [1, 2, {"f": "café"}], "g": True, "h": 1.5},
        {"tuple": (1, "a", None)},
        {"intkey": {1: "a", 2: "b"}},
        {"deep": {"deeper": {"s": "é", "n": 0, "f": 0.0, "b": False}}},
        {"empties": {"d": {}, "l": []}},
        [],
        "",
        None,
    ],
)
def test_previously_hashable_payloads_keep_their_digest(payload: object) -> None:
    """No stored identity moves: the JSON-native path is byte-for-byte unchanged.

    These digests were computed by running ``origin/master``'s
    ``_normalize_nested_for_hash`` (commit 0b1e99d69) against the same payloads
    through ``hash_payload``. Every value type the fix newly admits previously
    raised, and ``content_identity`` is computed on the write path before any
    row is inserted (``storage/sqlite/archive_tiers/write.py``), so no archived
    message can carry a shape whose identity this change moves.

    Anti-vacuity: any change to the JSON-native lowering -- reordering keys,
    dropping a sentinel, tagging a container -- turns these literals red.
    """
    expected = {
        "{'a': 1, 'b': 'x', 'c': None, 'd': '', 'e': [1, 2, {'f': 'café'}], 'g': True, 'h': 1.5}": "c5f49a915545f513",
        "{'tuple': (1, 'a', None)}": "1197ba6cb48bb886",
        "{'intkey': {1: 'a', 2: 'b'}}": "a9328ce48b1a01ce",
        "{'deep': {'deeper': {'s': 'é', 'n': 0, 'f': 0.0, 'b': False}}}": "9c17a8e354f05ef4",
        "{'empties': {'d': {}, 'l': []}}": "8e4d0667b1e2aafe",
        "[]": "4f53cda18c2baa0c",
        "''": "3c5c29debc2c0150",
        "None": "641924dd2a7bb104",
    }[repr(payload)]
    assert hash_payload(_normalize_nested_for_hash(payload)).startswith(expected)
