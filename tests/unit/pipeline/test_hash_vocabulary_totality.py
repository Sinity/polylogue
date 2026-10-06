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


def test_non_finite_set_members_sort_by_their_digest_tokens() -> None:
    """NaN and infinities must not tie as null in the sort-key encoder."""
    from polylogue.pipeline.ids import _normalize_nested_for_hash

    positive = float("inf")
    negative = float("-inf")
    nan = float("nan")
    first = {positive, nan, negative}
    second = {negative, positive, nan}
    assert hash_payload(_normalize_nested_for_hash(first)) == hash_payload(_normalize_nested_for_hash(second))


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
    assert "payload.'outer'[]" in message
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
def test_json_native_payloads_hash_to_pinned_digests(payload: object) -> None:
    """The JSON-native lowering is pinned byte for byte.

    ``None`` and ``""`` lower to the encoder's own ``null`` and ``""``, and
    strings and keys stay exact, so these digests are ``hash_payload`` of the
    payload as written.

    Anti-vacuity: any change to the JSON-native lowering -- reordering keys,
    reintroducing a null or empty marker, folding a string, tagging a
    container -- turns these literals red.
    """
    expected = {
        "{'a': 1, 'b': 'x', 'c': None, 'd': '', 'e': [1, 2, {'f': 'café'}], 'g': True, 'h': 1.5}": "dcfe6ba4e4dd365c",
        "{'tuple': (1, 'a', None)}": "b1c5b57a4c9acbb5",
        "{'intkey': {1: 'a', 2: 'b'}}": "a9328ce48b1a01ce",
        "{'deep': {'deeper': {'s': 'é', 'n': 0, 'f': 0.0, 'b': False}}}": "9c17a8e354f05ef4",
        "{'empties': {'d': {}, 'l': []}}": "8e4d0667b1e2aafe",
        "[]": "4f53cda18c2baa0c",
        "''": "12ae32cb1ec02d01",
        "None": "74234e98afe7498f",
    }[repr(payload)]
    assert hash_payload(_normalize_nested_for_hash(payload)).startswith(expected)


def test_fast_walk_matches_the_declared_walk_over_mixed_payloads() -> None:
    """The concrete-type fast walk lowers exactly as the declared walk does.

    Anti-vacuity: folding ``dict`` keys or string values in only one walk, or
    passing a ``set``/``Decimal``/``bytes`` leaf through unlowered, makes the
    lowered values differ.
    """
    import random
    from collections.abc import Mapping as MappingABC
    from enum import IntEnum

    from polylogue.pipeline.ids import _normalize_declared_for_hash

    class Level(IntEnum):
        HIGH = 2

    class Frozen(MappingABC[str, object]):
        def __init__(self, data: dict[str, object]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> object:
            return self._data[key]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    leaves: list[object] = [
        "",
        "plain",
        "café",
        None,
        0,
        -1.5e-9,
        True,
        Level.HIGH,
        Role.USER,
        Decimal("2.5"),
        b"\x00",
        date(2024, 1, 2),
        frozenset({"b", "a"}),
    ]
    rng = random.Random(11)

    def tree(depth: int) -> object:
        if depth == 0:
            return rng.choice(leaves)
        shape = rng.randrange(4)
        children = {f"k{index}́" if index % 2 else f"k{index}": tree(depth - 1) for index in range(3)}
        if shape == 0:
            return children
        if shape == 1:
            return Frozen(children)
        if shape == 2:
            return tuple(children.values())
        return list(children.values())

    for _ in range(300):
        payload = tree(4)
        assert _normalize_nested_for_hash(payload) == _normalize_declared_for_hash(payload, path="payload")
        assert hash_payload(_normalize_nested_for_hash(payload)) == hash_payload(
            _normalize_declared_for_hash(payload, path="payload")
        )


def test_fast_walk_still_names_the_path_of_a_refused_value() -> None:
    """Anti-vacuity: re-raising the fast walk's internal signal loses the path."""
    with pytest.raises(UnhashablePayloadValueError) as caught:
        _normalize_nested_for_hash({"outer": {"inner": [1, {"leaf": object()}]}})
    assert "payload.'outer'.'inner'[].'leaf'" in str(caught.value)


def test_decimals_beyond_float_precision_keep_distinct_identities() -> None:
    """Two decimals a float cannot tell apart must not hash alike.

    Anti-vacuity: lower every ``Decimal`` through ``float`` again and both
    values become ``9007199254740992.0``, so the payloads compare equal.
    """
    low = _normalize_nested_for_hash({"k": Decimal("9007199254740992")})
    high = _normalize_nested_for_hash({"k": Decimal("9007199254740993")})
    assert low != high
    assert hash_payload(low) != hash_payload(high)
    # Nor may the exact decimal hash like the equal string.
    assert hash_payload(high) != hash_payload(_normalize_nested_for_hash({"k": "9007199254740993"}))


@pytest.mark.parametrize("key", ["$decimal", "$$decimal"])
def test_a_mapping_cannot_construct_the_exact_decimal_tag(key: str) -> None:
    """Anti-vacuity: stop escaping the reserved key and ``{"$decimal": ...}`` hashes like the Decimal."""
    exact = hash_payload(_normalize_nested_for_hash({"k": Decimal("9007199254740993")}))
    mapping = hash_payload(_normalize_nested_for_hash({"k": {key: "9007199254740993"}}))
    assert exact != mapping
    assert hash_payload(_normalize_nested_for_hash({key: 1})) != hash_payload(
        _normalize_nested_for_hash({"$" + key: 1})
    )


def test_plain_key_ordering_equals_encoded_token_ordering() -> None:
    """Ordering plain ``str`` keys as text orders their QUERY tokens identically.

    Red if the plain-key fast path admits a key whose encoded token sorts
    differently from its text (a space, a quote, a backslash, a control or
    non-ASCII character), or if it sorts by the raw key rather than the legacy
    token (the ``$decimal`` escape).
    """
    import itertools

    from polylogue.core.digest import QUERY, canonical_bytes
    from polylogue.pipeline import ids

    alphabet = ["", "a", "b", "#", "~", "[", "]", " ", "!", '"', "\\", "\x01", "é", "$", "decimal", "$decimal", "A"]
    keys = {"".join(parts) for parts in itertools.product(alphabet, repeat=2)} | {"$$decimal", "z" * 40}

    def reference(value: dict[object, object]) -> list[tuple[object, object]]:
        entries = list(value.items())
        entries.sort(
            key=lambda pair: (
                canonical_bytes(ids._legacy_json_key(pair[0]), QUERY),
                canonical_bytes(ids._typed_identity_value(pair[0]), QUERY),
            )
        )
        return entries

    mapping: dict[object, object] = dict.fromkeys(sorted(keys), 0)
    plain: dict[object, object] = {key: 0 for key in sorted(keys) if ids._PLAIN_HASH_KEY.fullmatch(key)}
    mixed: dict[object, object] = {1: "x", "1": "y", "a": "z"}
    assert len(plain) > 20
    assert ids._hash_ordered_entries(plain) == reference(plain)
    assert ids._hash_ordered_entries(mapping) == reference(mapping)
    assert ids._hash_ordered_entries(mixed) == reference(mixed)
