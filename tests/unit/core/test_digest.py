"""The digest kernel reproduces every encoder it replaced, byte for byte.

The ``_reference_*`` functions below are the pre-kernel encoders, copied
verbatim from the domains that owned them. They are the anti-vacuity
condition: each profile is asserted equal to an independent implementation of
the bytes it inherited, so weakening a profile field -- dropping NFC, flipping
``ensure_ascii``, moving a domain between the stdlib and ``core.json``
encoders -- turns at least one assertion red rather than silently reminting
every digest that domain has written.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata

import pytest

from polylogue.browser_capture.capture_jobs import CaptureJobError, canonical_digest, canonical_json
from polylogue.core.digest import (
    CAPTURE,
    EXACT_INTEGER_LIMIT,
    IDENTITY,
    PROFILES,
    QUERY,
    RECEIPT,
    REFERENCE,
    CanonicalizationError,
    KeyCollisionError,
    canonical_bytes,
    digest,
    profile_for,
)
from polylogue.core.hashing import hash_payload
from polylogue.core.json import dumps_bytes
from polylogue.material_protocol.v1.canonical import canonical_bytes as material_canonical_bytes

# "café" and "crème brûlée" spelled with combining marks: the input that tells
# a normalizing profile from a non-normalizing one.
DECOMPOSED_KEY = "cafe\u0301"
DECOMPOSED_TEXT = "cre\u0300me bru\u0302le\u0301e"

CORPUS: dict[str, object] = {
    "empty_object": {},
    "empty_list": [],
    "scalars": {"t": True, "f": False, "n": None, "i": 0, "neg": -17},
    "nested": {"b": [1, {"z": "x", "a": "y"}], "a": {"k": [True, None]}},
    "decomposed": {DECOMPOSED_KEY: DECOMPOSED_TEXT},
    "precomposed": {unicodedata.normalize("NFC", DECOMPOSED_KEY): unicodedata.normalize("NFC", DECOMPOSED_TEXT)},
    "non_ascii_key_sort": {"\u00e9": 1, "e": 2, "z": 3},
    "astral": {"k": "\U0001f600 tail"},
    "exact_integer_limit": {"n": EXACT_INTEGER_LIMIT},
    "deep": {"a": [[[{"b": [1, 2, {"c": "d"}]}]]]},
    "escapes": {"q": 'line\nbreak\ttab "quote" back\\slash'},
    "control_character": {"c": "\u0001"},
    "latin_supplement": {"k": "\u00ff\u0100"},
}

FLOAT_CORPUS: dict[str, object] = {
    "floats": {"a": 1.0, "b": 0.1, "c": 1e30, "d": 5e-324, "e": -0.0},
    "beyond_exact_integer": {"n": EXACT_INTEGER_LIMIT + 1},
}


def _reference_nfc(value: object) -> object:
    """``annotations.write._nfc_json_value`` as it stood before the kernel."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_reference_nfc(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError(f"NFC-normalized JSON keys collide at {normalized_key!r}")
            normalized[normalized_key] = _reference_nfc(item)
        return normalized
    return value


def _reference_receipt_bytes(value: object) -> bytes:
    """``annotations.{batch,write,schema}`` and ``scenarios.workload`` framing."""
    return json.dumps(
        _reference_nfc(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reference_reference_bytes(value: object) -> bytes:
    """``AnnotationBatch`` provenance and ``context.compiler`` framing: no NFC."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reference_query_bytes(value: object) -> bytes:
    """``core.hashing.hash_payload`` framing: ASCII-escaped, unnormalized."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reference_identity_bytes(value: object) -> bytes:
    """``material_protocol.v1.canonical`` framing: NFC through ``core.json``."""

    def normalize(item: object) -> object:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            return {unicodedata.normalize("NFC", key): normalize(child) for key, child in item.items()}
        return item

    return dumps_bytes(normalize(value), sort_keys=True)


def _reference_capture_json(value: object) -> str:
    """``browser_capture.capture_jobs.canonical_json`` as it stood before the kernel."""
    if value is None or isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        return json.dumps(unicodedata.normalize("NFC", value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= EXACT_INTEGER_LIMIT:
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_reference_capture_json(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        entries = sorted(
            ((unicodedata.normalize("NFC", key), item) for key, item in value.items()),
            key=lambda entry: entry[0],
        )
        if any(entries[index - 1][0] == entries[index][0] for index in range(1, len(entries))):
            raise CaptureJobError(400, "non_canonical_key_collision")
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False) + ":" + _reference_capture_json(item) for key, item in entries
            )
            + "}"
        )
    raise CaptureJobError(400, "non_canonical_json")


@pytest.mark.parametrize("name", sorted(CORPUS | FLOAT_CORPUS))
class TestProfilesReproduceTheEncodersTheyReplaced:
    def test_receipt(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        assert canonical_bytes(payload, RECEIPT) == _reference_receipt_bytes(payload)

    def test_reference(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        assert canonical_bytes(payload, REFERENCE) == _reference_reference_bytes(payload)

    def test_query(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        assert canonical_bytes(payload, QUERY) == _reference_query_bytes(payload)

    def test_identity(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        assert canonical_bytes(payload, IDENTITY) == _reference_identity_bytes(payload)

    def test_capture(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        try:
            expected = _reference_capture_json(payload)
        except CaptureJobError as exc:
            with pytest.raises(CaptureJobError) as raised:
                canonical_json(payload)
            assert raised.value.code == exc.code
            return
        assert canonical_json(payload) == expected
        assert canonical_digest(payload) == "sha256:" + hashlib.sha256(expected.encode("utf-8")).hexdigest()


class TestDomainSurfacesKeepTheirBytes:
    """The public entry points, not just the profiles, still emit inherited bytes."""

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_material_protocol_canonical_bytes(self, name: str) -> None:
        assert material_canonical_bytes(CORPUS[name]) == _reference_identity_bytes(CORPUS[name])  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", sorted(CORPUS | FLOAT_CORPUS))
    def test_hash_payload(self, name: str) -> None:
        payload = (CORPUS | FLOAT_CORPUS)[name]
        expected = hashlib.sha256(_reference_query_bytes(payload)).hexdigest()
        assert hash_payload(payload) == expected


class TestProfilesAreNotInterchangeable:
    """Each declared difference is load-bearing on some real payload."""

    def test_query_escapes_where_receipt_emits_utf8(self) -> None:
        payload = {"k": "\u00e9"}
        assert canonical_bytes(payload, QUERY) == b'{"k":"\\u00e9"}'
        assert canonical_bytes(payload, RECEIPT) == b'{"k":"\xc3\xa9"}'

    def test_query_does_not_normalize_where_receipt_does(self) -> None:
        decomposed = {DECOMPOSED_KEY: "x"}
        precomposed = {unicodedata.normalize("NFC", DECOMPOSED_KEY): "x"}
        assert canonical_bytes(decomposed, QUERY) != canonical_bytes(precomposed, QUERY)
        assert canonical_bytes(decomposed, RECEIPT) == canonical_bytes(precomposed, RECEIPT)

    def test_reference_does_not_normalize_where_receipt_does(self) -> None:
        decomposed = {DECOMPOSED_KEY: "x"}
        precomposed = {unicodedata.normalize("NFC", DECOMPOSED_KEY): "x"}
        assert canonical_bytes(decomposed, REFERENCE) != canonical_bytes(precomposed, REFERENCE)
        assert canonical_bytes(decomposed, RECEIPT) == canonical_bytes(precomposed, RECEIPT)

    def test_identity_and_receipt_disagree_on_small_exponents(self) -> None:
        """``core.json`` writes ``1e-9`` where stdlib writes ``1e-09``.

        This is why the encoder is a profile field: moving the material
        protocol onto the stdlib encoder would rewrite content hashes for
        roughly one finite double in a hundred.
        """
        payload = {"v": 2.488520846603813e-09}
        assert canonical_bytes(payload, IDENTITY) == b'{"v":2.488520846603813e-9}'
        assert canonical_bytes(payload, RECEIPT) == b'{"v":2.488520846603813e-09}'


class TestAdmissibility:
    def test_receipt_rejects_non_finite(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_bytes({"n": float("nan")}, RECEIPT)

    def test_query_encodes_non_finite_as_stdlib_json_always_has(self) -> None:
        assert canonical_bytes({"n": float("nan")}, QUERY) == b'{"n":NaN}'

    def test_receipt_rejects_keys_that_collide_under_nfc(self) -> None:
        with pytest.raises(KeyCollisionError, match="NFC-normalized JSON keys collide"):
            canonical_bytes({DECOMPOSED_KEY: 1, unicodedata.normalize("NFC", DECOMPOSED_KEY): 2}, RECEIPT)

    def test_query_keeps_keys_that_collide_only_under_nfc(self) -> None:
        payload = {DECOMPOSED_KEY: 1, unicodedata.normalize("NFC", DECOMPOSED_KEY): 2}
        assert canonical_bytes(payload, QUERY).count(b":") == 2

    def test_capture_rejects_floats(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_bytes({"n": 1.5}, CAPTURE)

    def test_capture_rejects_integers_a_double_cannot_hold(self) -> None:
        assert canonical_bytes({"n": EXACT_INTEGER_LIMIT}, CAPTURE) == b'{"n":9007199254740991}'
        with pytest.raises(CanonicalizationError):
            canonical_bytes({"n": EXACT_INTEGER_LIMIT + 1}, CAPTURE)

    def test_capture_carries_its_digest_prefix(self) -> None:
        assert digest({}, CAPTURE).startswith("sha256:")
        assert not digest({}, RECEIPT).startswith("sha256:")

    def test_receipt_rejects_non_string_object_keys(self) -> None:
        with pytest.raises(CanonicalizationError):
            canonical_bytes({1: "x"}, RECEIPT)


class TestRegistry:
    def test_every_profile_is_registered_under_its_own_name(self) -> None:
        assert set(PROFILES) == {"identity", "receipt", "reference", "query", "capture"}
        for name, profile in PROFILES.items():
            assert profile.name == name
            assert profile_for(name) is profile

    def test_unknown_profile_fails_closed(self) -> None:
        with pytest.raises(CanonicalizationError):
            profile_for("no-such-profile")

    def test_digest_is_the_sha256_of_the_canonical_bytes(self) -> None:
        payload = CORPUS["nested"]
        for profile in (IDENTITY, RECEIPT, REFERENCE, QUERY, CAPTURE):
            expected = hashlib.sha256(canonical_bytes(payload, profile)).hexdigest()
            assert digest(payload, profile) == profile.digest_prefix + expected
