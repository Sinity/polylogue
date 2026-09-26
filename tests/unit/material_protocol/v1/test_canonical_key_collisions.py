"""Material-protocol encoding must refuse lossy NFC object-key normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.digest import IDENTITY, KeyCollisionError
from polylogue.core.digest import canonical_bytes as profile_canonical_bytes
from polylogue.material_protocol.v1.canonical import canonical_bytes, canonical_line, nfc_normalize


@pytest.mark.parametrize(
    "payload",
    [
        {"é": 1, "e\u0301": 2},
        {"outer": {"é": 1, "e\u0301": 2}},
        [{"outer": {"é": "first", "e\u0301": "second"}}],
    ],
)
def test_material_encoding_rejects_nfc_key_collisions_before_overwrite(payload: object) -> None:
    # The generic identity profile remains historically permissive. Material
    # framing must not inherit that overwrite behavior.
    assert profile_canonical_bytes(payload, IDENTITY)
    with pytest.raises(KeyCollisionError):
        canonical_bytes(payload)  # type: ignore[arg-type]
    with pytest.raises(KeyCollisionError):
        canonical_line(payload)  # type: ignore[arg-type]
    with pytest.raises(KeyCollisionError):
        nfc_normalize(payload)  # type: ignore[arg-type]


def test_material_encoding_keeps_v1_bytes_for_unambiguous_values() -> None:
    payload = {"nested": {"cafe\u0301": ["e\u0301", 9007199254740993]}}
    assert canonical_bytes(payload) == profile_canonical_bytes(payload, IDENTITY)
    assert nfc_normalize(payload) == {"nested": {"café": ["é", 9007199254740993]}}


def test_actual_producer_helper_emits_shared_numeric_unicode_boundary_vector() -> None:
    expected = (Path(__file__).parents[3] / "fixtures/material_protocol/v1/canonical_boundaries.json").read_bytes()
    payload = {
        "decimal_float": 0.00001,
        "large_float": 1e30,
        "large_integer": 9007199254740993,
        "nested": {"cafe\u0301": ["e\u0301", "日本語"]},
        "small_float": 1e-9,
        "u64_max": 18446744073709551615,
    }
    assert canonical_line(payload) == expected
