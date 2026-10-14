"""Source witness commitments preserve coordinates and reject ambiguity."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from polylogue.sinex.models import PublicationAcquisition, PublicationOriginal
from polylogue.sinex.provenance import originals_commitment


def original() -> PublicationOriginal:
    return PublicationOriginal("provider_native", "polylogue.raw:cafe\u0301", "a" * 64, 3, "application/octet-stream")


def test_commitment_binds_order_version_and_exact_acquisition_coordinates() -> None:
    first = original()
    second = replace(first, source_ref="polylogue.raw:other", acquisition=PublicationAcquisition("byte_window", 4, 7))
    expected = originals_commitment((first, second), "normalizer/v1")
    assert expected != originals_commitment((second, first), "normalizer/v1")
    assert expected != originals_commitment((first, second), "normalizer/v2")
    assert expected != originals_commitment(
        (replace(first, source_ref="polylogue.raw:caf\u00e9"), second), "normalizer/v1"
    )
    assert expected != originals_commitment(
        (first, replace(second, acquisition=PublicationAcquisition("full_snapshot"))), "normalizer/v1"
    )
    with pytest.raises(ValueError, match="duplicate"):
        originals_commitment((first, first), "normalizer/v1")


@pytest.mark.parametrize(
    "change",
    [
        {"role": "normalized"},
        {"blob_hash": "A" * 64},
        {"size_bytes": True},
        {"size_bytes": -1},
        {"source_ref": ""},
        {"source_ref": " \t"},
        {"media_type": " \t"},
        {"acquisition": PublicationAcquisition("byte_window", 4, 8)},
    ],
)
def test_invalid_original_cannot_enter_an_outbox_commitment(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="publication_original"):
        replace(original(), **change)


def test_acquisition_preserves_empty_windows_and_distinguishes_logical_exports() -> None:
    window = PublicationAcquisition("byte_window", 4, 4)
    assert PublicationAcquisition.from_dict(window.to_dict()) == window
    logical = PublicationAcquisition("logical_snapshot", addressing_mode="logical_rows", tables=("messages",))
    assert PublicationAcquisition.from_dict(logical.to_dict()) == logical
    with pytest.raises(ValueError, match="window"):
        PublicationAcquisition("byte_window", False, 4)
    with pytest.raises(ValueError, match="tables"):
        PublicationAcquisition("full_snapshot", tables=("messages",))


def test_window_size_and_normalizer_refusal_preserve_exact_valid_values() -> None:
    empty = replace(original(), size_bytes=0, acquisition=PublicationAcquisition("byte_window", 4, 4))
    assert empty.acquisition.byte_end == empty.acquisition.byte_start
    exact = replace(original(), source_ref=" polylogue.raw:cafe\u0301 ", media_type=" application/octet-stream ")
    version = " normalizer/cafe\u0301 "
    assert originals_commitment((exact,), version) != originals_commitment((exact,), version.strip())
    with pytest.raises(ValueError, match="normalizer_version"):
        originals_commitment((original(),), " \t")
