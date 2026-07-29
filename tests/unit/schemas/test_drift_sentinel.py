"""Tests for the format-drift sentinel classification (polylogue-da1)."""

from __future__ import annotations

from polylogue.schemas.drift_sentinel import (
    BENIGN_CLASSIFICATIONS,
    FIELD_CHANGED,
    KNOWN_FIELD_UNREAD,
    NEW_FIELD,
    RISKY_CLASSIFICATIONS,
    UNSEEN_SHAPE,
    classify_schema_drift,
    is_risky,
    unseen_key_signature,
)


def test_exact_match_with_no_drift_classifies_as_none() -> None:
    """A payload that resolved to a real package and validated cleanly, with
    no unknown fields, carries no drift signal at all."""
    assert (
        classify_schema_drift(
            resolution_reason="exact_structure",
            is_valid=True,
            drift_warnings=[],
        )
        is None
    )


def test_new_optional_field_classifies_as_benign_new_field() -> None:
    """An additional field the schema permits (has_additional truthy) is the
    common, benign case: valid payload but with drift warnings."""
    classification = classify_schema_drift(
        resolution_reason="exact_structure",
        is_valid=True,
        drift_warnings=["Unexpected field: metadata.newThing"],
    )
    assert classification == NEW_FIELD
    assert classification in BENIGN_CLASSIFICATIONS
    assert not is_risky(classification)


def test_failed_validation_classifies_as_risky_field_changed() -> None:
    """A required field disappearing or changing type fails JSON Schema
    validation -- this is the risky case regardless of resolution reason."""
    classification = classify_schema_drift(
        resolution_reason="exact_structure",
        is_valid=False,
        drift_warnings=[],
    )
    assert classification == FIELD_CHANGED
    assert classification in RISKY_CLASSIFICATIONS
    assert is_risky(classification)


def test_unseen_shape_resolution_classifies_as_risky_unseen_shape() -> None:
    """A payload that matched no real package/profile/bundle (fell back to
    package_default) but still validates is an unseen shape, not benign."""
    classification = classify_schema_drift(
        resolution_reason="package_default",
        is_valid=True,
        drift_warnings=[],
    )
    assert classification == UNSEEN_SHAPE
    assert is_risky(classification)


def test_validation_failure_outranks_unseen_shape_resolution() -> None:
    """A payload can simultaneously be an unseen shape AND fail validation;
    the failure (more actionable) wins the classification."""
    classification = classify_schema_drift(
        resolution_reason="package_default",
        is_valid=False,
        drift_warnings=["Unexpected field: foo"],
    )
    assert classification == FIELD_CHANGED


def test_unseen_key_signature_is_order_independent_and_deduplicated() -> None:
    warnings_a = [
        "Unexpected field: metadata.newThing",
        "Unexpected field: root.otherThing",
    ]
    warnings_b = list(reversed(warnings_a)) + ["Unexpected field: metadata.newThing"]
    assert unseen_key_signature(warnings_a) == unseen_key_signature(warnings_b)
    assert unseen_key_signature([]) == ""


def test_unseen_key_signature_falls_back_to_raw_text_for_unrecognized_prefix() -> None:
    assert unseen_key_signature(["some other warning text"]) == "some other warning text"


def test_clean_payload_with_unread_known_field_classifies_as_risky_known_field_unread() -> None:
    """The fourth classification (polylogue-2qx.3): a payload that clears
    every other axis -- real resolution, valid, no unknown fields -- but
    carries a field the schema declares and no parser reads is a defect,
    not "no drift". Without this, such payloads silently classified as
    ``None`` (benign)."""
    classification = classify_schema_drift(
        resolution_reason="exact_structure",
        is_valid=True,
        drift_warnings=[],
        unread_known_fields=["structuredPatch"],
    )
    assert classification == KNOWN_FIELD_UNREAD
    assert classification in RISKY_CLASSIFICATIONS
    assert classification not in BENIGN_CLASSIFICATIONS
    assert is_risky(classification)


def test_unread_known_fields_is_a_fallback_behind_the_other_three_axes() -> None:
    """unread_known_fields only matters once validation failure, unseen
    shape, and new-field warnings are all ruled out -- those three are
    each a stronger, more specific signal than the coverage join."""
    assert (
        classify_schema_drift(
            resolution_reason="exact_structure",
            is_valid=False,
            drift_warnings=[],
            unread_known_fields=["structuredPatch"],
        )
        == FIELD_CHANGED
    )
    assert (
        classify_schema_drift(
            resolution_reason="package_default",
            is_valid=True,
            drift_warnings=[],
            unread_known_fields=["structuredPatch"],
        )
        == UNSEEN_SHAPE
    )
    assert (
        classify_schema_drift(
            resolution_reason="exact_structure",
            is_valid=True,
            drift_warnings=["Unexpected field: metadata.newThing"],
            unread_known_fields=["structuredPatch"],
        )
        == NEW_FIELD
    )


def test_no_unread_known_fields_still_classifies_as_no_drift() -> None:
    assert (
        classify_schema_drift(
            resolution_reason="exact_structure",
            is_valid=True,
            drift_warnings=[],
            unread_known_fields=[],
        )
        is None
    )
