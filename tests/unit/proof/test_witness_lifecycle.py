"""Tests for WitnessLifecycle state machine and privacy gate."""

from __future__ import annotations

from polylogue.proof.witnesses import (
    WITNESS_SCHEMA_VERSION,
    PrivacyRecord,
    WitnessLifecycle,
    WitnessMetadata,
)


def test_witness_lifecycle_new_is_discovered() -> None:
    lc = WitnessLifecycle.new()
    assert lc.state == "discovered"
    assert lc.discovered_at != ""


def test_witness_lifecycle_transitions() -> None:
    lc = WitnessLifecycle.new()
    assert lc.state == "discovered"
    assert lc.minimized_at is None

    lc = lc.transition("minimized")
    assert lc.state == "minimized"
    assert lc.minimized_at is not None

    lc = lc.transition("committed")
    assert lc.state == "committed"
    assert lc.committed_at is not None

    lc = lc.transition("exercised")
    assert lc.state == "exercised"
    assert lc.last_exercised_at is not None

    lc = lc.transition("retired")
    assert lc.state == "retired"
    assert lc.retired_at is not None


def test_witness_lifecycle_roundtrip() -> None:
    original = WitnessLifecycle.new().transition("minimized").transition("committed")
    payload = original.to_payload()
    restored = WitnessLifecycle.from_payload(payload)
    assert restored.state == "committed"
    assert restored.discovered_at == original.discovered_at
    assert restored.minimized_at == original.minimized_at
    assert restored.committed_at == original.committed_at


def test_witness_metadata_privacy_gate() -> None:
    """Committed witnesses without privacy_classification fail validation."""
    metadata = WitnessMetadata(
        witness_id="test-witness",
        path="tests/witnesses/test.witness.json",
        origin="synthetic",
        provenance={},
        preserved_semantic_facts=("deterministic_output",),
        minimization_status="minimized",
        lifecycle=WitnessLifecycle.new().transition("minimized").transition("committed"),
        committed=True,
        schema_version=WITNESS_SCHEMA_VERSION,
    )
    errors = metadata.validation_errors()
    assert any("privacy_classification" in e for e in errors), f"expected privacy_classification error, got: {errors}"


def test_witness_metadata_privacy_gate_passes() -> None:
    """Committed witnesses with privacy_classification pass validation."""
    metadata = WitnessMetadata(
        witness_id="test-witness",
        path="tests/witnesses/test.witness.json",
        origin="synthetic",
        provenance={},
        preserved_semantic_facts=("deterministic_output",),
        minimization_status="minimized",
        privacy_classification="synthetic",
        lifecycle=WitnessLifecycle.new().transition("minimized").transition("committed"),
        committed=True,
        schema_version=WITNESS_SCHEMA_VERSION,
    )
    errors = metadata.validation_errors()
    assert not errors, f"unexpected validation errors: {errors}"


def test_witness_metadata_roundtrip_with_lifecycle() -> None:
    """WitnessMetadata round-trips through JSON with lifecycle and privacy_classification."""
    original = WitnessMetadata(
        witness_id="roundtrip-test",
        path="tests/witnesses/rt.witness.json",
        origin="regression",
        provenance={"source": "pytest"},
        preserved_semantic_facts=("fact1", "fact2"),
        minimization_status="minimized",
        privacy=PrivacyRecord(private_material="not_observed"),
        privacy_classification="synthetic",
        lifecycle=WitnessLifecycle.new().transition("minimized"),
        committed=True,
        schema_version=WITNESS_SCHEMA_VERSION,
    )
    payload = original.to_payload()
    restored = WitnessMetadata.from_payload(payload)
    assert restored.witness_id == original.witness_id
    assert restored.privacy_classification == "synthetic"
    assert restored.lifecycle is not None
    assert restored.lifecycle.state == "minimized"


def test_verify_witness_lifecycle_no_witnesses_is_ok() -> None:
    """verify-witness-lifecycle succeeds when no witnesses exist."""
    import tempfile
    from unittest.mock import patch

    from devtools.verify_witness_lifecycle import main as verify_main

    with patch("devtools.verify_witness_lifecycle.COMMITTED_WITNESS_DIR") as mock_dir:
        with tempfile.TemporaryDirectory() as tmp:
            mock_dir.return_value = type(mock_dir)(tmp)
            mock_dir.resolve.return_value = type(mock_dir)(tmp)
            exit_code = verify_main([])
            assert exit_code == 0
