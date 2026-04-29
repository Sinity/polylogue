from __future__ import annotations

from pathlib import Path

from polylogue.proof.witnesses import (
    COMMITTED_WITNESS_DIR,
    PrivacyRecord,
    WitnessMetadata,
    committed_witness_errors,
    load_committed_witnesses,
)


def test_committed_witness_catalog_classifies_existing_fixture_by_provenance() -> None:
    witnesses = load_committed_witnesses()
    witness = next(item for item in witnesses if item.witness_id == "golden.tool-use-json")

    assert committed_witness_errors(witnesses) == {}
    assert Path(witness.path).exists()
    assert Path(witness.path) == Path("tests/data/golden/tool-use-json.md")
    assert witness.origin == "golden-surface-snapshot"
    assert witness.provenance["fixture_kind"] == "golden-rendering-snapshot"
    assert witness.provenance["raw_material"] == "synthetic"


def test_committed_witness_records_semantic_facts_and_minimization_status() -> None:
    witness = COMMITTED_WITNESS_DIR / "golden-tool-use-json.witness.json"
    metadata = WitnessMetadata.read(witness)

    assert metadata.minimization_status == "minimized"
    assert metadata.preserved_semantic_facts == (
        "tool-use JSON remains rendered as parse-visible fenced content",
        "message role ordering survives markdown rendering",
    )
    assert metadata.validation_errors() == ()


def test_live_derived_witnesses_require_privacy_metadata_before_commit() -> None:
    missing_privacy = WitnessMetadata(
        witness_id="live.missing-privacy",
        path="tests/witnesses/live-missing.json",
        origin="live-derived",
        provenance={"source": "live archive probe"},
        preserved_semantic_facts=("provider remains parseable after minimization",),
        minimization_status="minimized",
        privacy=None,
    )

    assert "committed live-derived witnesses require privacy metadata" in missing_privacy.validation_errors()

    redacted = WitnessMetadata(
        witness_id="live.redacted",
        path="tests/witnesses/live-redacted.json",
        origin="live-derived",
        provenance={"source": "live archive probe"},
        preserved_semantic_facts=("provider remains parseable after minimization",),
        minimization_status="minimized",
        privacy_classification="redacted",
        privacy=PrivacyRecord(
            private_material="observed",
            transformed=True,
            redacted=True,
            discarded=True,
            retained=False,
        ),
    )

    assert redacted.validation_errors() == ()


def test_known_failing_witnesses_require_strict_xfail_or_rejection() -> None:
    failing = WitnessMetadata(
        witness_id="live.failing",
        path="tests/witnesses/live-failing.json",
        origin="live-derived",
        provenance={"source": "live archive probe"},
        preserved_semantic_facts=("parser keeps error boundary stable",),
        minimization_status="minimized",
        privacy=PrivacyRecord(private_material="not_observed"),
        known_failing=True,
    )

    assert "known failing witnesses require strict xfail with linked issue or rejection reason" in (
        failing.validation_errors()
    )

    strict_xfail = WitnessMetadata(
        witness_id="live.failing-xfail",
        path="tests/witnesses/live-failing.json",
        origin="live-derived",
        provenance={"source": "live archive probe"},
        preserved_semantic_facts=("parser keeps error boundary stable",),
        minimization_status="minimized",
        privacy_classification="redacted",
        privacy=PrivacyRecord(private_material="not_observed"),
        known_failing=True,
        xfail_strict=True,
        linked_issue="#335",
    )

    assert strict_xfail.validation_errors() == ()
