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


def test_committed_witness_corpus_covers_high_value_surfaces() -> None:
    witnesses = {item.witness_id: item for item in load_committed_witnesses()}

    expected = {
        "cli.command-surface": "tests/data/witnesses/cli-command-surface.json",
        "mcp.tool-schemas": "tests/data/witnesses/mcp-tool-schemas.json",
        "site.publication-contract": "tests/data/witnesses/site-publication-contract.json",
        "storage.blob-store-layout": "tests/data/witnesses/blob-store-layout.json",
        "browser-capture.sequence": "tests/data/witnesses/browser-capture-sequence.json",
    }

    assert expected.items() <= {witness_id: item.path for witness_id, item in witnesses.items()}.items()
    for witness_id in expected:
        witness = witnesses[witness_id]
        assert witness.privacy_classification == "synthetic"
        assert witness.validation_errors() == ()
        assert witness.preserved_semantic_facts


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


def test_known_failing_witnesses_require_rejection_reason() -> None:
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

    assert "known failing witnesses require an explicit rejection_reason" in failing.validation_errors()

    rejected = WitnessMetadata(
        witness_id="live.failing-rejected",
        path="tests/witnesses/live-failing.json",
        origin="live-derived",
        provenance={"source": "live archive probe"},
        preserved_semantic_facts=("parser keeps error boundary stable",),
        minimization_status="minimized",
        privacy_classification="redacted",
        privacy=PrivacyRecord(private_material="not_observed"),
        known_failing=True,
        xfail_strict=True,
        rejection_reason="fixture documents an unsupported provider shape",
    )

    assert rejected.validation_errors() == ()
