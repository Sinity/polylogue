from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.scenarios import ScenarioMetadata


def test_scenario_metadata_from_payload_normalizes_strings_and_targets() -> None:
    metadata = ScenarioMetadata.from_payload(
        {
            "origin": "generated.contract",
            "artifact_targets": ["doctor_runtime", "message_fts"],
            "operation_targets": ("cli.doctor",),
            "tags": ["generated", "json-contract"],
        }
    )

    assert metadata.origin == "generated.contract"
    assert metadata.artifact_targets == ("doctor_runtime", "message_fts")
    assert metadata.operation_targets == ("cli.doctor",)
    assert metadata.tags == ("generated", "json-contract")


def test_scenario_metadata_from_object_falls_back_for_mock_attributes() -> None:
    obj = MagicMock()

    metadata = ScenarioMetadata.from_object(obj)

    assert metadata.origin == "authored"
    assert metadata.artifact_targets == ()
    assert metadata.operation_targets == ()
    assert metadata.tags == ()


def test_scenario_metadata_payload_omits_empty_collections() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        artifact_targets=("doctor_runtime",),
        operation_targets=(),
        tags=("generated",),
    )

    assert metadata.to_payload() == {
        "origin": "generated.contract",
        "artifact_targets": ["doctor_runtime"],
        "tags": ["generated"],
    }
