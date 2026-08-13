from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.scenarios import ScenarioMetadata


def test_scenario_metadata_from_payload_preserves_human_facing_fields() -> None:
    metadata = ScenarioMetadata.from_payload(
        {
            "origin": "generated.contract",
            "tags": ["generated", "json-contract"],
            "docs_role": "quickstart",
            "caption": "Doctor detects repairable session-insight gaps.",
            "narrative_order": 10,
            "audience": ["operator"],
            "demonstrates": ["repair-preview", "json-envelope"],
            "privacy_level": "synthetic",
            "media": ["terminal"],
            "visual_style": "plain-terminal",
        }
    )

    assert metadata.origin == "generated.contract"
    assert metadata.tags == ("generated", "json-contract")
    assert metadata.docs_role == "quickstart"
    assert metadata.caption == "Doctor detects repairable session-insight gaps."
    assert metadata.narrative_order == 10
    assert metadata.audience == ("operator",)
    assert metadata.demonstrates == ("repair-preview", "json-envelope")
    assert metadata.privacy_level == "synthetic"
    assert metadata.media == ("terminal",)
    assert metadata.visual_style == "plain-terminal"


def test_scenario_metadata_from_object_uses_empty_defaults() -> None:
    metadata = ScenarioMetadata.from_object(MagicMock())

    assert metadata == ScenarioMetadata()


def test_scenario_metadata_payload_omits_empty_collections() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        tags=("generated",),
        docs_role="reference",
        caption="Session insights are visible to docs projections.",
        narrative_order=20,
        audience=("maintainer",),
        demonstrates=("session-insights",),
        privacy_level="synthetic",
        media=("markdown",),
        visual_style="reference-table",
    )

    assert metadata.to_payload() == {
        "origin": "generated.contract",
        "tags": ["generated"],
        "docs_role": "reference",
        "caption": "Session insights are visible to docs projections.",
        "narrative_order": 20,
        "audience": ["maintainer"],
        "demonstrates": ["session-insights"],
        "privacy_level": "synthetic",
        "media": ["markdown"],
        "visual_style": "reference-table",
    }


def test_scenario_metadata_merges_presentation_fields() -> None:
    defaults = ScenarioMetadata(
        docs_role="tour",
        caption="Default caption",
        narrative_order=30,
        audience=("operator",),
        demonstrates=("query",),
        privacy_level="synthetic",
        media=("terminal",),
        visual_style="plain",
        tags=("default",),
    )
    explicit = ScenarioMetadata(
        caption="Explicit caption",
        audience=("maintainer",),
        demonstrates=("repair",),
        media=("screenshot",),
        tags=("explicit",),
    )

    merged = explicit.with_defaults(defaults)

    assert merged.docs_role == "tour"
    assert merged.caption == "Explicit caption"
    assert merged.narrative_order == 30
    assert merged.audience == ("maintainer",)
    assert merged.demonstrates == ("repair",)
    assert merged.privacy_level == "synthetic"
    assert merged.media == ("screenshot",)
    assert merged.visual_style == "plain"
    assert merged.tags == ("explicit", "default")
