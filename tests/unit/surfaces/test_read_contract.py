from __future__ import annotations

import pytest

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.surfaces.projection_spec import BodyPolicy, RenderFormat
from polylogue.surfaces.read_contract import (
    READ_PRESETS,
    ReadRequest,
    read_contract_schema,
    read_preset,
    read_preset_catalog,
)


def test_normalizer_owns_selection_projection_and_render() -> None:
    request = ReadRequest.normalize(
        {
            "preset": "dialogue",
            "query": "needle",
            "origin": "chatgpt-export",
            "output_format": "json",
            "destination": "file",
            "out": "dialogue.json",
            "max_tokens": "80",
        }
    )

    assert isinstance(request.selection, SessionQuerySpec)
    assert request.selection.query_terms == ("needle",)
    assert request.selection.origins == ("chatgpt-export",)
    assert request.projection.body_policy is BodyPolicy.AUTHORED_DIALOGUE
    assert request.render.format is RenderFormat.JSON
    assert request.render.out == "dialogue.json"


def test_preset_catalog_and_schema_are_derived_from_the_registry() -> None:
    catalog = read_preset_catalog()
    schema = read_contract_schema()

    assert tuple(entry["name"] for entry in catalog) == tuple(preset.name for preset in READ_PRESETS)
    assert schema["properties"]["preset"]["enum"] == sorted(preset.name for preset in READ_PRESETS)
    assert schema["properties"]["projection"]["fields"]
    assert schema["properties"]["render"]["fields"]


def test_every_declared_read_view_is_normalizable() -> None:
    for preset in READ_PRESETS:
        request = ReadRequest.normalize({}, preset=preset.name)
        assert request.preset == preset.name
        assert request.projection.families


def test_unknown_preset_reports_discovery_choices() -> None:
    with pytest.raises(ValueError, match="unknown read preset"):
        read_preset("not-a-read")
