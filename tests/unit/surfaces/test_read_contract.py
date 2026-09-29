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


def test_normalizer_preserves_already_lowered_selection() -> None:
    selection = SessionQuerySpec.from_params({"query": "needle", "origin": "chatgpt-export", "filter_has_paste": True})

    request = ReadRequest.normalize({"selection": selection}, preset="summary")

    assert request.selection is selection
    assert request.selection.filter_has_paste is True


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


@pytest.mark.parametrize("raw", [False, "false", "False", "0", 0])
def test_read_boolean_false_overrides_are_not_truthy_strings(raw: object) -> None:
    """bool('false') enables assertions and redaction instead of honoring false."""
    request = ReadRequest.normalize({"include_assertions": raw, "redact_paths": raw})
    assert request.projection.include_assertions is False
    assert request.projection.redact_paths is False


def test_null_read_boolean_overrides_preserve_defaults() -> None:
    """Converting explicit None with bool() incorrectly disables redaction."""
    request = ReadRequest.normalize({"include_assertions": None, "redact_paths": None})
    assert request.projection.include_assertions is False
    assert request.projection.redact_paths is True


@pytest.mark.parametrize("raw", [False, "false", "0", 0])
def test_read_github_correlation_false_stays_disabled(raw: object) -> None:
    """bool('false') incorrectly enables the optional GitHub API route."""
    request = ReadRequest.normalize({"correlation_github_api": raw})
    assert request.projection.correlation_github_api is False


@pytest.mark.parametrize("field", ["include_assertions", "redact_paths", "correlation_github_api"])
def test_invalid_read_boolean_overrides_are_refused(field: str) -> None:
    """Unknown boolean spellings must not silently enable an external operation."""
    with pytest.raises(ValueError):
        ReadRequest.normalize({field: "perhaps"})


def test_default_read_formats_belong_to_their_profile() -> None:
    """A universal Markdown preset fails for each JSON-only production view."""
    from polylogue.archive.viewport import READ_VIEW_PROFILES
    from polylogue.surfaces.projection_spec import RENDER_FORMAT_ALIASES

    for profile in READ_VIEW_PROFILES:
        request = ReadRequest.normalize({}, preset=profile.view_id)
        formats = {
            RENDER_FORMAT_ALIASES[value] if value in RENDER_FORMAT_ALIASES else RenderFormat(value)
            for value in profile.formats
        }
        assert request.render.format in formats
        assert read_preset(profile.view_id).format == request.render.format
