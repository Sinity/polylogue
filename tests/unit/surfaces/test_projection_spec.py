from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.cli.read_view_handlers import READ_VIEW_HANDLERS
from polylogue.surfaces.projection_spec import (
    READ_VIEW_PROJECTION_FAMILIES,
    BodyPolicy,
    EvidenceFamily,
    ProjectionSpec,
    RenderDestination,
    RenderFormat,
    RenderSpec,
    projection_from_legacy_view,
)


def test_legacy_messages_view_maps_to_message_block_projection() -> None:
    spec = projection_from_legacy_view("messages", format="json", destination="stdout", max_tokens=1200)

    assert spec.projection.families == (EvidenceFamily.MESSAGES, EvidenceFamily.BLOCKS)
    assert spec.projection.max_tokens == 1200
    assert spec.render.format is RenderFormat.JSON
    assert spec.render.destination is RenderDestination.STDOUT


def test_tool_output_omission_is_projection_policy_not_cli_flag() -> None:
    spec = ProjectionSpec(body_policy=BodyPolicy.OMIT_TOOL_OUTPUTS)

    assert "tool_result" in spec.exclude_block_kinds
    assert "function_call_output" in spec.exclude_block_kinds


def test_authored_dialogue_policy_omits_tool_blocks_and_tool_role() -> None:
    spec = ProjectionSpec(
        body_policy=BodyPolicy.AUTHORED_DIALOGUE,
        include_roles=("user", "assistant", "tool"),
    )

    assert spec.include_roles == ("user", "assistant")
    assert {"tool_use", "tool_result", "function_call", "function_call_output"} <= set(spec.exclude_block_kinds)


def test_file_render_destination_requires_path() -> None:
    with pytest.raises(ValidationError):
        RenderSpec(destination=RenderDestination.FILE)

    spec = RenderSpec(destination=RenderDestination.FILE, out="artifact.md")
    assert spec.out == "artifact.md"


def test_unknown_legacy_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown projection view"):
        projection_from_legacy_view("recovery")


def test_executable_read_views_have_projection_mapping() -> None:
    mapped = {view: projection_from_legacy_view(view).projection.families for view in READ_VIEW_HANDLERS}

    assert set(READ_VIEW_PROJECTION_FAMILIES) == set(READ_VIEW_HANDLERS)
    assert set(mapped) == set(READ_VIEW_HANDLERS)
    assert "recovery" not in READ_VIEW_PROJECTION_FAMILIES
    assert "context-pack" not in READ_VIEW_PROJECTION_FAMILIES
