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
    projection_from_view,
    projection_from_views,
)


def test_messages_view_maps_to_message_block_projection() -> None:
    spec = projection_from_views(
        ("messages",),
        format="json",
        destination="stdout",
        layout="compact",
        max_tokens=1200,
        body_limit=7,
        body_offset=2,
    )

    assert spec.projection.families == (EvidenceFamily.MESSAGES, EvidenceFamily.BLOCKS)
    assert spec.projection.max_tokens == 1200
    assert spec.projection.body_limit == 7
    assert spec.projection.body_offset == 2
    assert spec.render.format is RenderFormat.JSON
    assert spec.render.destination is RenderDestination.STDOUT
    assert spec.render.layout == "compact"


def test_chronicle_view_maps_to_authored_dialogue_projection() -> None:
    spec = projection_from_views(("chronicle",), edge_limit=3)

    assert spec.projection.families == (
        EvidenceFamily.CHRONICLE,
        EvidenceFamily.SESSIONS,
        EvidenceFamily.MESSAGES,
    )
    assert spec.projection.body_policy is BodyPolicy.AUTHORED_DIALOGUE
    assert spec.projection.edge_limit == 3
    assert {"tool_use", "tool_result", "function_call", "function_call_output"} <= set(
        spec.projection.exclude_block_kinds
    )


def test_neighbors_view_maps_to_neighbor_projection_policy() -> None:
    spec = projection_from_views(("neighbors",), neighbor_limit=4, neighbor_window_hours=12)

    assert spec.projection.families == (EvidenceFamily.NEIGHBORS, EvidenceFamily.SESSIONS)
    assert spec.projection.neighbor_limit == 4
    assert spec.projection.neighbor_window_hours == 12


def test_multi_view_projection_dedupes_families_and_preserves_body_policy() -> None:
    spec = projection_from_views(
        ("temporal", "chronicle"),
        format="json",
        destination="stdout",
        query="repo:polylogue",
        origin="claude-code-session",
        max_tokens=2000,
        limit=8,
    )

    assert spec.selection.query == "repo:polylogue"
    assert spec.selection.origin == "claude-code-session"
    assert spec.selection.limit == 8
    assert spec.projection.families == (
        EvidenceFamily.TEMPORAL,
        EvidenceFamily.SESSIONS,
        EvidenceFamily.CHRONICLE,
        EvidenceFamily.MESSAGES,
    )
    assert spec.projection.body_policy is BodyPolicy.AUTHORED_DIALOGUE
    assert spec.projection.max_tokens == 2000
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


def test_unknown_projection_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown projection view"):
        projection_from_view("recovery")


def test_executable_read_views_have_projection_mapping() -> None:
    mapped = {view: projection_from_view(view).projection.families for view in READ_VIEW_HANDLERS}

    assert set(READ_VIEW_PROJECTION_FAMILIES) == set(READ_VIEW_HANDLERS)
    assert set(mapped) == set(READ_VIEW_HANDLERS)
    assert "recovery" not in READ_VIEW_PROJECTION_FAMILIES
    assert "context-pack" not in READ_VIEW_PROJECTION_FAMILIES
