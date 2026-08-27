"""The shared request and preset contract for archive reads.

Read surfaces may choose a named preset, but they do not define another
selection, projection, or render vocabulary.  This module is deliberately
storage-free; execution remains the responsibility of the query transaction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.surfaces.projection_spec import (
    ProjectionSpec,
    QueryProjectionSpec,
    RenderDestination,
    RenderFormat,
    RenderSpec,
    projection_from_views,
)


@dataclass(frozen=True, slots=True)
class ReadPreset:
    """A named, surface-independent default for a read."""

    name: str
    description: str
    views: tuple[str, ...]
    format: RenderFormat = RenderFormat.MARKDOWN
    destination: RenderDestination = RenderDestination.TERMINAL
    layout: str = "standard"

    def projection(self, params: Mapping[str, object] | None = None) -> QueryProjectionSpec:
        """Build the canonical projection for this preset and raw overrides."""

        params = params or {}
        configured_views = params.get("views")
        views = (
            tuple(str(view) for view in configured_views) if isinstance(configured_views, tuple | list) else self.views
        )
        return projection_from_views(
            views,
            format=str(params.get("output_format", self.format.value)),
            destination=str(params.get("destination", self.destination.value)),
            layout=str(params.get("layout", self.layout)),
            timestamps=str(params["timestamps"]) if params.get("timestamps") is not None else None,
            max_tokens=_optional_int(params.get("max_tokens")),
            out=str(params["out"]) if params.get("out") is not None else None,
            query=str(params["query"]) if params.get("query") is not None else None,
            origin=str(params["origin"]) if params.get("origin") is not None else None,
            since=str(params["since"]) if params.get("since") is not None else None,
            until=str(params["until"]) if params.get("until") is not None else None,
            project_path=str(params["project_path"]) if params.get("project_path") is not None else None,
            project_repo=str(params["project_repo"]) if params.get("project_repo") is not None else None,
            limit=_optional_int(params.get("limit")),
            edge_limit=_optional_int(params.get("edge_limit")),
            body_limit=_optional_int(params.get("body_limit")),
            body_offset=_optional_int(params.get("body_offset")),
            neighbor_limit=_optional_int(params.get("neighbor_limit")),
            neighbor_window_hours=_optional_int(params.get("neighbor_window_hours")),
            redact_paths=bool(params.get("redact_paths", True)),
            include_assertions=bool(params.get("include_assertions", False)),
        )


@dataclass(frozen=True, slots=True)
class ReadRequest:
    """Canonical Selection × Projection × Render request."""

    selection: SessionQuerySpec
    projection: ProjectionSpec
    render: RenderSpec
    preset: str = "summary"

    @classmethod
    def normalize(
        cls,
        params: Mapping[str, object] | None = None,
        *,
        preset: str | None = None,
    ) -> ReadRequest:
        """Normalize a surface payload into one request contract."""

        raw = params or {}
        preset_name = str(preset or raw.get("preset") or "summary")
        selected = read_preset(preset_name)
        projection = selected.projection(raw)
        return cls(
            selection=SessionQuerySpec.from_params(raw),
            projection=projection.projection,
            render=projection.render,
            preset=selected.name,
        )


READ_PRESETS: tuple[ReadPreset, ...] = tuple(
    ReadPreset(name=view, description=f"Read the {view} view.", views=(view,))
    for view in (
        "summary",
        "transcript",
        "dialogue",
        "messages",
        "raw",
        "hooks",
        "events",
        "file-edits",
        "agent-policies",
        "web-content",
        "context",
        "context-image",
        "neighbors",
        "correlation",
        "temporal",
        "chronicle",
    )
)
_PRESETS = {preset.name: preset for preset in READ_PRESETS}


def read_preset(name: str) -> ReadPreset:
    """Return a declared preset or raise a useful contract error."""

    try:
        return _PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown read preset {name!r}; choose one of: {available}") from exc


def read_preset_catalog() -> tuple[dict[str, object], ...]:
    """Return generated discovery metadata for every public read preset."""

    return tuple(
        {
            "name": preset.name,
            "description": preset.description,
            "views": list(preset.views),
            "format": preset.format.value,
            "destination": preset.destination.value,
            "layout": preset.layout,
        }
        for preset in READ_PRESETS
    )


def read_contract_schema() -> dict[str, Any]:
    """Describe the request fields without duplicating field inventories."""

    return {
        "type": "object",
        "required": [field.name for field in fields(ReadRequest) if field.name != "preset"],
        "properties": {
            "selection": {"type": "object", "fields": [field.name for field in fields(SessionQuerySpec)]},
            "projection": {"type": "object", "fields": _field_names(ProjectionSpec)},
            "render": {"type": "object", "fields": _field_names(RenderSpec)},
            "preset": {"type": "string", "enum": sorted(_PRESETS)},
        },
    }


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


def _field_names(model: type[object]) -> list[str]:
    """Read a model's authoritative field declaration for discovery output."""

    return list(model.model_fields)  # type: ignore[attr-defined]


__all__ = [
    "READ_PRESETS",
    "ReadPreset",
    "ReadRequest",
    "read_contract_schema",
    "read_preset",
    "read_preset_catalog",
]
