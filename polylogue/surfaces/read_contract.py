"""The shared request and preset contract for archive reads.

Read surfaces may choose a named preset, but they do not define another
selection, projection, or render vocabulary.  This module is deliberately
storage-free; execution remains the responsibility of the query transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    create_model,
    field_validator,
)

from polylogue.archive.query.spec import QUERY_PARAMETER_NAMES, SessionQuerySpec
from polylogue.archive.viewport import READ_VIEW_PROFILES
from polylogue.surfaces.projection_spec import (
    RENDER_FORMAT_ALIASES,
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
            format=str(params["output_format"] if params.get("output_format") is not None else self.format.value),
            destination=str(params.get("destination") or self.destination.value),
            layout=str(params.get("layout") or self.layout),
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
            context_related_limit=_optional_int(params.get("context_related_limit")),
            context_max_sessions=_optional_int(params.get("context_max_sessions")),
            correlation_repo_path=(
                str(params["correlation_repo_path"]) if params.get("correlation_repo_path") is not None else None
            ),
            correlation_since_hours=_optional_int(params.get("correlation_since_hours")),
            correlation_confidence_threshold=(
                float(str(params["correlation_confidence_threshold"]))
                if params.get("correlation_confidence_threshold") is not None
                else None
            ),
            correlation_github_api=(
                _boolean(params["correlation_github_api"], default=False)
                if params.get("correlation_github_api") is not None
                else None
            ),
            redact_paths=_boolean(params.get("redact_paths"), default=True),
            include_assertions=_boolean(params.get("include_assertions"), default=False),
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
        selection: SessionQuerySpec | None = None,
    ) -> ReadRequest:
        """Normalize a surface payload into one request contract."""

        raw = _READ_INPUT.model_validate(params if params is not None else {}).model_dump(exclude_unset=True)
        for key in (
            "latest",
            "reverse",
            "filter_has_tool_use",
            "filter_has_thinking",
            "filter_has_paste",
            "typed_only",
            "root",
        ):
            if raw.get(key) is not None:
                raw[key] = _boolean(raw[key], default=False)
        preset_name = str(preset or raw.get("preset") or "summary")
        selected = read_preset(preset_name)
        projection = selected.projection(raw)
        selection = (
            selection
            if selection is not None
            else SessionQuerySpec.from_params(
                {
                    **raw,
                    "cwd_prefix": raw.get("cwd_prefix", raw.get("project_path")),
                    "repo": raw.get("repo", raw.get("project_repo")),
                }
            )
        )
        return cls(
            selection=selection,
            projection=projection.projection,
            render=projection.render,
            preset=selected.name,
        )


def _default_render_format(formats: tuple[str, ...]) -> RenderFormat:
    """Markdown when the view renders it, else the view's first declared format."""
    if "markdown" in formats:
        return RenderFormat.MARKDOWN
    first = formats[0]
    return RENDER_FORMAT_ALIASES[first] if first in RENDER_FORMAT_ALIASES else RenderFormat(first)


#: One preset per declared read view, generated from the single view
#: declaration rather than restated here.  The hand-maintained list this
#: replaces had drifted: it omitted ``lineage`` and ``effective_context``
#: entirely, so two views the CLI offers had no public preset and nothing
#: compared the two lists (polylogue-dutav).  ``archive.viewport`` is the
#: shared declaration ``cli.read_view_registry`` already validates against, so
#: deriving from it makes the drift impossible rather than merely detectable.
READ_PRESETS: tuple[ReadPreset, ...] = tuple(
    ReadPreset(
        name=profile.view_id,
        description=f"Read the {profile.view_id} view.",
        views=(profile.view_id,),
        format=_default_render_format(profile.formats),
    )
    for profile in READ_VIEW_PROFILES
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


# The machine input is flat. These annotations describe input spellings;
# domain/projection owners still decide semantic validity (for example DSL
# predicates and destination/path combinations).
_InputInteger = StrictInt | Annotated[str, StringConstraints(pattern=r"^(?:[+-]?[0-9]+)?$")] | None
_InputBoolean = StrictBool | Literal[0, 1, "true", "false", "True", "False", "0", "1", "yes", "no", "on", "off"] | None
_InputText = StrictStr | None
_InputTerms = StrictStr | tuple[StrictStr, ...] | None
_INPUT_TYPES: dict[str, Any] = dict.fromkeys(QUERY_PARAMETER_NAMES, _InputText)
for _name in (
    "query",
    "contains",
    "exclude_text",
    "referenced_path",
    "action",
    "exclude_action",
    "action_sequence",
    "action_text",
    "tool",
    "exclude_tool",
    "origin",
    "exclude_origin",
    "tag",
    "exclude_tag",
    "repo",
    "project",
    "has_type",
):
    _INPUT_TYPES[_name] = _InputTerms
for _name in (
    "limit",
    "sample",
    "min_messages",
    "max_messages",
    "min_words",
    "max_words",
    "offset",
    "max_tokens",
    "edge_limit",
    "body_limit",
    "body_offset",
    "neighbor_limit",
    "neighbor_window_hours",
    "context_related_limit",
    "context_max_sessions",
    "correlation_since_hours",
):
    _INPUT_TYPES[_name] = _InputInteger
for _name in (
    "latest",
    "reverse",
    "filter_has_tool_use",
    "filter_has_thinking",
    "filter_has_paste",
    "typed_only",
    "root",
    "correlation_github_api",
    "redact_paths",
    "include_assertions",
):
    _INPUT_TYPES[_name] = _InputBoolean
for _name in ("project_path", "project_repo", "layout", "out", "correlation_repo_path"):
    _INPUT_TYPES[_name] = _InputText
_INPUT_CHOICES = {
    "preset": sorted(_PRESETS),
    "views": sorted(_PRESETS),
    "output_format": sorted({value.value for value in RenderFormat} | set(RENDER_FORMAT_ALIASES)),
    "destination": [value.value for value in RenderDestination],
    "timestamps": ["renderer-default", "include-available", "omit"],
}
for _name, _choices in _INPUT_CHOICES.items():
    _choice_type = Annotated[StrictStr, Field(json_schema_extra={"enum": _choices})]
    _INPUT_TYPES[_name] = tuple[_choice_type, ...] | None if _name == "views" else _choice_type | None
_INPUT_TYPES["correlation_confidence_threshold"] = (
    StrictFloat
    | StrictInt
    | Annotated[str, StringConstraints(pattern=r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")]
    | None
)


def _validate_input_choice(value: object, info: ValidationInfo) -> object:
    if value is None:
        return value
    choices = _INPUT_CHOICES[info.field_name or ""]
    values = value if isinstance(value, tuple) else (value,)
    if any(item not in choices for item in values):
        raise ValueError(f"{info.field_name} must choose from {', '.join(choices)}")
    return value


_INPUT_DEFINITIONS: dict[str, Any] = {name: (annotation, None) for name, annotation in _INPUT_TYPES.items()}
_INPUT_VALIDATORS: dict[str, Callable[..., Any]] = {
    "input_choice": field_validator(*_INPUT_CHOICES)(_validate_input_choice),
}
_READ_INPUT: type[BaseModel] = create_model(
    "ReadInput",
    __config__=ConfigDict(extra="forbid"),
    __validators__=_INPUT_VALIDATORS,
    **_INPUT_DEFINITIONS,
)


def read_input_fields() -> frozenset[str]:
    """The flat read keys a surface adapter may forward."""
    return frozenset(_READ_INPUT.model_fields)


def read_contract_schema() -> dict[str, Any]:
    """Describe the same flat structural input validated by normalization."""
    return _READ_INPUT.model_json_schema()


_BOOLEAN = TypeAdapter(bool)


def _boolean(value: object, *, default: bool) -> bool:
    """Parse an explicit override; absence retains the preset default."""
    return default if value is None else _BOOLEAN.validate_python(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))


__all__ = [
    "READ_PRESETS",
    "ReadPreset",
    "ReadRequest",
    "read_contract_schema",
    "read_input_fields",
    "read_preset",
    "read_preset_catalog",
]
