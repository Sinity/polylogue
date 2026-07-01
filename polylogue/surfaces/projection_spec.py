"""Typed query projection and rendering contracts.

This module names the algebra that should replace view-specific flag clusters:
query selection chooses evidence rows, projection chooses which evidence
families and bodies are included, and rendering chooses format/layout/destination.
It is intentionally storage-free so CLI, MCP, daemon, and demos can share it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from polylogue.surfaces.payloads import SurfacePayloadModel


class EvidenceFamily(str, Enum):
    """Archive evidence families that a projection can include."""

    SESSIONS = "sessions"
    MESSAGES = "messages"
    BLOCKS = "blocks"
    ACTIONS = "actions"
    RAW = "raw"
    CONTEXT = "context"
    CHRONICLE = "chronicle"
    NEIGHBORS = "neighbors"
    CORRELATION = "correlation"
    TEMPORAL = "temporal"
    ASSERTIONS = "assertions"


class BodyPolicy(str, Enum):
    """How much body text a projection should expose."""

    FULL = "full"
    OMIT_TOOL_OUTPUTS = "omit-tool-outputs"
    AUTHORED_DIALOGUE = "authored-dialogue"
    METADATA_ONLY = "metadata-only"


class RenderFormat(str, Enum):
    """Supported render encodings shared by read/export/report surfaces."""

    MARKDOWN = "markdown"
    JSON = "json"
    NDJSON = "ndjson"
    HTML = "html"
    OBSIDIAN = "obsidian"
    ORG = "org"
    YAML = "yaml"
    PLAINTEXT = "plaintext"
    CSV = "csv"


class RenderDestination(str, Enum):
    """Where rendered output is delivered."""

    TERMINAL = "terminal"
    STDOUT = "stdout"
    BROWSER = "browser"
    CLIPBOARD = "clipboard"
    FILE = "file"


class SelectionSpec(SurfacePayloadModel):
    """Evidence selection independent of projection and rendering."""

    refs: tuple[str, ...] = ()
    query: str | None = None
    origin: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = Field(default=None, ge=1)


class ProjectionSpec(SurfacePayloadModel):
    """Evidence families and body policies to include in a result."""

    families: tuple[EvidenceFamily, ...] = (EvidenceFamily.SESSIONS,)
    fields: tuple[str, ...] = ()
    body_policy: BodyPolicy = BodyPolicy.FULL
    include_roles: tuple[str, ...] = ()
    exclude_block_kinds: tuple[str, ...] = ()
    max_tokens: int | None = Field(default=None, ge=1)
    redact_paths: bool = True

    @model_validator(mode="after")
    def _normalize_policy_exclusions(self) -> ProjectionSpec:
        exclusions = set(self.exclude_block_kinds)
        if self.body_policy is BodyPolicy.OMIT_TOOL_OUTPUTS:
            exclusions.add("tool_result")
            exclusions.add("function_call_output")
        if self.body_policy is BodyPolicy.AUTHORED_DIALOGUE:
            exclusions.update({"tool_use", "tool_result", "function_call", "function_call_output"})
            object.__setattr__(self, "include_roles", tuple(role for role in self.include_roles if role != "tool"))
        object.__setattr__(self, "exclude_block_kinds", tuple(sorted(exclusions)))
        return self


class RenderSpec(SurfacePayloadModel):
    """Output encoding, layout, and destination."""

    format: RenderFormat = RenderFormat.MARKDOWN
    destination: RenderDestination = RenderDestination.TERMINAL
    layout: str = "standard"
    out: str | None = None

    @model_validator(mode="after")
    def _file_destination_requires_path(self) -> RenderSpec:
        if self.destination is RenderDestination.FILE and not self.out:
            raise ValueError("file render destination requires out")
        return self


class QueryProjectionSpec(SurfacePayloadModel):
    """Composable replacement for read/export/recovery view-specific flags."""

    selection: SelectionSpec = Field(default_factory=SelectionSpec)
    projection: ProjectionSpec = Field(default_factory=ProjectionSpec)
    render: RenderSpec = Field(default_factory=RenderSpec)


READ_VIEW_PROJECTION_FAMILIES: dict[str, tuple[EvidenceFamily, ...]] = {
    "summary": (EvidenceFamily.SESSIONS,),
    "transcript": (EvidenceFamily.MESSAGES, EvidenceFamily.BLOCKS),
    "messages": (EvidenceFamily.MESSAGES, EvidenceFamily.BLOCKS),
    "raw": (EvidenceFamily.RAW,),
    "context": (EvidenceFamily.CONTEXT, EvidenceFamily.MESSAGES),
    "context-image": (EvidenceFamily.CONTEXT, EvidenceFamily.MESSAGES, EvidenceFamily.ASSERTIONS),
    "chronicle": (EvidenceFamily.CHRONICLE, EvidenceFamily.SESSIONS, EvidenceFamily.MESSAGES),
    "neighbors": (EvidenceFamily.NEIGHBORS, EvidenceFamily.SESSIONS),
    "correlation": (EvidenceFamily.CORRELATION, EvidenceFamily.ACTIONS),
    "temporal": (EvidenceFamily.TEMPORAL, EvidenceFamily.SESSIONS),
}
"""Projection mapping for executable read views.

Additional named projections may exist outside this map, but every executable
read view must be represented here and this map must not carry obsolete views
that are no longer executable.
"""

NAMED_PROJECTION_FAMILIES: dict[str, tuple[EvidenceFamily, ...]] = {
    **READ_VIEW_PROJECTION_FAMILIES,
    "timeline": (EvidenceFamily.TEMPORAL, EvidenceFamily.SESSIONS, EvidenceFamily.MESSAGES),
}
"""All named projection shortcuts accepted by the projection bridge."""


def projection_from_view(
    view: str,
    *,
    format: str = "markdown",
    destination: str = "terminal",
    max_tokens: int | None = None,
) -> QueryProjectionSpec:
    """Map executable projection/read-view names into the shared vocabulary."""

    try:
        families = NAMED_PROJECTION_FAMILIES[view]
    except KeyError as exc:
        raise ValueError(f"unknown projection view: {view}") from exc
    body_policy = BodyPolicy.AUTHORED_DIALOGUE if view == "chronicle" else BodyPolicy.FULL
    return QueryProjectionSpec(
        projection=ProjectionSpec(families=families, body_policy=body_policy, max_tokens=max_tokens),
        render=RenderSpec(format=RenderFormat(format), destination=RenderDestination(destination)),
    )


__all__ = [
    "BodyPolicy",
    "EvidenceFamily",
    "NAMED_PROJECTION_FAMILIES",
    "ProjectionSpec",
    "QueryProjectionSpec",
    "READ_VIEW_PROJECTION_FAMILIES",
    "RenderDestination",
    "RenderFormat",
    "RenderSpec",
    "SelectionSpec",
    "projection_from_view",
]
