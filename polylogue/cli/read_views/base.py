"""Shared contracts for executable CLI read views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID
from polylogue.cli.read_view_registry import ReadViewSessionPolicy
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.surfaces.projection_spec import QueryProjectionSpec


ReadViewOptionName = str


@dataclass(frozen=True, slots=True)
class ReadViewMessageOptions:
    """Options owned by the messages/raw read views."""

    limit: int | None = None
    offset: int = 0
    full: bool = False


@dataclass(frozen=True, slots=True)
class ReadViewContextOptions:
    """Options owned by the context preamble view."""

    related_limit: int = 5


@dataclass(frozen=True, slots=True)
class ReadViewContextImageOptions:
    """Options owned by the project context-image view."""

    project_path: str | None = None
    project_repo: str | None = None
    since: str | None = None
    until: str | None = None
    origin: str | None = None
    query: str | None = None
    max_sessions: int = 5
    no_redact: bool = False


@dataclass(frozen=True, slots=True)
class ReadViewNeighborOptions:
    """Options owned by the neighbor discovery view."""

    limit: int | None = None
    window_hours: int = 24


@dataclass(frozen=True, slots=True)
class ReadViewCorrelationOptions:
    """Options owned by the correlation evidence view."""

    repo_path: str | None = None
    since_hours: int = 2
    confidence_threshold: float = 0.3
    github_api: bool = True
    otlp: bool = False


@dataclass(frozen=True, slots=True)
class ReadViewChronicleOptions:
    """Options owned by the bounded chronicle read view."""

    edge_limit: int = 8


@dataclass(frozen=True, slots=True)
class ReadViewEventsOptions:
    """Options owned by the raw session-events read view."""

    limit: int | None = None


ReadViewOptions = (
    ReadViewMessageOptions
    | ReadViewContextOptions
    | ReadViewContextImageOptions
    | ReadViewNeighborOptions
    | ReadViewCorrelationOptions
    | ReadViewChronicleOptions
    | ReadViewEventsOptions
)
ReadViewOptionValues = Mapping[str, object]
ReadViewOptionBuilder = Callable[[ReadViewOptionValues], ReadViewOptions | None]


@dataclass(frozen=True, slots=True)
class ReadViewInvocation:
    """Common transport and selection fields for one read-view execution."""

    view: str
    session_id: str | None
    output_format: str | None
    destination: str
    out_path: str | None
    options: ReadViewOptions | None = None
    explicit_options: frozenset[ReadViewOptionName] = frozenset()
    projection_spec: QueryProjectionSpec | None = None


ReadViewHandlerFunc = Callable[[AppEnv, "RootModeRequest", ReadViewInvocation], None]


@dataclass(frozen=True, slots=True)
class ReadViewHandler:
    """Executable handler contract for a read-view id."""

    view_id: str
    session_policy: ReadViewSessionPolicy
    handler: ReadViewHandlerFunc
    default_format: str | None = None
    accepted_options: frozenset[ReadViewOptionName] = frozenset()
    option_builder: ReadViewOptionBuilder | None = None
    accepts_query_set: bool = False

    def validate(self, invocation: ReadViewInvocation, request: RootModeRequest) -> None:
        """Validate cross-view selection rules before executing the handler."""

        if self.session_policy == "required" and invocation.session_id is None:
            raise click.UsageError(
                f"read --view {self.view_id} requires a session ID (use --id, id:prefix, or --latest)."
            )
        if self.session_policy == "query_or_session" and invocation.session_id is None:
            query_seed = " ".join(request.query_terms).strip()
            if not query_seed:
                raise click.UsageError(
                    f"read --view {self.view_id} requires a seed (use --id, id:prefix, --latest, or a query)."
                )
        if invocation.output_format is not None:
            profile = READ_VIEW_PROFILE_BY_ID[self.view_id]
            if invocation.output_format not in profile.formats:
                supported = ", ".join(profile.formats)
                raise click.UsageError(
                    f"read --view {self.view_id} does not support --format {invocation.output_format}. "
                    f"Supported formats: {supported}."
                )
        unsupported_options = sorted(invocation.explicit_options - self.accepted_options)
        if unsupported_options:
            options = ", ".join(f"--{option.replace('_', '-')}" for option in unsupported_options)
            raise click.UsageError(f"read --view {self.view_id} does not use {options}.")

    def build_options(self, values: ReadViewOptionValues) -> ReadViewOptions | None:
        """Build typed view options from raw Click option values."""

        if self.option_builder is None:
            return None
        return self.option_builder(values)


def _warn_on_secret_candidates(env: AppEnv, content: str, *, label: str) -> None:
    """Warn (never block) when rendered/exported content has credential shapes.

    Candidate-only, same detector and never-log-the-literal invariant as
    ``polylogue ops scan-secrets`` (``polylogue/security/secret_scan.py``).
    This is the render/export half of polylogue-t9xd (leak-surfaces L11):
    before this, nothing scanned a session at the point it left the archive
    as a file on disk, whatever credentials happened to be pasted into the
    original conversation went out unexamined.
    """
    from polylogue.security.secret_scan import describe_secret_candidate_spans, scan_text_for_secret_candidates

    spans = scan_text_for_secret_candidates(content)
    if not spans:
        return
    env.ui.console.print(
        f"[yellow]secret-scan: {label}: {describe_secret_candidate_spans(spans)} -- "
        "review before sharing this file (candidate detector, not proof of a real secret).[/yellow]"
    )


def deliver_content(
    env: AppEnv,
    content: str,
    *,
    destination: str,
    out_path: str | None,
    output_format: str = "markdown",
    session: Session | None = None,
) -> None:
    """Deliver captured content to the requested destination.

    ``destination`` must be one of the values in ``_READ_DESTINATIONS``
    (``query_verbs.py``): ``terminal``/``stdout``, ``clipboard``, ``file``, or
    ``browser``. An unrecognized destination raises rather than silently
    falling back to terminal output -- a prior version of this dispatch let
    any unhandled destination (including the valid-but-unwired ``browser``)
    degrade to a plain ``click.echo`` (polylogue-bvnz).
    """

    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        _warn_on_secret_candidates(env, content, label=out_path)
        Path(out_path).write_text(content, encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
    elif destination == "clipboard":
        from polylogue.cli.query_output import copy_to_clipboard

        copy_to_clipboard(env, content)
    elif destination == "browser":
        from polylogue.cli.query_output import open_in_browser

        open_in_browser(env, content, output_format, session)
    elif destination in ("stdout", "terminal"):
        click.echo(content, nl=False)
    else:
        raise click.UsageError(f"Unrecognized read destination: {destination!r}.")


def execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Execute query with verb-modified params."""

    from polylogue.cli.query import execute_query_request as _execute_query_request

    _execute_query_request(env, request)


__all__ = [
    "ReadViewContextOptions",
    "ReadViewContextImageOptions",
    "ReadViewCorrelationOptions",
    "ReadViewChronicleOptions",
    "ReadViewEventsOptions",
    "ReadViewHandler",
    "ReadViewHandlerFunc",
    "ReadViewInvocation",
    "ReadViewOptionName",
    "ReadViewMessageOptions",
    "ReadViewNeighborOptions",
    "ReadViewOptionBuilder",
    "ReadViewOptions",
    "ReadViewOptionValues",
    "deliver_content",
    "execute_query_request",
]
