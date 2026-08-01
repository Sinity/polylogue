"""Shared scope-filter option/decorator helpers for maintenance commands."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import Origin
from polylogue.maintenance.scope import MaintenanceScopeFilter

#: Human-readable description of where each config layer's value came from,
#: keyed by the layer names `PolylogueConfig.layer_of` returns
#: (`default`/`site`/`user`/`env`/`cli`).
_LAYER_DESCRIPTIONS: dict[str, str] = {
    "env": "POLYLOGUE_ARCHIVE_ROOT environment variable",
    "cli": "CLI flag",
    "default": "built-in default (no config file or environment override)",
}


def archive_root_provenance_line(env: AppEnv) -> str:
    """Return a one-line "archive root + provenance" description.

    Every ``ops maintenance`` command resolves its archive root through the
    same 5-layer precedence as the rest of the CLI (see
    ``polylogue/config.py``'s module docstring), but a break-glass operator
    running in an unfamiliar shell has no way to tell whether that root came
    from ``polylogue.toml`` or from a stray ``POLYLOGUE_ARCHIVE_ROOT`` left
    over from a devshell/demo session (polylogue-l1qg: an operator drew
    "archive looks clean" conclusions from a scratch archive this way). This
    is printed unconditionally at the top of every maintenance invocation --
    not gated behind ``--verbose`` -- because the whole point is that the
    operator should not have to know to ask for it.
    """
    settings = env.runtime.settings
    layer = settings.layer_of("archive_root")
    archive_root = env.config.archive_root
    if layer in ("site", "user"):
        config_path = settings.layer_paths.get(layer)
        source = f"{layer} config file ({config_path})" if config_path else f"{layer} config file"
    else:
        source = _LAYER_DESCRIPTIONS.get(layer, layer)
    return f"Archive root: {archive_root} [source: {source}]"


def print_archive_root_provenance(env: AppEnv) -> None:
    """Print :func:`archive_root_provenance_line` to stderr.

    Stderr (not stdout) so the banner never corrupts a subcommand's
    ``--output-format json`` payload piped downstream, while still landing
    in the operator's terminal by default.
    """
    click.echo(archive_root_provenance_line(env), err=True)


def _build_scope_filter(
    *,
    session_ids: tuple[str, ...],
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> MaintenanceScopeFilter:
    """Translate CLI options into a :class:`MaintenanceScopeFilter`.

    Helper exists so the CLI ``plan`` and ``run`` commands share one
    parsing path and one error surface.
    """

    time_range: tuple[datetime, datetime] | None
    if since is not None or until is not None:
        if since is None or until is None:
            raise click.UsageError("--since and --until must be supplied together")
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError as exc:
            raise click.UsageError(f"--since/--until must be ISO-8601 timestamps: {exc}") from exc
        time_range = (since_dt, until_dt)
    else:
        time_range = None

    return MaintenanceScopeFilter(
        session_ids=session_ids if session_ids else None,
        origin=Origin(origin).value if origin is not None else None,
        source_family=source_family,
        source_root=Path(source_root) if source_root else None,
        time_range=time_range,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )


_SCOPE_FILTER_OPTIONS = [
    click.option(
        "--session-id",
        "session_ids",
        multiple=True,
        help="Restrict scope to one or more session ids (repeatable).",
    ),
    click.option("--origin", "-o", type=str, default=None, help="Restrict scope to one origin token."),
    click.option(
        "--source-family",
        type=str,
        default=None,
        help="Restrict scope to one source family (e.g. claude-code-session).",
    ),
    click.option(
        "--source-root",
        type=str,
        default=None,
        help="Restrict scope to one source runtime root (e.g. ~/.claude/projects).",
    ),
    click.option("--since", type=str, default=None, help="Inclusive ISO-8601 lower bound of the time range."),
    click.option("--until", type=str, default=None, help="Inclusive ISO-8601 upper bound of the time range."),
    click.option(
        "--failure-kind",
        type=str,
        default=None,
        help="Restrict scope to attempts that failed with one kind.",
    ),
    click.option(
        "--parser-version",
        type=str,
        default=None,
        help="Restrict scope to one parser/materializer version.",
    ),
]


def _apply_scope_filter_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator stacking the shared scope-filter options onto a command."""
    for option in reversed(_SCOPE_FILTER_OPTIONS):
        fn = option(fn)
    return fn
