"""CLI package public API shim."""

from ..commands import CommandEnv
from .context import default_html_mode, resolve_html_enabled, resolve_html_settings
from .app import (  # pylint: disable=unused-import
    build_parser,
    main,
    run_completions_cli,
    run_dashboards_cli,
    run_complete_cli,
    run_help_cli,
    run_import_cli,
    run_inspect_branches,
    run_inspect_search,
    run_prune_cli,
    run_stats_cli,
    run_sync_cli,
    run_watch_cli,
    summarize_import,
    _run_sync_drive,
)

__all__ = [
    "CommandEnv",
    "build_parser",
    "default_html_mode",
    "main",
    "resolve_html_enabled",
    "resolve_html_settings",
    "run_completions_cli",
    "run_complete_cli",
    "run_help_cli",
    "run_import_cli",
    "run_inspect_branches",
    "run_inspect_search",
    "run_prune_cli",
    "run_stats_cli",
    "run_sync_cli",
    "run_watch_cli",
    "run_dashboards_cli",
    "summarize_import",
    "_run_sync_drive",
]

# Backwards compatibility for older imports
_resolve_html_settings = resolve_html_settings
