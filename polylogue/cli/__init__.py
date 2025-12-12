"""CLI package public API shim."""

from ..commands import CommandEnv
from .context import default_html_mode, resolve_html_enabled, resolve_html_settings
from .click_app import cli, main  # Click entrypoint

_resolve_html_settings = resolve_html_settings

__all__ = [
    "CommandEnv",
    "cli",
    "default_html_mode",
    "main",
    "resolve_html_enabled",
    "resolve_html_settings",
    "_resolve_html_settings",
]
