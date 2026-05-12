"""Command registration for the root Click application.

Commands that pull in the heavy archive/storage/daemon import chain are
loaded lazily so ``--help`` and tab-completion never pay the ~2.5 s
startup cost.  Lightweight commands (no archive/storage/daemon deps)
are imported eagerly so their Click metadata is available immediately.
"""

from __future__ import annotations

import importlib

import click

# ── lightweight (eager) commands ────────────────────────────────────
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.config import config_command
from polylogue.cli.commands.cost import cost_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.export import export_command
from polylogue.cli.commands.neighbors import neighbors_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.resume import resume_command
from polylogue.cli.commands.tags import tags_command

# ── lazy-command wrapper ───────────────────────────────────────────


class _LazyCommand(click.Command):
    """Click command that defers module import until invocation or help."""

    def __init__(self, module: str, attr: str) -> None:
        super().__init__(name=attr.replace("_command", "").replace("_group", ""))
        self.__lazy_module = module
        self.__lazy_attr = attr
        self.__resolved: click.Command | None = None

    def _resolve(self) -> click.Command:
        if self.__resolved is None:
            mod = importlib.import_module(self.__lazy_module)
            self.__resolved = getattr(mod, self.__lazy_attr)
        return self.__resolved

    def invoke(self, ctx: click.Context) -> object:
        return self._resolve().invoke(ctx)

    def get_help(self, ctx: click.Context) -> str:
        return self._resolve().get_help(ctx)

    def get_short_help_str(self, limit: int = 45) -> str:
        return self._resolve().get_short_help_str(limit)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        return self._resolve().get_params(ctx)

    def collect_usage_pieces(self, ctx: click.Context) -> list[str]:
        return self._resolve().collect_usage_pieces(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return self._resolve().parse_args(ctx, args)

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        return self._resolve().get_help_record(ctx)  # type: ignore[no-any-return,attr-defined]

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[click.shell_completion.CompletionItem]:
        return self._resolve().shell_complete(ctx, incomplete)

    def list_commands(self, ctx: click.Context):  # type: ignore[no-untyped-def]
        return self._resolve().list_commands(ctx)  # type: ignore[attr-defined]


def _L(name: str) -> _LazyCommand:  # noqa: N802
    """Shorthand for constructing lazy commands in the ROOT_COMMANDS tuple."""
    module = f"polylogue.cli.commands.{name}"
    attr = f"{name}_group" if name in ("diagnostics", "maintenance") else f"{name}_command"
    return _LazyCommand(module, attr)


# ── command list ───────────────────────────────────────────────────

ROOT_COMMANDS: tuple[click.Command, ...] = (
    _L("context_pack"),
    _L("backup"),
    _L("check"),
    config_command,
    cost_command,
    reset_command,
    _L("status"),
    _L("ingest"),
    _L("auth"),
    completions_command,
    dashboard_command,
    neighbors_command,
    export_command,
    resume_command,
    _L("insights"),
    tags_command,
    _L("schema"),
    _L("diagnostics"),
    _L("maintenance"),
)


def register_root_commands(group: click.Group) -> None:
    """Attach the canonical root subcommands to the main CLI group."""
    for command in ROOT_COMMANDS:
        group.add_command(command)


# Backward-compat aliases for callers that import individual commands
# directly (tests).  These resolve lazily like the tuple entries above.
maintenance_group: click.Command = _L("maintenance")

__all__ = [
    "completions_command",
    "dashboard_command",
    "export_command",
    "maintenance_group",
    "neighbors_command",
    "register_root_commands",
    "resume_command",
]
