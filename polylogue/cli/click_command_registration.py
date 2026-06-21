"""Lazy command registration for the root Click application."""

from __future__ import annotations

import importlib

import click

from polylogue.cli.commands.config import config_command

# ── lazy-command wrapper ───────────────────────────────────────────


class _LazyCommand(click.Command):
    """Click command that defers module import until invocation or help."""

    def __init__(self, name: str, module: str, attr: str, *, short_help: str | None = None) -> None:
        super().__init__(name=name, short_help=short_help)
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
        return super().get_short_help_str(limit)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        return self._resolve().get_params(ctx)

    def collect_usage_pieces(self, ctx: click.Context) -> list[str]:
        return self._resolve().collect_usage_pieces(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return self._resolve().parse_args(ctx, args)

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[click.shell_completion.CompletionItem]:
        return self._resolve().shell_complete(ctx, incomplete)


class _LazyGroup(_LazyCommand, click.Group):
    """Lazy proxy for Click groups that need nested command dispatch."""

    def invoke(self, ctx: click.Context) -> object:
        return self._resolve().invoke(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        return click.Group.parse_args(self, ctx, args)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        resolved = self._resolve()
        if not isinstance(resolved, click.Group):
            return None
        return resolved.get_command(ctx, cmd_name)

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        resolved = self._resolve()
        if not isinstance(resolved, click.Group):
            return super().resolve_command(ctx, args)
        return resolved.resolve_command(ctx, args)

    def add_command(self, cmd: click.Command, name: str | None = None) -> None:
        resolved = self._resolve()
        if not isinstance(resolved, click.Group):
            raise TypeError(f"{self.name} is not a Click group")
        resolved.add_command(cmd, name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        resolved = self._resolve()
        if not isinstance(resolved, click.Group):
            return []
        return resolved.list_commands(ctx)


_SHORT_HELP: dict[str, str] = {
    "auth": "Authenticate optional external services.",
    "backup": "Create a timestamped durability-tier backup.",
    "check": "Run archive health checks and repairs.",
    "completions": "Emit shell completion setup for polylogue.",
    "config": "Show configuration paths and resolved settings.",
    "dashboard": "Open the local dashboard.",
    "demo": "Seed and verify the deterministic demo archive.",
    "debt": "List archive work that needs operator attention.",
    "diagnostics": "Run archive and session diagnostics.",
    "embed": "Enable, preflight, and backfill the embedding pipeline.",
    "import_command": "Import sessions from configured sources.",
    "init": "Detect chat sources and write a starter polylogue.toml.",
    "insights": "Rebuild and inspect derived session insights.",
    "maintenance": "Preview and run maintenance backfill operations.",
    "ops": "Run operational archive and daemon commands.",
    "paths": "Print canonical archive paths and bind-mount detection.",
    "reset": "Reset local archive state.",
    "schema": "Inspect and audit provider schemas.",
    "status": "Show daemon and archive status.",
    "tutorial": "Interactive first-run walk-through.",
}

_COMMAND_NAMES: dict[str, str] = {
    "check": "doctor",
    "import_command": "import",
}

_GROUP_ATTRS: dict[str, str] = {
    "debt": "debt_command",
    "demo": "demo_command",
    "diagnostics": "diagnostics_group",
    "embed": "embed_command",
    "insights": "insights_command",
    "maintenance": "maintenance_group",
    "ops": "ops_command",
    "schema": "schema_command",
}

_COMMAND_ATTRS: dict[str, str] = {
    "import_command": "import_command",
}


def _L(name: str) -> _LazyCommand:  # noqa: N802
    """Shorthand for constructing lazy commands in the ROOT_COMMANDS tuple."""
    module = f"polylogue.cli.commands.{name}"
    attr = _GROUP_ATTRS.get(name, _COMMAND_ATTRS.get(name, f"{name}_command"))
    command_name = _COMMAND_NAMES.get(name, name.replace("_", "-"))
    cls = _LazyGroup if name in _GROUP_ATTRS else _LazyCommand
    return cls(command_name, module, attr, short_help=_SHORT_HELP.get(name))


# ── command list ───────────────────────────────────────────────────

ROOT_COMMANDS: tuple[click.Command, ...] = (
    config_command,
    _L("dashboard"),
    _L("demo"),
    _L("import_command"),
    _L("init"),
    _L("ops"),
    _L("tutorial"),
)

OPS_COMMANDS: tuple[click.Command, ...] = (
    _L("auth"),
    _L("backup"),
    _L("check"),
    _L("debt"),
    _L("diagnostics"),
    _L("embed"),
    _L("insights"),
    _L("maintenance"),
    _L("reset"),
    _L("schema"),
    _L("status"),
)


def register_root_commands(group: click.Group) -> None:
    """Attach the canonical root subcommands to the main CLI group."""
    for command in ROOT_COMMANDS:
        group.add_command(command)


def register_ops_commands(group: click.Group) -> None:
    """Attach operational commands under ``polylogue ops``."""
    for command in OPS_COMMANDS:
        group.add_command(command)


__all__ = ["register_ops_commands", "register_root_commands"]
