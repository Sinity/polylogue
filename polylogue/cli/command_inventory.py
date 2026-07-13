"""Recursive Click command-path inventory helpers."""

from __future__ import annotations

from dataclasses import dataclass

import click


@dataclass(frozen=True, slots=True)
class CommandPath:
    """A discovered command path in the Click tree."""

    path: tuple[str, ...]
    command: click.Command

    @property
    def display_name(self) -> str:
        return " ".join(self.path)

    @property
    def help_inventory_id(self) -> str:
        return "help-" + "-".join(self.path)


@dataclass(frozen=True, slots=True)
class RootCommandRoleSection:
    """Human-facing command grouping for the root help screen."""

    title: str
    commands: tuple[str, ...]
    footer: str | None = None


ROOT_COMMAND_ROLE_SECTIONS: tuple[RootCommandRoleSection, ...] = (
    RootCommandRoleSection(
        title="Search, read, and action workflows",
        commands=("find", "read", "select", "mark", "note", "analyze", "facets", "delete", "continue"),
        footer="Use `find QUERY then ACTION`; `facets` is the direct archive aggregate command.",
    ),
    RootCommandRoleSection(
        title="Setup, import, and evidence",
        commands=("config", "init", "hooks", "import", "demo", "tutorial"),
        footer="Use these for first-run setup, hook wiring, source import, demo archives, and onboarding checks.",
    ),
    RootCommandRoleSection(
        title="Reader and local UI",
        commands=("agents", "dashboard"),
        footer=(
            "`agents` exposes JSON-first coordination views for agent loops; `dashboard` launches the terminal TUI."
        ),
    ),
    RootCommandRoleSection(
        title="Operations and maintenance",
        commands=("status", "ops"),
        footer="`polylogue status` is the root shortcut for `polylogue ops status`; deeper maintenance stays under `ops`.",
    ),
)


def iter_command_paths(command: click.Command, *, include_root: bool = False) -> tuple[CommandPath, ...]:
    """Return the recursive command-path inventory for a Click command tree."""
    paths: list[CommandPath] = []

    def walk(current: click.Command, path: tuple[str, ...]) -> None:
        if path:
            paths.append(CommandPath(path=path, command=current))
        elif include_root:
            paths.append(CommandPath(path=(), command=current))
        if not isinstance(current, click.Group):
            return
        ctx = click.Context(current)
        for name in current.list_commands(ctx):
            child = current.get_command(ctx, name)
            if child is None or child.hidden:
                continue
            walk(child, (*path, name))

    walk(command, ())
    return tuple(paths)


__all__ = [
    "CommandPath",
    "ROOT_COMMAND_ROLE_SECTIONS",
    "RootCommandRoleSection",
    "iter_command_paths",
]
