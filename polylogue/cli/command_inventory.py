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
    def help_exercise_name(self) -> str:
        return "help-" + "-".join(self.path)


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


__all__ = ["CommandPath", "iter_command_paths"]
