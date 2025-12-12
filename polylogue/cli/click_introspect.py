from __future__ import annotations

from typing import List, Tuple

import click


def click_command_entries(group: click.Group) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for name, command in sorted(group.commands.items()):
        if getattr(command, "hidden", False):
            continue
        desc = command.help or command.short_help or (command.__doc__ or "")
        description = " ".join((desc or "").split())
        entries.append((name, description))
    return entries


__all__ = ["click_command_entries"]

