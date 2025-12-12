from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

CommandHandler = Callable[..., object]


@dataclass
class CommandInfo:
    name: str
    handler: CommandHandler
    help_text: Optional[str] = None
    aliases: Tuple[str, ...] = ()


class CommandRegistry:
    """Lightweight registry for CLI command handlers."""

    def __init__(self) -> None:
        self._commands: Dict[str, CommandInfo] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        help_text: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        if name in self._commands:
            raise ValueError(f"Command '{name}' already registered")
        alias_tuple = tuple(aliases or ())
        info = CommandInfo(name=name, handler=handler, help_text=help_text, aliases=alias_tuple)
        self._commands[name] = info
        for alias in alias_tuple:
            if alias in self._aliases:
                raise ValueError(f"Alias '{alias}' already registered for {self._aliases[alias]}")
            self._aliases[alias] = name

    def resolve(self, name: str) -> Optional[CommandHandler]:
        if name in self._commands:
            return self._commands[name].handler
        mapped = self._aliases.get(name)
        if mapped:
            return self._commands[mapped].handler
        return None

    def info(self, name: str) -> Optional[CommandInfo]:
        if name in self._commands:
            return self._commands[name]
        mapped = self._aliases.get(name)
        if not mapped:
            return None
        return self._commands.get(mapped)

    def items(self) -> List[CommandInfo]:
        return sorted(self._commands.values(), key=lambda info: info.name)

    def names(self) -> List[str]:
        return sorted(self._commands.keys())


def build_default_registry() -> CommandRegistry:
    """Builds the default registry with high-level CLI commands.

    The Click entrypoint in polylogue.cli.click_app wires the dispatchers.
    """
