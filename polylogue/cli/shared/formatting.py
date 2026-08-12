"""CLI output formatting utilities."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.config import Source


def should_use_plain(*, plain: bool, force_plain: bool = False, no_color: bool = False) -> bool:
    if plain or force_plain or no_color:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())


def format_sources_summary(sources: list[Source]) -> str:
    if not sources:
        return "none"
    labels: list[str] = []
    for source in sources:
        if source.folder:
            labels.append(f"{source.name} (drive)")
        elif source.path:
            labels.append(source.name)
        else:
            labels.append(f"{source.name} (missing)")
    if len(labels) > 8:
        extra = len(labels) - 8
        labels = labels[:8] + [f"+{extra} more"]
    return ", ".join(labels)
