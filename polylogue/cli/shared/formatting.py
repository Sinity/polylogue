"""CLI output formatting utilities."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.config import Source


def should_use_plain(*, plain: bool, force_plain: bool = False, no_color: bool = False) -> bool:
    if plain or force_plain or no_color:
        return True
    return not (sys.stdout.isatty() and sys.stderr.isatty())


# The renderers retain their long-standing internal vocabulary.  The CLI
# accepts the short spellings at its boundary and lowers them once, before a
# request reaches a view or renderer.
OUTPUT_DIALECT_ALIASES: dict[str, str] = {
    "md": "markdown",
    "jsonl": "ndjson",
}


def normalize_output_dialect(value: str | None) -> str | None:
    """Lower a public output-dialect spelling to the renderer vocabulary."""

    if value is None:
        return None
    return OUTPUT_DIALECT_ALIASES.get(value, value)


def output_dialect_choices(formats: Iterable[str]) -> click.Choice[str]:
    """Return a Click choice that includes applicable public short spellings."""

    choices = set(formats)
    for alias, canonical in OUTPUT_DIALECT_ALIASES.items():
        if canonical in choices:
            choices.add(alias)
    return click.Choice(sorted(choices))


def json_output_option(func: Callable[..., object]) -> Callable[..., object]:
    """Add the shared ``--json`` alias for a command's ``output_format``."""

    return click.option(
        "--json",
        "output_format",
        flag_value="json",
        default=None,
        help="Alias for --format json.",
    )(func)


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
