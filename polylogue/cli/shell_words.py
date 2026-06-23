"""Shared shell-completion command-line parsing helpers."""

from __future__ import annotations

import os

from click.shell_completion import split_arg_string


def completion_words() -> tuple[str, ...]:
    """Return shell completion words using Click's shell quoting rules."""

    raw_words = os.environ.get("COMP_WORDS", "")
    words = tuple(part for part in split_arg_string(raw_words) if part)
    if words and words[0] == "polylogue":
        return words[1:]
    return words
