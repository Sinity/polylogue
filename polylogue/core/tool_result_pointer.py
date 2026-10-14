"""The public persistence pointer carried by a Claude tool-result preview.

The provider's fixed marker and reference precedence are declared here.
Filenames and output bodies remain original field operands; they are never
a vocabulary or a codebook.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from posixpath import basename
from typing import Generic, Literal, TypeVar

PERSISTED_OUTPUT_PREFIX = "Full output saved to"
PERSISTED_OUTPUT_POINTER_PATTERN = re.compile(r"(?:Full output saved to|Output has been saved to):?\s*(\S+)")


def persisted_output_shape(value: str) -> tuple[int, int, int, int, int, int] | None:
    """Measure numeric pointer coordinates without retaining its spelling.

    Total length/newlines, token start/prefix newlines, token width and basename
    width are one observed joint. Paths are provider POSIX wire paths.
    """
    match = PERSISTED_OUTPUT_POINTER_PATTERN.search(value)
    if match is None:
        return None
    token = match.group(1)
    start = match.start(1)
    return (len(value), value.count("\n"), start, value[:start].count("\n"), len(token), len(basename(token)))


def pointer_width_relation(identity_width: int, basename_width: int) -> str:
    """Public numeric ordering needed by the direct/persisted join geometry."""
    difference = basename_width - identity_width
    return (
        "shorter"
        if difference < 0
        else "equal"
        if difference == 0
        else "one-more"
        if difference == 1
        else "two-more"
        if difference == 2
        else "longer"
    )


_Identity = TypeVar("_Identity", covariant=True, default=str)


@dataclass(frozen=True, slots=True)
class SidecarReferenceMatch(Generic[_Identity]):
    """A filename's exact reference under the provider's precedence rule."""

    tool_use_id: _Identity
    reference_kind: Literal["direct", "persisted"]
    was_truncated: bool


def matching_tool_result_reference(
    filename: str,
    tool_reference: Callable[[str | _Identity], tuple[int, bool] | None],
    persisted_reference: Callable[[str], _Identity | None],
) -> SidecarReferenceMatch[str | _Identity] | None:
    """Share one join law between retained inputs and neutral generated inputs.

    A direct tool ID wins even when a saved-to pointer claims the same name.
    Earlier saved-to names remain meaningful after a later inline result.
    """
    if filename.startswith("hook-"):
        return None
    stem = filename.rsplit(".", 1)[0]
    direct = tool_reference(stem)
    if direct is not None:
        return SidecarReferenceMatch(stem, "direct", bool(direct[1]))
    tool_use_id = persisted_reference(filename)
    if tool_use_id is None:
        tool_use_id = persisted_reference(stem)
    if tool_use_id is None:
        return None
    result = tool_reference(tool_use_id)
    if result is None:
        return None
    return SidecarReferenceMatch(tool_use_id, "persisted", bool(result[1]))
