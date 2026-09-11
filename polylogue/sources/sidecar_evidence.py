"""Retained tool-output sidecar evidence (polylogue-cq1ql).

A provider that overflows a tool result to a sibling file leaves only a
truncated preview in the transcript. Both join implementations
(``sources/live/tool_result_sidecars.py`` for Claude Code,
``sources/live/gemini_tool_output_sidecars.py`` for Gemini CLI) used to
enumerate that sibling directory on the original filesystem *while deriving*,
so the same retained transcript produced the full output or only its preview
depending on whether the original tree still existed
(``docs/design/retained-inputs-and-supersession.md`` D3/S8/S9).

This module is the seam that removes that dependency. A join never touches a
path: it is handed a :class:`RetainedSidecarScope` -- the sidecar files of one
ownership scope, plus the sibling transcripts whose ids decide ownership
inside it -- and reads only what the scope carries. Who resolves a scope is a
route decision:

- **Acquisition** resolves it from the source tree
  (``sources/live/sidecar_resolution.py:FilesystemSidecarResolver``); the tree
  is the input at that point, and the bytes are retained as ordinary
  ``tool_result_sidecar`` raw artifacts by the same walk.
- **Derivation** resolves it from those retained bytes
  (``RetainedSidecarResolver``), which is what makes a reparse reproduce the
  full text after the original tree is gone.

Scope identity is the durable ownership coordinate, not a live lookup: the
Claude Code session directory (shared by a parent transcript and every
subagent under it) and the Gemini CLI ``tool-outputs/session-<id>/``
directory. It is derived from the transcript's own retained ``source_path``,
so it reproduces without the tree.

``available`` distinguishes "this scope was never observed" (no events at
all -- the historical behaviour when the directory did not exist) from "this
scope resolved and holds these files", where a pointer with no file is an
explicit absence rather than silence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "RetainedSidecarFile",
    "RetainedSidecarScope",
    "SiblingTranscript",
    "SidecarResolver",
    "UNRESOLVED_SIDECAR_SCOPE",
    "iter_jsonl_records",
]


@dataclass(frozen=True, slots=True)
class RetainedSidecarFile:
    """One sidecar file of a scope, addressed by name rather than by path.

    ``read_text`` is deliberately deferred: a join reads only the files it
    actually matches, exactly as the directory walk it replaces did, so an
    unmatched multi-megabyte sidecar is never decoded. It raises ``OSError``
    when the retained bytes cannot be read; the join turns that into typed
    debt rather than failing the parse.
    """

    filename: str
    byte_size: int
    file_mtime_ms: int | None
    read_text: Callable[[], str]


@dataclass(frozen=True, slots=True)
class SiblingTranscript:
    """A transcript sharing this scope, as a re-openable record stream.

    Only the ``tool_use_id``\\s matter: a root Claude Code transcript folds
    its subagents' ids into a union ownership index so a file it does not own
    is not reported as debt. ``open_records`` streams rather than returning a
    list because a subagent transcript can be large and the index needs one
    pass.
    """

    coordinate: str
    open_records: Callable[[], Iterator[object]]


@dataclass(frozen=True, slots=True)
class RetainedSidecarScope:
    """Every sidecar input one ownership scope can offer a join.

    ``available`` is False when the scope itself could not be resolved -- no
    directory, no retained artifact, no archive to ask. A join returns an
    empty result for it and records nothing, which is what the directory
    enumeration did when the directory was absent.
    """

    scope_key: str = ""
    files: tuple[RetainedSidecarFile, ...] = field(default_factory=tuple)
    siblings: tuple[SiblingTranscript, ...] = field(default_factory=tuple)
    available: bool = False


#: The scope a route resolves when it has no evidence to offer at all.
UNRESOLVED_SIDECAR_SCOPE = RetainedSidecarScope()


class SidecarResolver(Protocol):
    """Resolves the sidecar inputs of a transcript's ownership scope."""

    def claude_code_scope(self, source_path: str | Path | None) -> RetainedSidecarScope:
        """Scope for a Claude Code ``<session>/tool-results/`` directory."""

    def gemini_cli_scope(self, source_path: str | Path | None, session_id: str | None) -> RetainedSidecarScope:
        """Scope for a Gemini CLI ``tool-outputs/session-<id>/`` directory."""


def iter_jsonl_records(open_binary: Callable[[], Iterator[bytes]]) -> Iterator[object]:
    """Decode a JSONL byte-line stream, skipping lines that do not decode.

    Shared by both resolvers so a sibling transcript read from the source tree
    and the same transcript read from its retained blob build an identical
    ownership index. A malformed line contributes nothing rather than failing
    the caller's own parse -- the sibling index is corroboration, never the
    primary parse path.
    """
    for line in open_binary():
        text = line.strip()
        if not text:
            continue
        try:
            yield json.loads(text)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            continue
