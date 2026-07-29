"""Claude Code ``tool-results/`` sidecar acquisition (polylogue-rujy).

Claude Code persists a tool result to ``<session>/tool-results/<name>.<ext>``
whenever the inline transcript envelope would be too large (a Bash/MCP
overflow: ``<persisted-output>Output too large ... Full output saved to:
<path>``/``Output has been saved to <path>``), leaving only a truncated
preview inline. It *also* mirrors many small, never-truncated tool results to
the same directory unconditionally, using the tool's own ``tool_use_id`` as
the filename stem -- those sidecars duplicate content already fully present
in the parsed ``tool_result`` block and add no new evidence.

Measured against a live ``~/.claude/projects`` corpus (12,588 sidecar files,
1.34 GB, sampled ~330 MB across 80 sessions): genuinely-truncated overflow
sidecars are a minority of *files* (~12%) but the majority of *bytes*
(~60-65%), the never-truncated mirror sidecars are ~85% of files but <2% of
bytes (already fully present inline), and a residual (~1-5% of files, highly
session-dependent) has no owning ``tool_result`` block left in the retained
transcript at all -- almost always because Claude Code's own compaction
rewrote the JSONL and pruned the turn that referenced it. That residual is
acquisition debt, not silently-dropped content.

This module performs the join only: given the raw JSONL records already read
for parsing (``payload``) and the session's ``tool-results/`` directory, it
matches each sidecar file to the ``tool_result`` block it belongs to by
``tool_use_id`` -- directly (filename stem) or via the "Full output saved
to"/"Output has been saved to" pointer embedded in the block's own inline
preview text. It performs no acquisition-tier writes itself: the parser
(``sources/parsers/claude/code_parser.py:apply_tool_result_sidecars``)
decides what to do with the result (replace truncated block text, record
session events). ``hook-*`` files under the same directory are a distinct,
already-tracked capture surface (raw hook stdout, polylogue-qqyg / #2781) and
are always skipped here, never counted as debt.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.core.hashing import hash_text

# Anthropic Messages API tool_use ids are ``toolu_...``; Claude Code's own
# overflow-persistence path and MCP tool bridge mint other id shapes (a random
# short slug, ``call_NN_...``, ``mcp-<server>-<tool>-<ts>``) for the *file*,
# while the *owning* tool_result block still carries a real tool_use_id --
# recovered from this pointer line embedded in the block's own preview text.
_SAVED_TO_RE = re.compile(r"(?:Full output saved to|Output has been saved to):?\s*(\S+)")
_HOOK_FILE_PREFIX = "hook-"


@dataclass(frozen=True)
class SidecarMatch:
    """A sidecar file successfully joined to its owning ``tool_result`` block."""

    tool_use_id: str
    filename: str
    byte_size: int
    content_hash: str
    was_truncated: bool
    full_text: str


@dataclass(frozen=True)
class SidecarDebt:
    """A sidecar file with no owning ``tool_result`` block in the retained transcript."""

    filename: str
    byte_size: int
    reason: str


@dataclass(frozen=True)
class SidecarJoinResult:
    matched: tuple[SidecarMatch, ...] = field(default_factory=tuple)
    debt: tuple[SidecarDebt, ...] = field(default_factory=tuple)


def _tool_result_index(payload: Sequence[object]) -> tuple[dict[str, tuple[int, bool]], dict[str, str]]:
    """Return (tool_use_id -> (inline_len, is_truncated), persisted_basename -> tool_use_id)."""
    by_tool_use_id: dict[str, tuple[int, bool]] = {}
    by_persisted_name: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        message = item.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for seg in content:
            if not isinstance(seg, dict) or seg.get("type") != "tool_result":
                continue
            tool_use_id = seg.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                continue
            raw = seg.get("content")
            if isinstance(raw, str):
                inline_len = len(raw)
                pointer = _SAVED_TO_RE.search(raw)
                is_truncated = pointer is not None
                if pointer:
                    by_persisted_name[os.path.basename(pointer.group(1))] = tool_use_id
            elif isinstance(raw, list):
                inline_len = sum(
                    len(part.get("text", "")) for part in raw if isinstance(part, dict) and part.get("type") == "text"
                )
                is_truncated = False
            else:
                inline_len, is_truncated = 0, False
            by_tool_use_id[tool_use_id] = (inline_len, is_truncated)
    return by_tool_use_id, by_persisted_name


def join_tool_result_sidecars(payload: Sequence[object], tool_results_dir: Path) -> SidecarJoinResult:
    """Join ``tool-results/*`` sidecar files to the ``tool_result`` blocks in ``payload``.

    Read-only: never mutates ``tool_results_dir`` or ``payload``. Returns
    matches (with the full sidecar text, ready for a parser to attach to its
    owning block) and typed debt for files whose owner cannot be found.
    """
    if not tool_results_dir.is_dir():
        return SidecarJoinResult()

    by_tool_use_id, by_persisted_name = _tool_result_index(payload)

    matched: list[SidecarMatch] = []
    debt: list[SidecarDebt] = []

    for entry in sorted(tool_results_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith(_HOOK_FILE_PREFIX):
            continue

        stem = name.rsplit(".", 1)[0]
        tool_use_id = stem if stem in by_tool_use_id else by_persisted_name.get(name, by_persisted_name.get(stem))

        try:
            byte_size = entry.stat().st_size
        except OSError:
            byte_size = 0

        if tool_use_id is None or tool_use_id not in by_tool_use_id:
            debt.append(SidecarDebt(filename=name, byte_size=byte_size, reason="no_owning_tool_result_block"))
            continue

        try:
            full_text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            debt.append(SidecarDebt(filename=name, byte_size=byte_size, reason=f"read_error:{type(exc).__name__}"))
            continue

        inline_len, is_truncated = by_tool_use_id[tool_use_id]
        matched.append(
            SidecarMatch(
                tool_use_id=tool_use_id,
                filename=name,
                byte_size=len(full_text.encode("utf-8")),
                content_hash=hash_text(full_text),
                was_truncated=is_truncated or len(full_text) > inline_len,
                full_text=full_text,
            )
        )

    return SidecarJoinResult(matched=tuple(matched), debt=tuple(debt))


__all__ = ["SidecarDebt", "SidecarJoinResult", "SidecarMatch", "join_tool_result_sidecars"]
