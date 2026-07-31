"""Claude Code ``tool-results/`` sidecar acquisition (polylogue-rujy).

Claude Code persists a tool result to ``<session>/tool-results/<name>.<ext>``
whenever the inline transcript envelope would be too large (a Bash/MCP
overflow: ``<persisted-output>Output too large ... Full output saved to:
<path>``/``Output has been saved to <path>``), leaving only a truncated
preview inline. It *also* mirrors many small, never-truncated tool results to
the same directory unconditionally, using the tool's own ``tool_use_id`` as
the filename stem -- those sidecars duplicate content already fully present
in the parsed ``tool_result`` block and add no new evidence.

This module performs the join: given the raw JSONL records already read for
parsing (``payload``) and the session's ``tool-results/`` directory, it
matches each sidecar file to the ``tool_result`` block it belongs to by
``tool_use_id`` -- directly (filename stem) or via the "Full output saved
to"/"Output has been saved to" pointer embedded in the block's own inline
preview text. It performs no acquisition-tier writes itself: the parser
(``sources/parsers/claude/code_parser.py:apply_tool_result_sidecars``)
decides what to do with the result (replace truncated block text, record
session events). ``hook-*`` files under the same directory are a distinct,
already-tracked capture surface (raw hook stdout, polylogue-qqyg / #2781) and
are always skipped here, never counted as debt.

**Session-scoped join (polylogue debt-audit, 2026-07-30).** Claude Code
subagents (``Task`` tool invocations) get their own JSONL transcript at
``<session>/subagents/agent-*.jsonl``, but every subagent shares the SAME
session-level ``tool-results/`` directory as the parent (see
``resolve_tool_results_dir``). A single-transcript join only has that one
transcript's own ``tool_use_id``\\s in scope, so when it enumerates the full
shared directory it misclassifies every sibling-owned file as
``no_owning_tool_result_block`` -- multiplying false debt once per sibling
transcript that doesn't own the file. Measured against the live archive: a
naive per-transcript join reported 556,871 debt events, but only 14,209 are
physically distinct ``(session, filename)`` pairs (a ~39x inflation -- most of
the reported ~72GB of "lost" bytes was the same handful of files re-counted
under every sibling subagent's session_id). Verified on 3 sampled
multi-subagent sessions (172, 73, and 49 physical sidecar files
respectively): 100% resolved once sibling transcripts were joined into a
session-wide index -- the "debt" was a join-scope bug, not lost content.
``join_tool_result_sidecars_session_scoped`` (and the streaming accumulator's
``join_session_scoped``) fix this: a subagent transcript never originates
debt for a directory it doesn't own (it only ever matches its own ids,
silently skipping files it doesn't recognize -- the true owner's own pass
will match them), and the root/parent transcript classifies debt against a
session-wide union index built from every sibling ``.jsonl`` transcript on
disk, so each physical file is judged, and reported as debt at most, once.

**Acquisition-time instrumentation.** Every ``SidecarMatch``/``SidecarDebt``
now carries the sidecar *file's own mtime* (``file_mtime_ms``) -- the
filesystem's own record of when Claude Code wrote it. The parser threads this
into the emitted ``claude_tool_result_sidecar`` session event's ``timestamp``,
so ``occurred_at_ms`` is populated instead of permanently NULL (the join has
no better source of truth: sidecar files never carry an embedded timestamp,
and for genuine debt the owning ``tool_result`` block -- if one ever existed
-- is by definition not resolvable). This is what makes it possible to tell
whether sidecar debt is a closed historical cohort or an actively accruing
one.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator, Sequence
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
    file_mtime_ms: int | None = None


@dataclass(frozen=True)
class SidecarDebt:
    """A sidecar file with no owning ``tool_result`` block in the retained transcript."""

    filename: str
    byte_size: int
    reason: str
    file_mtime_ms: int | None = None


@dataclass(frozen=True)
class SidecarJoinResult:
    matched: tuple[SidecarMatch, ...] = field(default_factory=tuple)
    debt: tuple[SidecarDebt, ...] = field(default_factory=tuple)


class ToolResultIndexAccumulator:
    """Incrementally builds the ``tool_use_id`` index one record at a time.

    ``join_tool_result_sidecars`` needs ``tool_use_id -> (inline_len,
    is_truncated)`` (plus the "saved to" pointer reverse-index) built from the
    full payload. The eager/batch ingest path already holds the full payload
    in memory, so building this in one pass over ``Sequence[object]`` is free.
    The streaming ingest path (``parse_code_stream``, used for multi-GiB
    Claude Code JSONL) deliberately does *not* retain the raw payload -- that
    is the whole point of streaming it. This accumulator lets a caller observe
    each record as it flows past (see ``observe_tool_result_stream``) and join
    against the resulting index afterward, without ever materializing the
    full record list.
    """

    def __init__(self) -> None:
        self._by_tool_use_id: dict[str, tuple[int, bool]] = {}
        self._by_persisted_name: dict[str, str] = {}

    def observe(self, item: object) -> None:
        if not isinstance(item, dict):
            return
        message = item.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return
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
                    self._by_persisted_name[os.path.basename(pointer.group(1))] = tool_use_id
            elif isinstance(raw, list):
                inline_len = sum(
                    len(part.get("text", "")) for part in raw if isinstance(part, dict) and part.get("type") == "text"
                )
                is_truncated = False
            else:
                inline_len, is_truncated = 0, False
            self._by_tool_use_id[tool_use_id] = (inline_len, is_truncated)

    def join(self, tool_results_dir: Path) -> SidecarJoinResult:
        """Join the observed index against ``tool_results_dir``. See ``join_tool_result_sidecars``."""
        return _join_from_index(self._by_tool_use_id, self._by_persisted_name, tool_results_dir)

    def join_session_scoped(self, tool_results_dir: Path, source_path: str | Path) -> SidecarJoinResult:
        """Session-scoped variant of :meth:`join`. See ``join_tool_result_sidecars_session_scoped``."""
        return _session_scoped_join(self._by_tool_use_id, self._by_persisted_name, tool_results_dir, source_path)


def observe_tool_result_stream(records: Iterable[object], accumulator: ToolResultIndexAccumulator) -> Iterator[object]:
    """Tee a record stream through ``accumulator.observe`` without buffering it.

    Yields each record unchanged so it can be interposed transparently in
    front of the existing streaming parser (``_parse_code_records`` via
    ``parse_code_stream``); the accumulator is complete only once the caller
    has fully exhausted this iterator.
    """
    for item in records:
        accumulator.observe(item)
        yield item


def _tool_result_index(payload: Sequence[object]) -> tuple[dict[str, tuple[int, bool]], dict[str, str]]:
    """Return (tool_use_id -> (inline_len, is_truncated), persisted_basename -> tool_use_id)."""
    accumulator = ToolResultIndexAccumulator()
    for item in payload:
        accumulator.observe(item)
    return accumulator._by_tool_use_id, accumulator._by_persisted_name


def _index_from_jsonl_file(path: Path) -> tuple[dict[str, tuple[int, bool]], dict[str, str]]:
    """Build the same index as :func:`_tool_result_index`, reading a sibling transcript from disk.

    Used only by the session-scoped join to fold a *sibling* transcript's
    ``tool_use_id``\\s into the union index -- it is never the primary parse
    path, so a malformed line or unreadable file degrades to "contributes
    nothing" rather than failing the caller's own parse.
    """
    accumulator = ToolResultIndexAccumulator()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                accumulator.observe(item)
    except OSError:
        return {}, {}
    return accumulator._by_tool_use_id, accumulator._by_persisted_name


def resolve_tool_results_dir(source_path: str | Path | None) -> Path | None:
    """Return the session-level ``tool-results/`` dir for a Claude Code JSONL path.

    Claude Code persists sidecars to ``<project>/<session-uuid>/tool-results/``,
    a directory that sits *alongside* ``<project>/<session-uuid>.jsonl`` (same
    stem, not nested inside it). Subagent transcripts live one level deeper at
    ``<project>/<session-uuid>/subagents/agent-*.jsonl`` -- their sidecars are
    **not** per-subagent; they persist to the same session-level directory, so
    a subagent source path resolves to its grandparent's ``tool-results/``, not
    a ``subagents/tool-results/`` that Claude Code never creates. Verified
    against a live ``~/.claude/projects`` corpus (polylogue-rujy).

    Returns ``None`` when ``source_path`` is absent (no directory to derive).
    """
    if not source_path:
        return None
    path = Path(source_path)
    session_dir = path.parent.parent if path.parent.name == "subagents" else path.parent / path.stem
    return session_dir / "tool-results"


def _is_root_transcript(source_path: str | Path) -> bool:
    """A transcript is the session root unless it lives under a ``subagents/`` dir."""
    return Path(source_path).parent.name != "subagents"


def resolve_sibling_transcript_paths(source_path: str | Path) -> list[Path]:
    """Return every sibling transcript sharing this session's ``tool-results/`` dir.

    That is: the parent (root) ``.jsonl`` plus every ``subagents/agent-*.jsonl``
    underneath it, excluding ``source_path`` itself. Claude Code fans a
    session's tool calls across all of these but persists every one of their
    sidecars into the single shared directory ``resolve_tool_results_dir``
    resolves to -- this is what a session-scoped join needs to build a
    complete ownership index.
    """
    path = Path(source_path)
    session_dir = path.parent.parent if path.parent.name == "subagents" else path.parent / path.stem
    root_path = session_dir.parent / f"{session_dir.name}.jsonl"

    siblings: list[Path] = []
    if root_path.is_file():
        siblings.append(root_path)
    subagents_dir = session_dir / "subagents"
    if subagents_dir.is_dir():
        siblings.extend(sorted(subagents_dir.glob("agent-*.jsonl")))
    return [candidate for candidate in siblings if candidate != path]


def join_tool_result_sidecars(payload: Sequence[object], tool_results_dir: Path) -> SidecarJoinResult:
    """Join ``tool-results/*`` sidecar files to the ``tool_result`` blocks in ``payload``.

    Read-only: never mutates ``tool_results_dir`` or ``payload``. Returns
    matches (with the full sidecar text, ready for a parser to attach to its
    owning block) and typed debt for files whose owner cannot be found.

    Single-transcript scope: for a Claude Code session with subagent
    transcripts, prefer :func:`join_tool_result_sidecars_session_scoped` --
    this function alone will misclassify every sibling-owned sidecar as debt
    (see the module docstring).
    """
    by_tool_use_id, by_persisted_name = _tool_result_index(payload)
    return _join_from_index(by_tool_use_id, by_persisted_name, tool_results_dir)


def join_tool_result_sidecars_session_scoped(
    payload: Sequence[object],
    tool_results_dir: Path,
    source_path: str | Path,
) -> SidecarJoinResult:
    """Session-scoped ``tool-results/`` join across a parent transcript and its subagents.

    See the module docstring for the bug this fixes and its measured impact.
    Behavior:

    - A **subagent** transcript (``source_path`` under ``subagents/``) only
      ever matches sidecars it owns (its own ``tool_use_id``\\s); anything it
      doesn't recognize in the shared directory is silently skipped -- no
      event, not debt. The transcript that actually owns the file records the
      match when *it* is parsed.
    - The **root/parent** transcript builds a session-wide union index from
      every sibling ``.jsonl`` on disk (``resolve_sibling_transcript_paths``)
      and classifies debt against that union: a file is debt only if no
      transcript anywhere in the session owns it. Files it doesn't personally
      own (matched via the union but not the root's own payload) are skipped
      the same way, so each physical file is matched by its true owner and
      reported as debt, if at all, exactly once.
    """
    own_by_id, own_by_name = _tool_result_index(payload)
    return _session_scoped_join(own_by_id, own_by_name, tool_results_dir, source_path)


def _session_scoped_join(
    own_by_id: dict[str, tuple[int, bool]],
    own_by_name: dict[str, str],
    tool_results_dir: Path,
    source_path: str | Path,
) -> SidecarJoinResult:
    if _is_root_transcript(source_path):
        union_by_id: dict[str, tuple[int, bool]] = dict(own_by_id)
        union_by_name: dict[str, str] = dict(own_by_name)
        for sibling_path in resolve_sibling_transcript_paths(source_path):
            sibling_by_id, sibling_by_name = _index_from_jsonl_file(sibling_path)
            for tool_use_id, index_entry in sibling_by_id.items():
                union_by_id.setdefault(tool_use_id, index_entry)
            for basename, tool_use_id in sibling_by_name.items():
                union_by_name.setdefault(basename, tool_use_id)

        union_result = _join_from_index(union_by_id, union_by_name, tool_results_dir)
        matched = tuple(match for match in union_result.matched if match.tool_use_id in own_by_id)
        return SidecarJoinResult(matched=matched, debt=union_result.debt)

    # A subagent transcript doesn't own the shared directory: it never
    # originates debt for files it doesn't recognize, only matches for its own.
    own_result = _join_from_index(own_by_id, own_by_name, tool_results_dir)
    return SidecarJoinResult(matched=own_result.matched, debt=())


def _join_from_index(
    by_tool_use_id: dict[str, tuple[int, bool]],
    by_persisted_name: dict[str, str],
    tool_results_dir: Path,
) -> SidecarJoinResult:
    if not tool_results_dir.is_dir():
        return SidecarJoinResult()

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
            stat_result = entry.stat()
            byte_size = stat_result.st_size
            file_mtime_ms = int(stat_result.st_mtime * 1000)
        except OSError:
            byte_size = 0
            file_mtime_ms = None

        if tool_use_id is None or tool_use_id not in by_tool_use_id:
            debt.append(
                SidecarDebt(
                    filename=name,
                    byte_size=byte_size,
                    reason="no_owning_tool_result_block",
                    file_mtime_ms=file_mtime_ms,
                )
            )
            continue

        try:
            full_text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            debt.append(
                SidecarDebt(
                    filename=name,
                    byte_size=byte_size,
                    reason=f"read_error:{type(exc).__name__}",
                    file_mtime_ms=file_mtime_ms,
                )
            )
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
                file_mtime_ms=file_mtime_ms,
            )
        )

    return SidecarJoinResult(matched=tuple(matched), debt=tuple(debt))


__all__ = [
    "SidecarDebt",
    "SidecarJoinResult",
    "SidecarMatch",
    "ToolResultIndexAccumulator",
    "join_tool_result_sidecars",
    "join_tool_result_sidecars_session_scoped",
    "observe_tool_result_stream",
    "resolve_sibling_transcript_paths",
    "resolve_tool_results_dir",
]
