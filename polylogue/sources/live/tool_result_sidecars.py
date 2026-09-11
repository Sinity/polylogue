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
``<session>/subagents/agent-*.jsonl`` -- and, separately, an
``agent-*.meta.json`` metadata companion (``AGENT_SIDECAR_META``, ingested as
its own lightweight quasi-session) -- but every one of them shares the SAME
session-level ``tool-results/`` directory as the parent (see
``resolve_tool_results_dir``). A single-transcript join only has that one
transcript's own ``tool_use_id``\\s in scope, so when it enumerates the full
shared directory it misclassifies every sibling-owned file as
``no_owning_tool_result_block`` -- multiplying false debt once per sibling
transcript that doesn't own the file (the ``.meta.json`` companion owns
*nothing*, so it multiplied it by a full extra false-debt copy of the
directory on top of the ``.jsonl`` fanout).

The unit of "debt" is the physical file, not the event: the same file
re-observed by every non-owning sibling is one piece of missing evidence, not
N. Measured against the live archive, per-event counting reported 556,871
debt events; per-file (matching sidecar basenames against files actually
present under a live ``~/.claude/projects`` corpus) only ~12,000 are
genuinely distinct, and every one of them is still on disk -- none of the
"~72GB of lost bytes" the event count implied was ever lost, it was ~1.4GB of
real content re-counted roughly 46x by transcript fanout. That reconciles
with this module's original file-counted docstring claim of a 1-5% debt rate
(the two numbers were never in conflict -- they counted different
denominators). Verified on 3 sampled multi-subagent sessions (172, 73, and 49
physical sidecar files respectively): 100% resolved once sibling transcripts
were joined into a session-wide index -- the "debt" was a join-scope bug, not
lost content. ``join_tool_result_sidecars_session_scoped`` (and the streaming
accumulator's ``join_session_scoped``) fix this: a non-root transcript
(subagent ``.jsonl`` or ``.meta.json`` companion, anything under
``subagents/``) never originates debt for a directory it doesn't own -- it
only ever matches its own ids, silently skipping files it doesn't recognize
so the true owner's own pass can match them -- and the root/parent transcript
classifies debt against a session-wide union index built from every sibling
transcript on disk, so each physical file is judged, and reported as debt at
most, once.

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

**Retained-input resolution (polylogue-cq1ql).** This module no longer reads
the ``tool-results/`` directory. It is handed a
:class:`~polylogue.sources.sidecar_evidence.RetainedSidecarScope` -- the
scope's sidecar files and its sibling transcripts -- and joins against that.
Acquisition resolves a scope from the source tree; derivation resolves it from
the bytes that walk retained, so a reparse reproduces the full output after
the original tree is gone (``docs/design/retained-inputs-and-supersession.md``
D3). ``resolve_tool_results_dir``/``resolve_sibling_transcript_paths`` stay
here as the path law both resolvers share.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.core.hashing import hash_text
from polylogue.sources.sidecar_evidence import (
    RetainedSidecarFile,
    RetainedSidecarScope,
    SiblingTranscript,
    iter_jsonl_records,
)

# Anthropic Messages API tool_use ids are ``toolu_...``; Claude Code's own
# overflow-persistence path and MCP tool bridge mint other id shapes (a random
# short slug, ``call_NN_...``, ``mcp-<server>-<tool>-<ts>``) for the *file*,
# while the *owning* tool_result block still carries a real tool_use_id --
# recovered from this pointer line embedded in the block's own preview text.
_SAVED_TO_RE = re.compile(r"(?:Full output saved to|Output has been saved to):?\s*(\S+)")
_HOOK_FILE_PREFIX = "hook-"

#: A ``tool_result`` preview names a persisted output the scope does not hold.
#: Distinct from ``no_owning_tool_result_block`` (a retained file no transcript
#: claims): this is a claim with no file, which is what "the sidecar tree was
#: lost" looks like from the transcript's side (polylogue-cq1ql).
_EXPECTED_SIDECAR_ABSENT = "expected_sidecar_not_retained"


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

    def join(self, scope: RetainedSidecarScope) -> SidecarJoinResult:
        """Join the observed index against ``scope``. See ``join_tool_result_sidecars``."""
        return _join_from_index(self._by_tool_use_id, self._by_persisted_name, scope, self._by_persisted_name)

    def join_session_scoped(self, scope: RetainedSidecarScope, source_path: str | Path) -> SidecarJoinResult:
        """Session-scoped variant of :meth:`join`. See ``join_tool_result_sidecars_session_scoped``."""
        return _session_scoped_join(self._by_tool_use_id, self._by_persisted_name, scope, source_path)


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


def _index_from_sibling(sibling: SiblingTranscript) -> tuple[dict[str, tuple[int, bool]], dict[str, str]]:
    """Build the same index as :func:`_tool_result_index` from a sibling transcript.

    Used only by the session-scoped join to fold a *sibling* transcript's
    ``tool_use_id``\\s into the union index -- it is never the primary parse
    path, so an unreadable stream degrades to "contributes nothing" rather
    than failing the caller's own parse. The stream itself already skips
    lines that do not decode (``iter_jsonl_records``), so a sibling read from
    the source tree and the same sibling read from its retained blob produce
    the same index.
    """
    accumulator = ToolResultIndexAccumulator()
    try:
        for item in sibling.open_records():
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


def join_tool_result_sidecars(payload: Sequence[object], scope: RetainedSidecarScope) -> SidecarJoinResult:
    """Join ``scope``'s sidecar files to the ``tool_result`` blocks in ``payload``.

    Read-only: never mutates ``scope`` or ``payload``. Returns matches (with
    the full sidecar text, ready for a parser to attach to its owning block)
    and typed debt for files whose owner cannot be found.

    Single-transcript scope: for a Claude Code session with subagent
    transcripts, prefer :func:`join_tool_result_sidecars_session_scoped` --
    this function alone will misclassify every sibling-owned sidecar as debt
    (see the module docstring).
    """
    by_tool_use_id, by_persisted_name = _tool_result_index(payload)
    return _join_from_index(by_tool_use_id, by_persisted_name, scope, by_persisted_name)


def join_tool_result_sidecars_session_scoped(
    payload: Sequence[object],
    scope: RetainedSidecarScope,
    source_path: str | Path,
) -> SidecarJoinResult:
    """Session-scoped join across a parent transcript and its subagents.

    See the module docstring for the bug this fixes and its measured impact.
    Behavior:

    - A **subagent** transcript (``source_path`` under ``subagents/``) only
      ever matches sidecars it owns (its own ``tool_use_id``\\s); anything it
      doesn't recognize in the shared scope is silently skipped -- no event,
      not debt. The transcript that actually owns the file records the match
      when *it* is parsed.
    - The **root/parent** transcript builds a session-wide union index from
      every sibling transcript the scope carries and classifies debt against
      that union: a file is debt only if no transcript anywhere in the session
      owns it. Files it doesn't personally own (matched via the union but not
      the root's own payload) are skipped the same way, so each physical file
      is matched by its true owner and reported as debt, if at all, exactly
      once.
    """
    own_by_id, own_by_name = _tool_result_index(payload)
    return _session_scoped_join(own_by_id, own_by_name, scope, source_path)


def _session_scoped_join(
    own_by_id: dict[str, tuple[int, bool]],
    own_by_name: dict[str, str],
    scope: RetainedSidecarScope,
    source_path: str | Path,
) -> SidecarJoinResult:
    if _is_root_transcript(source_path):
        union_by_id: dict[str, tuple[int, bool]] = dict(own_by_id)
        union_by_name: dict[str, str] = dict(own_by_name)
        for sibling in scope.siblings:
            sibling_by_id, sibling_by_name = _index_from_sibling(sibling)
            for tool_use_id, index_entry in sibling_by_id.items():
                union_by_id.setdefault(tool_use_id, index_entry)
            for basename, tool_use_id in sibling_by_name.items():
                union_by_name.setdefault(basename, tool_use_id)

        union_result = _join_from_index(union_by_id, union_by_name, scope, own_by_name)
        matched = tuple(match for match in union_result.matched if match.tool_use_id in own_by_id)
        return SidecarJoinResult(matched=matched, debt=union_result.debt)

    # A subagent transcript doesn't own the shared scope: it never originates
    # file-level debt for files it doesn't recognize, only matches for its own.
    # Its *own* unresolved pointer is a different statement -- a fact about a
    # block in this transcript, not a judgement about a shared directory -- so
    # it survives, and a subagent whose output was never retained says so
    # instead of going silent.
    own_result = _join_from_index(own_by_id, own_by_name, scope, own_by_name)
    own_absence = tuple(debt for debt in own_result.debt if debt.reason == _EXPECTED_SIDECAR_ABSENT)
    return SidecarJoinResult(matched=own_result.matched, debt=own_absence)


def _join_from_index(
    by_tool_use_id: dict[str, tuple[int, bool]],
    by_persisted_name: dict[str, str],
    scope: RetainedSidecarScope,
    expected_by_persisted_name: dict[str, str],
) -> SidecarJoinResult:
    """Join one ownership index against the files ``scope`` retained.

    ``expected_by_persisted_name`` is the *own* pointer index of the
    transcript being parsed (never the sibling union): each pointer in it
    names a file that transcript's own preview says exists, so a pointer with
    no retained file is that transcript's explicit
    ``expected_sidecar_not_retained`` outcome rather than silence. Reported
    only when the scope resolved at all -- an unresolved scope is "we never
    observed this directory", which is not evidence that anything is missing.
    """
    if not scope.available:
        return SidecarJoinResult()

    matched: list[SidecarMatch] = []
    debt: list[SidecarDebt] = []
    present: set[str] = set()

    for entry in sorted(scope.files, key=lambda candidate: candidate.filename):
        name = entry.filename
        if name.startswith(_HOOK_FILE_PREFIX):
            continue
        present.add(name)

        stem = name.rsplit(".", 1)[0]
        tool_use_id = stem if stem in by_tool_use_id else by_persisted_name.get(name, by_persisted_name.get(stem))
        byte_size = entry.byte_size
        file_mtime_ms = entry.file_mtime_ms

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
            full_text = entry.read_text()
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

    matched_ids = {match.tool_use_id for match in matched}
    for expected_name, expected_tool_use_id in sorted(expected_by_persisted_name.items()):
        if expected_name in present or expected_name.startswith(_HOOK_FILE_PREFIX):
            continue
        if expected_tool_use_id in matched_ids:
            # The same call's output was recovered under the name the provider
            # actually wrote. A preview can cite a spelling that was never a
            # file, so a pointer whose call already resolved is not evidence
            # that anything is missing.
            continue
        debt.append(
            SidecarDebt(
                filename=expected_name,
                byte_size=0,
                reason=_EXPECTED_SIDECAR_ABSENT,
                file_mtime_ms=None,
            )
        )

    return SidecarJoinResult(matched=tuple(matched), debt=tuple(debt))


def sidecar_files_from_directory(tool_results_dir: Path) -> tuple[RetainedSidecarFile, ...]:
    """Read one ``tool-results/`` directory into scope files.

    The single filesystem enumeration in the Claude Code sidecar path, used
    by the acquisition-time resolver. Derivation never calls it.
    """
    if not tool_results_dir.is_dir():
        return ()
    files: list[RetainedSidecarFile] = []
    for entry in sorted(tool_results_dir.iterdir()):
        if not entry.is_file():
            continue
        try:
            stat_result = entry.stat()
            byte_size = stat_result.st_size
            file_mtime_ms: int | None = int(stat_result.st_mtime * 1000)
        except OSError:
            byte_size, file_mtime_ms = 0, None
        files.append(
            RetainedSidecarFile(
                filename=entry.name,
                byte_size=byte_size,
                file_mtime_ms=file_mtime_ms,
                read_text=_read_text_from_path(entry),
            )
        )
    return tuple(files)


def _read_text_from_path(path: Path) -> Callable[[], str]:
    def read() -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    return read


def sibling_transcripts_from_directory(source_path: str | Path) -> tuple[SiblingTranscript, ...]:
    """Open every on-disk sibling transcript of ``source_path`` as a record stream."""
    return tuple(
        SiblingTranscript(coordinate=str(path), open_records=_records_from_path(path))
        for path in resolve_sibling_transcript_paths(source_path)
    )


def _records_from_path(path: Path) -> Callable[[], Iterator[object]]:
    def open_records() -> Iterator[object]:
        with path.open("rb") as handle:
            yield from iter_jsonl_records(lambda: iter(handle))

    return open_records


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
    "sidecar_files_from_directory",
    "sibling_transcripts_from_directory",
]
