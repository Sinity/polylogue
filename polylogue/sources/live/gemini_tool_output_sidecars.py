"""Gemini CLI ``tool-outputs/`` sidecar acquisition (polylogue-rlw4h).

Gemini CLI persists an oversized tool result to
``<project>/tool-outputs/session-<sessionId>/<file>.txt`` and leaves a masking
envelope in the transcript's own ``functionResponse.response.output``::

    <tool_output_masked>
    Output too large. Showing first 8,000 and last 32,000 characters.
    For full output see: <path>
    ...
    </tool_output_masked>

Two properties of that envelope shape the join. It is a **both-ends**
truncation -- first 8,000 and last 32,000 characters -- so the inline text is
not a prefix of the file and the recovered full text replaces it rather than
extending it. And its pointer wording (``For full output see:``) is a third
form that Claude Code's ``tool_result_sidecars`` pointer expression does not
match, so it cannot simply be routed through that module.

The primary join is nevertheless the same as Claude Code's: **filename stem to
tool id**. Gemini CLI mints a tool id of the form ``<tool>_<epoch-ms>_<n>``
and names the sidecar either exactly that or ``<tool>_<id>_<slug>``, so the id
is recoverable from the stem for 191 of the 218 files in the measured corpus,
with no ambiguity. The pointer is the fallback reverse index for the rest --
and it is genuinely needed, because one envelope's head and tail may cite two
different spellings of the same persisted output, only one of which Gemini CLI
actually wrote.

The directory is session-scoped -- one per ``sessionId``, never shared with
another transcript -- so a file in it that neither index resolves has no owner
anywhere and is reported as debt.

Read-only: this module performs no acquisition-tier writes. The parser
(``sources/parsers/local_agent.py:apply_gemini_tool_output_sidecars``) decides
what to do with the result.

**Retained-input resolution (polylogue-cq1ql).** Like its Claude Code sibling,
the join reads a
:class:`~polylogue.sources.sidecar_evidence.RetainedSidecarScope` rather than
the directory: acquisition resolves that scope from the source tree and
retains the bytes as ``tool_result_sidecar`` raw artifacts (declared by the
gemini-cli ``OriginSpec``), derivation resolves it from those retained bytes.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from polylogue.core.hashing import hash_text
from polylogue.core.json import JSONDocument, json_document
from polylogue.sources.live.tool_result_sidecars import SidecarDebt, SidecarJoinResult, SidecarMatch
from polylogue.sources.sidecar_evidence import RetainedSidecarFile, RetainedSidecarScope

# The envelope's pointer line, and the bare path as it also appears inside the
# retained head/tail excerpt. Both spellings resolve to the same basename.
_POINTER_RE = re.compile(r"tool-outputs/[^\s\"'\\,)]+")

# Gemini CLI's masking envelope. Either marker alone identifies a truncated
# inline rendering: the wrapper tag is absent on some tools that emit only the
# "Output too large" preamble.
_MASK_RE = re.compile(
    r"<tool_output_masked>|Output too large\. Showing first [\d,]+ and last [\d,]+ characters",
)


def is_masked_tool_output(text: str | None) -> bool:
    """True when ``text`` is Gemini CLI's truncated rendering of a larger output."""
    return bool(text) and _MASK_RE.search(text or "") is not None


def resolve_tool_outputs_dir(source_path: str | Path | None, session_id: str | None) -> Path | None:
    """Return the ``tool-outputs/session-<id>/`` dir for a Gemini CLI chat snapshot.

    Snapshots live at ``<project>/chats/session-*.json``; sidecars at
    ``<project>/tool-outputs/session-<sessionId>/``. Returns ``None`` when
    either coordinate is missing -- there is no directory to derive.
    """
    if not source_path or not session_id:
        return None
    return Path(source_path).parent.parent / "tool-outputs" / f"session-{session_id}"


def _iter_tool_calls(payload: JSONDocument) -> list[JSONDocument]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    tool_calls: list[JSONDocument] = []
    for message in messages:
        raw_calls = json_document(message).get("toolCalls")
        if not isinstance(raw_calls, list):
            continue
        tool_calls.extend(json_document(tool_call) for tool_call in raw_calls)
    return tool_calls


def _tool_call_texts(tool_record: JSONDocument) -> list[str]:
    """Every string a tool call could carry a sidecar pointer in."""
    texts: list[str] = []
    results = tool_record.get("result")
    for result_item in results if isinstance(results, list) else []:
        response = json_document(json_document(result_item).get("functionResponse")).get("response")
        texts.extend(value for value in json_document(response).values() if isinstance(value, str))
    display = tool_record.get("resultDisplay")
    if isinstance(display, str):
        texts.append(display)
    elif display is not None:
        texts.append(json.dumps(display))
    return texts


def _tool_output_index(payload: JSONDocument) -> tuple[dict[str, tuple[int, bool]], dict[str, str]]:
    """Return (tool id -> (inline output length, inline output is masked), cited basename -> tool id)."""
    by_tool_id: dict[str, tuple[int, bool]] = {}
    by_pointer_name: dict[str, str] = {}
    for tool_record in _iter_tool_calls(payload):
        tool_id = tool_record.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        inline = ""
        masked = False
        results = tool_record.get("result")
        for result_item in results if isinstance(results, list) else []:
            response = json_document(json_document(result_item).get("functionResponse")).get("response")
            output = json_document(response).get("output")
            if isinstance(output, str):
                inline = output if len(output) > len(inline) else inline
                masked = masked or is_masked_tool_output(output)
        by_tool_id[tool_id] = (len(inline), masked)
        for text in _tool_call_texts(tool_record):
            for pointer in _POINTER_RE.findall(text):
                by_pointer_name.setdefault(os.path.basename(pointer), tool_id)
    return by_tool_id, by_pointer_name


def _tool_id_for_stem(stem: str, by_tool_id: dict[str, tuple[int, bool]]) -> str | None:
    """Resolve a sidecar filename stem to the tool id that produced it.

    The stem is either the tool id verbatim or the id embedded between
    underscore-delimited runs (``<tool>_<id>_<slug>``). Longest match wins, so
    a shorter id that happens to be a prefix of a longer one never steals the
    file.
    """
    if stem in by_tool_id:
        return stem
    candidates = [
        tool_id
        for tool_id in by_tool_id
        if f"_{tool_id}_" in stem or stem.endswith(f"_{tool_id}") or stem.startswith(f"{tool_id}_")
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def join_gemini_tool_output_sidecars(payload: JSONDocument, scope: RetainedSidecarScope) -> SidecarJoinResult:
    """Join ``scope``'s ``tool-outputs/session-<id>/*`` files to the tool calls that produced them.

    Read-only. Returns matches carrying the sidecar's full text, ready for the
    parser to attach to the owning ``tool_result`` block, and typed debt for
    files no tool call in this session's transcript claims. A pointer the
    transcript cites with no retained file is the explicit
    ``expected_sidecar_not_retained`` outcome -- the directory is
    session-exclusive, so there is no sibling that could own it instead.
    """
    if not scope.available:
        return SidecarJoinResult()

    by_tool_id, by_pointer_name = _tool_output_index(payload)

    matched: list[SidecarMatch] = []
    debt: list[SidecarDebt] = []
    present: set[str] = set()
    for entry in sorted(scope.files, key=lambda candidate: candidate.filename):
        present.add(entry.filename)
        byte_size = entry.byte_size
        file_mtime_ms = entry.file_mtime_ms

        stem = entry.filename.rsplit(".", 1)[0]
        tool_id = _tool_id_for_stem(stem, by_tool_id) or by_pointer_name.get(entry.filename, by_pointer_name.get(stem))
        if tool_id is None or tool_id not in by_tool_id:
            debt.append(
                SidecarDebt(
                    filename=entry.filename,
                    byte_size=byte_size,
                    reason="no_owning_tool_call",
                    file_mtime_ms=file_mtime_ms,
                )
            )
            continue

        try:
            full_text = entry.read_text()
        except OSError as exc:
            debt.append(
                SidecarDebt(
                    filename=entry.filename,
                    byte_size=byte_size,
                    reason=f"read_error:{type(exc).__name__}",
                    file_mtime_ms=file_mtime_ms,
                )
            )
            continue

        inline_len, masked = by_tool_id[tool_id]
        matched.append(
            SidecarMatch(
                tool_use_id=tool_id,
                filename=entry.filename,
                byte_size=len(full_text.encode("utf-8")),
                content_hash=hash_text(full_text),
                was_truncated=masked or len(full_text) > inline_len,
                full_text=full_text,
                file_mtime_ms=file_mtime_ms,
            )
        )

    matched_ids = {match.tool_use_id for match in matched}
    for expected_name, expected_tool_id in sorted(by_pointer_name.items()):
        if expected_name in present or expected_tool_id in matched_ids:
            # One envelope's head and tail can cite two spellings of the same
            # persisted output, only one of which Gemini CLI wrote (see the
            # module docstring). A pointer whose tool call already resolved is
            # not a missing sidecar.
            continue
        debt.append(
            SidecarDebt(
                filename=expected_name,
                byte_size=0,
                reason="expected_sidecar_not_retained",
                file_mtime_ms=None,
            )
        )
    return SidecarJoinResult(matched=tuple(matched), debt=tuple(debt))


def tool_output_files_from_directory(tool_outputs_dir: Path) -> tuple[RetainedSidecarFile, ...]:
    """Read one ``tool-outputs/session-<id>/`` directory into scope files.

    The single filesystem enumeration in the Gemini CLI sidecar path, used by
    the acquisition-time resolver. Derivation never calls it.
    """
    if not tool_outputs_dir.is_dir():
        return ()
    files: list[RetainedSidecarFile] = []
    for entry in sorted(tool_outputs_dir.iterdir()):
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


__all__ = [
    "is_masked_tool_output",
    "join_gemini_tool_output_sidecars",
    "resolve_tool_outputs_dir",
    "tool_output_files_from_directory",
]
