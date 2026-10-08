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

import os
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

from polylogue.core.json import JSONDocument
from polylogue.logging import WARNING, emit
from polylogue.sources.live.tool_result_sidecars import SidecarDebt, SidecarJoinResult, SidecarMatch
from polylogue.sources.prepared_message_sink import GeminiToolOutputIndex, is_masked_tool_output
from polylogue.sources.sidecar_evidence import RetainedSidecarFile, RetainedSidecarScope


def resolve_tool_outputs_dir(source_path: str | Path | None, session_id: str | None) -> Path | None:
    """Return the ``tool-outputs/session-<id>/`` dir for a Gemini CLI chat snapshot.

    Snapshots live at ``<project>/chats/session-*.json``; sidecars at
    ``<project>/tool-outputs/session-<sessionId>/``. Returns ``None`` when
    either coordinate is missing -- there is no directory to derive.

    ``session_id`` is untrusted export content. It names exactly one directory
    component under ``tool-outputs/``, so a value carrying a path separator or
    a ``.``/``..`` traversal component names a directory this source does not
    own. That is refused (``None`` plus a logged refusal), never resolved --
    an escaping value would read sidecar bytes from outside the snapshot's own
    project tree into the archive.
    """
    if not source_path or not session_id:
        return None
    directory_name = f"session-{session_id}"
    if not _is_single_path_component(directory_name):
        emit(
            "sources.gemini.tool_output_dir_refused",
            level=WARNING,
            reason="session_id_not_a_path_component",
            source_path=str(source_path),
        )
        return None
    return Path(source_path).parent.parent / "tool-outputs" / directory_name


def _is_single_path_component(name: str) -> bool:
    """True when ``name`` names one ordinary directory entry and nothing else."""
    if not name or name in {".", ".."}:
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    return Path(name).name == name


def join_gemini_tool_output_sidecars(payload: JSONDocument, scope: RetainedSidecarScope) -> SidecarJoinResult:
    """Join ``scope``'s ``tool-outputs/session-<id>/*`` files to the tool calls that produced them.

    Read-only. Returns matches carrying a deferred reader for the sidecar's
    full text, ready for the parser to attach to the owning ``tool_result`` block, and typed debt for
    files no tool call in this session's transcript claims. A pointer the
    transcript cites with no retained file is the explicit
    ``expected_sidecar_not_retained`` outcome -- the directory is
    session-exclusive, so there is no sibling that could own it instead.
    """
    if not scope.available:
        return SidecarJoinResult()

    messages = payload.get("messages")
    with tempfile.TemporaryDirectory() as directory, sqlite3.connect(Path(directory) / "index.db") as conn:
        index = GeminiToolOutputIndex(conn)
        for message in messages if isinstance(messages, list) else []:
            index.observe(message)
        outcomes = list(index.join(scope))
    return SidecarJoinResult(
        matched=tuple(item for item in outcomes if isinstance(item, SidecarMatch)),
        debt=tuple(item for item in outcomes if isinstance(item, SidecarDebt)),
    )


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
        identity: tuple[int, int, int, int, int] | None
        try:
            stat_result = entry.stat()
            byte_size = stat_result.st_size
            file_mtime_ms: int | None = int(stat_result.st_mtime * 1000)
            identity = _file_identity(stat_result)
        except OSError:
            byte_size, file_mtime_ms, identity = 0, None, None
        files.append(
            RetainedSidecarFile(
                filename=entry.name,
                byte_size=byte_size,
                file_mtime_ms=file_mtime_ms,
                read_text=_read_text_from_path(entry, expected_size=byte_size, enumerated=identity),
            )
        )
    return tuple(files)


class SidecarChangedDuringReadError(OSError):
    """The sidecar's bytes moved while it was read; the read is not evidence.

    Gemini CLI may still be writing a tool output when ingest observes it.
    The join records this as read-error debt, and the file's next change
    brings it back through intake.
    """


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _read_text_from_path(
    path: Path, *, expected_size: int, enumerated: tuple[int, int, int, int, int] | None = None
) -> Callable[[], str]:
    """Read the sidecar the enumeration observed, or refuse it as changed.

    The opened handle is compared with the enumerated device, inode, size,
    mtime and ctime, not only with itself: a same-length rewrite or atomic
    replacement between enumeration and read would otherwise pair new bytes
    with the enumerated mtime.
    """

    def read() -> str:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            # One byte past the enumerated size proves growth without
            # reading whatever a still-running writer appended.
            payload = handle.read(expected_size + 1)
            after = os.fstat(handle.fileno())
        if (
            len(payload) != expected_size
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or after.st_size != expected_size
            or (enumerated is not None and _file_identity(before) != enumerated)
        ):
            raise SidecarChangedDuringReadError(f"sidecar changed while it was read: {path.name}")
        return payload.decode("utf-8", errors="replace")

    return read


__all__ = [
    "is_masked_tool_output",
    "join_gemini_tool_output_sidecars",
    "resolve_tool_outputs_dir",
    "tool_output_files_from_directory",
]
