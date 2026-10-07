"""Provider assembly inputs resolved from retained archive evidence.

Convergence must be able to rebuild a session's provider metadata and its
attachment joins from what the archive durably holds, not from files that
happen to still sit beside the original source path. Carrying discovery
results to one worker does not make those results available when derived
state is absent, and re-reading a live sibling file makes the same retained
transcript derive different content depending on whether the original tree
still exists (``docs/design/retained-inputs-and-supersession.md``, D3).

This module is the retained-evidence side of that contract for the two
families polylogue-ximhz owns:

* **Claude Code** — ``projects/<project>/sessions-index.json`` (curated title
  and branch) and ``history.jsonl`` (paste evidence no transcript records).
* **ChatGPT** — ``library_files.json`` /
  ``conversation_asset_file_names.json`` (asset names and sandbox files) and
  the id-bearing export members that carry the asset bytes.

Three rules from the decision record shape every lookup here:

* **Scope is identity** (R3). A lookup is anchored on the session's own
  durable ``source_path``: the project directory a Claude session was
  acquired from, or the export a ChatGPT shard is a member of. The same
  basename under two installs, and the same asset id in two exports, are
  different objects and never cross-bind.
* **Currency follows durable receipt order** (R5). Reobserving an earlier
  byte revision renews its ``raw_payload`` receipt while preserving its raw
  identity. The existing receipt-order owner decides currency; wall clocks
  and raw-row insertion order do not. A ZIP artifact additionally requires
  exact completed acquisition membership. Groups with conflicting complete
  sets produce a typed gap, and custody remains intact.
* **Absence is an outcome, never a guess** (R6/S6). A missing map resolves to
  an empty bundle and the parsed-content fallbacks apply; nothing is
  reconstructed from the recorded source path.

The anchors these lookups use are the same ones live discovery uses
(``assembly_claude_code.py``, ``assembly_chatgpt.py``), so a replay resolves
the evidence a live ingest resolved rather than a differently-scoped
approximation.
"""

from __future__ import annotations

import errno
import sqlite3
import threading
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Protocol, TypeVar, cast

import ijson

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.archive.revision_authority import raw_receipt_order_sql
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, read_captured_zip_coordinate_receipt
from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError
from polylogue.logging import DEBUG, emit, get_logger
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_items import (
    CompletedSourceItemRead,
    retained_completed_source_item_for_raw,
)

from .assembly import (
    ClaudeCodeHistoryPasteIndex,
    ClaudeCodeSessionIndex,
    SidecarData,
)

logger = get_logger(__name__)

#: ``~/.claude/history.jsonl`` is global to one Claude Code install and sits
#: two levels above a project's session files, exactly as
#: ``ClaudeCodeAssemblySpec`` resolves it on the live path.
_HISTORY_RELATIVE = Path("..") / ".." / "history.jsonl"
_SESSIONS_INDEX_NAME = "sessions-index.json"
_CODEX_SESSION_INDEX_NAME = "session_index.jsonl"
_CODEX_HISTORY_NAME = "history.jsonl"


@dataclass(frozen=True, slots=True)
class RetainedArtifact:
    """One retained observation of a declared non-session source artifact."""

    raw_id: str
    source_path: str
    blob_hash: str
    blob_size: int
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None


def _like_prefix(prefix: str) -> str:
    """Escape a literal path prefix for a ``LIKE ... ESCAPE '\\'`` match."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _select_retained(
    source_read: RetainedAssemblyRead,
    *,
    origin: Origin,
    artifact_kind: ArtifactKind,
    coordinate: str,
    prefix: bool = False,
) -> Generator[tuple[str, RetainedArtifact], None, None]:
    """Page current exact coordinates with no SQL cursor across consumption."""
    after: str | None = None
    while True:
        check_compute_cancelled()
        page = source_read.retained_artifact_page(
            origin.value,
            artifact_kind.value,
            coordinate,
            prefix=prefix,
            after=after,
        )
        if not page:
            return
        for raw_id, source_path, blob_hash, blob_size, receipt, ordinal, captured in page:
            check_compute_cancelled()
            if receipt is None:
                raise OSError(errno.ENODATA, "retained artifact currency has no source receipt")
            if ordinal is not None:
                generation, item = retained_completed_source_item_for_raw(source_read, raw_id)
                member = source_read.retained_group_member(
                    generation,
                    item,
                    source_path,
                    origin.value,
                    artifact_kind.value,
                )
                if member is None:
                    raise RetainedZipMembershipUnprovedError("completed ZIP group lacks its selected artifact")
                raw_id, blob_hash, blob_size, captured = member
                if captured is None:
                    raise RetainedZipMembershipUnprovedError(
                        "retained ZIP artifact lacks its captured namespace/member receipt"
                    )
            coordinate_receipt = None if captured is None else read_captured_zip_coordinate_receipt(captured)
            if coordinate_receipt is not None and coordinate_receipt.declared_member != source_path:
                raise RetainedZipMembershipUnprovedError(
                    "retained ZIP artifact coordinate differs from its captured member receipt"
                )
            yield source_path, RetainedArtifact(raw_id, source_path, blob_hash, blob_size, coordinate_receipt)
        after = page[-1][1]


def _first_retained_artifact(records: Generator[tuple[str, RetainedArtifact], None, None]) -> RetainedArtifact | None:
    try:
        first = next(records, None)
        return None if first is None else first[1]
    finally:
        records.close()


def _read(source_read: RetainedAssemblyRead, artifact: RetainedArtifact) -> bytes | None:
    try:
        with source_read.open_sidecar_payload(artifact.raw_id, bytes.fromhex(artifact.blob_hash)) as payload:
            return payload.read()
    except (OSError, ValueError) as exc:
        emit(
            "retained_assembly_blob_unavailable",
            level=DEBUG,
            source_path=artifact.source_path,
            failure=exc,
            outcome="unavailable",
        )
        return None


_Parsed = TypeVar("_Parsed")
#: Install-global sidecars (``history.jsonl``, session indexes) are shared by
#: every session file of one install. Live intake enriches each file, so
#: reparsing the same retained history per file makes a fresh build pay
#: O(files x history bytes). A retained blob is content-addressed and
#: immutable, so its parsed form is keyed by hash, parser and anchor path.
#: One parsed artifact is kept per kind -- the one the last enrichment of
#: that kind already had to hold -- and only for one origin at a time
#: (``claude_code.*`` or ``codex.*``): enriching another origin's file
#: releases the previous origin's artifacts first, so residency never
#: exceeds what a single enrichment needs, whatever the parsed size.
_parsed_retained_cache: dict[str, tuple[tuple[str, str, str], object]] = {}
_parsed_retained_lock = threading.Lock()
#: Held across one read-and-parse, so parsed residency stays one origin's.
_parsed_retained_fill_lock = threading.Lock()


def _read_parsed(
    source_read: RetainedAssemblyRead,
    blob_store: BlobStore,
    artifact: RetainedArtifact,
    kind: str,
    parse: Callable[[bytes], _Parsed],
) -> _Parsed | None:
    key = (str(blob_store.root), artifact.blob_hash, artifact.source_path)
    with _parsed_retained_lock:
        cached = _parsed_retained_cache.get(kind)
        if cached is not None and cached[0] == key:
            return cast("_Parsed", cached[1])
    # Fills are serialized: two workers filling different origins at once
    # would each hold an unbounded parse and publish both. Under the fill
    # lock, the previous artifact of this kind and every artifact of another
    # origin are released before the next is parsed.
    with _parsed_retained_fill_lock:
        with _parsed_retained_lock:
            cached = _parsed_retained_cache.get(kind)
            if cached is not None and cached[0] == key:
                return cast("_Parsed", cached[1])
            origin = kind.split(".", 1)[0]
            for held in [held for held in _parsed_retained_cache if held == kind or held.split(".", 1)[0] != origin]:
                del _parsed_retained_cache[held]
        payload = _read(source_read, artifact)
        if payload is None:
            return None
        parsed = parse(payload)
        with _parsed_retained_lock:
            _parsed_retained_cache[kind] = (key, parsed)
        return parsed


# --------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------


def claude_code_sidecar_coordinates(session_source_path: str) -> tuple[str, str] | None:
    """Return ``(sessions-index path, history path)`` for one session path.

    Resolved exactly as ``ClaudeCodeAssemblySpec.discover_sidecars`` resolves
    them on the live path, so replay anchors on the same two coordinates a
    live ingest read rather than a differently-scoped approximation.
    """
    if not session_source_path or ":" in PurePosixPath(session_source_path).name:
        return None
    path = Path(session_source_path)
    parent = path.parent
    if str(parent) in {"", ".", path.anchor}:
        return None
    return str(parent / _SESSIONS_INDEX_NAME), str((parent / _HISTORY_RELATIVE).resolve())


def retained_claude_code_sidecars(
    source_read: RetainedAssemblyRead,
    blob_store: BlobStore,
    *,
    session_source_path: str,
) -> SidecarData:
    """Rebuild the Claude Code assembly bundle from retained bytes."""
    coordinates = claude_code_sidecar_coordinates(session_source_path)
    if coordinates is None:
        return cast(SidecarData, {})
    index_path, history_path = coordinates
    resolved: SidecarData = {}

    indexes = _select_retained(
        source_read,
        origin=Origin.CLAUDE_CODE_SESSION,
        artifact_kind=ArtifactKind.SESSION_INDEX,
        coordinate=index_path,
    )
    artifact = _first_retained_artifact(indexes)
    if artifact is not None:
        from .parsers.claude.index import parse_sessions_index_bytes

        entries: ClaudeCodeSessionIndex | None = _read_parsed(
            source_read, blob_store, artifact, "claude_code.session_index", parse_sessions_index_bytes
        )
        if entries:
            resolved["session_index"] = entries

    histories = _select_retained(
        source_read,
        origin=Origin.CLAUDE_CODE_SESSION,
        artifact_kind=ArtifactKind.PROMPT_HISTORY_LOG,
        coordinate=history_path,
    )
    artifact = _first_retained_artifact(histories)
    if artifact is not None:
        from .parsers.claude.history import build_session_paste_index_bytes

        pastes: ClaudeCodeHistoryPasteIndex | None = _read_parsed(
            source_read,
            blob_store,
            artifact,
            "claude_code.history_paste_index",
            lambda payload: build_session_paste_index_bytes(payload, origin=history_path),
        )
        if pastes:
            resolved["history_paste_index"] = pastes
    return resolved


# --------------------------------------------------------------------------
# Codex
# --------------------------------------------------------------------------


def codex_sidecar_coordinates(session_source_path: str) -> tuple[str, str] | None:
    """Return the two root-sidecar coordinates for one Codex rollout.

    Codex writes both title inputs at the install root, immediately above its
    ``sessions/`` tree.  Finding that tree in the session's own coordinate is
    deliberate: two installs may have identical thread ids and sidecar names,
    but never share retained title evidence.
    """
    if not session_source_path:
        return None
    # Use the acquired coordinate spelling, independent of the host OS.
    windows = PureWindowsPath(session_source_path)
    path = windows if "\\" in session_source_path and windows.is_absolute() else PurePosixPath(session_source_path)
    sessions_root = next((parent for parent in path.parents if parent.name == "sessions"), None)
    if sessions_root is None:
        return None
    install_root = sessions_root.parent
    if str(install_root) in {"", ".", sessions_root.anchor}:
        return None
    return str(install_root / _CODEX_SESSION_INDEX_NAME), str(install_root / _CODEX_HISTORY_NAME)


def retained_codex_sidecars(
    source_read: RetainedAssemblyRead,
    blob_store: BlobStore,
    *,
    session_source_path: str,
) -> SidecarData:
    """Rebuild Codex title inputs from exact retained root-sidecar bytes."""
    coordinates = codex_sidecar_coordinates(session_source_path)
    if coordinates is None:
        return cast(SidecarData, {})
    index_path, history_path = coordinates
    resolved: SidecarData = {}

    indexes = _select_retained(
        source_read,
        origin=Origin.CODEX_SESSION,
        artifact_kind=ArtifactKind.SESSION_INDEX,
        coordinate=index_path,
    )
    artifact = _first_retained_artifact(indexes)
    if artifact is not None:
        from .assembly_codex import parse_codex_session_index_bytes

        names = _read_parsed(source_read, blob_store, artifact, "codex.session_index", parse_codex_session_index_bytes)
        if names:
            resolved["thread_names"] = names

    histories = _select_retained(
        source_read,
        origin=Origin.CODEX_SESSION,
        artifact_kind=ArtifactKind.PROMPT_HISTORY_LOG,
        coordinate=history_path,
    )
    artifact = _first_retained_artifact(histories)
    if artifact is not None:
        from .assembly_codex import parse_codex_history_bytes

        titles = _read_parsed(source_read, blob_store, artifact, "codex.history_titles", parse_codex_history_bytes)
        if titles:
            resolved["history_titles"] = titles
    return resolved


# --------------------------------------------------------------------------
# ChatGPT
# --------------------------------------------------------------------------


def chatgpt_export_scope(
    session_source_path: str,
    *,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
) -> str | None:
    """Scope exact acquired ZIP members separately from ordinary literal paths."""
    if not session_source_path:
        return None
    if captured_zip_coordinate is not None:
        if session_source_path != captured_zip_coordinate.declared_member:
            raise ValueError("session path differs from its captured ZIP coordinate")
        return captured_zip_coordinate.declared_container + ":"
    parent = Path(session_source_path).parent
    return None if str(parent) in {"", "."} else f"{parent}/"


def _member_basename(artifact: RetainedArtifact) -> str:
    member = (
        artifact.captured_zip_coordinate.member_name
        if artifact.captured_zip_coordinate is not None
        else artifact.source_path
    )
    return PurePosixPath(member.replace("\\", "/")).name


def retained_chatgpt_sidecars(
    source_read: RetainedAssemblyRead,
    blob_store: BlobStore,
    *,
    session_source_path: str,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
) -> SidecarData:
    """Rebuild the ChatGPT asset index and asset-blob map from retained bytes."""
    scope = chatgpt_export_scope(session_source_path, captured_zip_coordinate=captured_zip_coordinate)
    if scope is None:
        return cast(SidecarData, {})
    from .assembly_chatgpt import _member_asset_id
    from .parsers.chatgpt_sidecars import ChatGPTAssetIndex

    index = ChatGPTAssetIndex()
    claimed: set[str] = set()
    indexes = _select_retained(
        source_read,
        origin=Origin.CHATGPT_EXPORT,
        artifact_kind=ArtifactKind.EXPORT_ASSET_INDEX,
        coordinate=_like_prefix(scope),
        prefix=True,
    )
    try:
        try:
            for path, artifact in indexes:
                name = _member_basename(artifact)
                if name not in {"library_files.json", "conversation_asset_file_names.json"} or name in claimed:
                    continue
                try:
                    with source_read.open_sidecar_payload(artifact.raw_id, bytes.fromhex(artifact.blob_hash)) as source:
                        if index.load_stream(source, library=name == "library_files.json"):
                            claimed.add(name)
                except (OSError, ValueError, ijson.JSONError) as exc:
                    emit(
                        "retained_chatgpt_asset_index_unavailable",
                        level=DEBUG,
                        source_path=path,
                        failure=exc,
                        outcome="unavailable",
                    )
        finally:
            indexes.close()
    except BaseException:
        index.close()
        raise

    try:
        group = index.begin_asset_group()
        assets = _select_retained(
            source_read,
            origin=Origin.CHATGPT_EXPORT,
            artifact_kind=ArtifactKind.EXPORT_ASSET,
            coordinate=_like_prefix(scope),
            prefix=True,
        )
        # Key members exactly as live discovery does: the bare asset id until a
        # second member proves it ambiguous, then ``asset_id#member`` for every
        # member, with the member named relative to its export scope.
        try:
            for path, artifact in assets:
                asset_id = _member_asset_id(_member_basename(artifact))
                if asset_id is None:
                    continue
                member = path[len(scope) :] if path.startswith(scope) else _member_basename(artifact)
                index.record_asset(group, asset_id, member, (artifact.blob_hash, artifact.blob_size))

        finally:
            assets.close()

        index.finish_asset_group(group)
        index.seal()
        asset_blobs = index.asset_blobs
        if not claimed and not asset_blobs:
            index.close()
            return cast(SidecarData, {})
        resolved: SidecarData = {"chatgpt_asset_index": index}
        if asset_blobs:
            resolved["chatgpt_asset_blobs"] = asset_blobs
        return resolved
    except BaseException:
        index.close()
        raise


# --------------------------------------------------------------------------
# Shared entry point
# --------------------------------------------------------------------------


def with_retained_assembly_evidence(
    sidecar_data: SidecarData,
    *,
    provider: Provider | None,
    source_read: RetainedAssemblyRead,
    blob_store: BlobStore,
    source_path: str | None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None,
) -> SidecarData:
    """Fill assembly inputs the caller does not already carry.

    A caller that resolved a key during acquisition stays authoritative for
    it; this only supplies what is absent, which is every key on the replay
    and convergence routes where no live discovery ran.
    """
    if not source_path:
        return sidecar_data
    if provider is Provider.CLAUDE_CODE:
        wanted: tuple[str, ...] = ("session_index", "history_paste_index")
    elif provider is Provider.CODEX:
        wanted = ("thread_names", "history_titles")
    elif provider is Provider.CHATGPT:
        wanted = ("chatgpt_asset_index", "chatgpt_asset_blobs")
    else:
        return sidecar_data
    # An empty live Codex snapshot supplies no title evidence. Unlike the
    # other sidecar families, its two maps are often carried explicitly as
    # empty dicts, so key presence alone must not hide retained evidence.
    if provider is not Provider.CODEX and all(key in sidecar_data for key in wanted):
        return sidecar_data
    if provider is Provider.CLAUDE_CODE:
        retained = retained_claude_code_sidecars(source_read, blob_store, session_source_path=source_path)
    elif provider is Provider.CODEX:
        # The acquisition snapshot is the live authority.  Retained sidecars
        # are replay evidence only, so never mix them into a bundle that
        # already carries any live Codex title input.
        if any(sidecar_data.get(key) for key in ("thread_names", "history_titles", "state_titles")):
            return sidecar_data
        retained = retained_codex_sidecars(source_read, blob_store, session_source_path=source_path)
    else:
        retained = retained_chatgpt_sidecars(
            source_read, blob_store, session_source_path=source_path, captured_zip_coordinate=captured_zip_coordinate
        )
    if not retained:
        return sidecar_data
    merged: dict[str, object] = dict(sidecar_data)
    for key, value in retained.items():
        merged.setdefault(key, value)
    result = cast(SidecarData, merged)
    from .assembly import close_sidecar_data

    # A supplement whose keys were all already authoritative must settle its
    # private artifact. Any retained asset view kept in result carries that
    # same owner through the last consumer instead.
    close_sidecar_data(retained, borrowed=result)
    return result


__all__ = [
    "RetainedArtifact",
    "codex_sidecar_coordinates",
    "chatgpt_export_scope",
    "claude_code_sidecar_coordinates",
    "retained_codex_sidecars",
    "retained_chatgpt_sidecars",
    "retained_claude_code_sidecars",
    "with_retained_assembly_evidence",
]


class RetainedAssemblyRead(CompletedSourceItemRead, Protocol):
    """Finite retained artifact selection and exact acquired-byte capability."""

    def retained_artifact_page(
        self,
        origin: str,
        artifact_kind: str,
        coordinate: str,
        *,
        prefix: bool,
        after: str | None,
    ) -> list[tuple[str, str, str, int, int | None, int | None, str | None]]: ...

    def retained_group_member(
        self,
        generation: str,
        item: str,
        source_path: str,
        origin: str,
        artifact_kind: str,
    ) -> tuple[str, str, int, str | None] | None: ...

    def open_sidecar_payload(self, raw_id: str, blob_hash: bytes) -> AbstractContextManager[BinaryIO]: ...


def _retained_artifact_predicate(prefix: bool) -> str:
    return "a.source_path LIKE ? ESCAPE '\\'" if prefix else "a.source_path=?"


def _retained_artifact_page_sql(prefix: bool) -> str:
    receipt_order = raw_receipt_order_sql("r")
    # Cut complete path partitions before ranking. This preserves the exact
    # winner and avoids rereading earlier paths on subsequent keyset pages.
    return f"""
        WITH candidates AS (
            SELECT r.raw_id, a.source_path, lower(hex(r.blob_hash)) AS blob_hash, r.blob_size,
                   {receipt_order} AS receipt_order, c.entry_ordinal, c.captured_coordinate,
                   ROW_NUMBER() OVER (PARTITION BY a.source_path ORDER BY {receipt_order} DESC, r.raw_id) AS rank
            FROM raw_artifacts a JOIN raw_sessions r ON r.raw_id=a.raw_id
            LEFT JOIN raw_container_coordinates c ON c.raw_id=r.raw_id
            WHERE a.origin=? AND ({_retained_artifact_predicate(prefix)}) AND a.artifact_kind=?
              AND r.blob_hash IS NOT NULL AND (? IS NULL OR a.source_path>?)
        ) SELECT raw_id, source_path, blob_hash, blob_size, receipt_order, entry_ordinal, captured_coordinate
          FROM candidates WHERE rank=1 ORDER BY source_path LIMIT 256
    """


_RETAINED_GROUP_MEMBER_SQL = (
    "SELECT r.raw_id, lower(hex(r.blob_hash)), r.blob_size, c.captured_coordinate FROM source_item_raw_members m "
    "JOIN raw_sessions r ON r.raw_id=m.raw_id AND r.blob_hash=m.raw_blob_hash "
    "JOIN raw_container_coordinates c ON c.raw_id=r.raw_id "
    "JOIN raw_artifacts a ON a.raw_id=r.raw_id "
    "WHERE m.source_generation_id=? AND m.source_item_id=? AND a.source_path=? "
    "AND a.origin=? AND a.artifact_kind=? ORDER BY c.entry_ordinal, c.split_index LIMIT 1"
)


def _retained_artifact_page_from_rows(
    rows: sqlite3.Cursor,
) -> list[tuple[str, str, str, int, int | None, int | None, str | None]]:
    page: list[tuple[str, str, str, int, int | None, int | None, str | None]] = []
    for row in rows:
        check_compute_cancelled()
        page.append(
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                None if row[4] is None else int(row[4]),
                None if row[5] is None else int(row[5]),
                None if row[6] is None else str(row[6]),
            )
        )
    return page


def _retained_group_member_from_row(
    row: sqlite3.Row | tuple[Any, ...] | None,
) -> tuple[str, str, int, str | None] | None:
    return None if row is None else (str(row[0]), str(row[1]), int(row[2]), None if row[3] is None else str(row[3]))
