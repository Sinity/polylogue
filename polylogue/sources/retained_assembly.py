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
* **Currency follows durable receipt order** (R5). A rewritten index or map
  re-mints its content-derived raw id when its bytes return to an earlier
  value, so ``raw_sessions.acquired_at_ms`` is the first sighting, not the
  newest. The newest ``raw_payload`` receipt decides, exactly as
  ``sources/codex_state_projection.py`` orders one raw.
* **Absence is an outcome, never a guess** (R6/S6). A missing map resolves to
  an empty bundle and the parsed-content fallbacks apply; nothing is
  reconstructed from the recorded source path.

The anchors these lookups use are the same ones live discovery uses
(``assembly_claude_code.py``, ``assembly_chatgpt.py``), so a replay resolves
the evidence a live ingest resolved rather than a differently-scoped
approximation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.core.enums import Origin, Provider
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore

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


def _like_prefix(prefix: str) -> str:
    """Escape a literal path prefix for a ``LIKE ... ESCAPE '\\'`` match."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


_RECEIPT_ORDER = """
    SELECT b.{column}
    FROM blob_refs AS b
    WHERE b.ref_id = r.raw_id AND b.ref_type = 'raw_payload'
    ORDER BY b.acquired_at_ms DESC, b.rowid DESC
    LIMIT 1
"""


def _select_retained(
    source_conn: sqlite3.Connection,
    *,
    origin: Origin,
    artifact_kind: ArtifactKind,
    where: str,
    parameters: list[object],
) -> dict[str, RetainedArtifact]:
    """Return the newest retained observation per exact ``source_path``.

    One row per coordinate: an older observation of the same coordinate is
    still archived and readable, it is simply not the current value.
    """
    # The coordinate predicate is expressed on ``raw_artifacts`` so
    # ``idx_raw_artifacts_source_identity`` (origin, source_path, source_index)
    # serves both the exact-path and the export-prefix form; ``raw_sessions``
    # is joined only for the retained bytes and the receipt order.
    sql = f"""
        SELECT
            r.raw_id,
            a.source_path,
            lower(hex(r.blob_hash)),
            r.blob_size,
            COALESCE(({_RECEIPT_ORDER.format(column="acquired_at_ms")}), r.acquired_at_ms),
            COALESCE(({_RECEIPT_ORDER.format(column="rowid")}), r.rowid)
        FROM raw_artifacts AS a
        JOIN raw_sessions AS r ON r.raw_id = a.raw_id
        WHERE a.origin = ?
          AND ({where})
          AND a.artifact_kind = ?
          AND r.blob_hash IS NOT NULL
          AND r.parse_error IS NULL
        ORDER BY 5 DESC, 6 DESC, r.raw_id DESC
    """
    # A read failure here is infrastructure state, not an answer: it
    # propagates so the ingesting pass records a retryable outcome instead of
    # resolving to "no evidence" and writing a session that silently lost its
    # provider metadata.
    rows = source_conn.execute(sql, [origin.value, *parameters, artifact_kind.value]).fetchall()
    newest: dict[str, RetainedArtifact] = {}
    for raw_id, source_path, blob_hash, blob_size, _observed, _order in rows:
        path = str(source_path)
        if path in newest:
            continue
        newest[path] = RetainedArtifact(str(raw_id), path, str(blob_hash), int(blob_size))
    return newest


def _read(blob_store: BlobStore, artifact: RetainedArtifact) -> bytes | None:
    try:
        return blob_store.read_all(artifact.blob_hash)
    except (OSError, ValueError) as exc:
        logger.debug("retained assembly blob unavailable (%s): %s", artifact.source_path, exc)
        return None


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
    source_conn: sqlite3.Connection,
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
        source_conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        artifact_kind=ArtifactKind.SESSION_INDEX,
        where="a.source_path = ?",
        parameters=[index_path],
    )
    artifact = indexes.get(index_path)
    if artifact is not None:
        payload = _read(blob_store, artifact)
        if payload is not None:
            from .parsers.claude.index import parse_sessions_index_bytes

            entries: ClaudeCodeSessionIndex = parse_sessions_index_bytes(payload)
            if entries:
                resolved["session_index"] = entries

    histories = _select_retained(
        source_conn,
        origin=Origin.CLAUDE_CODE_SESSION,
        artifact_kind=ArtifactKind.PROMPT_HISTORY_LOG,
        where="a.source_path = ?",
        parameters=[history_path],
    )
    artifact = histories.get(history_path)
    if artifact is not None:
        payload = _read(blob_store, artifact)
        if payload is not None:
            from .parsers.claude.history import build_session_paste_index_bytes

            pastes: ClaudeCodeHistoryPasteIndex = build_session_paste_index_bytes(payload, origin=history_path)
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
    if not session_source_path or ":" in PurePosixPath(session_source_path).name:
        return None
    path = Path(session_source_path)
    sessions_root = next((parent for parent in path.parents if parent.name == "sessions"), None)
    if sessions_root is None:
        return None
    install_root = sessions_root.parent
    if str(install_root) in {"", ".", sessions_root.anchor}:
        return None
    return str(install_root / _CODEX_SESSION_INDEX_NAME), str(install_root / _CODEX_HISTORY_NAME)


def retained_codex_sidecars(
    source_conn: sqlite3.Connection,
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
        source_conn,
        origin=Origin.CODEX_SESSION,
        artifact_kind=ArtifactKind.SESSION_INDEX,
        where="a.source_path = ?",
        parameters=[index_path],
    )
    artifact = indexes.get(index_path)
    if artifact is not None:
        payload = _read(blob_store, artifact)
        if payload is not None:
            from .assembly_codex import parse_codex_session_index_bytes

            names = parse_codex_session_index_bytes(payload)
            if names:
                resolved["thread_names"] = names

    histories = _select_retained(
        source_conn,
        origin=Origin.CODEX_SESSION,
        artifact_kind=ArtifactKind.PROMPT_HISTORY_LOG,
        where="a.source_path = ?",
        parameters=[history_path],
    )
    artifact = histories.get(history_path)
    if artifact is not None:
        payload = _read(blob_store, artifact)
        if payload is not None:
            from .assembly_codex import parse_codex_history_bytes

            titles = parse_codex_history_bytes(payload)
            if titles:
                resolved["history_titles"] = titles
    return resolved


# --------------------------------------------------------------------------
# ChatGPT
# --------------------------------------------------------------------------


def chatgpt_export_scope(session_source_path: str) -> str | None:
    """Return the coordinate prefix of the export a shard is a member of.

    A ZIP member is recorded as ``<archive>:<member>``, so the export scope is
    everything up to and including that separator. A directory export's shard
    is scoped by its own containing directory -- the same anchor
    ``ChatGPTAssemblySpec.discover_sidecars`` climbs to. Two exports are two
    scopes in either acquisition order.
    """
    if not session_source_path:
        return None
    head, separator, _member = session_source_path.partition(".zip:")
    if separator:
        return f"{head}.zip:"
    parent = Path(session_source_path).parent
    if str(parent) in {"", "."}:
        return None
    return f"{parent}/"


def retained_chatgpt_sidecars(
    source_conn: sqlite3.Connection,
    blob_store: BlobStore,
    *,
    session_source_path: str,
) -> SidecarData:
    """Rebuild the ChatGPT asset index and asset-blob map from retained bytes."""
    scope = chatgpt_export_scope(session_source_path)
    if scope is None:
        return cast(SidecarData, {})
    from .assembly_chatgpt import _member_asset_id
    from .parsers.chatgpt_sidecars import ChatGPTAssetIndex

    library_payload: object | None = None
    asset_names_payload: object | None = None
    indexes = _select_retained(
        source_conn,
        origin=Origin.CHATGPT_EXPORT,
        artifact_kind=ArtifactKind.EXPORT_ASSET_INDEX,
        where="a.source_path LIKE ? ESCAPE '\\'",
        parameters=[_like_prefix(scope)],
    )
    for path, artifact in sorted(indexes.items()):
        payload = _read(blob_store, artifact)
        if payload is None:
            continue
        from polylogue.core.json import JSONDecodeError
        from polylogue.core.json import loads as json_loads

        try:
            document = json_loads(payload)
        except (JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            logger.debug("retained chatgpt asset index is not JSON (%s): %s", path, exc)
            continue
        name = PurePosixPath(path.replace("\\", "/")).name
        if name == "library_files.json" and library_payload is None:
            library_payload = document
        elif name == "conversation_asset_file_names.json" and asset_names_payload is None:
            asset_names_payload = document

    asset_blobs: dict[str, tuple[str, int]] = {}
    assets = _select_retained(
        source_conn,
        origin=Origin.CHATGPT_EXPORT,
        artifact_kind=ArtifactKind.EXPORT_ASSET,
        where="a.source_path LIKE ? ESCAPE '\\'",
        parameters=[_like_prefix(scope)],
    )
    for path, artifact in sorted(assets.items()):
        asset_id = _member_asset_id(PurePosixPath(path.replace("\\", "/")).name)
        if asset_id is None or asset_id in asset_blobs:
            continue
        asset_blobs[asset_id] = (artifact.blob_hash, artifact.blob_size)

    if library_payload is None and asset_names_payload is None and not asset_blobs:
        return cast(SidecarData, {})
    resolved: SidecarData = {
        "chatgpt_asset_index": ChatGPTAssetIndex.build(
            library_files_payload=library_payload,
            asset_file_names_payload=asset_names_payload,
        )
    }
    if asset_blobs:
        resolved["chatgpt_asset_blobs"] = asset_blobs
    return resolved


# --------------------------------------------------------------------------
# Shared entry point
# --------------------------------------------------------------------------


def with_retained_assembly_evidence(
    sidecar_data: SidecarData,
    *,
    provider: Provider | None,
    source_conn: sqlite3.Connection,
    blob_store: BlobStore,
    source_path: str | None,
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
        retained = retained_claude_code_sidecars(source_conn, blob_store, session_source_path=source_path)
    elif provider is Provider.CODEX:
        # The acquisition snapshot is the live authority.  Retained sidecars
        # are replay evidence only, so never mix them into a bundle that
        # already carries any live Codex title input.
        if any(sidecar_data.get(key) for key in ("thread_names", "history_titles", "state_titles")):
            return sidecar_data
        retained = retained_codex_sidecars(source_conn, blob_store, session_source_path=source_path)
    else:
        retained = retained_chatgpt_sidecars(source_conn, blob_store, session_source_path=source_path)
    if not retained:
        return sidecar_data
    merged: dict[str, object] = dict(sidecar_data)
    for key, value in retained.items():
        merged.setdefault(key, value)
    return cast(SidecarData, merged)


def resolve_retained_assembly_evidence(
    sidecar_data: SidecarData,
    *,
    provider: Provider | None,
    archive_root: Path,
    source_path: str | None,
) -> SidecarData:
    """Fill missing assembly inputs from the archive at ``archive_root``.

    The archive is the evidence carrier, so both the pipeline ingest worker
    and retained-raw replay resolve these inputs from it rather than from any
    live file beside the original source path. A read-only connection keeps
    this safe beside the daemon's single writer, and an absent or unreadable
    source tier degrades to no evidence rather than to a rediscovery.
    """
    if provider not in {Provider.CLAUDE_CODE, Provider.CODEX, Provider.CHATGPT} or not source_path:
        return sidecar_data
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return sidecar_data
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=5.0)
    try:
        return with_retained_assembly_evidence(
            sidecar_data,
            provider=provider,
            source_conn=conn,
            blob_store=BlobStore(archive_root / "blob"),
            source_path=source_path,
        )
    finally:
        conn.close()


__all__ = [
    "RetainedArtifact",
    "resolve_retained_assembly_evidence",
    "codex_sidecar_coordinates",
    "chatgpt_export_scope",
    "claude_code_sidecar_coordinates",
    "retained_codex_sidecars",
    "retained_chatgpt_sidecars",
    "retained_claude_code_sidecars",
    "with_retained_assembly_evidence",
]
