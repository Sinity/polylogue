"""Source detection, provider parsing, and conversation iteration."""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

import ijson
from pydantic import BaseModel

from polylogue.config import Source
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.log import get_logger
from polylogue.types import Provider

from ..storage.store import AttachmentRecord, ContentBlockRecord, ConversationRecord, MessageRecord
from .parsers import chatgpt, claude, codex, drive
from .parsers.base import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
    extract_messages_from_list,
)
from .parsers.claude import (
    SessionIndexEntry,
    enrich_conversation_from_index,
    parse_sessions_index,
)

if TYPE_CHECKING:
    from ..storage.repository import ConversationRepository

logger = get_logger(__name__)


class RecordBundle(BaseModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]
    content_blocks: list[ContentBlockRecord] = []


class SaveResult(BaseModel):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


async def save_bundle(bundle: RecordBundle, repository: ConversationRepository) -> SaveResult:
    """Save a bundle of records into the repository.

    Args:
        bundle: Bundle containing conversation, messages, and attachments
        repository: Storage repository to save records to

    Returns:
        SaveResult with counts of imported/skipped items
    """
    counts = await repository.save_conversation(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
        content_blocks=bundle.content_blocks,
    )
    return SaveResult(**counts)


_ENCODING_GUESSES: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
)

# ZIP bomb protection constants
MAX_COMPRESSION_RATIO = 1000  # 1000x — JSON/JSONL compresses extremely well (100-500x typical)
MAX_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024  # 10GB — multi-year chat archives can be large


def _decode_json_bytes(blob: bytes) -> str | None:
    """Decode a JSON payload from bytes, trying multiple encodings."""

    for encoding in _ENCODING_GUESSES:
        try:
            decoded = blob.decode(encoding)
        except UnicodeError:
            continue
        cleaned = decoded.replace("\x00", "").lstrip("\ufeff")
        if cleaned:
            return cleaned
    try:
        decoded = blob.decode("utf-8", errors="ignore").replace("\x00", "")
        return decoded if decoded else None
    except (AttributeError, UnicodeDecodeError):
        logger.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def detect_provider(payload: Any, path: Path) -> Provider | None:
    if isinstance(payload, dict):
        if chatgpt.looks_like(payload):
            return Provider.CHATGPT
        if claude.looks_like_ai(payload):
            return Provider.CLAUDE
        if claude.looks_like_code([payload]):
            return Provider.CLAUDE_CODE
        if codex.looks_like([payload]):
            return Provider.CODEX
        # Gemini content-based detection (chunkedPrompt or chunks list)
        if "chunkedPrompt" in payload or ("chunks" in payload and isinstance(payload.get("chunks"), list)):
            return Provider.GEMINI
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            first = payload[0]
            # ChatGPT bundle export: list[conversation]
            if isinstance(first.get("mapping"), dict):
                return Provider.CHATGPT
            # Claude AI bundle export: list[conversation]
            if isinstance(first.get("chat_messages"), list):
                return Provider.CLAUDE
            # Gemini/Drive grouped list of conversations/chunks
            if "chunkedPrompt" in first or ("chunks" in first and isinstance(first.get("chunks"), list)):
                return Provider.GEMINI
        if claude.looks_like_code(payload):
            return Provider.CLAUDE_CODE
        if codex.looks_like(payload):
            return Provider.CODEX

    # Check filename and parent directory for provider hints
    name = path.name.lower()
    path_str = str(path).lower()

    if "chatgpt" in name or "chatgpt" in path_str:
        return Provider.CHATGPT
    if "claude-code" in name or "claude_code" in name or "claude-code" in path_str or "claude_code" in path_str:
        return Provider.CLAUDE_CODE
    if "claude" in name or "/claude/" in path_str:
        return Provider.CLAUDE
    if "codex" in name or "codex" in path_str:
        return Provider.CODEX
    if "gemini" in name or "gemini" in path_str:
        return Provider.GEMINI
    return None


_MAX_PARSE_DEPTH = 10


def _looks_like_chunked_conversation(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        drive.looks_like(payload)
        or isinstance(payload.get("chunks"), list)
    )


def _looks_like_chunked_conversation_list(payload: list[Any]) -> bool:
    return bool(payload) and all(_looks_like_chunked_conversation(item) for item in payload)


def parse_payload(provider: str, payload: Any, fallback_id: str, _depth: int = 0) -> list[ParsedConversation]:
    """Dispatch parsed payload to the appropriate provider parser.

    This function is the central routing point for all conversation parsing.
    Each provider has different wire formats:
    - ChatGPT: nested mapping with node IDs (``parse``)
    - Claude AI: ``{"conversations": [{"chat_messages": [...]}]}`` (``parse_ai``)
    - Claude Code: JSONL entries with ``sessionId`` (``parse_code``)
    - Codex: flat ``[{role, content}]`` message lists (``parse``)
    - Gemini/Drive: ``{"chunkedPrompt": {"chunks": [...]}}`` (``parse_chunked_prompt``)

    Signatures differ across providers because wire formats vary significantly.
    Unifying them would require an abstraction layer that adds complexity
    without value — the dispatch here IS the abstraction.

    Args:
        provider: Detected provider name (e.g. "chatgpt", "claude", "gemini")
        payload: Deserialized JSON content (dict or list depending on format)
        fallback_id: ID to use if the payload lacks an explicit conversation ID
        _depth: Recursion guard for nested structures (max: _MAX_PARSE_DEPTH)

    Returns:
        List of ParsedConversation objects extracted from the payload
    """
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing %s (provider=%s)", fallback_id, provider)
        return []
    if isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
        results: list[ParsedConversation] = []
        for i, item in enumerate(payload["conversations"]):
            if isinstance(item, dict):
                    results.extend(parse_payload(provider, item, f"{fallback_id}-{i}", _depth + 1))
        return results
    if provider == Provider.CHATGPT:
        if isinstance(payload, list):
            results: list[ParsedConversation] = []
            for i, item in enumerate(payload):
                if isinstance(item, dict):
                    results.append(chatgpt.parse(item, f"{fallback_id}-{i}"))
            return results
        return [chatgpt.parse(payload, fallback_id)]
    if provider == Provider.CLAUDE:
        if isinstance(payload, list):
            results: list[ParsedConversation] = []
            for i, item in enumerate(payload):
                if isinstance(item, dict):
                    results.append(claude.parse_ai(item, f"{fallback_id}-{i}"))
            return results
        if isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
            results: list[ParsedConversation] = []
            for i, item in enumerate(payload["conversations"]):
                if isinstance(item, dict):
                    results.append(claude.parse_ai(item, f"{fallback_id}-{i}"))
            if results:
                return results
        return [claude.parse_ai(payload, fallback_id)]
    if provider == Provider.CLAUDE_CODE:
        if isinstance(payload, list):
            return [claude.parse_code(payload, fallback_id)]
        if isinstance(payload, dict):
            if isinstance(payload.get("messages"), list):
                return [claude.parse_code(payload["messages"], fallback_id)]
            return [claude.parse_code([payload], fallback_id)]
    if provider == Provider.CODEX:
        if isinstance(payload, list):
            return [codex.parse(payload, fallback_id)]
        if isinstance(payload, dict):
            return [codex.parse([payload], fallback_id)]
    if provider in (Provider.GEMINI, Provider.DRIVE) and isinstance(payload, list):
        if _looks_like_chunked_conversation_list(payload):
            results = []
            for i, item in enumerate(payload):
                results.extend(parse_payload(provider, item, f"{fallback_id}-{i}", _depth + 1))
            return results
        return [drive.parse_chunked_prompt(provider, {"chunks": payload}, fallback_id)]

    # Fallback / Generic
    if isinstance(payload, dict):
        # Generic "messages" support
        if "messages" in payload and isinstance(payload["messages"], list):
            messages = extract_messages_from_list(payload["messages"])
            title = payload.get("title") or payload.get("name") or fallback_id
            return [
                ParsedConversation(
                    provider_name=provider,
                    provider_conversation_id=str(payload.get("id") or fallback_id),
                    title=str(title),
                    created_at=None,
                    updated_at=None,
                    messages=messages,
                )
            ]

        if chatgpt.looks_like(payload):
            return [chatgpt.parse(payload, fallback_id)]
        if _looks_like_chunked_conversation(payload):
            return [drive.parse_chunked_prompt(provider, payload, fallback_id)]
        return []

    return []


def parse_drive_payload(provider: str, payload: Any, fallback_id: str, _depth: int = 0) -> list[ParsedConversation]:
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing drive payload %s", fallback_id)
        return []
    if isinstance(payload, list):
        # Check if it looks like a list of conversations or a list of messages
        # For drive/gemini, if it's a list, it's often a list of chunks
        if payload and isinstance(payload[0], dict) and ("role" in payload[0] or "text" in payload[0]):
            return [drive.parse_chunked_prompt(provider, {"chunks": payload}, fallback_id)]

        results = []
        for i, item in enumerate(payload):
            results.extend(parse_drive_payload(provider, item, f"{fallback_id}-{i}", _depth + 1))
        return results
    if isinstance(payload, dict):
        if "chunkedPrompt" in payload or "chunks" in payload:
            return [drive.parse_chunked_prompt(provider, payload, fallback_id)]
        detected = detect_provider(payload, Path(fallback_id)) or provider
        return parse_payload(detected, payload, fallback_id)
    return []


def _iter_json_stream(handle: BinaryIO | IO[bytes], path_name: str, unpack_lists: bool = True) -> Iterable[Any]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        error_count = 0
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            if isinstance(raw, bytes):
                decoded = _decode_json_bytes(raw)
                if not decoded:
                    logger.warning("Skipping undecodable line from %s", path_name)
                    continue
            else:
                decoded = raw
            try:
                yield json.loads(decoded)
            except json.JSONDecodeError as exc:
                # Log first few errors, then summarize
                error_count += 1
                if error_count <= 3:
                    logger.warning("Skipping invalid JSON line in %s: %s", path_name, exc)
                elif error_count == 4:
                    logger.warning("Skipping further invalid JSON lines in %s...", path_name)
                continue
        if error_count > 3:
            logger.warning("Skipped %d invalid JSON lines in %s", error_count, path_name)
        return

    if unpack_lists:
        # Strategy 1: Try streaming root list
        try:
            found_any = False
            for item in ijson.items(handle, "item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates
        except Exception as exc:
            logger.debug("Strategy 1 (ijson items) failed for %s: %s", path_name, exc)
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates

        handle.seek(0)
        # Strategy 2: Try streaming conversations list
        try:
            found_any = False
            for item in ijson.items(handle, "conversations.item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates
        except Exception as exc:
            logger.debug("Strategy 2 (ijson conversations.item) failed for %s: %s", path_name, exc)
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates

        handle.seek(0)
    # Strategy 3: Load full object (fallback for single dicts or unknown structures)
    # Let JSONDecodeError propagate so outer handler can track failed files
    data = json.load(handle)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        if unpack_lists:
            yield from data
        else:
            yield data


_SUPPORTED_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson", ".zip"})
_SUPPORTED_DOUBLE_EXTENSIONS = frozenset({".jsonl.txt"})

# Directories to skip during source parsing.
# These contain derived/analysis artifacts, not raw conversation data.
_SKIP_DIRS = frozenset({"analysis", "__pycache__", ".git", "node_modules"})


def _has_supported_extension(path: Path) -> bool:
    """Check if path has a supported file extension (case-insensitive)."""
    name_lower = path.name.lower()
    # Check double extensions first (e.g., .jsonl.txt)
    for ext in _SUPPORTED_DOUBLE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True
    # Check single extensions
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


# =============================================================================
# Decomposition helpers
# =============================================================================


def _get_file_mtime(path: Path) -> str | None:
    """Get ISO-format mtime for a path, or None on OSError."""
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _record_cursor_failure(
    cursor_state: dict[str, Any] | None,
    path: str,
    error: str,
) -> None:
    """Record a file processing failure in cursor_state."""
    if cursor_state is not None:
        cursor_state["failed_files"].append({"path": path, "error": error})
        cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1


def _walk_source_paths(base: Path) -> list[Path]:
    """Walk a directory and return sorted paths with supported extensions.

    Prunes ``_SKIP_DIRS`` during traversal.
    """
    paths: list[Path] = []
    for root, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for filename in files:
            file_path = Path(root) / filename
            if _has_supported_extension(file_path):
                paths.append(file_path)
    return sorted(paths)


def _build_session_indices(paths: list[Path]) -> dict[Path, dict[str, SessionIndexEntry]]:
    """Load ``sessions-index.json`` for each unique parent directory."""
    indices: dict[Path, dict[str, SessionIndexEntry]] = {}
    for path in paths:
        parent = path.parent
        if parent not in indices:
            index_path = parent / "sessions-index.json"
            indices[parent] = parse_sessions_index(index_path)
    return indices


def _resolve_source_paths(source: Source) -> list[Path]:
    """Resolve a source path into sorted candidate files."""
    if not source.path:
        return []

    base = source.path.expanduser()
    if base.is_dir():
        return _walk_source_paths(base)
    if base.is_file():
        return [base]
    return []


def _initialize_cursor_state(
    cursor_state: dict[str, Any] | None,
    paths: list[Path],
) -> None:
    """Populate cursor bookkeeping for a source iteration pass."""
    if cursor_state is None:
        return

    cursor_state["file_count"] = len(paths)
    cursor_state.setdefault("failed_files", [])
    cursor_state.setdefault("failed_count", 0)
    if not paths:
        return

    try:
        latest = max(paths, key=lambda p: p.stat().st_mtime)
        cursor_state["latest_mtime"] = latest.stat().st_mtime
        cursor_state["latest_path"] = str(latest)
    except OSError:
        pass


def _select_paths_for_processing(
    paths: list[Path],
    *,
    include_file_mtime: bool,
    known_mtimes: dict[str, str] | None = None,
) -> tuple[list[tuple[Path, str | None]], int]:
    """Filter unchanged files and return `(path, file_mtime)` tuples."""
    selected: list[tuple[Path, str | None]] = []
    skipped_mtime = 0

    for path in paths:
        file_mtime = _get_file_mtime(path) if include_file_mtime else None
        if known_mtimes and file_mtime and known_mtimes.get(str(path)) == file_mtime:
            skipped_mtime += 1
            continue
        selected.append((path, file_mtime))

    return selected, skipped_mtime


def _log_source_iteration_summary(
    *,
    source_name: str,
    total_paths: int,
    skipped_mtime: int,
    failed_count: int,
    failure_kind: str,
) -> None:
    """Emit common skip/failure summaries for source iterators."""
    if skipped_mtime > 0:
        logger.info(
            "Skipped %d of %d files from source %r (unchanged mtime)",
            skipped_mtime,
            total_paths,
            source_name,
        )

    if failed_count > 0:
        logger.warning(
            "Skipped %d of %d files from source %r due to %s errors. "
            "Run with --verbose for details.",
            failed_count,
            total_paths,
            source_name,
            failure_kind,
        )


# =============================================================================
# Parse context and conversation emitter
# =============================================================================

_GROUP_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE})


@dataclass
class _SourceWalkSetup:
    """Result of shared source-path setup used by both public iterators."""

    paths: list[Path]
    paths_to_process: list[tuple[Path, str | None]]
    skipped_mtime: int
    session_indices: dict[Path, dict[str, SessionIndexEntry]]


def _setup_source_walk(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None,
    include_mtime: bool,
    known_mtimes: dict[str, str] | None,
    build_session_indices: bool,
) -> _SourceWalkSetup | None:
    """Resolve source paths and prepare iteration state.

    Returns None when the source yields no paths (callers should return early).
    """
    paths = _resolve_source_paths(source)
    _initialize_cursor_state(cursor_state, paths)
    if not paths:
        return None
    paths_to_process, skipped_mtime = _select_paths_for_processing(
        paths,
        include_file_mtime=include_mtime,
        known_mtimes=known_mtimes,
    )
    session_indices = _build_session_indices(paths) if build_session_indices else {}
    return _SourceWalkSetup(
        paths=paths,
        paths_to_process=paths_to_process,
        skipped_mtime=skipped_mtime,
        session_indices=session_indices,
    )


@dataclass
class _ParseContext:
    """All context needed to parse a stream and yield conversations."""

    provider_hint: Provider
    should_group: bool
    source_path_str: str  # For RawConversationData.source_path
    fallback_id: str  # path.stem, used as fallback conversation ID
    file_mtime: str | None
    capture_raw: bool
    session_index: dict[str, SessionIndexEntry]
    detect_path: Path  # For detect_provider calls on individual payloads


class _ConversationEmitter:
    """Parse a binary stream and yield ``(raw, conv)`` tuples.

    Unifies the grouped-JSONL, individual-items, and raw-capture logic
    that was previously duplicated across ZIP and filesystem code paths.
    """

    __slots__ = ("_ctx",)

    def __init__(self, ctx: _ParseContext) -> None:
        self._ctx = ctx

    def emit(
        self,
        handle: BinaryIO,
        stream_name: str,
        *,
        pre_read_bytes: bytes | None = None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        """Parse a stream and yield ``(raw, conv)`` tuples.

        Args:
            handle: Binary stream to read from.
            stream_name: Filename for ``_iter_json_stream`` (determines
                JSONL vs JSON parsing strategy).
            pre_read_bytes: If provided, already-read bytes for raw capture.
                Used when the caller pre-read the whole file for grouped
                providers with ``capture_raw=True``.
        """
        lower = stream_name.lower()
        is_jsonl = lower.endswith((".jsonl", ".jsonl.txt", ".ndjson"))

        if is_jsonl and self._ctx.should_group:
            yield from self._emit_grouped(handle, stream_name, pre_read_bytes)
        else:
            yield from self._emit_individual(handle, stream_name, pre_read_bytes=pre_read_bytes)

    def _emit_grouped(
        self,
        handle: BinaryIO,
        stream_name: str,
        pre_read_bytes: bytes | None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        """Grouped JSONL: entire file = one conversation."""
        if self._ctx.capture_raw and pre_read_bytes is None:
            raw_bytes = handle.read()
            handle = BytesIO(raw_bytes)  # type: ignore[assignment]
        else:
            raw_bytes = pre_read_bytes
            if raw_bytes is not None:
                handle = BytesIO(raw_bytes)  # type: ignore[assignment]

        payloads = list(_iter_json_stream(handle, stream_name))
        if not payloads:
            return

        raw_data = self._make_raw(raw_bytes) if raw_bytes else None
        provider = detect_provider(payloads, Path(stream_name)) or self._ctx.provider_hint
        for conv in parse_payload(provider, payloads, self._ctx.fallback_id):
            yield (raw_data, self._maybe_enrich(conv))

    def _emit_individual(
        self,
        handle: BinaryIO,
        stream_name: str,
        *,
        pre_read_bytes: bytes | None = None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        """Individual items: each payload = one conversation."""
        unpack = not (stream_name.lower().endswith(".json") and self._ctx.should_group)

        # If caller pre-read the whole file, use that as one raw capture
        # (for should_group + capture_raw + non-JSONL files)
        whole_file_raw = self._make_raw(pre_read_bytes) if pre_read_bytes is not None else None

        source_index = 0
        for payload in _iter_json_stream(handle, stream_name, unpack_lists=unpack):
            try:
                provider = detect_provider(payload, self._ctx.detect_path) or self._ctx.provider_hint

                if whole_file_raw is not None:
                    raw_data: RawConversationData | None = whole_file_raw
                elif self._ctx.capture_raw:
                    raw_bytes = json_dumps(payload).encode("utf-8")
                    raw_data = self._make_raw(raw_bytes, source_index=source_index, provider_override=provider)
                else:
                    raw_data = None

                for conv in parse_payload(provider, payload, self._ctx.fallback_id):
                    yield (raw_data, self._maybe_enrich(conv, provider))
                source_index += 1
            except Exception:
                logger.exception("Error processing payload from %s", stream_name)
                raise

    def _make_raw(
        self,
        raw_bytes: bytes | None,
        *,
        source_index: int | None = None,
        provider_override: str | None = None,
    ) -> RawConversationData | None:
        """Construct ``RawConversationData``, or ``None`` if no bytes."""
        if raw_bytes is None or not self._ctx.capture_raw:
            return None
        return RawConversationData(
            raw_bytes=raw_bytes,
            source_path=self._ctx.source_path_str,
            source_index=source_index,
            file_mtime=self._ctx.file_mtime,
            provider_hint=provider_override or self._ctx.provider_hint,
        )

    def _maybe_enrich(
        self,
        conv: ParsedConversation,
        provider: str | None = None,
    ) -> ParsedConversation:
        """Apply Claude Code session index enrichment if applicable."""
        p = provider or self._ctx.provider_hint
        idx = self._ctx.session_index
        if p == Provider.CLAUDE_CODE and conv.provider_conversation_id in idx:
            return enrich_conversation_from_index(conv, idx[conv.provider_conversation_id])
        return conv


# =============================================================================
# ZIP processing
# =============================================================================


class _ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_provider_hint", "_cursor_state", "_zip_path")

    def __init__(
        self,
        provider_hint: str,
        *,
        cursor_state: dict[str, Any] | None,
        zip_path: Path,
    ) -> None:
        self._provider_hint = provider_hint
        self._cursor_state = cursor_state
        self._zip_path = zip_path

    def filter_entries(self, entries: list[zipfile.ZipInfo]) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries.  Record failures in cursor_state."""
        for info in entries:
            if info.is_dir():
                continue
            name = info.filename
            lower_name = name.lower()

            # Filter Claude AI ZIP: only process conversations.json
            if self._provider_hint in ("claude", "claude-ai"):
                basename = lower_name.split("/")[-1]
                if basename not in ("conversations.json",):
                    continue

            # ZIP bomb protection: compression ratio
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    logger.warning(
                        "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                        name,
                        self._zip_path,
                        ratio,
                    )
                    _record_cursor_failure(
                        self._cursor_state,
                        f"{self._zip_path}:{name}",
                        f"Suspicious compression ratio: {ratio:.1f}",
                    )
                    continue

            # ZIP bomb protection: uncompressed size
            if info.file_size > MAX_UNCOMPRESSED_SIZE:
                logger.warning(
                    "Skipping oversized file %s in %s: %d bytes exceeds limit",
                    name,
                    self._zip_path,
                    info.file_size,
                )
                _record_cursor_failure(
                    self._cursor_state,
                    f"{self._zip_path}:{name}",
                    f"File size {info.file_size} exceeds limit",
                )
                continue

            # Only yield entries with supported JSON extensions
            if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                yield info


def _process_zip(
    zip_path: Path,
    *,
    provider_hint: str,
    should_group: bool,
    file_mtime: str | None,
    capture_raw: bool,
    cursor_state: dict[str, Any] | None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Process a ZIP file, yielding conversations from its entries."""
    validator = _ZipEntryValidator(provider_hint, cursor_state=cursor_state, zip_path=zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for info in validator.filter_entries(zf.infolist()):
            name = info.filename
            entry_provider_hint = detect_provider(None, Path(name)) or provider_hint
            entry_should_group = entry_provider_hint in _GROUP_PROVIDERS
            ctx = _ParseContext(
                provider_hint=entry_provider_hint,
                should_group=entry_should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                session_index={},  # ZIP files don't have session indices
                detect_path=Path(name),
            )
            emitter = _ConversationEmitter(ctx)
            with zf.open(name) as handle:
                yield from emitter.emit(handle, name)


# =============================================================================
# Public iteration API
# =============================================================================


def iter_source_conversations(
    source: Source, *, cursor_state: dict[str, Any] | None = None
) -> Iterable[ParsedConversation]:
    """Iterate over conversations from a source.

    Delegates to iter_source_conversations_with_raw with capture_raw=False
    and strips the raw data from the yielded tuples.
    """
    for _raw, conv in iter_source_conversations_with_raw(source, cursor_state=cursor_state, capture_raw=False):
        yield conv


def iter_source_conversations_with_raw(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    capture_raw: bool = True,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Iterate over conversations with optional raw byte capture.

    This is the raw-capturing version of iter_source_conversations(). It yields
    tuples of (raw_data, parsed_conversation) where raw_data contains the original
    JSON bytes that produced the conversation.

    For JSONL files where one file = one conversation (claude-code, codex), the
    entire file is captured. For bundle files (chatgpt conversations.json), each
    conversation dict is re-serialized to capture individual raw bytes.

    Args:
        source: Source configuration to iterate
        cursor_state: Optional state dict for tracking progress
        capture_raw: Whether to capture raw bytes (True by default)
        known_mtimes: Optional dict of {source_path: file_mtime} from previous runs.
            Files whose current mtime matches the known mtime are skipped entirely,
            replacing a full read+SHA256 with a single stat() call.

    Yields:
        Tuples of (RawConversationData | None, ParsedConversation)
    """
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=capture_raw,
        known_mtimes=known_mtimes,
        build_session_indices=True,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            provider_hint = detect_provider(None, path) or Provider.from_string(source.name)
            should_group = provider_hint in _GROUP_PROVIDERS

            if path.suffix.lower() == ".zip":
                yield from _process_zip(
                    path,
                    provider_hint=provider_hint,
                    should_group=should_group,
                    file_mtime=file_mtime,
                    capture_raw=capture_raw,
                    cursor_state=cursor_state,
                )
            else:
                ctx = _ParseContext(
                    provider_hint=provider_hint,
                    should_group=should_group,
                    source_path_str=str(path),
                    fallback_id=path.stem,
                    file_mtime=file_mtime,
                    capture_raw=capture_raw,
                    session_index=walk.session_indices.get(path.parent, {}),
                    detect_path=path,
                )
                emitter = _ConversationEmitter(ctx)

                if capture_raw and should_group:
                    raw_bytes = path.read_bytes()
                    yield from emitter.emit(
                        BytesIO(raw_bytes), path.name, pre_read_bytes=raw_bytes,
                    )
                else:
                    with path.open("rb") as handle:
                        yield from emitter.emit(handle, path.name)
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(cursor_state, str(path), f"File not found (may have been deleted): {exc}")
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            failed_count += 1
            logger.warning("Failed to parse %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error processing %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="parse/read",
    )


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[RawConversationData]:
    """Iterate raw source payloads without parsing provider payload semantics.

    This iterator is intended for acquisition-stage storage only. It yields one
    RawConversationData item per file (or per ZIP JSON entry) and performs no
    provider parser dispatching.

    Args:
        source: Source configuration to iterate
        cursor_state: Optional state dict for tracking progress
        known_mtimes: Optional dict of {source_path: file_mtime} from previous runs.
            Files whose current mtime matches the known mtime are skipped entirely.

    Yields:
        RawConversationData blobs suitable for raw_conversations storage.
    """
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=True,
        known_mtimes=known_mtimes,
        build_session_indices=False,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            provider_hint = detect_provider(None, path) or Provider.from_string(source.name)

            if path.suffix.lower() == ".zip":
                validator = _ZipEntryValidator(provider_hint, cursor_state=cursor_state, zip_path=path)
                with zipfile.ZipFile(path) as zf:
                    for info in validator.filter_entries(zf.infolist()):
                        entry_path = f"{path}:{info.filename}"
                        with zf.open(info.filename) as handle:
                            raw_bytes = handle.read()
                        yield RawConversationData(
                            raw_bytes=raw_bytes,
                            source_path=entry_path,
                            source_index=None,
                            file_mtime=file_mtime,
                            provider_hint=provider_hint,
                        )
            else:
                yield RawConversationData(
                    raw_bytes=path.read_bytes(),
                    source_path=str(path),
                    source_index=None,
                    file_mtime=file_mtime,
                    provider_hint=provider_hint,
                )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(cursor_state, str(path), f"File not found (may have been deleted): {exc}")
        except (UnicodeDecodeError, zipfile.BadZipFile, OSError) as exc:
            failed_count += 1
            logger.warning("Failed to read %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error reading %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="read",
    )


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "RawConversationData",
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
    "iter_source_raw_data",
    "parse_payload",
    "parse_drive_payload",
]
