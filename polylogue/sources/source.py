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

from ..storage.store import AttachmentRecord, ConversationRecord, MessageRecord
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
        cleaned = decoded.replace("\x00", "")
        if cleaned:
            return cleaned
    try:
        decoded = blob.decode("utf-8", errors="ignore").replace("\x00", "")
        return decoded if decoded else None
    except (AttributeError, UnicodeDecodeError):
        logger.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def detect_provider(payload: Any, path: Path) -> str | None:
    if isinstance(payload, dict):
        if chatgpt.looks_like(payload):
            return "chatgpt"
        if claude.looks_like_ai(payload):
            return "claude"
        # Gemini content-based detection (chunkedPrompt or chunks list)
        if "chunkedPrompt" in payload or ("chunks" in payload and isinstance(payload.get("chunks"), list)):
            return "gemini"
    if isinstance(payload, list):
        if claude.looks_like_code(payload):
            return "claude-code"
        if codex.looks_like(payload):
            return "codex"

    # Check filename and parent directory for provider hints
    name = path.name.lower()
    path_str = str(path).lower()

    if "chatgpt" in name or "chatgpt" in path_str:
        return "chatgpt"
    if "claude-code" in name or "claude_code" in name or "claude-code" in path_str or "claude_code" in path_str:
        return "claude-code"
    if "claude" in name or "/claude/" in path_str:
        return "claude"
    if "codex" in name or "codex" in path_str:
        return "codex"
    if "gemini" in name or "gemini" in path_str:
        return "gemini"
    return None


_MAX_PARSE_DEPTH = 10


def _parse_json_payload(provider: str, payload: Any, fallback_id: str, _depth: int = 0) -> list[ParsedConversation]:
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
    if provider == "chatgpt":
        return [chatgpt.parse(payload, fallback_id)]
    if provider == "claude":
        return [claude.parse_ai(payload, fallback_id)]
    if provider == "claude-code":
        if isinstance(payload, list):
            return [claude.parse_code(payload, fallback_id)]
        # If payload is a dict with messages, extract them
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            return [claude.parse_code(payload["messages"], fallback_id)]
    if provider == "codex":
        if isinstance(payload, list):
            return [codex.parse(payload, fallback_id)]
        if isinstance(payload, dict) and ("prompt" in payload or "completion" in payload):
                return [codex.parse([payload], fallback_id)]
    if (provider == "gemini" or provider == "drive") and isinstance(payload, list):
        # Check if items are conversation dicts (have 'chunks' key) or raw chunks
        if payload and isinstance(payload[0], dict) and "chunks" in payload[0]:
            # List of conversation dicts from grouped JSONL - parse each one
            results = []
            for i, item in enumerate(payload):
                results.extend(_parse_json_payload(provider, item, f"{fallback_id}-{i}", _depth + 1))
            return results
        # Treat list as chunks for a single conversation
        return [drive.parse_chunked_prompt(provider, {"chunks": payload}, fallback_id)]

    # Fallback / Generic
    if isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            results = []
            for i, item in enumerate(payload["conversations"]):
                parsed = _parse_json_payload(provider, item, f"{fallback_id}-{i}", _depth + 1)
                results.extend(parsed)
            return results

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

        return [
            chatgpt.parse(payload, fallback_id)
            if chatgpt.looks_like(payload)
            else drive.parse_chunked_prompt(provider, payload, fallback_id)
        ]

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
        return _parse_json_payload(detected, payload, fallback_id)
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


# =============================================================================
# Parse context and conversation emitter
# =============================================================================

_GROUP_PROVIDERS = frozenset({"claude-code", "codex", "gemini", "drive"})


@dataclass
class _ParseContext:
    """All context needed to parse a stream and yield conversations."""

    provider_hint: str
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
        for conv in _parse_json_payload(self._ctx.provider_hint, payloads, self._ctx.fallback_id):
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

                for conv in _parse_json_payload(provider, payload, self._ctx.fallback_id):
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
        if p == "claude-code" and conv.provider_conversation_id in idx:
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
            ctx = _ParseContext(
                provider_hint=provider_hint,
                should_group=should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                session_index={},  # ZIP files don't have session indices
                detect_path=zip_path,
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
    source: Source, *, cursor_state: dict[str, Any] | None = None, capture_raw: bool = True
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

    Yields:
        Tuples of (RawConversationData | None, ParsedConversation)
    """
    if not source.path:
        return
    base = source.path.expanduser()

    # Phase 1: Resolve paths
    if base.is_dir():
        paths = _walk_source_paths(base)
    elif base.is_file():
        paths = [base]
    else:
        paths = []

    session_indices = _build_session_indices(paths)

    # Initialize cursor state
    if cursor_state is not None:
        cursor_state["file_count"] = len(paths)
        cursor_state.setdefault("failed_files", [])
        cursor_state.setdefault("failed_count", 0)
        if paths:
            try:
                latest = max(paths, key=lambda p: p.stat().st_mtime)
                cursor_state["latest_mtime"] = latest.stat().st_mtime
                cursor_state["latest_path"] = str(latest)
            except OSError:
                pass

    # Phase 2: Process each file
    failed_count = 0
    for path in paths:
        try:
            provider_hint = detect_provider(None, path) or source.name
            should_group = provider_hint in _GROUP_PROVIDERS
            file_mtime = _get_file_mtime(path) if capture_raw else None

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
                    session_index=session_indices.get(path.parent, {}),
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

    # Emit a prominent summary if any files were skipped
    if failed_count > 0:
        logger.warning(
            "Skipped %d of %d files from source %r due to parse/read errors. "
            "Run with --verbose for details.",
            failed_count,
            len(paths),
            source.name,
        )


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "RawConversationData",
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
    "parse_drive_payload",
]
