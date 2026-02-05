from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

import ijson
from pydantic import BaseModel

from polylogue.config import Source
from polylogue.lib.json import dumps as json_dumps

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

LOGGER = logging.getLogger(__name__)


class IngestBundle(BaseModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


class IngestResult(BaseModel):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


def ingest_bundle(bundle: IngestBundle, repository: ConversationRepository) -> IngestResult:
    """Ingest a bundle of records into the repository.

    Args:
        bundle: Bundle containing conversation, messages, and attachments
        repository: Storage repository to safe records to

    Returns:
        IngestResult with counts of imported/skipped items
    """
    counts = repository.save_conversation(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
    )
    return IngestResult(**counts)


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
MAX_COMPRESSION_RATIO = 100  # Reject if uncompressed/compressed > 100x
MAX_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB limit per file


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
        LOGGER.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def detect_provider(payload: Any, path: Path) -> str | None:
    if isinstance(payload, dict):
        if chatgpt.looks_like(payload):
            return "chatgpt"
        if claude.looks_like_ai(payload):
            return "claude"
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
    if "claude-code" in name or "claude_code" in name or "claude-code" in path_str:
        return "claude-code"
    if "claude" in name or "/claude/" in path_str:
        return "claude"
    if "codex" in name or "codex" in path_str:
        return "codex"
    if "gemini" in name or "gemini" in path_str:
        return "gemini"
    return None


def _parse_json_payload(provider: str, payload: Any, fallback_id: str) -> list[ParsedConversation]:
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
                results.extend(_parse_json_payload(provider, item, f"{fallback_id}-{i}"))
            return results
        # Treat list as chunks for a single conversation
        return [drive.parse_chunked_prompt(provider, {"chunks": payload}, fallback_id)]

    # Fallback / Generic
    if isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            results = []
            for i, item in enumerate(payload["conversations"]):
                parsed = _parse_json_payload(provider, item, f"{fallback_id}-{i}")
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


def parse_drive_payload(provider: str, payload: Any, fallback_id: str) -> list[ParsedConversation]:
    if isinstance(payload, list):
        # Check if it looks like a list of conversations or a list of messages
        # For drive/gemini, if it's a list, it's often a list of chunks
        if payload and isinstance(payload[0], dict) and ("role" in payload[0] or "text" in payload[0]):
            return [drive.parse_chunked_prompt(provider, {"chunks": payload}, fallback_id)]

        results = []
        for i, item in enumerate(payload):
            results.extend(parse_drive_payload(provider, item, f"{fallback_id}-{i}"))
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
                    LOGGER.warning("Skipping undecodable line from %s", path_name)
                    continue
            else:
                decoded = raw
            try:
                yield json.loads(decoded)
            except json.JSONDecodeError as exc:
                # Log first few errors, then summarize
                error_count += 1
                if error_count <= 3:
                    LOGGER.warning("Skipping invalid JSON line in %s: %s", path_name, exc)
                elif error_count == 4:
                    LOGGER.warning("Skipping further invalid JSON lines in %s...", path_name)
                continue
        if error_count > 3:
            LOGGER.warning("Skipped %d invalid JSON lines in %s", error_count, path_name)
        return

    if unpack_lists:
        # Strategy 1: Try streaming root list
        try:
            # LOGGER.info("Strategy 1: ijson items(item) for %s", path_name)
            found_any = False
            for item in ijson.items(handle, "item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            pass  # Expected, try next strategy
        except Exception as exc:
            LOGGER.debug("Strategy 1 (ijson items) failed for %s: %s", path_name, exc)

        handle.seek(0)
        # Strategy 2: Try streaming conversations list
        try:
            # LOGGER.info("Strategy 2: ijson items(conversations.item) for %s", path_name)
            found_any = False
            for item in ijson.items(handle, "conversations.item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            pass  # Expected, try next strategy
        except Exception as exc:
            LOGGER.debug("Strategy 2 (ijson conversations.item) failed for %s: %s", path_name, exc)

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


_INGEST_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson", ".zip"})
_INGEST_DOUBLE_EXTENSIONS = frozenset({".jsonl.txt"})


def _has_ingest_extension(path: Path) -> bool:
    """Check if path has a supported ingest extension (case-insensitive)."""
    name_lower = path.name.lower()
    # Check double extensions first (e.g., .jsonl.txt)
    for ext in _INGEST_DOUBLE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True
    # Check single extensions
    return path.suffix.lower() in _INGEST_EXTENSIONS


def iter_source_conversations(
    source: Source, *, cursor_state: dict[str, Any] | None = None
) -> Iterable[ParsedConversation]:
    if not source.path:
        return
    base = source.path.expanduser()
    paths: list[Path] = []
    if base.is_dir():
        # Iterate all files and filter by extension (case-insensitive)
        # Use os.walk with followlinks=True to traverse symlinked directories
        import os

        paths = []
        for root, _, files in os.walk(base, followlinks=True):
            for filename in files:
                file_path = Path(root) / filename
                if _has_ingest_extension(file_path):
                    paths.append(file_path)
        paths = sorted(paths)
    elif base.is_file():
        paths.append(base)

    # Build session index lookup for Claude Code enrichment
    # Group paths by directory and load sessions-index.json for each
    session_indices: dict[Path, dict[str, SessionIndexEntry]] = {}
    for path in paths:
        parent = path.parent
        if parent not in session_indices:
            index_path = parent / "sessions-index.json"
            session_indices[parent] = parse_sessions_index(index_path)

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

    for path in paths:
        try:
            # Detect provider from path name first to decide iteration strategy
            provider_hint = detect_provider(None, path) or source.name
            # Providers where one file (even JSONL) is one conversation
            group_providers = {"claude-code", "codex", "gemini", "drive"}
            should_group = provider_hint in group_providers

            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    for info in zf.infolist():
                        # Skip directories
                        if info.is_dir():
                            continue

                        name = info.filename
                        lower_name = name.lower()

                        # ZIP bomb protection
                        if info.compress_size > 0:
                            ratio = info.file_size / info.compress_size
                            if ratio > MAX_COMPRESSION_RATIO:
                                LOGGER.warning(
                                    "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                                    name,
                                    path,
                                    ratio,
                                )
                                if cursor_state is not None:
                                    cursor_state["failed_files"].append(
                                        {
                                            "path": f"{path}:{name}",
                                            "error": f"Suspicious compression ratio: {ratio:.1f}",
                                        }
                                    )
                                    cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
                                continue

                        if info.file_size > MAX_UNCOMPRESSED_SIZE:
                            LOGGER.warning(
                                "Skipping oversized file %s in %s: %d bytes exceeds limit", name, path, info.file_size
                            )
                            if cursor_state is not None:
                                cursor_state["failed_files"].append(
                                    {"path": f"{path}:{name}", "error": f"File size {info.file_size} exceeds limit"}
                                )
                                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
                            continue

                        if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                            with zf.open(name) as handle:
                                if lower_name.endswith((".jsonl", ".jsonl.txt", ".ndjson")) and should_group:
                                    # Group all lines into one conversation
                                    payloads = list(_iter_json_stream(handle, name))
                                    if payloads:
                                        yield from _parse_json_payload(provider_hint, payloads, path.stem)
                                else:
                                    # For .json files, we might need to disable unpacking if it's a grouped provider
                                    unpack = not (lower_name.endswith(".json") and should_group)
                                    for payload in _iter_json_stream(handle, name, unpack_lists=unpack):
                                        try:
                                            provider = detect_provider(payload, path) or provider_hint
                                            yield from _parse_json_payload(provider, payload, path.stem)
                                        except Exception:
                                            LOGGER.exception("Error processing payload from %s", name)
                                            raise
            else:
                # Get session index for this directory (for claude-code enrichment)
                dir_index = session_indices.get(path.parent, {})

                with path.open("rb") as handle:
                    is_jsonl = path.suffix.lower() in (".jsonl", ".ndjson") or path.name.lower().endswith(".jsonl.txt")
                    if is_jsonl and should_group:
                        # Group all lines into one conversation
                        payloads = list(_iter_json_stream(handle, path.name))
                        if payloads:
                            for conv in _parse_json_payload(provider_hint, payloads, path.stem):
                                # Enrich claude-code with session index metadata
                                if provider_hint == "claude-code" and conv.provider_conversation_id in dir_index:
                                    conv = enrich_conversation_from_index(
                                        conv, dir_index[conv.provider_conversation_id]
                                    )
                                yield conv
                            continue

                    unpack = not (path.suffix.lower() == ".json" and should_group)
                    for payload in _iter_json_stream(handle, path.name, unpack_lists=unpack):
                        try:
                            provider = detect_provider(payload, path) or provider_hint
                            for conv in _parse_json_payload(provider, payload, path.stem):
                                # Enrich claude-code with session index metadata
                                if provider == "claude-code" and conv.provider_conversation_id in dir_index:
                                    conv = enrich_conversation_from_index(
                                        conv, dir_index[conv.provider_conversation_id]
                                    )
                                yield conv
                        except Exception:
                            LOGGER.exception("Error processing payload from %s", path)
                            raise
        except FileNotFoundError as exc:
            # TOCTOU race condition: file existed during directory scan but was deleted before read
            LOGGER.warning("File disappeared during processing (TOCTOU race): %s", path)
            if cursor_state is not None:
                cursor_state["failed_files"].append(
                    {"path": str(path), "error": f"File not found (may have been deleted): {exc}"}
                )
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
            if cursor_state is not None:
                cursor_state["failed_files"].append({"path": str(path), "error": str(exc)})
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue
        except Exception as exc:
            LOGGER.error("Unexpected error processing %s: %s", path, exc)
            if cursor_state is not None:
                cursor_state["failed_files"].append({"path": str(path), "error": str(exc)})
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue


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
    paths: list[Path] = []
    if base.is_dir():
        import os

        paths = []
        for root, _, files in os.walk(base, followlinks=True):
            for filename in files:
                file_path = Path(root) / filename
                if _has_ingest_extension(file_path):
                    paths.append(file_path)
        paths = sorted(paths)
    elif base.is_file():
        paths.append(base)

    # Build session index lookup for Claude Code enrichment
    session_indices: dict[Path, dict[str, SessionIndexEntry]] = {}
    for path in paths:
        parent = path.parent
        if parent not in session_indices:
            index_path = parent / "sessions-index.json"
            session_indices[parent] = parse_sessions_index(index_path)

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

    for path in paths:
        try:
            # Detect provider from path name first
            provider_hint = detect_provider(None, path) or source.name
            group_providers = {"claude-code", "codex", "gemini", "drive"}
            should_group = provider_hint in group_providers

            # Get file mtime for raw capture
            file_mtime: str | None = None
            if capture_raw:
                try:
                    stat = path.stat()
                    from datetime import datetime, timezone

                    file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                except OSError:
                    pass

            if path.suffix.lower() == ".zip":
                # Get ZIP mtime for raw capture
                zip_mtime: str | None = None
                if capture_raw:
                    try:
                        stat = path.stat()
                        from datetime import datetime, timezone

                        zip_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    except OSError:
                        pass

                with zipfile.ZipFile(path) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        name = info.filename
                        lower_name = name.lower()

                        # Filter Claude AI ZIP: only process conversations.json
                        if provider_hint in ("claude", "claude-ai"):
                            basename = lower_name.split("/")[-1]
                            if basename not in ("conversations.json",):
                                continue

                        # ZIP bomb protection
                        if info.compress_size > 0:
                            ratio = info.file_size / info.compress_size
                            if ratio > MAX_COMPRESSION_RATIO:
                                LOGGER.warning(
                                    "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                                    name,
                                    path,
                                    ratio,
                                )
                                if cursor_state is not None:
                                    cursor_state["failed_files"].append(
                                        {
                                            "path": f"{path}:{name}",
                                            "error": f"Suspicious compression ratio: {ratio:.1f}",
                                        }
                                    )
                                    cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
                                continue

                        if info.file_size > MAX_UNCOMPRESSED_SIZE:
                            LOGGER.warning(
                                "Skipping oversized file %s in %s: %d bytes exceeds limit", name, path, info.file_size
                            )
                            if cursor_state is not None:
                                cursor_state["failed_files"].append(
                                    {"path": f"{path}:{name}", "error": f"File size {info.file_size} exceeds limit"}
                                )
                                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
                            continue

                        if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                            with zf.open(name) as handle:
                                if lower_name.endswith((".jsonl", ".jsonl.txt", ".ndjson")) and should_group:
                                    # Grouped JSONL - read entire file for raw capture
                                    raw_bytes = handle.read() if capture_raw else None
                                    from io import BytesIO

                                    stream = BytesIO(raw_bytes) if raw_bytes else handle
                                    payloads = list(_iter_json_stream(stream, name))
                                    if payloads:
                                        raw_data = (
                                            RawConversationData(
                                                raw_bytes=raw_bytes,
                                                source_path=f"{path}:{name}",
                                                source_index=None,
                                                file_mtime=zip_mtime,
                                                provider_hint=provider_hint,
                                            )
                                            if capture_raw and raw_bytes
                                            else None
                                        )
                                        for conv in _parse_json_payload(provider_hint, payloads, path.stem):
                                            yield (raw_data, conv)
                                else:
                                    # Individual items - capture each separately
                                    unpack = not (lower_name.endswith(".json") and should_group)
                                    source_index = 0
                                    for payload in _iter_json_stream(handle, name, unpack_lists=unpack):
                                        try:
                                            provider = detect_provider(payload, path) or provider_hint

                                            # Capture raw for each individual conversation
                                            raw_data_item: RawConversationData | None = None
                                            if capture_raw:
                                                raw_bytes_item = json_dumps(payload).encode("utf-8")
                                                raw_data_item = RawConversationData(
                                                    raw_bytes=raw_bytes_item,
                                                    source_path=f"{path}:{name}",
                                                    source_index=source_index,
                                                    file_mtime=zip_mtime,
                                                    provider_hint=provider,
                                                )

                                            for conv in _parse_json_payload(provider, payload, path.stem):
                                                yield (raw_data_item, conv)
                                            source_index += 1
                                        except Exception:
                                            LOGGER.exception("Error processing payload from %s", name)
                                            raise
            else:
                dir_index = session_indices.get(path.parent, {})

                # For raw capture: read entire file at once for grouped providers
                if capture_raw and should_group:
                    raw_bytes = path.read_bytes()
                    raw_data = RawConversationData(
                        raw_bytes=raw_bytes,
                        source_path=str(path),
                        source_index=None,  # Single conversation per file
                        file_mtime=file_mtime,
                        provider_hint=provider_hint,
                    )

                    # Parse and yield with raw
                    from io import BytesIO

                    handle = BytesIO(raw_bytes)
                    if path.suffix.lower() in (".jsonl", ".ndjson") or path.name.lower().endswith(".jsonl.txt"):
                        payloads = list(_iter_json_stream(handle, path.name))
                        if payloads:
                            for conv in _parse_json_payload(provider_hint, payloads, path.stem):
                                if provider_hint == "claude-code" and conv.provider_conversation_id in dir_index:
                                    conv = enrich_conversation_from_index(
                                        conv, dir_index[conv.provider_conversation_id]
                                    )
                                yield (raw_data, conv)
                    else:
                        unpack = not (path.suffix.lower() == ".json" and should_group)
                        for payload in _iter_json_stream(handle, path.name, unpack_lists=unpack):
                            try:
                                provider = detect_provider(payload, path) or provider_hint
                                for conv in _parse_json_payload(provider, payload, path.stem):
                                    if provider == "claude-code" and conv.provider_conversation_id in dir_index:
                                        conv = enrich_conversation_from_index(
                                            conv, dir_index[conv.provider_conversation_id]
                                        )
                                    yield (raw_data, conv)
                            except Exception:
                                LOGGER.exception("Error processing payload from %s", path)
                                raise
                else:
                    # Non-grouped providers or no raw capture
                    with path.open("rb") as handle:
                        is_jsonl = path.suffix.lower() in (".jsonl", ".ndjson") or path.name.lower().endswith(".jsonl.txt")
                        if is_jsonl and should_group:
                            payloads = list(_iter_json_stream(handle, path.name))
                            if payloads:
                                for conv in _parse_json_payload(provider_hint, payloads, path.stem):
                                    if (
                                        provider_hint == "claude-code"
                                        and conv.provider_conversation_id in dir_index
                                    ):
                                        conv = enrich_conversation_from_index(
                                            conv, dir_index[conv.provider_conversation_id]
                                        )
                                    yield (None, conv)
                            continue

                        unpack = not (path.suffix.lower() == ".json" and should_group)
                        source_index = 0
                        for payload in _iter_json_stream(handle, path.name, unpack_lists=unpack):
                            try:
                                provider = detect_provider(payload, path) or provider_hint

                                # Capture raw for individual items in bundles
                                raw_data_item: RawConversationData | None = None
                                if capture_raw:
                                    # Re-serialize individual payload to JSON bytes
                                    # Uses json_dumps which handles Decimal from ijson
                                    raw_bytes_item = json_dumps(payload).encode("utf-8")
                                    raw_data_item = RawConversationData(
                                        raw_bytes=raw_bytes_item,
                                        source_path=str(path),
                                        source_index=source_index,
                                        file_mtime=file_mtime,
                                        provider_hint=provider,
                                    )

                                for conv in _parse_json_payload(provider, payload, path.stem):
                                    if provider == "claude-code" and conv.provider_conversation_id in dir_index:
                                        conv = enrich_conversation_from_index(
                                            conv, dir_index[conv.provider_conversation_id]
                                        )
                                    yield (raw_data_item, conv)
                                source_index += 1
                            except Exception:
                                LOGGER.exception("Error processing payload from %s", path)
                                raise
        except FileNotFoundError as exc:
            LOGGER.warning("File disappeared during processing (TOCTOU race): %s", path)
            if cursor_state is not None:
                cursor_state["failed_files"].append(
                    {"path": str(path), "error": f"File not found (may have been deleted): {exc}"}
                )
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
            if cursor_state is not None:
                cursor_state["failed_files"].append({"path": str(path), "error": str(exc)})
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue
        except Exception as exc:
            LOGGER.error("Unexpected error processing %s: %s", path, exc)
            if cursor_state is not None:
                cursor_state["failed_files"].append({"path": str(path), "error": str(exc)})
                cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1
            continue


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "RawConversationData",
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
    "parse_drive_payload",
]
