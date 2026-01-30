from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, BinaryIO, TYPE_CHECKING

import ijson

from pydantic import BaseModel

from ..config import Source
from ..importers import chatgpt, claude, codex, drive
from ..importers.base import ParsedAttachment, ParsedConversation, ParsedMessage, extract_messages_from_list
from ..storage.store import AttachmentRecord, ConversationRecord, MessageRecord

if TYPE_CHECKING:
    from ..storage.repository import StorageRepository

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


def ingest_bundle(bundle: IngestBundle, repository: StorageRepository) -> IngestResult:
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

    name = path.name.lower()
    if "chatgpt" in name:
        return "chatgpt"
    if "claude-code" in name or "claude_code" in name:
        return "claude-code"
    if "claude" in name:
        return "claude"
    if "codex" in name:
        return "codex"
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
        if isinstance(payload, dict):
            # If it's a single dict that looks like a codex item, wrap it
            if "prompt" in payload or "completion" in payload:
                return [codex.parse([payload], fallback_id)]
    if provider == "gemini" or provider == "drive":
        if isinstance(payload, list):
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
                with path.open("rb") as handle:
                    if path.suffix.lower() in (".jsonl", ".ndjson") or path.name.lower().endswith(".jsonl.txt"):
                        if should_group:
                            # Group all lines into one conversation
                            payloads = list(_iter_json_stream(handle, path.name))
                            if payloads:
                                yield from _parse_json_payload(provider_hint, payloads, path.stem)
                            continue

                    unpack = not (path.suffix.lower() == ".json" and should_group)
                    for payload in _iter_json_stream(handle, path.name, unpack_lists=unpack):
                        try:
                            provider = detect_provider(payload, path) or provider_hint
                            yield from _parse_json_payload(provider, payload, path.stem)
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


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "iter_source_conversations",
    "parse_drive_payload",
]
