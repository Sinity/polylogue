"""Conversation emitter — parses a binary stream and yields (raw, conv) tuples."""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from polylogue.lib.json import dumps as json_dumps
from polylogue.logging import get_logger
from polylogue.types import Provider

from .cursor import _ParseContext
from .decoders import _iter_json_stream
from .parsers.base import ParsedConversation, RawConversationData
from .parsers.claude import enrich_conversation_from_index

logger = get_logger(__name__)


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
            return

        if is_jsonl:
            sniff_bytes = pre_read_bytes if pre_read_bytes is not None else handle.read()
            sniff_payloads = list(_iter_json_stream(BytesIO(sniff_bytes), stream_name))
            sniff_provider = detect_provider(sniff_payloads) or self._ctx.provider_hint
            from .source import _GROUP_PROVIDERS
            if sniff_provider in _GROUP_PROVIDERS:
                yield from self._emit_grouped(
                    BytesIO(sniff_bytes),
                    stream_name,
                    sniff_bytes,
                )
                return
            handle = BytesIO(sniff_bytes)
            pre_read_bytes = sniff_bytes

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
        from .source import detect_provider, parse_payload
        provider = detect_provider(payloads) or self._ctx.provider_hint
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
        from .source import detect_provider, parse_payload
        for payload in _iter_json_stream(handle, stream_name, unpack_lists=unpack):
            try:
                provider = detect_provider(payload) or self._ctx.provider_hint

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
        provider_override: Provider | None = None,
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
        provider: Provider | None = None,
    ) -> ParsedConversation:
        """Apply Claude Code session index enrichment if applicable."""
        p = provider or self._ctx.provider_hint
        idx = self._ctx.session_index
        if p is Provider.CLAUDE_CODE and conv.provider_conversation_id in idx:
            return enrich_conversation_from_index(conv, idx[conv.provider_conversation_id])
        return conv


__all__ = [
    "_ConversationEmitter",
]
