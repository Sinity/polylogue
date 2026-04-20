"""Conversation emitter — parses a binary stream and yields (raw, conv) tuples."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from itertools import chain
from typing import TYPE_CHECKING, BinaryIO

from polylogue.lib.artifact_taxonomy import ArtifactClassification, classify_artifact
from polylogue.lib.json import dumps_bytes as json_dumps_bytes
from polylogue.logging import get_logger
from polylogue.types import Provider

from .assembly import get_assembly_spec
from .cursor import _ParseContext
from .decoder_json import JsonValue
from .decoders import _iter_json_stream
from .dispatch import GROUP_PROVIDERS, detect_provider, parse_payload
from .parsers.base import ParsedConversation, RawConversationData

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.runtime_registry import SchemaRegistry as SchemaRegistryType

logger = get_logger(__name__)


def _schema_registry_factory() -> SchemaRegistryType:
    from polylogue.schemas.runtime_registry import SchemaRegistry

    return SchemaRegistry()


@dataclass(frozen=True, slots=True)
class _SniffResult:
    provider: Provider
    payloads: Iterable[JsonValue]
    grouped_payloads: list[JsonValue] | None = None

    @property
    def is_grouped(self) -> bool:
        return self.grouped_payloads is not None


@dataclass(frozen=True, slots=True)
class _PreparedJsonlState:
    sniff_handle: BinaryIO
    pre_read_bytes: bytes | None
    stream_start: int | None


@dataclass(frozen=True, slots=True)
class _ResolvedPayload:
    provider: Provider
    artifact: ArtifactClassification
    schema_resolution: SchemaResolution | None


class _ConversationEmitter:
    """Parse a binary stream and yield ``(raw, conv)`` tuples.

    Unifies the grouped-JSONL, individual-items, and raw-capture logic
    that was previously duplicated across ZIP and filesystem code paths.
    """

    __slots__ = (
        "_ctx",
        "_schema_registry",
    )

    def __init__(self, ctx: _ParseContext) -> None:
        self._ctx = ctx
        self._schema_registry: SchemaRegistryType | None = None

    def emit(
        self,
        handle: BinaryIO,
        stream_name: str,
        *,
        pre_read_bytes: bytes | None = None,
        precomputed_raw: RawConversationData | None = None,
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
            yield from self._emit_grouped(
                handle,
                stream_name,
                pre_read_bytes,
                precomputed_raw=precomputed_raw,
            )
            return

        if is_jsonl:
            yield from self._emit_jsonl(
                handle,
                stream_name,
                pre_read_bytes=pre_read_bytes,
                precomputed_raw=precomputed_raw,
            )
            return

        yield from self._emit_individual(handle, stream_name, pre_read_bytes=pre_read_bytes)

    def _emit_grouped(
        self,
        handle: BinaryIO,
        stream_name: str,
        pre_read_bytes: bytes | None,
        *,
        precomputed_raw: RawConversationData | None = None,
        precomputed_payloads: list[JsonValue] | None = None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        """Grouped JSONL: entire file = one conversation."""
        if precomputed_raw is not None:
            raw_bytes = None
        elif self._ctx.capture_raw and pre_read_bytes is None:
            raw_bytes = handle.read()
            handle = BytesIO(raw_bytes)
        else:
            raw_bytes = pre_read_bytes
            if raw_bytes is not None:
                handle = BytesIO(raw_bytes)

        payloads = (
            precomputed_payloads if precomputed_payloads is not None else list(_iter_json_stream(handle, stream_name))
        )
        if not payloads:
            return

        raw_data = precomputed_raw or (self._make_raw(raw_bytes) if raw_bytes else None)
        resolved = self._resolve_payload(payloads)
        if not resolved.artifact.parse_as_conversation:
            return
        for conv in parse_payload(
            resolved.provider,
            payloads,
            self._ctx.fallback_id,
            schema_resolution=resolved.schema_resolution,
        ):
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

        yield from self._emit_individual_payloads(
            _iter_json_stream(handle, stream_name, unpack_lists=unpack),
            stream_name=stream_name,
            whole_file_raw=whole_file_raw,
        )

    def _emit_individual_payloads(
        self,
        payloads: Iterable[JsonValue],
        *,
        stream_name: str,
        whole_file_raw: RawConversationData | None = None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        source_index = 0
        for payload in payloads:
            try:
                resolved = self._resolve_payload(payload)
                if not resolved.artifact.parse_as_conversation:
                    continue

                if whole_file_raw is not None:
                    raw_data: RawConversationData | None = whole_file_raw
                elif self._ctx.capture_raw:
                    raw_bytes = json_dumps_bytes(payload)
                    raw_data = self._make_raw(
                        raw_bytes,
                        source_index=source_index,
                        provider_override=resolved.provider,
                    )
                else:
                    raw_data = None

                for conv in parse_payload(
                    resolved.provider,
                    payload,
                    self._ctx.fallback_id,
                    schema_resolution=resolved.schema_resolution,
                ):
                    yield (raw_data, self._maybe_enrich(conv, resolved.provider))
                source_index += 1
            except Exception:
                logger.exception("Error processing payload from %s", stream_name)
                raise

    def _sniff_jsonl_payloads(
        self,
        handle: BinaryIO,
        stream_name: str,
    ) -> _SniffResult:
        payload_iter = iter(_iter_json_stream(handle, stream_name))
        buffered_payloads: list[JsonValue] = []
        for payload in payload_iter:
            buffered_payloads.append(payload)
            detected_provider = detect_provider(payload)
            if detected_provider is None:
                continue
            if detected_provider in GROUP_PROVIDERS:
                buffered_payloads.extend(payload_iter)
                return _SniffResult(
                    provider=detected_provider,
                    payloads=buffered_payloads,
                    grouped_payloads=buffered_payloads,
                )
            return _SniffResult(provider=detected_provider, payloads=chain(buffered_payloads, payload_iter))

        detected_provider = detect_provider(buffered_payloads) or self._ctx.provider_hint
        if detected_provider in GROUP_PROVIDERS:
            return _SniffResult(
                provider=detected_provider,
                payloads=buffered_payloads,
                grouped_payloads=buffered_payloads,
            )
        return _SniffResult(provider=detected_provider, payloads=iter(buffered_payloads))

    def _prepare_jsonl_state(
        self,
        handle: BinaryIO,
        *,
        pre_read_bytes: bytes | None,
    ) -> _PreparedJsonlState:
        if pre_read_bytes is not None:
            return _PreparedJsonlState(
                sniff_handle=BytesIO(pre_read_bytes),
                pre_read_bytes=pre_read_bytes,
                stream_start=None,
            )
        return _PreparedJsonlState(
            sniff_handle=handle,
            pre_read_bytes=None,
            stream_start=self._capture_stream_start(handle),
        )

    def _non_seekable_jsonl_sniff(
        self,
        handle: BinaryIO,
        stream_name: str,
    ) -> tuple[bytes, _SniffResult]:
        sniff_bytes = handle.read()
        return sniff_bytes, self._sniff_jsonl_payloads(BytesIO(sniff_bytes), stream_name)

    def _grouped_jsonl_source(
        self,
        handle: BinaryIO,
        state: _PreparedJsonlState,
        *,
        precomputed_raw: RawConversationData | None,
    ) -> tuple[BinaryIO, bytes | None]:
        grouped_bytes = state.pre_read_bytes
        grouped_handle: BinaryIO = state.sniff_handle
        if (
            grouped_bytes is None
            and self._ctx.capture_raw
            and precomputed_raw is None
            and state.stream_start is not None
        ):
            self._restore_stream_start(handle, state.stream_start)
            grouped_bytes = handle.read()
            grouped_handle = BytesIO(grouped_bytes)
        return grouped_handle, grouped_bytes

    def _emit_jsonl(
        self,
        handle: BinaryIO,
        stream_name: str,
        *,
        pre_read_bytes: bytes | None,
        precomputed_raw: RawConversationData | None,
    ) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
        state = self._prepare_jsonl_state(handle, pre_read_bytes=pre_read_bytes)
        if (
            state.pre_read_bytes is None
            and state.stream_start is None
            and self._ctx.capture_raw
            and precomputed_raw is None
        ):
            sniff_bytes, sniffed = self._non_seekable_jsonl_sniff(handle, stream_name)
            if sniffed.is_grouped:
                yield from self._emit_grouped(
                    BytesIO(sniff_bytes),
                    stream_name,
                    sniff_bytes,
                    precomputed_raw=precomputed_raw,
                    precomputed_payloads=sniffed.grouped_payloads,
                )
                return
            yield from self._emit_individual_payloads(
                sniffed.payloads,
                stream_name=stream_name,
            )
            return

        sniffed = self._sniff_jsonl_payloads(state.sniff_handle, stream_name)
        if sniffed.is_grouped:
            grouped_handle, grouped_bytes = self._grouped_jsonl_source(
                handle,
                state,
                precomputed_raw=precomputed_raw,
            )
            yield from self._emit_grouped(
                grouped_handle,
                stream_name,
                grouped_bytes,
                precomputed_raw=precomputed_raw,
                precomputed_payloads=sniffed.grouped_payloads,
            )
            return
        yield from self._emit_individual_payloads(
            sniffed.payloads,
            stream_name=stream_name,
        )

    def _capture_stream_start(self, handle: BinaryIO) -> int | None:
        seekable = getattr(handle, "seekable", None)
        if callable(seekable):
            try:
                if not seekable():
                    return None
            except OSError:
                return None
        try:
            return int(handle.tell())
        except (AttributeError, OSError, ValueError):
            return None

    def _restore_stream_start(self, handle: BinaryIO, stream_start: int) -> None:
        handle.seek(stream_start)

    def _resolve_schema(
        self,
        provider: Provider,
        payload: JsonValue,
    ) -> SchemaResolution | None:
        """Resolve schema metadata for the payload, if schemas are available."""
        if self._schema_registry is None:
            self._schema_registry = _schema_registry_factory()
        try:
            return self._schema_registry.resolve_payload(
                provider,
                payload,
                source_path=self._ctx.source_path_str,
            )
        except Exception as exc:
            logger.debug(
                "Schema resolution failed for %s in %s: %s",
                provider,
                self._ctx.source_path_str,
                exc,
            )
            return None

    def _resolve_payload(self, payload: JsonValue) -> _ResolvedPayload:
        provider = detect_provider(payload) or self._ctx.provider_hint
        artifact = classify_artifact(
            payload,
            provider=provider,
            source_path=self._ctx.source_path_str,
        )
        return _ResolvedPayload(
            provider=provider,
            artifact=artifact,
            schema_resolution=self._resolve_schema(provider, payload),
        )

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
        """Apply provider-specific enrichment via assembly layer."""
        p = provider or self._ctx.provider_hint
        spec = get_assembly_spec(p)
        if spec is not None:
            return spec.enrich_conversation(conv, self._ctx.sidecar_data)
        return conv


__all__ = [
    "_ConversationEmitter",
]
