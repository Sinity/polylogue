"""Session emitter — parses a binary stream and yields (raw, conv) tuples."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, cast

from polylogue.archive.artifact_taxonomy import ArtifactClassification, classify_artifact
from polylogue.core.content_identity import ContentIdentityRefusal, stream_payload_content_identity
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.logging import get_logger

from .acquisition_boundary import (
    bind_stream,
    bound_profile_identity,
    bound_source_observation,
    drain_bound,
)
from .assembly import get_assembly_spec
from .cursor import _ParseContext
from .decoder_json import DecodedRecordSequence, JsonValue
from .decoders import owned_json_records
from .dispatch import GROUP_PROVIDERS, detect_provider, is_jsonl_source_path, parse_payload
from .parsers.base import ParsedSession, RawSessionData
from .staged_raw_payload import StagedRawPayload

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
    grouped_payloads: DecodedRecordSequence | None = None

    @property
    def is_grouped(self) -> bool:
        return self.grouped_payloads is not None


@dataclass(frozen=True, slots=True)
class _ResolvedPayload:
    provider: Provider
    artifact: ArtifactClassification
    schema_resolution: SchemaResolution | None


class _SessionEmitter:
    """Parse a binary stream and yield ``(raw, conv)`` tuples.

    Unifies the grouped-JSONL, individual-items, and raw-capture logic
    that was previously duplicated across ZIP and filesystem code paths.
    """

    __slots__ = (
        "_ctx",
        "_schema_registry",
        "_profile_identity",
    )

    def __init__(self, ctx: _ParseContext) -> None:
        self._ctx = ctx
        self._schema_registry: SchemaRegistryType | None = None
        self._profile_identity: str | None = None

    def emit(
        self,
        handle: IO[bytes],
        stream_name: str,
        *,
        precomputed_raw: RawSessionData | None = None,
        session_artifact: ArtifactClassification | None = None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        """Parse under the source owner; private captures survive creator pickup."""
        handle = bind_stream(handle, stream_name, self._ctx.bound_provider)
        canonical, observed = bound_source_observation(handle)
        profile = bound_profile_identity(handle)
        self._profile_identity = (
            profile.key
            if profile is not None
            else precomputed_raw.captured_profile_key
            if precomputed_raw is not None
            else None
        )

        with ExitStack() as input_lifetime:
            input_stage = None
            if self._ctx.capture_raw and precomputed_raw is None:
                input_stage = StagedRawPayload.from_stream(handle, directory=self._ctx.raw_directory)
                handle = input_lifetime.enter_context(input_stage.path.open("rb"))

            def captured_records() -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
                for raw, session in self._emit_stream(
                    handle, stream_name, input_stage, precomputed_raw, session_artifact
                ):
                    if raw is not None:
                        if raw.canonical_source_path is None:
                            raw.canonical_source_path = canonical
                        if raw.captured_file_observation is None:
                            raw.captured_file_observation = observed
                        if profile is not None and session.source_name is Provider.HERMES:
                            raw.captured_profile_key = profile.key
                            raw.captured_profile_source_path = str(profile.source_path)
                        if raw.staged_payload is not None:
                            raw.staged_payload.retained = True
                    yield raw, session

            emitted = captured_records()
            collected: list[tuple[RawSessionData | None, ParsedSession]] = []
            delivered = 0

            def handoff() -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
                nonlocal delivered
                for item in collected:
                    delivered += 1
                    yield item

            try:
                if self._ctx.bound_provider is None:
                    yield from emitted
                    return
                # A bound source is one admission unit. Its captured bytes
                # have settled before a parser result can leave preparation.
                try:
                    collected.extend(emitted)
                except ContentIdentityRefusal:
                    drain_bound(handle)
                    yield from handoff()
                    raise
                except BaseException:
                    for raw, _session in collected:
                        if raw is not None and raw.staged_payload is not None:
                            raw.staged_payload.discard()
                    raise
                yield from handoff()
            finally:
                for raw, _session in collected[delivered:]:
                    if raw is not None and raw.staged_payload is not None:
                        raw.staged_payload.discard()
                if input_stage is not None and not input_stage.retained:
                    input_stage.discard()

    def _emit_stream(
        self,
        handle: IO[bytes],
        stream_name: str,
        input_stage: StagedRawPayload | None,
        precomputed_raw: RawSessionData | None,
        session_artifact: ArtifactClassification | None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        is_jsonl = is_jsonl_source_path(stream_name)

        if is_jsonl and self._ctx.should_group:
            yield from self._emit_grouped(
                handle,
                stream_name,
                input_stage,
                precomputed_raw=precomputed_raw,
                session_artifact=session_artifact,
            )
            return

        if is_jsonl:
            yield from self._emit_jsonl(
                handle,
                stream_name,
                input_stage=input_stage,
                precomputed_raw=precomputed_raw,
            )
            return

        yield from self._emit_individual(
            handle,
            stream_name,
            whole_file_raw=(precomputed_raw or self._make_raw(input_stage) if self._ctx.should_group else None),
            session_artifact=session_artifact,
        )

    def _emit_grouped(
        self,
        handle: IO[bytes],
        stream_name: str,
        input_stage: StagedRawPayload | None,
        *,
        precomputed_raw: RawSessionData | None = None,
        precomputed_payloads: DecodedRecordSequence | None = None,
        session_artifact: ArtifactClassification | None = None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        """Grouped JSONL: entire file = one session."""
        with ExitStack() as lifetime:
            records = precomputed_payloads
            if records is None:
                records = cast(DecodedRecordSequence, lifetime.enter_context(owned_json_records(handle, stream_name)))
            if not records:
                return
            payloads = cast(JsonValue, records)
            raw_data = precomputed_raw or self._make_raw(input_stage)
            resolved = self._resolve_payload(payloads)
            if session_artifact is not None:
                resolved = _ResolvedPayload(
                    provider=resolved.provider,
                    artifact=session_artifact,
                    schema_resolution=resolved.schema_resolution,
                )
            if not resolved.artifact.parse_as_session:
                return
            for conv in parse_payload(
                resolved.provider,
                payloads,
                self._ctx.fallback_id,
                schema_resolution=resolved.schema_resolution,
                source_path=self._ctx.source_path_str,
                profile_identity=self._profile_identity,
            ):
                yield (raw_data, self._maybe_enrich(conv))

    def _emit_individual(
        self,
        handle: IO[bytes],
        stream_name: str,
        *,
        whole_file_raw: RawSessionData | None = None,
        session_artifact: ArtifactClassification | None = None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        """Individual items: each payload = one session."""
        unpack = not (stream_name.lower().endswith(".json") and self._ctx.should_group)

        with owned_json_records(handle, stream_name, unpack_lists=unpack) as payloads:
            yield from self._emit_individual_payloads(
                payloads,
                stream_name=stream_name,
                whole_file_raw=whole_file_raw,
                session_artifact=session_artifact,
            )

    def _emit_individual_payloads(
        self,
        payloads: Iterable[JsonValue],
        *,
        stream_name: str,
        whole_file_raw: RawSessionData | None = None,
        session_artifact: ArtifactClassification | None = None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        source_index = 0
        refusals: list[ContentIdentityRefusal] = []
        for payload in payloads:
            try:
                resolved = self._resolve_payload(payload)
                if session_artifact is not None:
                    resolved = _ResolvedPayload(
                        provider=resolved.provider,
                        artifact=session_artifact,
                        schema_resolution=resolved.schema_resolution,
                    )
                if not resolved.artifact.parse_as_session:
                    continue

                if whole_file_raw is not None:
                    raw_data: RawSessionData | None = whole_file_raw
                elif self._ctx.capture_raw:
                    staged = StagedRawPayload.from_value(payload, directory=self._ctx.raw_directory)
                    try:
                        raw_data = self._make_raw(
                            staged,
                            source_index=source_index,
                            provider_override=resolved.provider,
                        )
                    except ContentIdentityRefusal as refusal:
                        staged.discard()
                        # This element is the member's recorded gap; the
                        # elements after it are still parsed and captured.
                        refusals.append(refusal)
                        source_index += 1
                        continue
                    except BaseException:
                        staged.discard()
                        raise
                else:
                    raw_data = None

                try:
                    for conv in parse_payload(
                        resolved.provider,
                        payload,
                        self._ctx.fallback_id,
                        schema_resolution=resolved.schema_resolution,
                        source_path=self._ctx.source_path_str,
                        profile_identity=self._profile_identity,
                    ):
                        yield (raw_data, self._maybe_enrich(conv, resolved.provider))
                finally:
                    if (
                        raw_data is not None
                        and raw_data.staged_payload is not None
                        and not raw_data.staged_payload.retained
                    ):
                        raw_data.staged_payload.discard()
                source_index += 1
            except Exception:
                logger.exception("Error processing payload from %s", stream_name)
                raise
        if refusals:
            raise refusals[0]

    def _sniff_jsonl_payloads(self, records: DecodedRecordSequence) -> _SniffResult:
        for payload in records:
            detected = detect_provider(payload)
            if detected is not None:
                return _SniffResult(detected, records, records if detected in GROUP_PROVIDERS else None)
        detected = detect_provider(records) or self._ctx.provider_hint
        return _SniffResult(detected, records, records if detected in GROUP_PROVIDERS else None)

    def _emit_jsonl(
        self,
        handle: IO[bytes],
        stream_name: str,
        *,
        input_stage: StagedRawPayload | None,
        precomputed_raw: RawSessionData | None,
    ) -> Iterable[tuple[RawSessionData | None, ParsedSession]]:
        with owned_json_records(handle, stream_name) as records:
            sniffed = self._sniff_jsonl_payloads(cast(DecodedRecordSequence, records))
            if sniffed.is_grouped:
                yield from self._emit_grouped(
                    handle,
                    stream_name,
                    input_stage,
                    precomputed_raw=precomputed_raw,
                    precomputed_payloads=sniffed.grouped_payloads,
                )
                return
            yield from self._emit_individual_payloads(sniffed.payloads, stream_name=stream_name)

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
        artifact = classify_artifact(payload, provider=provider)
        if not artifact.parse_as_session:
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
        staged: StagedRawPayload | None,
        *,
        source_index: int | None = None,
        provider_override: Provider | None = None,
    ) -> RawSessionData | None:
        """Borrow the sealed capture until creator-owned publication pickup."""
        if staged is None or not self._ctx.capture_raw:
            return None
        content_identity = None
        if ":" in self._ctx.source_path_str:
            with staged.path.open("rb") as source:
                content_identity = stream_payload_content_identity(source)
        return RawSessionData(
            staged_payload=staged,
            source_path=self._ctx.source_path_str,
            source_index=source_index,
            addressing_mode=(
                MemberAddressingMode.ELEMENT_OF_CONTAINER
                if source_index is not None and ":" in self._ctx.source_path_str
                else None
            ),
            content_identity=content_identity,
            file_mtime=self._ctx.file_mtime,
            provider_hint=provider_override or self._ctx.provider_hint,
            sidecar_snapshot=(
                dict(self._ctx.sidecar_data)
                if (provider_override or self._ctx.provider_hint) is Provider.CODEX
                else None
            ),
        )

    def _maybe_enrich(
        self,
        conv: ParsedSession,
        provider: Provider | None = None,
    ) -> ParsedSession:
        """Apply provider-specific enrichment via assembly layer."""
        p = provider or self._ctx.provider_hint
        spec = get_assembly_spec(p)
        if spec is not None:
            conv = spec.enrich_session(conv, self._ctx.sidecar_data)
        return conv


__all__ = [
    "_SessionEmitter",
]
