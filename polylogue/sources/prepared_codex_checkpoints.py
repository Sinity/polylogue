"""Exact, source-bound artifacts for simple growing Codex JSONL revisions.

This module is deliberately conservative.  It only derives interior artifacts
when the complete byte stream is one session header followed by plain text
response messages.  Any richer valid Codex grammar belongs on the ordinary
retained preparation route.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Protocol, overload

from polylogue.archive.artifact_taxonomy import ArtifactStreamClassification
from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.compute import DaemonBackpressureError, DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.timestamp_authority import normalize_session_timestamps
from polylogue.core.timestamps import parse_timestamp_pair
from polylogue.pipeline.ids import session_content_hash
from polylogue.schemas.retained_validation import PrefixValidationState, RetainedValidationVerdict
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication
from polylogue.sources.parsers import codex
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import AdmissionUnit, ParseAccounting
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.storage.blob_publication import ArchiveBlobPublisher, BlobPublicationSourceRead


class _CodexCheckpointRead(Protocol):
    """Read only the pinned source evidence needed by prefix preparation."""

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]: ...

    def raw_profile_identity(self, raw_id: str) -> str | None: ...

    def raw_revision_file_mtime(self, raw_id: str) -> str | None: ...

    def open_raw_revision_material(
        self, raw_id: str
    ) -> AbstractContextManager[tuple[Provider, BinaryIO, str, RawRevisionKind]]: ...


class CodexCheckpointDisposition(StrEnum):
    READY = "ready"
    ORDINARY_FALLBACK = "ordinary_fallback"


@dataclass(frozen=True, slots=True)
class CodexCheckpointArtifactOptions:
    source_path: str
    fallback_timestamp: str | None
    classification: ArtifactStreamClassification | None = None
    enrichment_digest: str | None = None
    enrichment_index_path: str | None = None
    parsed_prefix_size: int | None = None
    captured_profile_key: str | None = None
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None
    # When supplied, this must be a uniquely owned workspace for this artifact.
    artifact_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class CodexPrefixPreparation:
    disposition: CodexCheckpointDisposition
    raw_ids: Sequence[str]
    reason: str | None
    _artifacts: Iterator[tuple[str, PreparedJsonl]] | None = None
    _head_blob: BinaryIO | None = None
    _consumed: bool = False

    def __enter__(self) -> CodexPrefixPreparation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        object.__setattr__(self, "_consumed", True)
        artifacts = self._artifacts
        if artifacts is not None:
            close = getattr(artifacts, "close", None)
            if close is not None:
                close()
        head_blob = self._head_blob
        object.__setattr__(self, "_head_blob", None)
        if head_blob is not None:
            head_blob.close()

    def iter_artifacts(self) -> Iterator[tuple[str, PreparedJsonl]]:
        if self.disposition is not CodexCheckpointDisposition.READY or self._artifacts is None:
            if self.disposition is CodexCheckpointDisposition.ORDINARY_FALLBACK:
                return iter(())
            raise RuntimeError("checkpoint artifacts have already been consumed")
        if self._consumed:
            raise RuntimeError("checkpoint artifacts have already been consumed")
        object.__setattr__(self, "_consumed", True)
        return self._artifacts


class _PrefixMessages(Sequence[ParsedMessage]):
    """Read-only view over one independently parsed head message prefix."""

    def __init__(self, messages: Sequence[ParsedMessage], count: int) -> None:
        self._messages = messages
        self._count = count

    @property
    def path(self) -> Path | None:
        return getattr(self._messages, "path", None)

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | Sequence[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        message = self._messages[ordinal]
        return message.model_copy(update={"is_active_leaf": ordinal == self._count - 1})


def _plain_text_header(record: object) -> str | None:
    if not isinstance(record, dict) or set(record) != {"type", "payload"} or record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        return None
    return identifier


def _plain_text_message(record: object) -> tuple[str, str, str] | None:
    if not isinstance(record, dict) or set(record) != {"type", "payload"} or record.get("type") != "response_item":
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if set(payload) - {"type", "id", "role", "content"} or payload.get("type") != "message":
        return None
    message_id = payload.get("id")
    role = payload.get("role")
    content = payload.get("content")
    if not isinstance(message_id, str) or not message_id or role not in {"user", "assistant"}:
        return None
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0].get("type") != "input_text"
        or not isinstance(content[0].get("text"), str)
        or not content[0]["text"]
    ):
        return None
    return message_id, role, content[0]["text"]


def _hash_and_compare_prefix(source_read: _CodexCheckpointRead, raw_id: str, prefix_file: BinaryIO) -> tuple[str, int]:
    provider, blob_hash, _source_path, kind, declared_size = source_read.raw_revision_descriptor(raw_id)
    allowed_kind = kind in {RawRevisionKind.FULL, RawRevisionKind.UNKNOWN}
    if provider is not Provider.CODEX or not allowed_kind:
        raise ValueError("checkpoint cohort contains a non-full Codex raw")
    digest = hashlib.sha256()
    actual_size = 0
    with source_read.open_raw_revision_material(raw_id) as (opened_provider, payload, _path, opened_kind):
        opened_kind_allowed = opened_kind in {RawRevisionKind.FULL, RawRevisionKind.UNKNOWN}
        if opened_provider is not Provider.CODEX or not opened_kind_allowed:
            raise ValueError("checkpoint raw changed while its original source witness was open")
        position = 0
        prefix_file.seek(0)
        while chunk := payload.read(1024 * 1024):
            check_compute_cancelled()
            digest.update(chunk)
            actual_size += len(chunk)
            prefix_file.seek(position)
            expected = prefix_file.read(len(chunk))
            if expected != chunk:
                raise ValueError("retained Codex revisions are not exact byte prefixes")
            position += len(chunk)
    if actual_size != declared_size or digest.hexdigest() != blob_hash:
        raise ValueError("retained Codex raw does not match its source witness hash and size")
    if actual_size > 0:
        prefix_file.seek(actual_size - 1)
        if prefix_file.read(1) != b"\n":
            raise ValueError("retained Codex revision ends inside a JSONL record")
    return blob_hash, actual_size


def _read_head_and_prove(
    source_read: _CodexCheckpointRead,
    raw_ids: Sequence[str],
    head_blob: BinaryIO,
    *,
    validation_mode: ValidationMode,
    validation_directory: Path,
    schema_registry: SchemaRegistry | None,
) -> tuple[BinaryIO, list[int], list[str], str, int, dict[str, RetainedValidationVerdict]]:
    if len(raw_ids) < 4 or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("checkpoint cohort needs three probes and an interior revision")
    descriptors = [source_read.raw_revision_descriptor(raw_id) for raw_id in raw_ids]
    first = descriptors[0]
    if any(
        descriptor[0] is not Provider.CODEX
        or descriptor[3] not in {RawRevisionKind.FULL, RawRevisionKind.UNKNOWN}
        or descriptor[2] != first[2]
        for index, descriptor in enumerate(descriptors)
    ):
        raise ValueError("checkpoint cohort does not share one full Codex source path")
    profile_keys = [source_read.raw_profile_identity(raw_id) for raw_id in raw_ids]
    if any(profile_key != profile_keys[0] for profile_key in profile_keys):
        raise ValueError("checkpoint cohort profile identity changed")

    hashes: list[str] = []
    sizes: list[int] = []
    for raw_id in raw_ids:
        check_compute_cancelled()
        digest = hashlib.sha256()
        size = 0
        with source_read.open_raw_revision_material(raw_id) as (provider, payload, _path, kind):
            if provider is not Provider.CODEX or kind not in {RawRevisionKind.FULL, RawRevisionKind.UNKNOWN}:
                raise ValueError("retained Codex revision changed kind")
            while chunk := payload.read(1024 * 1024):
                check_compute_cancelled()
                digest.update(chunk)
                size += len(chunk)
                if raw_id == raw_ids[-1]:
                    head_blob.write(chunk)
        expected_hash = descriptors[len(hashes)][1]
        expected_size = descriptors[len(sizes)][4]
        if size != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError("retained Codex raw does not match its pinned hash and size")
        if sizes and (size <= sizes[-1] or size > descriptors[-1][4]):
            raise ValueError("Codex revisions are not strictly increasing byte prefixes")
        hashes.append(expected_hash)
        sizes.append(size)

    for raw_index, (raw_id, size) in enumerate(zip(raw_ids, sizes, strict=True)):
        observed_hash, observed_size = _hash_and_compare_prefix(source_read, raw_id, head_blob)
        if observed_hash != descriptors[raw_index][1] or observed_size != size:
            raise ValueError("retained Codex revision changed during exact prefix verification")

    head_blob.seek(0)
    line_end = 0
    prefix_record_counts: list[int] = []
    prefix_verdicts: dict[str, RetainedValidationVerdict] = {}
    next_prefix = 0
    header_id: str | None = None
    line_number = 0
    registry = schema_registry or SchemaRegistry()
    snapshot = (
        registry.current_provider_snapshot(Provider.CODEX)
        if validation_mode is not ValidationMode.OFF
        else nullcontext()
    )
    with (
        snapshot,
        PrefixValidationState(
            provider=Provider.CODEX,
            source_path=first[2],
            mode=validation_mode,
            scratch_directory=validation_directory,
            registry=registry,
        ) as validation,
    ):
        for line_number, line in enumerate(head_blob, start=1):
            check_compute_cancelled()
            line_end += len(line)
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Codex checkpoint stream is not complete JSONL") from exc
            if line_number == 1:
                header_id = _plain_text_header(record)
                if header_id is None:
                    raise ValueError("Codex checkpoint stream has no unique leading session_meta")
            elif _plain_text_message(record) is None:
                raise ValueError("Codex record requires ordinary retained preparation")
            if not isinstance(record, dict):
                raise ValueError("Codex checkpoint record is not a JSON object")
            validation.observe(record)
            if next_prefix < len(sizes):
                if line_end == sizes[next_prefix]:
                    prefix_record_counts.append(line_number)
                    if 2 <= next_prefix < len(raw_ids) - 1:
                        raw_id = raw_ids[next_prefix]
                        prefix_verdicts[raw_id] = validation.verdict(
                            raw_id=raw_id,
                            revision_sha256=hashes[next_prefix],
                            evidence_id=raw_id,
                        )
                    next_prefix += 1
                elif line_end > sizes[next_prefix]:
                    raise ValueError("a retained revision does not end on a complete record boundary")
    if next_prefix != len(sizes) or line_end != sizes[-1]:
        raise ValueError("head Codex JSONL stream ended inside a record")
    if header_id is None or line_number == 0:
        raise ValueError("head Codex JSONL stream has no session header")
    if not prefix_record_counts or prefix_record_counts[0] != 1 or prefix_record_counts[-1] != line_number:
        raise ValueError("checkpoint prefixes must include the header and the complete head")
    return head_blob, prefix_record_counts, hashes, header_id, line_number - 1, prefix_verdicts


def _prefix_accounting(message_count: int) -> ParseAccounting:
    expected = {AdmissionUnit.OUTER_RECORD: message_count + 1}
    materialized = {AdmissionUnit.OUTER_RECORD: [(0, message_count + 1)]}
    if message_count:
        expected.update({AdmissionUnit.PART: message_count, AdmissionUnit.BLOCK: message_count})
        materialized.update(
            {
                AdmissionUnit.PART: [(0, message_count)],
                AdmissionUnit.BLOCK: [(0, message_count)],
            }
        )
    return ParseAccounting(
        expected=expected,
        materialized_ordinals=materialized,
    )


def _finalize_codex_prefix(
    head: ParsedSession,
    messages: Sequence[ParsedMessage],
    message_count: int,
    accounting: ParseAccounting,
    updated_at: str | None,
) -> ParsedSession:
    """Shared finalization for ordinary Codex parses and exact prefix views."""
    prefix_messages = _PrefixMessages(messages, message_count)
    leaf = prefix_messages[-1].provider_message_id if message_count else None
    return codex.finalize_codex_session(
        head,
        messages=prefix_messages,
        session_events=[],
        updated_at=updated_at,
        unit_accounting=accounting,
        mark_active_leaf=False,
        active_leaf_message_provider_id=leaf,
    )


def prepare_codex_prefix_checkpoints(
    source_read: _CodexCheckpointRead,
    raw_ids: Sequence[str],
    *,
    head_artifact: PreparedJsonl,
    artifact_directory: Path,
    validation_mode: ValidationMode,
    publication_publisher: ArchiveBlobPublisher | None,
    publication_source_read: BlobPublicationSourceRead | None,
    prepare_sessions: Callable[[str, Iterable[ParsedSession]], Iterable[ParsedSession]],
    artifact_options: Callable[[str, int], CodexCheckpointArtifactOptions],
    schema_registry: SchemaRegistry | None = None,
) -> CodexPrefixPreparation:
    """Prove the whole cohort, then lazily seal one exact interior at a time.

    The already prepared head is the sole canonical parse of the complete
    stream. The caller independently prepares the identity endpoints and uses
    those artifacts unchanged. This function only supplies interior artifacts.
    """
    head_blob: BinaryIO | None = None
    try:
        # The result object owns this spool from the moment it is opened, even
        # when source verification fails before `_read_head_and_prove` returns.
        head_blob = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        head_blob, prefix_record_counts, hashes, header_id, head_message_count, prefix_verdicts = _read_head_and_prove(
            source_read,
            raw_ids,
            head_blob,
            validation_mode=validation_mode,
            validation_directory=artifact_directory,
            schema_registry=schema_registry,
        )
        parser_head = head_artifact.parser_stage_artifact or head_artifact
        if parser_head.blob_hash != hashes[-1]:
            raise ValueError("canonical head parser stage is not bound to this cohort's exact head blob")
        if parser_head.captured_profile_key != source_read.raw_profile_identity(raw_ids[-1]):
            raise ValueError("canonical head parser profile differs from its captured source witness")
        parsed_sessions = parser_head.session_sequence()
        if len(parsed_sessions) != 1:
            raise ValueError("Codex checkpoint head must contain exactly one parsed session")
        head = parsed_sessions[0]
        if head.source_name is not Provider.CODEX or head.provider_session_id != header_id:
            raise ValueError("canonical head artifact disagrees with the proved Codex header")
        if len(head.messages) != head_message_count:
            raise ValueError("canonical head artifact message count disagrees with the proved grammar")
        if head.created_at_provenance == "fallback" or head.updated_at_provenance == "fallback":
            raise ValueError("canonical head artifact already carries head-specific fallback timestamps")
        if any(message.timestamp is not None for message in head.messages):
            raise ValueError("canonical head artifact message timestamps exceed the proved message grammar")
        if head.session_events:
            raise ValueError("canonical head artifact includes non-prefix-local Codex events")
        for index in range(2, len(raw_ids) - 1):
            raw_id = raw_ids[index]
            options = artifact_options(raw_id, prefix_record_counts[index])
            if options.captured_profile_key != source_read.raw_profile_identity(raw_id):
                raise ValueError("checkpoint artifact profile differs from its captured source witness")
    except (DaemonBackpressureError, DaemonOperationCancelled):
        if head_blob is not None:
            head_blob.close()
        raise
    except Exception as exc:
        if head_blob is not None:
            head_blob.close()
        return CodexPrefixPreparation(CodexCheckpointDisposition.ORDINARY_FALLBACK, raw_ids, str(exc))

    def artifacts() -> Iterator[tuple[str, PreparedJsonl]]:
        assert head_blob is not None
        last_count = prefix_record_counts[1] - 1
        timestamp_pair = parse_timestamp_pair(head.created_at)
        try:
            for index in range(2, len(raw_ids) - 1):
                check_compute_cancelled()
                message_count = prefix_record_counts[index] - 1
                options = artifact_options(raw_ids[index], prefix_record_counts[index])
                for message_index in range(last_count, message_count):
                    timestamp_pair = codex._newer_timestamp_pair(
                        timestamp_pair,
                        parse_timestamp_pair(head.messages[message_index].timestamp),
                    )
                last_count = message_count
                canonical = _finalize_codex_prefix(
                    head,
                    head.messages,
                    message_count,
                    _prefix_accounting(message_count),
                    timestamp_pair[1] if timestamp_pair is not None else None,
                )
                canonical = normalize_session_timestamps(
                    canonical,
                    fallback_timestamp=options.fallback_timestamp,
                )
                source_path = options.source_path
                finalized = iter(prepare_sessions(raw_ids[index], iter((canonical,))))

                def prepared_sessions(
                    sessions: Iterable[ParsedSession], source_path_for_raw: str = source_path
                ) -> Iterator[ParsedSession]:
                    for session in sessions:
                        admitted = admit_parsed_sessions_for_publication(
                            [session], provider=Provider.CODEX, source_path=source_path_for_raw
                        )
                        if not admitted:
                            continue
                        session = admitted[0]
                        session.content_hash = session_content_hash(session)
                        yield session

                prepared = PreparedJsonl.from_sessions(
                    prepared_sessions(finalized),
                    blob_hash=hashes[index],
                    artifact_directory=options.artifact_directory or artifact_directory,
                    publication_publisher=publication_publisher,
                    publication_source_read=publication_source_read,
                    classification=options.classification,
                    enrichment_digest=options.enrichment_digest,
                    enrichment_index_path=options.enrichment_index_path,
                    parsed_prefix_size=options.parsed_prefix_size,
                    resolved_provider=Provider.CODEX,
                    captured_profile_key=options.captured_profile_key,
                    preparation_dependency=options.preparation_dependency,
                )
                prepared = replace(prepared, validation_verdict=prefix_verdicts[raw_ids[index]])
                yield raw_ids[index], prepared
        finally:
            head_blob.close()

    return CodexPrefixPreparation(CodexCheckpointDisposition.READY, raw_ids, None, artifacts(), head_blob)
