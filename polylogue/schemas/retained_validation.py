"""Spill-backed schema validation for retained source revisions.

This module is the compact-result path used while preparing a retained raw
revision.  The caller owns one :class:`StreamedJSONDocument` context; all
sample scans, package fallbacks and drift reductions replay that same lazy
view before it is closed.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Generator, Iterator, KeysView, Mapping, Sequence
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, SupportsIndex, cast, overload

from jsonschema import Draft202012Validator, ValidationError, validators

from polylogue.archive.raw_payload.sampling_buckets import is_record_candidate, take_bucketed_samples
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.iterator_lifetime import settled_iterator
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.core.provider_identity import normalize_provider_token
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.sources import origin_from_provider
from polylogue.core.work_progress import (
    advance_work_progress,
    reports_work_progress,
    stable_productive_identity,
    utf8_byte_length,
)
from polylogue.schemas.drift_sentinel import (
    FIELD_CHANGED,
    KNOWN_FIELD_UNREAD,
    NEW_FIELD,
    UNSEEN_SHAPE,
    DriftClassification,
    DriftSignature,
    SchemaDriftObservation,
)
from polylogue.schemas.observation_models import ProfileToken
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaObservation, SchemaRegistry
from polylogue.schemas.schema_parser_coverage import unread_field_names
from polylogue.schemas.validator_resolution import (
    _choose_retained_schema,
    _historical_schemas,
    _load_package,
    _load_schema,
    canonical_provider,
    resolve_retained_schema,
)
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

if TYPE_CHECKING:
    pass


_ACTIVE_VALIDATION_CONNECTION: ContextVar[sqlite3.Connection | None] = ContextVar(
    "retained_validation_connection", default=None
)
_BOUNDED_VALIDATOR_CLASS: Any = None
_ACTIVE_VALIDATION_SCRATCH: ContextVar[_ValidationScratch | None] = ContextVar(
    "retained_validation_scratch", default=None
)


class _ValidationScratch:
    """Reducer tables owned by one live scratch connection and validation scope."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        _ensure_reducer_tables(connection)

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _ACTIVE_VALIDATION_SCRATCH.set(self)
        try:
            yield
        finally:
            _ACTIVE_VALIDATION_SCRATCH.reset(token)


def _validation_scratch(connection: sqlite3.Connection) -> _ValidationScratch:
    active = _ACTIVE_VALIDATION_SCRATCH.get()
    if active is not None and active.connection is connection:
        return active
    return _ValidationScratch(connection)


def _retained_validation_productive_identity(
    provider: str | Provider,
    path: Path,
    *,
    mode: ValidationMode,
    raw_id: str,
    revision_sha256: str,
    evidence_id: str,
    source_path: str | None = None,
    jsonl: bool = False,
    accepted_prefix_size: int | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    schema_resolution: SchemaResolution | None = None,
    schema_resolution_is_explicit: bool = False,
    registry: SchemaRegistry | None = None,
    signature_directory: Path,
) -> str:
    """Identify validation work by its retained source recipe, never scratch path."""
    del path, registry, signature_directory
    resolution = None
    if schema_resolution is not None:
        resolution = (
            schema_resolution.provider,
            schema_resolution.package_version,
            schema_resolution.element_kind,
            schema_resolution.exact_structure_id,
            schema_resolution.bundle_scope,
            schema_resolution.reason,
            schema_resolution.profile_score,
        )
    zip_coordinate = None
    if captured_zip_coordinate is not None:
        zip_coordinate = (
            "captured-zip-coordinate",
            captured_zip_coordinate.canonical_container,
            captured_zip_coordinate.declared_container,
            captured_zip_coordinate.member_name,
            captured_zip_coordinate.entry_ordinal,
            captured_zip_coordinate.split_index,
            captured_zip_coordinate.addressing_mode.value,
            captured_zip_coordinate.container_blob_hash,
            captured_zip_coordinate.decoder_fingerprint,
            captured_zip_coordinate.profile_namespace,
        )
    recipe = (
        normalize_provider_token(provider),
        raw_id,
        revision_sha256,
        evidence_id,
        source_path,
        jsonl,
        accepted_prefix_size,
        zip_coordinate,
        ValidationMode.from_string(mode).value,
        resolution,
        schema_resolution_is_explicit,
    )
    return stable_productive_identity(recipe)


class _ConnectionBoundValidator:
    """Run one shared validator class with its caller-owned spill connection."""

    def __init__(self, validator: Any, connection: sqlite3.Connection, scratch: _ValidationScratch) -> None:
        self._validator = validator
        self._connection = connection
        self._scratch = scratch

    def iter_errors(self, instance: object) -> Iterator[ValidationError]:
        token = _ACTIVE_VALIDATION_CONNECTION.set(self._connection)
        try:
            with self._scratch.activate():
                yield from self._validator.iter_errors(instance)
        finally:
            _ACTIVE_VALIDATION_CONNECTION.reset(token)

    def is_valid(self, instance: object) -> bool:
        token = _ACTIVE_VALIDATION_CONNECTION.set(self._connection)
        try:
            with self._scratch.activate():
                return bool(self._validator.is_valid(instance))
        finally:
            _ACTIVE_VALIDATION_CONNECTION.reset(token)

    def evolve(self, **kwargs: object) -> _ConnectionBoundValidator:
        return _ConnectionBoundValidator(self._validator.evolve(**kwargs), self._connection, self._scratch)


def _active_validation_connection() -> sqlite3.Connection:
    connection = _ACTIVE_VALIDATION_CONNECTION.get()
    if connection is None:
        raise RuntimeError("bounded schema keyword ran without an active spill connection")
    return connection


@dataclass(frozen=True, slots=True)
class RetainedValidationVerdict:
    """Bounded validation outcome attached to its retained source evidence."""

    raw_id: str
    revision_sha256: str
    evidence_id: str
    mode: ValidationMode
    status: ValidationStatus
    sample_count: int
    invalid_count: int
    error_count: int
    drift_count: int
    first_diagnostic: str | None
    schema_resolution: SchemaResolution | None
    drift_observation: SchemaDriftObservation | None
    strict_refusal: bool


@dataclass(frozen=True, slots=True)
class RetainedValidationBody:
    """Complete schema evidence before binding its acquired Source coordinates."""

    mode: ValidationMode
    status: ValidationStatus
    sample_count: int
    invalid_count: int
    error_count: int
    drift_count: int
    first_diagnostic: str | None
    schema_resolution: SchemaResolution | None
    drift_observation: SchemaDriftObservation | None
    strict_refusal: bool

    @classmethod
    def from_verdict(cls, verdict: RetainedValidationVerdict) -> RetainedValidationBody:
        drift = verdict.drift_observation
        if drift is not None:
            drift = replace(drift, raw_id="", native_id_example="")
        return cls(
            verdict.mode,
            verdict.status,
            verdict.sample_count,
            verdict.invalid_count,
            verdict.error_count,
            verdict.drift_count,
            verdict.first_diagnostic,
            verdict.schema_resolution,
            drift,
            verdict.strict_refusal,
        )

    def bind(
        self, *, raw_id: str, revision_sha256: str, evidence_id: str, source_path: str | None
    ) -> RetainedValidationVerdict:
        drift = self.drift_observation
        if drift is not None:
            drift = replace(drift, raw_id=raw_id, native_id_example=source_path or raw_id)
        return RetainedValidationVerdict(
            raw_id,
            revision_sha256,
            evidence_id,
            self.mode,
            self.status,
            self.sample_count,
            self.invalid_count,
            self.error_count,
            self.drift_count,
            self.first_diagnostic,
            self.schema_resolution,
            drift,
            self.strict_refusal,
        )


class _SampleValidationReducer:
    """Accumulate the exact per-sample validation and drift result."""

    def __init__(
        self,
        schema: Mapping[str, object],
        provider: Provider,
        resolution: SchemaResolution | None,
        connection: sqlite3.Connection,
        *,
        source_path: str | None,
        signature_directory: Path,
        validation_accepted: bool = False,
    ) -> None:
        self.signature_directory = signature_directory
        self.validation_accepted = validation_accepted
        self.schema = schema
        self.provider = provider
        self.resolution = resolution
        self.connection = connection
        self.source_path = source_path
        self.validator = _bounded_validator(schema, connection)
        self.sample_count = 0
        self.invalid_count = 0
        self.error_count = 0
        self.drift_count = 0
        self.first_diagnostic: str | None = None
        self.strongest: SchemaDriftObservation | None = None

    @property
    def accepts(self) -> bool:
        return self.invalid_count == 0

    def observe(self, sample: Mapping[str, object]) -> None:
        with self.validator._scratch.activate():
            self._observe(sample)

    def _observe(self, sample: Mapping[str, object]) -> None:
        check_compute_cancelled()
        self.sample_count += 1
        sample_errors = 0
        if not self.validation_accepted:
            normalized = _normalized(sample, self.schema, self.schema, self.connection)
            for error in self.validator.iter_errors(normalized):
                check_compute_cancelled()
                sample_errors += 1
                self.error_count += 1
                if self.first_diagnostic is None:
                    self.first_diagnostic = _diagnostic(error)
        valid = sample_errors == 0
        if not valid:
            self.invalid_count += 1
        self.drift_count += _collect_drift_paths(
            sample,
            self.schema,
            self.connection,
            self.sample_count,
        )
        sample_drift = _reduce_sample_drift(
            sample,
            self.schema,
            self.provider,
            self.resolution,
            self.connection,
            self.sample_count,
            raw_id="",
            native_id_example=self.source_path or "",
            is_valid=valid,
            signature_directory=self.signature_directory,
        )
        _clear_sample_drift(self.connection, self.sample_count)
        self.connection.execute("DELETE FROM retained_unread WHERE sample=?", (self.sample_count,))
        self.strongest = _stronger_drift(self.strongest, sample_drift)

    def verdict(
        self,
        *,
        raw_id: str,
        revision_sha256: str,
        evidence_id: str,
        mode: ValidationMode,
        resolution: SchemaResolution | None,
        source_path: str | None,
    ) -> RetainedValidationVerdict:
        status = (
            ValidationStatus.FAILED if mode is ValidationMode.STRICT and self.invalid_count else ValidationStatus.PASSED
        )
        diagnostic = (
            f"Schema validation failed: {self.first_diagnostic}"
            if status is ValidationStatus.FAILED and self.first_diagnostic is not None
            else self.first_diagnostic
        )
        drift = self.strongest
        if drift is not None:
            drift = replace(drift, raw_id=raw_id, native_id_example=source_path or raw_id)
        return RetainedValidationVerdict(
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            mode=mode,
            status=status,
            sample_count=self.sample_count,
            invalid_count=self.invalid_count,
            error_count=self.error_count,
            drift_count=self.drift_count,
            first_diagnostic=diagnostic,
            schema_resolution=resolution,
            drift_observation=drift,
            strict_refusal=status is ValidationStatus.FAILED,
        )


@dataclass(slots=True)
class _PrefixSchemaReducer:
    version: str
    schema: Mapping[str, object] | None
    reducer: _SampleValidationReducer | None
    error: FileNotFoundError | ImportError | None = None


class PrefixValidationState:
    """One-pass retained-schema validation for proven Codex JSONL prefixes.

    The caller feeds the exact wire records of a supported JSONL grammar once,
    then requests detached, source-bound verdict snapshots at raw revision
    boundaries. This state preserves ordinary schema precedence: each prefix
    resolves its own observation and selects the current schema or the first
    historical schema accepting that prefix.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        source_path: str,
        mode: ValidationMode,
        registry: SchemaRegistry | None = None,
        scratch_directory: Path | None = None,
        signature_directory: Path,
    ) -> None:
        self.signature_directory = signature_directory
        self.provider = canonical_provider(provider)
        self.source_path = source_path
        self.mode = ValidationMode.from_string(mode)
        self.registry = registry or SchemaRegistry()
        self._scratch_context = scratch_connection_context(
            prefix="polylogue-prefix-validation-",
            filename="validation.sqlite",
            directory=scratch_directory,
        )
        self.connection = self._scratch_context.__enter__()
        self._validation_scratch: _ValidationScratch | None = None
        self._header: JSONDocument | None = None
        self._base_observation: SchemaObservation | None = None
        self._base_resolution: SchemaResolution | None = None
        self._base_version: str | None = None
        self._base_element: str | None = None
        self._base_error: FileNotFoundError | ImportError | None = None
        self._reducers: list[_PrefixSchemaReducer] = []
        self._header_witness: tuple[str, ...] = ()
        self._message_witness: tuple[str, ...] = ()
        self._message_fingerprint: str | None = None
        self._message_profile: tuple[ProfileToken, ...] | None = None
        self._record_count = 0
        self._closed = False

    def __enter__(self) -> PrefixValidationState:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(exc_type, exc, traceback)

    def close(self) -> None:
        self._close(None, None, None)

    def _close(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._scratch_context.__exit__(exc_type, exc, traceback)

    def observe(self, record: JSONDocument) -> None:
        """Add one grammar-proved header/message record exactly once."""
        if self._closed:
            raise RuntimeError("prefix validation state is closed")
        self._record_count += 1
        if self.mode is ValidationMode.OFF:
            return
        if self._record_count == 1:
            self._header = record
            return
        if self._record_count == 2:
            self._initialize(record)
            return
        self._prove_message_shape(record)
        self._observe_validators(record)

    def _initialize(self, first_message: JSONDocument) -> None:
        from polylogue.schemas.observation_runtime import _record_profile_tokens
        from polylogue.schemas.runtime_registry import _structure_witnesses
        from polylogue.schemas.shape_fingerprint import fingerprint_parts

        if self._header is None:
            raise ValueError("prefix validation has no leading header")
        observations, _cluster = self.registry.observe_payload(
            str(self.provider),
            cast(JSONValue, [self._header, first_message]),
            source_path=self.source_path,
        )
        if len(observations) != 1:
            raise ValueError("Codex checkpoint prefix has no unique schema observation")
        self._base_observation = observations[0]
        witnesses = _structure_witnesses([self._header, first_message])
        if len(witnesses) != 2:
            raise ValueError("Codex checkpoint prefix schema witnesses are not record-local")
        self._header_witness, self._message_witness = witnesses
        self._message_profile = _record_profile_tokens([first_message], record_type_key="type")
        digest = hashlib.sha256()
        for part in fingerprint_parts(first_message, depth=1):
            digest.update(part.encode("utf-8"))
        self._message_fingerprint = digest.hexdigest()

        self._base_resolution = self.registry.resolve_observation(
            str(self.provider), (self._base_observation,), source_path=self.source_path
        )
        try:
            if self._base_resolution is None:
                package = _load_package(self.registry, self.provider, version="default")
                self._base_version = package.version
                self._base_element = package.default_element_kind
            else:
                self._base_version = self._base_resolution.package_version
                self._base_element = self._base_resolution.element_kind
            base_schema = _load_schema(
                self.registry,
                self.provider,
                package_version=self._base_version,
                element_kind=self._base_element,
            )
            self._reducers.append(self._new_reducer(self._base_version, base_schema))
        except (FileNotFoundError, ImportError) as exc:
            self._base_error = exc
            return

        historical = _historical_schemas(
            self.registry,
            self.provider,
            element_kind=self._base_element,
        )
        while True:
            try:
                version, schema = next(historical)
            except StopIteration:
                break
            except (FileNotFoundError, ImportError) as exc:
                self._reducers.append(_PrefixSchemaReducer("", None, None, error=exc))
                break
            self._reducers.append(self._new_reducer(version, schema))
        self._observe_validators(self._header)
        self._observe_validators(first_message)

    def _new_reducer(self, version: str, schema: Mapping[str, object]) -> _PrefixSchemaReducer:
        if self._validation_scratch is None:
            self._validation_scratch = _ValidationScratch(self.connection)
        with self._validation_scratch.activate():
            reducer = _SampleValidationReducer(
                schema,
                self.provider,
                self._base_resolution,
                self.connection,
                source_path=self.source_path,
                signature_directory=self.signature_directory,
            )
        return _PrefixSchemaReducer(version, schema, reducer)

    def _observe_validators(self, record: JSONDocument) -> None:
        for index, candidate in enumerate(self._reducers):
            if candidate.reducer is None or candidate.schema is None:
                continue
            # Appending records cannot restore a rejected record-local candidate.
            # The base still owns complete failure counts when every schema rejects.
            if index != 0 and not candidate.reducer.accepts:
                continue
            if any(_validation_samples(record, candidate.schema, self.provider)):
                candidate.reducer.observe(record)

    def _prove_message_shape(self, record: JSONDocument) -> None:
        from polylogue.schemas.observation_runtime import _record_profile_tokens
        from polylogue.schemas.runtime_registry import _structure_witnesses
        from polylogue.schemas.shape_fingerprint import fingerprint_parts

        digest = hashlib.sha256()
        for part in fingerprint_parts(record, depth=1):
            digest.update(part.encode("utf-8"))
        if digest.hexdigest() != self._message_fingerprint:
            raise ValueError("Codex checkpoint messages are not shape-uniform")
        if _record_profile_tokens([record], record_type_key="type") != self._message_profile:
            raise ValueError("Codex checkpoint message profile changed")
        if _structure_witnesses([record])[0] != self._message_witness:
            raise ValueError("Codex checkpoint message structure witness changed")

    def _prefix_observation(self) -> SchemaObservation:
        from dataclasses import replace as dataclass_replace

        assert self._base_observation is not None
        if self._record_count <= 64:
            witnesses = (self._header_witness, *((self._message_witness,) * (self._record_count - 1)))
        else:
            buckets: dict[str, list[JSONDocument]] = {
                "type:response_item": [{"ordinal": index} for index in range(1, 9)],
                "type:session_meta": [{"ordinal": 0}],
            }
            selected = take_bucketed_samples(buckets, 64)
            witnesses = tuple(
                self._header_witness if cast(int, index["ordinal"]) == 0 else self._message_witness
                for index in selected
            )
        return dataclass_replace(
            self._base_observation,
            source_witnesses=witnesses if self._base_observation.source_witnesses else (),
        )

    def verdict(self, *, raw_id: str, revision_sha256: str, evidence_id: str) -> RetainedValidationVerdict:
        if self._closed:
            raise RuntimeError("prefix validation state is closed")
        if self.mode is ValidationMode.OFF:
            return _verdict(raw_id, revision_sha256, evidence_id, self.mode, ValidationStatus.SKIPPED)
        if self._record_count < 2 or self._base_observation is None:
            raise ValueError("prefix validation requires a header and one message")
        if self._base_error is not None:
            return _verdict(
                raw_id,
                revision_sha256,
                evidence_id,
                self.mode,
                ValidationStatus.SKIPPED,
                schema_resolution=self._base_resolution,
            )
        resolved = self.registry.resolve_observation(
            str(self.provider), (self._prefix_observation(),), source_path=self.source_path
        )
        if self._base_resolution is None:
            if resolved is not None:
                raise ValueError("Codex prefix schema selection changed after its first message")
        elif resolved is None or (
            resolved.package_version != self._base_version
            or resolved.element_kind != self._base_element
            or resolved.reason != self._base_resolution.reason
        ):
            raise ValueError("Codex prefix base schema observation is not stable")

        candidates_by_version = {candidate.version: candidate for candidate in self._reducers}
        candidates_by_schema = {
            id(candidate.schema): candidate for candidate in self._reducers if candidate.schema is not None
        }

        def historical_schemas() -> Iterator[tuple[str, JSONDocument]]:
            for candidate in self._reducers[1:]:
                if candidate.error is not None:
                    raise candidate.error
                if candidate.schema is not None:
                    yield candidate.version, cast(JSONDocument, candidate.schema)

        def schema_accepts(schema: JSONDocument) -> bool:
            candidate = candidates_by_schema.get(id(schema))
            if candidate is None or candidate.reducer is None:
                raise ValueError("prefix validation candidate has no sample reducer")
            return candidate.reducer.accepts

        assert self._base_version is not None and self._reducers[0].schema is not None
        try:
            selected_version, _selected_schema = _choose_retained_schema(
                self._base_version,
                cast(JSONDocument, self._reducers[0].schema),
                historical_schemas(),
                schema_accepts=schema_accepts,
            )
        except (FileNotFoundError, ImportError):
            return _verdict(
                raw_id,
                revision_sha256,
                evidence_id,
                self.mode,
                ValidationStatus.SKIPPED,
                schema_resolution=self._base_resolution,
            )
        selected = candidates_by_version[selected_version]
        selected_resolution = resolved
        if selected.version != self._base_version and selected_resolution is not None:
            selected_resolution = replace(
                selected_resolution,
                package_version=selected.version,
            )
        assert selected.reducer is not None
        return selected.reducer.verdict(
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            mode=self.mode,
            resolution=selected_resolution,
            source_path=self.source_path,
        )


_DRIFT_STRENGTH: dict[DriftClassification, int] = {
    FIELD_CHANGED: 4,
    UNSEEN_SHAPE: 3,
    KNOWN_FIELD_UNREAD: 2,
    NEW_FIELD: 1,
}


@reports_work_progress("source_preparation", productive_identity=_retained_validation_productive_identity)
def validate_retained_document(
    provider: str | Provider,
    path: Path,
    *,
    mode: ValidationMode,
    raw_id: str,
    revision_sha256: str,
    evidence_id: str,
    source_path: str | None = None,
    jsonl: bool = False,
    accepted_prefix_size: int | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
    schema_resolution: SchemaResolution | None = None,
    schema_resolution_is_explicit: bool = False,
    registry: SchemaRegistry | None = None,
    signature_directory: Path,
) -> RetainedValidationVerdict:
    """Decode and validate a complete retained JSON document using disk spill.

    Validation mode controls schema validation only.  Decoding remains the
    caller's independent admission concern, and drift never changes whether
    the source revision can be published. The caller retains signature_directory
    until the verdict's drift observation has been sampled after publication.
    """
    mode = ValidationMode.from_string(mode)
    if mode is ValidationMode.OFF:
        return _verdict(
            raw_id,
            revision_sha256,
            evidence_id,
            mode,
            ValidationStatus.SKIPPED,
            schema_resolution=schema_resolution,
        )

    from polylogue.schemas.observation_spill import StreamedJSONDocument

    document = StreamedJSONDocument(path, jsonl=jsonl, accepted_prefix_size=accepted_prefix_size)
    with document as payload, _ValidationScratch(document.connection).activate():
        spill = _active_spill(document)
        active_registry = registry or SchemaRegistry()
        resolved: SchemaResolution | None = schema_resolution
        from polylogue.schemas.validator_resolution import canonical_provider

        canonical = canonical_provider(provider)
        selected_schema: Mapping[str, object] | None = None
        schema_key: tuple[str, str, str] | None = None
        accepted_schema: Mapping[str, object] | None = None

        def accepts(candidate: JSONDocument) -> bool:
            nonlocal accepted_schema
            accepted = _schema_accepts_document(payload, candidate, canonical, spill.connection)
            if accepted:
                accepted_schema = candidate
            return accepted

        try:
            canonical, selected, schema_key, resolved = resolve_retained_schema(
                provider,
                payload,
                source_path=source_path,
                schema_resolution=schema_resolution,
                schema_resolution_is_explicit=schema_resolution_is_explicit,
                registry=active_registry,
                schema_store=spill.store_schema,
                schema_accepts=accepts,
            )
            selected_schema = selected
        except (FileNotFoundError, ImportError):
            return _verdict(
                raw_id,
                revision_sha256,
                evidence_id,
                mode,
                ValidationStatus.SKIPPED,
                schema_resolution=resolved,
            )

        assert selected_schema is not None
        reducer = _SampleValidationReducer(
            selected_schema,
            canonical,
            resolved,
            spill.connection,
            source_path=source_path,
            signature_directory=signature_directory,
            # This proof belongs only to this completed immutable document and
            # the exact schema returned by the acceptance pass. Drift is still
            # reduced from every original sample below.
            validation_accepted=accepted_schema is selected_schema,
        )
        for sample in _validation_samples(payload, selected_schema, canonical):
            reducer.observe(sample)
        return reducer.verdict(
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            mode=mode,
            resolution=resolved,
            source_path=source_path,
        )


def _verdict(
    raw_id: str,
    revision_sha256: str,
    evidence_id: str,
    mode: ValidationMode,
    status: ValidationStatus,
    *,
    schema_resolution: SchemaResolution | None = None,
) -> RetainedValidationVerdict:
    return RetainedValidationVerdict(
        raw_id=raw_id,
        revision_sha256=revision_sha256,
        evidence_id=evidence_id,
        mode=mode,
        status=status,
        sample_count=0,
        invalid_count=0,
        error_count=0,
        drift_count=0,
        first_diagnostic=None,
        schema_resolution=schema_resolution,
        drift_observation=None,
        strict_refusal=False,
    )


def _active_spill(document: object) -> Any:
    # The root lazy object intentionally carries the same owner connection as
    # child objects.  The public context exposes it through the document, so
    # callers can use a single private database for tree and reducers.
    connection = getattr(document, "connection", None)
    if connection is None:
        raise RuntimeError("retained schema document has no live spill connection")

    class SpillAccess:
        def __init__(self, document: object, conn: sqlite3.Connection) -> None:
            self.document = document
            self.connection = conn

        def store_schema(self, value: object) -> JSONDocument:
            store = getattr(self.document, "store_schema", None)
            if store is None:
                raise RuntimeError("retained schema document does not expose schema spill storage")
            return cast(JSONDocument, store(value))

    return SpillAccess(document, cast(sqlite3.Connection, connection))


def _validation_samples(
    payload: object,
    schema: Mapping[str, object],
    provider: Provider,
) -> Iterator[Mapping[str, object]]:
    granularity = schema.get("x-polylogue-sample-granularity")
    if granularity not in {"record", "document"}:
        granularity = "record" if provider in {Provider.CLAUDE_CODE, Provider.CODEX} else "document"
    if isinstance(payload, Mapping):
        if granularity == "document" or is_record_candidate(cast(JSONDocument, payload)):
            advance_work_progress(messages=1)
            yield payload
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            check_compute_cancelled()
            if isinstance(value, Mapping) and (
                granularity == "document" or is_record_candidate(cast(JSONDocument, value))
            ):
                advance_work_progress(messages=1)
                yield value


def _diagnostic(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "root"
    keyword = str(error.validator or "schema")
    return f"{path}: {keyword} validation failed"


def _validation_value_size(value: object) -> int:
    """Count decoded JSON bytes reached by schema traversal, including container delimiters."""
    if isinstance(value, str):
        return utf8_byte_length(value)
    if value is None:
        return 4
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, (int, float, Decimal)):
        return len(str(value).encode("ascii"))
    if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
        return 2
    return 0


def _schema_accepts_document(
    payload: object,
    schema: Mapping[str, object],
    provider: Provider,
    connection: sqlite3.Connection,
) -> bool:
    validator = _bounded_validator(schema, connection)
    with validator._scratch.activate():
        for sample in _validation_samples(payload, schema, provider):
            check_compute_cancelled()
            if not validator.is_valid(_normalized(sample, schema, schema, connection)):
                return False
    return True


def _normalized(
    value: object,
    schema: object,
    root: Mapping[str, object],
    connection: sqlite3.Connection,
) -> object:
    from polylogue.schemas.validator import (
        _schema_allows_type,
        _schema_branch_for_value,
    )

    selected = _schema_branch_for_value(schema, value, root, connection=connection)
    if isinstance(value, dict):
        return _NormalizedObject(value, selected, root, connection)
    if isinstance(value, list):
        if not value and not _schema_allows_type(selected, "array") and _schema_allows_type(selected, "null"):
            return None
        return _NormalizedArray(value, selected, root, connection)
    return value


class _NormalizedObject(dict[str, object]):
    def __init__(
        self, value: dict[str, object], schema: object, root: Mapping[str, object], connection: sqlite3.Connection
    ) -> None:
        dict.__init__(self)
        self._value = value
        self._schema = schema
        self._root = root
        self._connection = connection

    def __iter__(self) -> Iterator[str]:
        for key in self._value:
            check_compute_cancelled()
            advance_work_progress(bytes=utf8_byte_length(key))
            yield key

    def __len__(self) -> int:
        return len(self._value)

    def __contains__(self, key: object) -> bool:
        return key in self._value

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        return KeysView(self)

    def sorted_keys(self) -> Iterator[str]:
        source = getattr(self._value, "sorted_keys", None)
        keys = source() if source is not None else iter(sorted(self._value))
        for key in keys:
            check_compute_cancelled()
            yield key

    def __getitem__(self, key: str) -> object:
        check_compute_cancelled()
        value = self._value[key]
        return self._normalize_member(key, value)

    def _normalize_member(self, key: str, value: object) -> object:
        advance_work_progress(bytes=utf8_byte_length(key) + _validation_value_size(value))
        from polylogue.schemas.validator import _schema_for_property

        child_schema = _schema_for_property(self._schema, key, value, self._root, connection=self._connection)
        return _normalized(value, child_schema, self._root, self._connection)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[str, object]]:  # type: ignore[override]
        with settled_iterator(self._value.items()) as items:
            while True:
                check_compute_cancelled()
                try:
                    key, value = next(items)
                except StopIteration:
                    return
                advance_work_progress(bytes=utf8_byte_length(key))
                yield key, self._normalize_member(key, value)

    def values(self) -> Iterator[object]:  # type: ignore[override]
        with settled_iterator(self._value.items()) as items:
            while True:
                check_compute_cancelled()
                try:
                    key, value = next(items)
                except StopIteration:
                    return
                advance_work_progress(bytes=utf8_byte_length(key))
                yield self._normalize_member(key, value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Mapping)
            and len(self) == len(other)
            and all(key in other and self[key] == other[key] for key in self)
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


class _NormalizedArray(list[object]):
    def __init__(
        self, value: list[object], schema: object, root: Mapping[str, object], connection: sqlite3.Connection
    ) -> None:
        list.__init__(self)
        self._value = value
        self._schema = schema
        self._root = root
        self._connection = connection

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self) -> Iterator[object]:
        with settled_iterator(self._value) as items:
            while True:
                check_compute_cancelled()
                try:
                    value = next(items)
                except StopIteration:
                    return
                yield self._normalize_item(value)

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> list[object]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> object:
        check_compute_cancelled()
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        value = self._value[int(index)]
        return self._normalize_item(value)

    def _normalize_item(self, value: object) -> object:
        from polylogue.schemas.validator import _schema_for_items

        advance_work_progress(bytes=_validation_value_size(value))
        return _normalized(value, _schema_for_items(self._schema, value, self._root), self._root, self._connection)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and not isinstance(other, (str, bytes))
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


def _bounded_validator(schema: Mapping[str, object], connection: sqlite3.Connection) -> Any:
    global _BOUNDED_VALIDATOR_CLASS

    scratch = _validation_scratch(connection)
    if _BOUNDED_VALIDATOR_CLASS is not None:
        return _ConnectionBoundValidator(_BOUNDED_VALIDATOR_CLASS(schema), connection, scratch)

    def any_of(
        validator: Any, schemas: Sequence[object], instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        accepted = False
        for branch in schemas:
            check_compute_cancelled()
            if validator.evolve(schema=branch).is_valid(instance):
                accepted = True
                break
        if not accepted:
            yield ValidationError("No anyOf branch accepted the value", instance=instance, schema=schema_node)

    def one_of(
        validator: Any, schemas: Sequence[object], instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        accepted = 0
        for branch in schemas:
            check_compute_cancelled()
            if validator.evolve(schema=branch).is_valid(instance):
                accepted += 1
                if accepted > 1:
                    break
        if accepted != 1:
            yield ValidationError("oneOf requires exactly one accepting branch", instance=instance, schema=schema_node)

    def additional_properties(
        validator: Any, additional: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "object") or not isinstance(instance, Mapping):
            return
        properties = schema_node.get("properties", {}) if isinstance(schema_node, Mapping) else {}
        patterns = schema_node.get("patternProperties", {}) if isinstance(schema_node, Mapping) else {}
        compiled = [(re.compile(pattern), subschema) for pattern, subschema in patterns.items()]
        if additional is True and not compiled:
            return
        for key in instance:
            check_compute_cancelled()
            if key in properties:
                continue
            matching = [subschema for pattern, subschema in compiled if pattern.search(key)]
            if matching:
                continue
            if additional is False:
                error = ValidationError("additional property is not allowed", instance=instance, schema=schema_node)
                error.path.append(key)
                yield error
            elif additional is not True:
                yield from validator.descend(instance[key], additional, path=key)

    def unique_items(
        validator: Any, enabled: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if enabled is not True or not validator.is_type(instance, "array") or not isinstance(instance, Sequence):
            return
        connection = _active_validation_connection()
        scope = _new_scope(connection)
        try:
            for index, item in enumerate(instance):
                check_compute_cancelled()
                digest = _json_equality_digest(item)
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO retained_unique(scope,digest) VALUES (?,?)", (scope, digest)
                )
                if cursor.rowcount == 0:
                    error = ValidationError("array items are not unique", instance=instance, schema=schema_node)
                    error.path.append(index)
                    yield error
                    break
        finally:
            connection.execute("DELETE FROM retained_unique WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    def unevaluated_properties(
        validator: Any, unevaluated: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "object") or not isinstance(instance, Mapping):
            return
        connection = _active_validation_connection()
        scope = _new_scope(connection)
        try:
            for key in _evaluated_property_keys(validator, instance, schema_node):
                connection.execute("INSERT OR IGNORE INTO retained_eval_props VALUES (?,?)", (scope, key))
            for key, value in instance.items():
                check_compute_cancelled()
                found = connection.execute(
                    "SELECT 1 FROM retained_eval_props WHERE scope=? AND property=?", (scope, key)
                ).fetchone()
                if found is None:
                    yield from validator.descend(value, unevaluated, path=key)
        finally:
            connection.execute("DELETE FROM retained_eval_props WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    def unevaluated_items(
        validator: Any, unevaluated: object, instance: object, schema_node: object
    ) -> Iterator[ValidationError]:
        if not validator.is_type(instance, "array") or not isinstance(instance, Sequence):
            return
        connection = _active_validation_connection()
        scope = _new_scope(connection)
        try:
            for index in _evaluated_item_indexes(validator, instance, schema_node):
                connection.execute("INSERT OR IGNORE INTO retained_eval_items VALUES (?,?)", (scope, index))
            for index, value in enumerate(instance):
                check_compute_cancelled()
                found = connection.execute(
                    "SELECT 1 FROM retained_eval_items WHERE scope=? AND item_index=?", (scope, index)
                ).fetchone()
                if found is None:
                    yield from validator.descend(value, unevaluated, path=index)
        finally:
            connection.execute("DELETE FROM retained_eval_items WHERE scope=?", (scope,))
            connection.execute("DELETE FROM retained_scope WHERE id=?", (scope,))

    _BOUNDED_VALIDATOR_CLASS = validators.extend(
        Draft202012Validator,
        validators={
            "anyOf": any_of,
            "oneOf": one_of,
            "additionalProperties": additional_properties,
            "uniqueItems": unique_items,
            "unevaluatedProperties": unevaluated_properties,
            "unevaluatedItems": unevaluated_items,
        },
    )
    return _ConnectionBoundValidator(_BOUNDED_VALIDATOR_CLASS(schema), connection, scratch)


def _ensure_reducer_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS retained_scope(id INTEGER PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_unique(scope INTEGER NOT NULL,digest BLOB NOT NULL,PRIMARY KEY(scope,digest)) WITHOUT ROWID"
    )
    drift_created = (
        connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='retained_drift'").fetchone()
        is None
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_drift(sample INTEGER NOT NULL,path INTEGER NOT NULL,digest BLOB NOT NULL,"
        "PRIMARY KEY(sample,path)) WITHOUT ROWID"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS retained_drift_digest ON retained_drift(sample,digest)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_drift_chunks(sample INTEGER NOT NULL,path INTEGER NOT NULL,ordinal INTEGER NOT NULL,"
        "data BLOB NOT NULL,PRIMARY KEY(sample,path,ordinal)) WITHOUT ROWID"
    )
    if drift_created:
        connection.create_collation(
            "retained_path_order", lambda left, right: _compare_stored_drift_paths(connection, left, right)
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_unread(sample INTEGER NOT NULL,path TEXT NOT NULL,PRIMARY KEY(sample,path)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_eval_props(scope INTEGER NOT NULL,property TEXT NOT NULL,PRIMARY KEY(scope,property)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retained_eval_items(scope INTEGER NOT NULL,item_index INTEGER NOT NULL,PRIMARY KEY(scope,item_index)) WITHOUT ROWID"
    )


def _new_scope(connection: sqlite3.Connection) -> int:
    lastrowid = connection.execute("INSERT INTO retained_scope DEFAULT VALUES").lastrowid
    assert lastrowid is not None
    return int(lastrowid)


def _evaluated_property_keys(validator: Any, instance: Mapping[str, object], schema: object) -> Iterator[str]:
    if validator.is_type(schema, "boolean") or not isinstance(schema, Mapping):
        return
    for ref_key in ("$ref", "$dynamicRef"):
        ref = schema.get(ref_key)
        if ref is not None:
            resolved = validator._resolver.lookup(ref)
            yield from _evaluated_property_keys(
                validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents
            )
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for key in properties:
            check_compute_cancelled()
            if key in instance:
                yield key
    for keyword in ("additionalProperties", "unevaluatedProperties"):
        sub = schema.get(keyword)
        if sub is None:
            continue
        for key, value in instance.items():
            check_compute_cancelled()
            if next(validator.descend(value, sub), None) is None:
                yield key
    patterns = schema.get("patternProperties")
    if isinstance(patterns, Mapping):
        compiled = tuple(re.compile(pattern) for pattern in patterns if isinstance(pattern, str))
        for key in instance:
            check_compute_cancelled()
            if any(pattern.search(key) for pattern in compiled):
                yield key
    dependent = schema.get("dependentSchemas")
    if isinstance(dependent, Mapping):
        for key, sub in dependent.items():
            check_compute_cancelled()
            if key in instance:
                yield from _evaluated_property_keys(validator, instance, sub)
    for keyword in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(keyword, ()):
            check_compute_cancelled()
            evolved = validator.evolve(schema=sub)
            if evolved.is_valid(instance):
                yield from _evaluated_property_keys(evolved, instance, sub)
    condition = schema.get("if")
    if condition is not None:
        if validator.evolve(schema=condition).is_valid(instance):
            yield from _evaluated_property_keys(validator, instance, condition)
            if "then" in schema:
                yield from _evaluated_property_keys(validator, instance, schema["then"])
        elif "else" in schema:
            yield from _evaluated_property_keys(validator, instance, schema["else"])


def _evaluated_item_indexes(validator: Any, instance: Sequence[object], schema: object) -> Iterator[int]:
    if validator.is_type(schema, "boolean") or not isinstance(schema, Mapping):
        return
    if "items" in schema:
        yield from range(len(instance))
        return
    for ref_key in ("$ref", "$dynamicRef"):
        ref = schema.get(ref_key)
        if ref is not None:
            resolved = validator._resolver.lookup(ref)
            yield from _evaluated_item_indexes(
                validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents
            )
    prefix = schema.get("prefixItems")
    if isinstance(prefix, Sequence) and not isinstance(prefix, (str, bytes)):
        for index in range(min(len(prefix), len(instance))):
            check_compute_cancelled()
            yield index
    condition = schema.get("if")
    if condition is not None:
        if validator.evolve(schema=condition).is_valid(instance):
            yield from _evaluated_item_indexes(validator, instance, condition)
            if "then" in schema:
                yield from _evaluated_item_indexes(validator, instance, schema["then"])
        elif "else" in schema:
            yield from _evaluated_item_indexes(validator, instance, schema["else"])
    for keyword in ("contains", "unevaluatedItems"):
        sub = schema.get(keyword)
        if sub is not None:
            for index, value in enumerate(instance):
                check_compute_cancelled()
                if validator.evolve(schema=sub).is_valid(value):
                    yield index
    for keyword in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(keyword, ()):
            check_compute_cancelled()
            evolved = validator.evolve(schema=sub)
            if evolved.is_valid(instance):
                yield from _evaluated_item_indexes(evolved, instance, sub)


def _json_equality_digest(value: object) -> bytes:
    digest = hashlib.sha256()

    def visit(node: object) -> None:
        check_compute_cancelled()
        if node is None:
            digest.update(b"null\0")
        elif isinstance(node, bool):
            digest.update(b"bool\1" if node else b"bool\0")
        elif isinstance(node, (int, float, Decimal)):
            digest.update(b"number\0")
            number = Decimal(str(node)).normalize()
            digest.update(str(number).encode("ascii"))
            digest.update(b"\0")
        elif isinstance(node, str):
            digest.update(b"string\0")
            data = node.encode("utf-8", "surrogatepass")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        elif isinstance(node, Mapping):
            digest.update(b"object\0")
            keys = node.sorted_keys() if hasattr(node, "sorted_keys") else iter(sorted(node))
            for key in keys:
                check_compute_cancelled()
                visit(str(key))
                visit(node[key])
            digest.update(b"end-object\0")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            digest.update(b"array\0")
            for item in node:
                check_compute_cancelled()
                visit(item)
            digest.update(b"end-array\0")
        else:
            digest.update(type(node).__qualname__.encode("utf-8"))

    visit(value)
    return digest.digest()


def _stored_drift_chunks(connection: sqlite3.Connection, sample: int, path: int) -> Generator[bytes, None, None]:
    from polylogue.schemas.observation_spill import _read_rows

    with closing(
        _read_rows(
            connection,
            "SELECT data FROM retained_drift_chunks WHERE sample=? AND path=? ORDER BY ordinal",
            (sample, path),
        )
    ) as rows:
        for (data,) in rows:
            check_compute_cancelled()
            yield bytes(data)


def _compare_stored_drift_paths(connection: sqlite3.Connection, left: str, right: str) -> int:
    from polylogue.schemas.observation_spill import _compare_key_chunks

    left_sample, left_path = (int(part) for part in left.split(":"))
    right_sample, right_path = (int(part) for part in right.split(":"))
    return _compare_key_chunks(
        _stored_drift_chunks(connection, left_sample, left_path),
        _stored_drift_chunks(connection, right_sample, right_path),
    )


def _clear_sample_drift(connection: sqlite3.Connection, sample: int) -> None:
    connection.execute("DELETE FROM retained_drift_chunks WHERE sample=?", (sample,))
    connection.execute("DELETE FROM retained_drift WHERE sample=?", (sample,))


def _collect_drift_paths(
    sample: Mapping[str, object],
    schema: Mapping[str, object],
    connection: sqlite3.Connection,
    sample_index: int,
) -> int:
    from polylogue.schemas.observation_spill import _compare_key_chunks, _read_rows
    from polylogue.schemas.validator import _DriftPath, _iter_drift_key_paths

    if (active := _ACTIVE_VALIDATION_SCRATCH.get()) is None or active.connection is not connection:
        _ensure_reducer_tables(connection)
    count = 0
    _clear_sample_drift(connection, sample_index)
    with closing(_iter_drift_key_paths(sample, schema, _DriftPath(), schema, connection)) as paths:
        for path in paths:
            check_compute_cancelled()
            count += 1
            digest = hashlib.sha256()
            with closing(path.iter_utf8_chunks()) as chunks:
                for chunk in chunks:
                    digest.update(chunk)
            duplicate = False
            with closing(
                _read_rows(
                    connection,
                    "SELECT path FROM retained_drift WHERE sample=? AND digest=?",
                    (sample_index, digest.digest()),
                )
            ) as candidates:
                for (candidate,) in candidates:
                    if (
                        _compare_key_chunks(
                            path.iter_utf8_chunks(), _stored_drift_chunks(connection, sample_index, int(candidate))
                        )
                        == 0
                    ):
                        duplicate = True
                        break
            if duplicate:
                continue
            connection.execute("INSERT INTO retained_drift VALUES (?,?,?)", (sample_index, count, digest.digest()))
            pending = bytearray()
            ordinal = 0
            with closing(path.iter_utf8_chunks()) as chunks:
                for chunk in chunks:
                    for offset in range(0, len(chunk), 4096):
                        pending.extend(chunk[offset : offset + 4096])
                        while len(pending) >= 4096:
                            connection.execute(
                                "INSERT INTO retained_drift_chunks VALUES (?,?,?,?)",
                                (sample_index, count, ordinal, bytes(pending[:4096])),
                            )
                            del pending[:4096]
                            ordinal += 1
                if pending:
                    connection.execute(
                        "INSERT INTO retained_drift_chunks VALUES (?,?,?,?)",
                        (sample_index, count, ordinal, bytes(pending)),
                    )
    return count


def _reduce_sample_drift(
    sample: Mapping[str, object],
    schema: Mapping[str, object],
    provider: Provider,
    resolution: SchemaResolution | None,
    connection: sqlite3.Connection,
    sample_index: int,
    *,
    raw_id: str,
    native_id_example: str,
    is_valid: bool,
    signature_directory: Path,
) -> SchemaDriftObservation | None:
    if resolution is None:
        return None
    path_count = int(
        connection.execute("SELECT COUNT(*) FROM retained_drift WHERE sample=?", (sample_index,)).fetchone()[0]
    )
    unread_count = 0
    if not path_count and is_valid:
        unread_names = unread_field_names(normalize_provider_token(str(provider)))
        connection.execute("DELETE FROM retained_unread WHERE sample=?", (sample_index,))
        from polylogue.schemas.observation_spill import SpilledObject

        keys = (
            sample.matching_field_names(unread_names)
            if isinstance(sample, SpilledObject)
            else (key for key in sample if key in unread_names)
        )
        for key in keys:
            check_compute_cancelled()
            connection.execute("INSERT OR IGNORE INTO retained_unread VALUES (?,?)", (sample_index, key))
        unread_count = int(
            connection.execute("SELECT COUNT(*) FROM retained_unread WHERE sample=?", (sample_index,)).fetchone()[0]
        )
    classification: DriftClassification | None
    if not is_valid:
        classification = FIELD_CHANGED
    elif resolution.reason == "package_default":
        classification = UNSEEN_SHAPE
    elif path_count:
        classification = NEW_FIELD
    elif unread_count:
        classification = KNOWN_FIELD_UNREAD
    else:
        classification = None
    if classification is None:
        return None
    table = "retained_unread" if classification == KNOWN_FIELD_UNREAD else "retained_drift"

    def signature_chunks() -> Generator[bytes, None, None]:
        from polylogue.schemas.observation_spill import _read_rows

        order = (
            "(CAST(sample AS TEXT)||':'||CAST(path AS TEXT)) COLLATE retained_path_order"
            if table == "retained_drift"
            else "path"
        )
        with closing(
            _read_rows(connection, f"SELECT path FROM {table} WHERE sample=? ORDER BY {order}", (sample_index,))
        ) as rows:
            first = True
            for (path,) in rows:
                check_compute_cancelled()
                if not first:
                    yield b","
                first = False
                if table == "retained_drift":
                    with closing(_stored_drift_chunks(connection, sample_index, int(path))) as chunks:
                        yield from chunks
                else:
                    yield path.encode("utf-8", "surrogatepass")

    signature = DriftSignature.from_utf8_chunks(signature_chunks(), directory=signature_directory)
    return SchemaDriftObservation(
        origin=str(origin_from_provider(provider)),
        element_kind=resolution.element_kind,
        classification=classification,
        unseen_key_signature=signature,
        native_id_example=native_id_example,
        raw_id=raw_id,
    )


def _stronger_drift(
    current: SchemaDriftObservation | None,
    candidate: SchemaDriftObservation | None,
) -> SchemaDriftObservation | None:
    if current is None:
        return candidate
    if candidate is None:
        return current
    left = _DRIFT_STRENGTH[current.classification]
    right = _DRIFT_STRENGTH[candidate.classification]
    if right != left:
        return candidate if right > left else current
    return candidate if candidate.unseen_key_signature.compare(current.unseen_key_signature) < 0 else current
