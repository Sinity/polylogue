"""Immutable provenance container for one annotation import batch.

Rows remain ordinary assertions. A batch supplies the durable source-result,
actor/model/prompt, validation, and count provenance shared by those rows and
is addressed through the existing ``annotation-batch:<id>`` ObjectRef kind.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from polylogue.core.json import JSONDocument, require_json_document
from polylogue.core.refs import ObjectRef, normalize_object_ref_text

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SCHEMA_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
ANNOTATION_BATCH_PROVENANCE_FORMAT = "polylogue.annotation-batch/v1"


class AnnotationBatchError(ValueError):
    """Raised when batch provenance is malformed or conflicts with durable state."""


def _normalized_ref(value: str, *, label: str, expected_kind: str | None = None) -> str:
    try:
        normalized = normalize_object_ref_text(value)
        parsed = ObjectRef.parse(normalized)
    except ValueError as exc:
        raise AnnotationBatchError(f"{label} must be a valid ObjectRef") from exc
    if expected_kind is not None and parsed.kind != expected_kind:
        raise AnnotationBatchError(f"{label} must use the {expected_kind!r} ObjectRef kind")
    return normalized


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nfc_json_value(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_nfc_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AnnotationBatchError("canonical JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise AnnotationBatchError(f"NFC-normalized JSON keys collide at {normalized_key!r}")
            normalized[normalized_key] = _nfc_json_value(item)
        return normalized
    return value


def _document_copy(value: object, *, context: str) -> JSONDocument:
    if not isinstance(value, dict):
        raise AnnotationBatchError(f"{context} must be a finite JSON object")
    try:
        canonical = _canonical_json_text(value, context=context)
        return require_json_document(json.loads(canonical), context=context)
    except AnnotationBatchError:
        raise
    except (TypeError, ValueError) as exc:
        raise AnnotationBatchError(f"{context} must be a finite JSON object") from exc


def _canonical_json_text(value: object, *, context: str) -> str:
    try:
        return json.dumps(
            _nfc_json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except AnnotationBatchError:
        raise
    except (TypeError, ValueError) as exc:
        raise AnnotationBatchError(f"{context} must be finite canonical JSON") from exc


def _canonical_json_bytes(value: object, *, context: str) -> bytes:
    try:
        return _canonical_json_text(value, context=context).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AnnotationBatchError(f"{context} must be valid UTF-8") from exc


@dataclass(frozen=True, slots=True)
class AnnotationBatch:
    """Complete, write-once provenance and accounting for one label batch."""

    batch_id: str
    schema_id: str
    schema_version: int
    target_ref: str
    source_result_ref: str
    actor_ref: str
    model_ref: str
    prompt_ref: str
    total_count: int
    valid_count: int
    invalid_count: int
    abstained_count: int
    assertion_refs: tuple[str, ...] = ()
    validation_failures: tuple[JSONDocument, ...] = ()
    metadata: JSONDocument = field(default_factory=dict)
    created_at_ms: int = 0
    _canonical_provenance: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _BATCH_ID_RE.fullmatch(self.batch_id):
            raise AnnotationBatchError(f"batch_id {self.batch_id!r} must match {_BATCH_ID_RE.pattern!r}")
        if not _SCHEMA_ID_RE.fullmatch(self.schema_id):
            raise AnnotationBatchError(f"schema_id {self.schema_id!r} must match {_SCHEMA_ID_RE.pattern!r}")
        if not _is_positive_int(self.schema_version):
            raise AnnotationBatchError("schema_version must be >= 1")
        if not _is_nonnegative_int(self.created_at_ms):
            raise AnnotationBatchError("created_at_ms cannot be negative")
        for name, count in (
            ("total_count", self.total_count),
            ("valid_count", self.valid_count),
            ("invalid_count", self.invalid_count),
            ("abstained_count", self.abstained_count),
        ):
            if not _is_nonnegative_int(count):
                raise AnnotationBatchError(f"{name} must be a non-negative integer")
        if self.valid_count + self.invalid_count != self.total_count:
            raise AnnotationBatchError("valid_count + invalid_count must equal total_count")
        if self.abstained_count > self.valid_count:
            raise AnnotationBatchError("abstained_count cannot exceed valid_count")
        if len(self.assertion_refs) != self.valid_count:
            raise AnnotationBatchError("assertion_refs count must equal valid_count")
        if len(self.validation_failures) != self.invalid_count:
            raise AnnotationBatchError("validation_failures count must equal invalid_count")

        object.__setattr__(self, "target_ref", _normalized_ref(self.target_ref, label="target_ref"))
        object.__setattr__(
            self,
            "source_result_ref",
            _normalized_ref(self.source_result_ref, label="source_result_ref", expected_kind="result-set"),
        )
        object.__setattr__(self, "actor_ref", _normalized_ref(self.actor_ref, label="actor_ref"))
        object.__setattr__(self, "model_ref", _normalized_ref(self.model_ref, label="model_ref"))
        object.__setattr__(self, "prompt_ref", _normalized_ref(self.prompt_ref, label="prompt_ref"))
        normalized_assertion_refs = tuple(
            _normalized_ref(ref, label="assertion_ref", expected_kind="assertion") for ref in self.assertion_refs
        )
        if len(set(normalized_assertion_refs)) != len(normalized_assertion_refs):
            raise AnnotationBatchError("assertion_refs must be unique after normalization")
        object.__setattr__(self, "assertion_refs", normalized_assertion_refs)
        object.__setattr__(
            self,
            "validation_failures",
            tuple(
                _document_copy(item, context="annotation batch validation failure") for item in self.validation_failures
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _document_copy(self.metadata, context="annotation batch metadata"),
        )
        object.__setattr__(
            self,
            "_canonical_provenance",
            _canonical_json_bytes(
                self._provenance_document_from_fields(),
                context="annotation batch provenance",
            ),
        )

    @property
    def qualified_schema_id(self) -> str:
        return f"{self.schema_id}@v{self.schema_version}"

    @property
    def batch_ref(self) -> str:
        return ObjectRef(kind="annotation-batch", object_id=self.batch_id).format()

    def _provenance_document_from_fields(self) -> JSONDocument:
        return {
            "abstained_count": self.abstained_count,
            "actor_ref": self.actor_ref,
            "assertion_refs": list(self.assertion_refs),
            "batch_id": self.batch_id,
            "created_at_ms": self.created_at_ms,
            "format": ANNOTATION_BATCH_PROVENANCE_FORMAT,
            "invalid_count": self.invalid_count,
            "metadata": self.metadata,
            "model_ref": self.model_ref,
            "prompt_ref": self.prompt_ref,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "source_result_ref": self.source_result_ref,
            "target_ref": self.target_ref,
            "total_count": self.total_count,
            "valid_count": self.valid_count,
            "validation_failures": list(self.validation_failures),
        }

    def provenance_document(self) -> JSONDocument:
        """Return a detached copy of the immutable versioned provenance."""

        return require_json_document(
            json.loads(self._canonical_provenance),
            context="annotation batch provenance",
        )

    def canonical_provenance_json(self) -> str:
        """Return the byte-stable finite JSON used for exact-retry identity."""

        return self._canonical_provenance.decode("utf-8")

    def canonical_provenance_bytes(self) -> bytes:
        """Return canonical UTF-8 provenance bytes, rejecting invalid Unicode."""

        return self._canonical_provenance


__all__ = ["ANNOTATION_BATCH_PROVENANCE_FORMAT", "AnnotationBatch", "AnnotationBatchError"]
