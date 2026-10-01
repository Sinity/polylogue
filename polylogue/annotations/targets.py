"""Annotation provenance vocabulary, independent of live target operations."""

from __future__ import annotations

from typing import Literal, get_args

from polylogue.core.refs import ObjectRefKind, normalize_object_ref_text

AnnotationTargetKind = ObjectRefKind | Literal["phase", "work_event"]
RETIRED_ANNOTATION_TARGET_KINDS = frozenset({"phase", "work_event"})
ANNOTATION_TARGET_KINDS = frozenset(get_args(ObjectRefKind)) | RETIRED_ANNOTATION_TARGET_KINDS


def normalize_annotation_target_ref(value: str) -> str:
    """Decode a provenance target without authorizing its live resolution.

    The two retired grains retain the opaque colon-form identity they had when
    written. New target operations still use the live ObjectRef boundary.
    """
    kind, separator, identity = value.partition(":")
    if kind in RETIRED_ANNOTATION_TARGET_KINDS:
        if not separator or not identity or identity.endswith(":"):
            raise ValueError("annotation provenance target must use 'kind:id' form")
        return value
    return normalize_object_ref_text(value)
