"""Format-drift classification for the live-ingest sentinel (polylogue-da1).

Providers change their export shapes without notice. Today drift surfaces
as silent parse degradation, discovered manually weeks later. This module
turns the per-record schema resolution + validation signal that ingest
already computes (`polylogue.schemas.runtime_registry.SchemaRegistry`,
`polylogue.schemas.validator`) into one of a small set of drift
classifications, so a health check / status line can report *rates* instead
of an operator having to notice a silent regression.

Two axes of "not an exact known-shape match":

- ``UNSEEN_SHAPE`` -- the schema registry had no real candidate at all and
  fell back to ``package_default`` (see ``SchemaResolutionReason``). This is
  the strongest signal: the payload's structure did not resemble any
  committed package for the provider.
- ``NEW_FIELD`` -- the payload validated successfully against a resolved
  schema, but carries at least one field the schema does not declare
  (``ValidationResult.drift_warnings``/``detect_drift``). This is common and
  usually benign (providers add optional fields constantly).
- ``FIELD_CHANGED`` -- the payload *failed* JSON Schema validation against
  the resolved schema (a required field disappeared, or a field's type no
  longer matches). This is the risky case: a parser assumption is now
  false and something may be silently mis-parsed rather than merely
  ignored.

A payload that resolves to a real candidate, validates cleanly, and has no
unknown fields classifies as no drift at all (``None`` -- nothing to
record).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

DriftClassification: TypeAlias = Literal["unseen_shape", "new_field", "field_changed"]

UNSEEN_SHAPE: DriftClassification = "unseen_shape"
NEW_FIELD: DriftClassification = "new_field"
FIELD_CHANGED: DriftClassification = "field_changed"

# Reason value SchemaRegistry.resolve_payload falls back to when no real
# candidate (exact structure / bundle scope / profile family) matched.
_UNSEEN_SHAPE_RESOLUTION_REASON = "package_default"

# Risky classifications outrank benign ones for alerting purposes -- a
# rate dominated by field_changed/unseen_shape should never read as "fine"
# just because new_field volume swamps it.
RISKY_CLASSIFICATIONS: frozenset[DriftClassification] = frozenset({FIELD_CHANGED, UNSEEN_SHAPE})
BENIGN_CLASSIFICATIONS: frozenset[DriftClassification] = frozenset({NEW_FIELD})


@dataclass(frozen=True, slots=True)
class SchemaDriftObservation:
    """One classified drift signal for a single ingested raw record."""

    origin: str
    element_kind: str
    classification: DriftClassification
    unseen_key_signature: str
    native_id_example: str
    raw_id: str


def classify_schema_drift(
    *,
    resolution_reason: str | None,
    is_valid: bool,
    drift_warnings: Sequence[str],
) -> DriftClassification | None:
    """Classify one validated payload sample's drift signal.

    Precedence: a JSON-Schema validation failure (``is_valid=False``) is
    always ``field_changed`` regardless of the resolution reason -- a
    payload can simultaneously be an "unseen shape" *and* fail validation,
    but the failure is the more actionable, riskier signal. Next, a
    fallback-to-default resolution (no real package/profile/bundle match)
    is ``unseen_shape``. Finally, a clean validation with unknown-but-
    permitted fields is the benign ``new_field`` case. Returns ``None``
    when there is no drift signal at all.
    """
    if not is_valid:
        return FIELD_CHANGED
    if resolution_reason == _UNSEEN_SHAPE_RESOLUTION_REASON:
        return UNSEEN_SHAPE
    if drift_warnings:
        return NEW_FIELD
    return None


def unseen_key_signature(drift_warnings: Sequence[str]) -> str:
    """Build a stable, order-independent signature for a set of drift warnings.

    ``detect_drift`` warnings look like ``"Unexpected field: <path>"``; the
    signature strips that fixed prefix and joins the sorted, deduplicated
    set of paths so identical drift shapes (independent of dict iteration
    order) group under the same ``unseen_key_signature`` key.
    """
    paths = sorted({_warning_path(warning) for warning in drift_warnings})
    return ",".join(paths)


def _warning_path(warning: str) -> str:
    prefix = "Unexpected field: "
    return warning[len(prefix) :] if warning.startswith(prefix) else warning


def is_risky(classification: DriftClassification) -> bool:
    return classification in RISKY_CLASSIFICATIONS


__all__ = [
    "BENIGN_CLASSIFICATIONS",
    "FIELD_CHANGED",
    "NEW_FIELD",
    "RISKY_CLASSIFICATIONS",
    "UNSEEN_SHAPE",
    "DriftClassification",
    "SchemaDriftObservation",
    "classify_schema_drift",
    "is_risky",
    "unseen_key_signature",
]
