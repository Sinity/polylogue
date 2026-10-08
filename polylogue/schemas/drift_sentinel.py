"""Format-drift vocabulary and observation payload for retained validation.

The retained validator classifies complete records in precedence order:
validation failure, unresolved shape, new field, then schema-known unread
field. It emits deterministic field signatures in ``SchemaDriftObservation``;
the ops sampler and status readers share this vocabulary. New fields are
benign, while the other classifications are risky. No drift yields no
observation, and telemetry never chooses source identity or gates publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

DriftClassification: TypeAlias = Literal["unseen_shape", "new_field", "field_changed", "known_field_unread"]

UNSEEN_SHAPE: DriftClassification = "unseen_shape"
NEW_FIELD: DriftClassification = "new_field"
FIELD_CHANGED: DriftClassification = "field_changed"
KNOWN_FIELD_UNREAD: DriftClassification = "known_field_unread"

# Risky classifications outrank benign ones for alerting purposes -- a
# rate dominated by field_changed/unseen_shape/known_field_unread should
# never read as "fine" just because new_field volume swamps it.
# known_field_unread is risky: it is not "the schema doesn't know about
# this yet" (the other three classifications' shared axis) but "the schema
# knows, and the parser silently drops it anyway" -- a defect, not drift.
RISKY_CLASSIFICATIONS: frozenset[DriftClassification] = frozenset({FIELD_CHANGED, UNSEEN_SHAPE, KNOWN_FIELD_UNREAD})
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


def is_risky(classification: DriftClassification) -> bool:
    return classification in RISKY_CLASSIFICATIONS


__all__ = [
    "BENIGN_CLASSIFICATIONS",
    "FIELD_CHANGED",
    "KNOWN_FIELD_UNREAD",
    "NEW_FIELD",
    "RISKY_CLASSIFICATIONS",
    "UNSEEN_SHAPE",
    "DriftClassification",
    "SchemaDriftObservation",
    "is_risky",
]
