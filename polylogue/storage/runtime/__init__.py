"""Stable storage record-model surface."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from polylogue.core.json import dumps as json_dumps
from polylogue.storage.derived.aggregate.records import (
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
)
from polylogue.storage.derived.session.records import (
    SessionLatencyProfileRecord,
    SessionProfileRecord,
    ThreadRecord,
)
from polylogue.storage.runtime.archive.records import (
    LINEAGE_TRUNCATION_CYCLE,
    LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT,
    AttachmentRecord,
    BlockRecord,
    FileEditRecord,
    LineageCompleteness,
    LineageTruncationReason,
    MessageRecord,
    SessionCommitRecord,
    SessionEventRecord,
    SessionRecord,
    SessionRefRecord,
    WebContentConstructRecord,
)
from polylogue.storage.runtime.raw.records import (
    ArtifactObservationRecord,
    RawSessionRecord,
)
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_EVENT_MATERIALIZER_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)


def _json_or_none(value: BaseModel | Mapping[str, object] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return json_dumps(value.model_dump(mode="json"))
    return json_dumps(dict(value))


def _json_array_or_none(value: tuple[str, ...] | list[str] | None) -> str | None:
    if not value:
        return None
    return json_dumps(list(value))


__all__ = [
    "LINEAGE_TRUNCATION_CYCLE",
    "AttachmentRecord",
    "ArtifactObservationRecord",
    "BlockRecord",
    "FileEditRecord",
    "SessionRecord",
    "SessionRefRecord",
    "SessionCommitRecord",
    "WebContentConstructRecord",
    "DaySessionSummaryRecord",
    "LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT",
    "LineageCompleteness",
    "LineageTruncationReason",
    "MessageRecord",
    "SESSION_EVENT_MATERIALIZER_VERSION",
    "SessionEventRecord",
    "RawSessionRecord",
    "SESSION_ENRICHMENT_FAMILY",
    "SESSION_ENRICHMENT_VERSION",
    "SESSION_INFERENCE_FAMILY",
    "SESSION_INFERENCE_VERSION",
    "SESSION_INSIGHT_MATERIALIZER_VERSION",
    "SessionLatencyProfileRecord",
    "SessionProfileRecord",
    "SessionTagRollupRecord",
    "ThreadRecord",
    "_json_array_or_none",
    "_json_or_none",
]
