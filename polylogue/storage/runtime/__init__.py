"""Stable storage record-model surface."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from pydantic import BaseModel

from polylogue.core.json import dumps as json_dumps
from polylogue.storage.insights.aggregate.records import (
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
)
from polylogue.storage.insights.session.records import (
    SessionLatencyProfileRecord,
    SessionProfileRecord,
    WorkThreadRecord,
)
from polylogue.storage.insights.timeline.records import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.runtime.action.records import ActionEventRecord
from polylogue.storage.runtime.archive.records import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionRecord,
)
from polylogue.storage.runtime.raw.records import (
    ArtifactObservationRecord,
    RawSessionRecord,
)
from polylogue.storage.runtime.store_constants import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    PROVIDER_EVENT_MATERIALIZER_VERSION,
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)
from polylogue.types import AttachmentId, MessageId, SessionId


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


def _make_ref_id(attachment_id: AttachmentId, session_id: SessionId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{session_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


__all__ = [
    "ACTION_EVENT_MATERIALIZER_VERSION",
    "ActionEventRecord",
    "AttachmentRecord",
    "ArtifactObservationRecord",
    "ContentBlockRecord",
    "SessionRecord",
    "DaySessionSummaryRecord",
    "MessageRecord",
    "PROVIDER_EVENT_MATERIALIZER_VERSION",
    "ProviderEventRecord",
    "RawSessionRecord",
    "SESSION_ENRICHMENT_FAMILY",
    "SESSION_ENRICHMENT_VERSION",
    "SESSION_INFERENCE_FAMILY",
    "SESSION_INFERENCE_VERSION",
    "SESSION_INSIGHT_MATERIALIZER_VERSION",
    "SessionPhaseRecord",
    "SessionLatencyProfileRecord",
    "SessionProfileRecord",
    "SessionTagRollupRecord",
    "SessionWorkEventRecord",
    "WorkThreadRecord",
    "_json_array_or_none",
    "_json_or_none",
    "_make_ref_id",
]
