"""Stable storage record-model surface."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from pydantic import BaseModel

from polylogue.lib.json import dumps as json_dumps
from polylogue.storage.products.aggregate.records import (
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
)
from polylogue.storage.products.session.records import SessionProfileRecord, WorkThreadRecord
from polylogue.storage.products.timeline.records import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.runtime.action.records import ActionEventRecord
from polylogue.storage.runtime.archive.records import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
)
from polylogue.storage.runtime.raw.records import (
    ArtifactObservationRecord,
    RawConversationRecord,
)
from polylogue.storage.runtime.store_constants import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
)
from polylogue.types import AttachmentId, ConversationId, MessageId


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


def _make_ref_id(attachment_id: AttachmentId, conversation_id: ConversationId, message_id: MessageId | None) -> str:
    seed = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ref-{digest}"


__all__ = [
    "ACTION_EVENT_MATERIALIZER_VERSION",
    "ActionEventRecord",
    "AttachmentRecord",
    "ArtifactObservationRecord",
    "ContentBlockRecord",
    "ConversationRecord",
    "DaySessionSummaryRecord",
    "MessageRecord",
    "PublicationRecord",
    "RawConversationRecord",
    "RunRecord",
    "SESSION_ENRICHMENT_FAMILY",
    "SESSION_ENRICHMENT_VERSION",
    "SESSION_INFERENCE_FAMILY",
    "SESSION_INFERENCE_VERSION",
    "SESSION_PRODUCT_MATERIALIZER_VERSION",
    "SessionPhaseRecord",
    "SessionProfileRecord",
    "SessionTagRollupRecord",
    "SessionWorkEventRecord",
    "WorkThreadRecord",
    "_json_array_or_none",
    "_json_or_none",
    "_make_ref_id",
]
