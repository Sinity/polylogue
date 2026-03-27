"""Stable storage record-model surface."""

from __future__ import annotations

from polylogue.storage.store_constants import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    MAX_ATTACHMENT_SIZE,
    MAX_RAW_CONTENT_SIZE,
    SESSION_ENRICHMENT_FAMILY,
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_PRODUCT_MATERIALIZER_VERSION,
)
from polylogue.storage.store_product_aggregate_records import (
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
)
from polylogue.storage.store_product_session_records import SessionProfileRecord, WorkThreadRecord
from polylogue.storage.store_product_timeline_records import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.store_runtime_action_records import ActionEventRecord
from polylogue.storage.store_runtime_archive_records import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    PublicationRecord,
    RunRecord,
)
from polylogue.storage.store_runtime_raw_records import (
    ArtifactObservationRecord,
    RawConversationRecord,
)
from polylogue.storage.store_support import (
    _json_array_or_none,
    _json_or_none,
    _make_ref_id,
)

__all__ = [
    "ACTION_EVENT_MATERIALIZER_VERSION",
    "ActionEventRecord",
    "AttachmentRecord",
    "ArtifactObservationRecord",
    "ContentBlockRecord",
    "ConversationRecord",
    "DaySessionSummaryRecord",
    "MAX_ATTACHMENT_SIZE",
    "MAX_RAW_CONTENT_SIZE",
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
