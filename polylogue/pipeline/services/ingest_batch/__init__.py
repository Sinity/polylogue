"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Split from a single 1326-line module into a package: _models (SQL + types)
and _core (write operations + worker pool + result pipeline). Internal
helpers are re-exported for test access.
"""

from polylogue.pipeline.services.ingest_batch._core import (
    _build_batch_memory_observation,
    _drain_ready_conversation_entries,
    _failed_raw_state_update,
    _iter_ingest_results_sync,
    _persist_batch_raw_state_updates,
    _process_ingest_batch_sync,
    _select_ingest_worker_count,
    _successful_raw_state_update,
    _topo_sort_conversation_entries,
    _unattributed_batch_elapsed_s,
    _write_conversation,
    process_ingest_batch,
    refresh_session_insights_bulk,
)
from polylogue.pipeline.services.ingest_batch._models import (
    _ConversationEntry,
    _IngestBatchSummary,
    _IngestWorkerRequest,
    _RawIngestOutcome,
)

__all__ = [
    "_ConversationEntry",
    "_IngestBatchSummary",
    "_IngestWorkerRequest",
    "_RawIngestOutcome",
    "_build_batch_memory_observation",
    "_drain_ready_conversation_entries",
    "_failed_raw_state_update",
    "_iter_ingest_results_sync",
    "_persist_batch_raw_state_updates",
    "_process_ingest_batch_sync",
    "_select_ingest_worker_count",
    "_successful_raw_state_update",
    "_topo_sort_conversation_entries",
    "_unattributed_batch_elapsed_s",
    "_write_conversation",
    "process_ingest_batch",
    "refresh_session_insights_bulk",
]
