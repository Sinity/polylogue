"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Split from a single 1326-line module into a package: _models (SQL + types)
and _core (write operations + worker pool + result pipeline). Internal
helpers are re-exported for test access.
"""

from polylogue.pipeline.services.ingest_batch._core import (
    _drain_ready_session_entries,
    _failed_raw_state_update,
    _iter_ingest_results_sync,
    _persist_batch_raw_state_updates,
    _process_ingest_batch_sync,
    _select_ingest_worker_count,
    _successful_raw_state_update,
    _topo_sort_session_entries,
    _write_session,
    process_ingest_batch,
    refresh_session_insights_bulk,
)
from polylogue.pipeline.services.ingest_batch._models import (
    _IngestBatchSummary,
    _IngestWorkerRequest,
    _RawIngestOutcome,
    _SessionEntry,
)
from polylogue.pipeline.services.ingest_batch._observations import (
    _build_batch_memory_observation,
    _unattributed_batch_elapsed_s,
)
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.pipeline.services.process_pool import process_pool_executor
from polylogue.storage.sqlite.connection_profile import WRITE_CONNECTION_PRAGMA_STATEMENTS

__all__ = [
    "_SessionEntry",
    "_IngestBatchSummary",
    "_IngestWorkerRequest",
    "_RawIngestOutcome",
    "_build_batch_memory_observation",
    "_drain_ready_session_entries",
    "_failed_raw_state_update",
    "_iter_ingest_results_sync",
    "_persist_batch_raw_state_updates",
    "_process_ingest_batch_sync",
    "_select_ingest_worker_count",
    "_successful_raw_state_update",
    "_topo_sort_session_entries",
    "_unattributed_batch_elapsed_s",
    "_write_session",
    "ingest_record",
    "process_ingest_batch",
    "process_pool_executor",
    "refresh_session_insights_bulk",
    "WRITE_CONNECTION_PRAGMA_STATEMENTS",
]
