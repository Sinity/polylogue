"""Batch ingest orchestration: ProcessPool workers + sync sqlite3 writes.

Split from a single 1326-line module into a package: _models (SQL + types)
and _core (write operations + worker pool + result pipeline). Internal
helpers are re-exported for test access.
"""

from polylogue.pipeline.services.ingest_batch._core import (
    process_ingest_batch,
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

__all__ = [
    "_SessionEntry",
    "_IngestBatchSummary",
    "_IngestWorkerRequest",
    "_RawIngestOutcome",
    "_build_batch_memory_observation",
    "_unattributed_batch_elapsed_s",
    "ingest_record",
    "process_ingest_batch",
]
