"""Timeline-oriented derived insight storage models."""

from __future__ import annotations

from pydantic import BaseModel

from polylogue.analysis.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.core.types import SessionId
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION


class SessionRunRecord(BaseModel):
    """Read-model row wrapping one :class:`ProjectedRun`.

    polylogue-dab/itvd: rows are source-derived on every read by
    ``run_projection_relations.py``'s CTE (computed from ``sessions`` and
    ``blocks``), not a materialized ``session_runs`` table. ``ProjectedRun``
    is hydrated directly from the CTE's typed columns, not from a stored
    ``payload_json`` blob.
    """

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    run: ProjectedRun
    search_text: str


class SessionObservedEventRecord(BaseModel):
    """Read-model row wrapping one :class:`ObservedEvent`, source-derived on every read (see :class:`SessionRunRecord`)."""

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    event: ObservedEvent
    search_text: str


class SessionContextSnapshotRecord(BaseModel):
    """Read-model row wrapping one :class:`ContextSnapshot`, source-derived on every read (see :class:`SessionRunRecord`)."""

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    snapshot: ContextSnapshot
    search_text: str


__all__ = [
    "SessionContextSnapshotRecord",
    "SessionObservedEventRecord",
    "SessionRunRecord",
]
