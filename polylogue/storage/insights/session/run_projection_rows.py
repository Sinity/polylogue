"""Run-projection row builders and hydration helpers.

polylogue-dab/itvd: run/observed-event/context-snapshot rows are no longer
materialized -- ``run_projection_relations.py``'s CTEs compute them on every
read from ``sessions``/``blocks``. ``hydrate_projected_run``/
``hydrate_observed_event``/``hydrate_context_snapshot`` here unwrap the
already-hydrated :class:`polylogue.insights.run_projection.RunProjection`
member models from a :class:`SessionRunRecord`/etc.
"""

from __future__ import annotations

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.core.types import SessionId
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun, RunProjection
from polylogue.storage.insights.session.profiles import now_iso
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION


def _join_search_text(*parts: str | None) -> str:
    return " \n".join(part.strip() for part in parts if part and part.strip())


def _ref_text(ref: ObjectRef | EvidenceRef | None) -> str | None:
    return ref.format() if ref is not None else None


# The ``search_text`` columns back the ``text:`` predicate for each unit. They
# reproduce the field coverage of the removed runtime-transform ``text`` matcher
# (every ref/identifier the runtime haystack searched) so ``text:`` behaviour is
# equivalent on the SQL path, not silently narrowed.
def run_search_text(run: ProjectedRun) -> str:
    # Exact field set of the removed runtime ``text`` matcher for runs (no more,
    # no less): run/parent/agent/context refs, lineage + evidence refs, transcript,
    # provider_origin, harness, role, title, cwd, git_branch, status, confidence.
    # native_session_id / native_parent_session_id were NOT in the runtime ``text``
    # haystack (they are reachable via their own field predicates), so they stay out.
    return _join_search_text(
        run.run_ref.format(),
        _ref_text(run.parent_run_ref),
        _ref_text(run.agent_ref),
        *(ref.format() for ref in run.lineage_refs),
        run.provider_origin,
        run.harness,
        run.role,
        run.title,
        run.cwd,
        run.git_branch,
        run.status,
        run.confidence,
        _ref_text(run.transcript_ref),
        *(ref.format() for ref in run.evidence_refs),
        _ref_text(run.context_snapshot_ref),
    )


def observed_event_search_text(event: ObservedEvent) -> str:
    # Exact field set of the removed runtime ``text`` matcher for observed events:
    # summary + subject/object/evidence refs only. kind / delivery_state / run_ref
    # are reachable via their own field predicates, not via ``text``.
    return _join_search_text(
        event.summary,
        _ref_text(event.subject_ref),
        *(ref.format() for ref in event.object_refs),
        *(ref.format() for ref in event.evidence_refs),
    )


def context_snapshot_search_text(snapshot: ContextSnapshot) -> str:
    return _join_search_text(
        snapshot.boundary,
        snapshot.inheritance_mode,
        snapshot.snapshot_ref.format(),
        snapshot.run_ref.format(),
        *(ref.format() for ref in snapshot.segment_refs),
        *(ref.format() for ref in snapshot.evidence_refs),
        *snapshot.metadata.values(),
    )


def build_session_run_records(
    projection: RunProjection,
    *,
    materialized_at: str | None = None,
    source_updated_at: str | None = None,
) -> list[SessionRunRecord]:
    built_at = materialized_at or now_iso()
    session_id = SessionId(projection.session_id)
    return [
        SessionRunRecord(
            session_id=session_id,
            position=index,
            materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
            materialized_at=built_at,
            source_updated_at=source_updated_at,
            run=run,
            search_text=run_search_text(run),
        )
        for index, run in enumerate(projection.runs)
    ]


def build_session_observed_event_records(
    projection: RunProjection,
    *,
    materialized_at: str | None = None,
    source_updated_at: str | None = None,
) -> list[SessionObservedEventRecord]:
    built_at = materialized_at or now_iso()
    session_id = SessionId(projection.session_id)
    return [
        SessionObservedEventRecord(
            session_id=session_id,
            position=index,
            materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
            materialized_at=built_at,
            source_updated_at=source_updated_at,
            event=event,
            search_text=observed_event_search_text(event),
        )
        for index, event in enumerate(projection.events)
    ]


def build_session_context_snapshot_records(
    projection: RunProjection,
    *,
    materialized_at: str | None = None,
    source_updated_at: str | None = None,
) -> list[SessionContextSnapshotRecord]:
    built_at = materialized_at or now_iso()
    session_id = SessionId(projection.session_id)
    return [
        SessionContextSnapshotRecord(
            session_id=session_id,
            position=index,
            materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
            materialized_at=built_at,
            source_updated_at=source_updated_at,
            snapshot=snapshot,
            search_text=context_snapshot_search_text(snapshot),
        )
        for index, snapshot in enumerate(projection.context_snapshots)
    ]


def hydrate_projected_run(record: SessionRunRecord) -> ProjectedRun:
    return record.run


def hydrate_observed_event(record: SessionObservedEventRecord) -> ObservedEvent:
    return record.event


def hydrate_context_snapshot(record: SessionContextSnapshotRecord) -> ContextSnapshot:
    return record.snapshot


__all__ = [
    "build_session_context_snapshot_records",
    "build_session_observed_event_records",
    "build_session_run_records",
    "context_snapshot_search_text",
    "hydrate_context_snapshot",
    "hydrate_observed_event",
    "hydrate_projected_run",
    "observed_event_search_text",
    "run_search_text",
]
