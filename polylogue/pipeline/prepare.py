"""Async parse preparation facade with explicit transform and enrichment stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.logging import get_logger
from polylogue.pipeline.ids import (
    materialize_attachment_path,
    move_attachment_to_archive,
)
from polylogue.pipeline.ids import (
    session_id as make_session_id,
)
from polylogue.pipeline.prepare_enrichment import _build_single_cache, enrich_bundle_from_db
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    PersistedSessionResult,
    PrepareCache,
    PreparedBundle,
    SaveResult,
    TransformResult,
    _timestamp_sort_key,
)
from polylogue.pipeline.prepare_transform import transform_to_records

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)


class PrepareRepository(Protocol):
    """Repository surface needed by record preparation."""

    @property
    def backend(self) -> SQLiteBackend: ...

    async def save_parsed_session(
        self,
        session: ParsedSession,
        content_hash: str,
    ) -> dict[str, int]: ...


async def prepare_bundle(
    convo: ParsedSession,
    source_name: str,
    *,
    archive_root: Path,
    backend: SQLiteBackend | None = None,
    repository: PrepareRepository | None = None,
    raw_id: str | None = None,
    cache: PrepareCache | None = None,
) -> PreparedBundle:
    """Convert a parsed session to a DB-aware prepared bundle."""
    if repository is None and backend is None:
        raise ValueError("prepare_bundle requires a repository or backend")
    if repository is None:
        from polylogue.storage.repository import SessionRepository

        repository = SessionRepository(backend=backend)
    if backend is None:
        backend = repository.backend

    transform = transform_to_records(convo, source_name, archive_root=archive_root)
    if cache is None:
        cache = await _build_single_cache(backend, convo, transform.candidate_cid, transform.candidate_cid)
    return enrich_bundle_from_db(convo, source_name, transform, cache, raw_id=raw_id)


async def persist_prepared_bundle(
    prepared: PreparedBundle,
    *,
    repository: PrepareRepository,
) -> PersistedSessionResult:
    """Persist a prepared session bundle, including attachment materialization."""
    applied_moves: list[tuple[Path, Path]] = []
    try:
        for source_path, target_path in prepared.materialization_plan.move_before_save:
            materialize_attachment_path(source_path, target_path)
            applied_moves.append((source_path, target_path))

        counts = await repository.save_parsed_session(prepared.session, str(prepared.content_hash))
        save_result = SaveResult(**counts)
    except Exception:
        for source_path, target_path in reversed(applied_moves):
            if target_path.exists():
                move_attachment_to_archive(target_path, source_path)
        raise

    for duplicate_source in prepared.materialization_plan.delete_after_save:
        if duplicate_source.exists():
            duplicate_source.unlink()

    return PersistedSessionResult(
        session_id=prepared.cid,
        save_result=save_result,
        content_changed=prepared.changed,
    )


async def prepare_records(
    convo: ParsedSession,
    source_name: str,
    *,
    archive_root: Path,
    backend: SQLiteBackend | None = None,
    repository: PrepareRepository | None = None,
    raw_id: str | None = None,
    cache: PrepareCache | None = None,
) -> PersistedSessionResult:
    """Convert a ParsedSession to storage records and persist them."""
    if repository is None and backend is None:
        raise ValueError("prepare_records requires a repository or backend")
    if repository is None:
        from polylogue.storage.repository import SessionRepository

        repository = SessionRepository(backend=backend)
    if backend is None:
        backend = repository.backend

    if not convo.messages:
        cid = make_session_id(convo.source_name, convo.provider_session_id)
        logger.debug("Skipping empty session (no messages)", session_id=cid)
        return PersistedSessionResult(
            session_id=cid,
            save_result=SaveResult(
                sessions=0,
                messages=0,
                attachments=0,
                skipped_sessions=1,
                skipped_messages=0,
                skipped_attachments=0,
            ),
            content_changed=False,
        )

    prepared = await prepare_bundle(
        convo,
        source_name,
        archive_root=archive_root,
        backend=backend,
        repository=repository,
        raw_id=raw_id,
        cache=cache,
    )
    return await persist_prepared_bundle(prepared, repository=repository)


__all__ = [
    "AttachmentMaterializationPlan",
    "PersistedSessionResult",
    "PrepareCache",
    "PrepareRepository",
    "PreparedBundle",
    "SaveResult",
    "TransformResult",
    "_timestamp_sort_key",
    "persist_prepared_bundle",
    "prepare_bundle",
    "enrich_bundle_from_db",
    "prepare_records",
    "transform_to_records",
]
