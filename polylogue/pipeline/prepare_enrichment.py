"""DB-aware enrichment for transformed parsed-session writes."""

from __future__ import annotations

from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.prepare_models import (
    PrepareCache,
    PreparedBundle,
    TransformResult,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.archive_views import ExistingSession
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.types import SessionId


def enrich_bundle_from_db(
    convo: ParsedSession,
    source_name: str,
    transform: TransformResult,
    cache: PrepareCache,
    *,
    raw_id: str | None = None,
) -> PreparedBundle:
    del source_name, raw_id
    candidate_cid = transform.candidate_cid
    content_hash = transform.content_hash

    existing = cache.existing.get(candidate_cid)
    if existing:
        cid = SessionId(existing.session_id)
        changed = existing.content_hash != content_hash
    else:
        cid = candidate_cid
        changed = False

    prepared_session = transform.session.model_copy(
        update={
            "working_directories": list(transform.session.working_directories),
            "git_branch": transform.session.git_branch,
            "git_repository_url": transform.session.git_repository_url,
        }
    )

    return PreparedBundle(
        session=prepared_session,
        materialization_plan=transform.materialization_plan,
        content_hash=content_hash,
        cid=cid,
        changed=changed,
    )


async def _build_single_cache(
    backend: SQLiteBackend,
    convo: ParsedSession,
    candidate_cid: SessionId,
    _unused: object,
) -> PrepareCache:
    cache = PrepareCache()

    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT session_id, content_hash FROM sessions WHERE session_id = ? LIMIT 1",
            (candidate_cid,),
        )
        row = await cursor.fetchone()
    if row:
        cid = row["session_id"]
        raw_content_hash = row["content_hash"]
        content_hash = raw_content_hash.hex() if isinstance(raw_content_hash, bytes) else str(raw_content_hash)
        cache.existing[cid] = ExistingSession(session_id=cid, content_hash=content_hash)
        cache.known_ids.add(cid)

    if convo.parent_session_provider_id:
        candidate_parent = make_session_id(convo.source_name, convo.parent_session_provider_id)
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (candidate_parent,),
            )
            if await cursor.fetchone():
                cache.known_ids.add(candidate_parent)

    return cache


__all__ = [
    "_build_single_cache",
    "enrich_bundle_from_db",
]
