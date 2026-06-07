"""Tests for current parsed-session preparation enrichment."""

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.pipeline.prepare_enrichment import enrich_bundle_from_db
from polylogue.pipeline.prepare_models import AttachmentMaterializationPlan, PrepareCache, TransformResult
from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession
from polylogue.storage.archive_views import ExistingSession
from polylogue.types import ContentHash, Provider, SessionId


def _session() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="conv-1",
        title="Session",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
        working_directories=["/realm/project/polylogue"],
        git_branch="feature/refactor/current-archive-runtime",
        git_repository_url="https://github.com/Sinity/polylogue",
    )


def _transform(session: ParsedSession, *, content_hash: str = "hash-1") -> TransformResult:
    return TransformResult(
        session=session,
        materialization_plan=AttachmentMaterializationPlan(),
        content_hash=ContentHash(content_hash),
        candidate_cid=SessionId("codex-session:conv-1"),
    )


def test_enrich_bundle_carries_parsed_session_directly() -> None:
    session = _session()
    result = enrich_bundle_from_db(session, "ignored-source", _transform(session), PrepareCache())

    assert result.session is session or result.session.model_dump() == session.model_dump()
    assert result.cid == "codex-session:conv-1"
    assert result.changed is False
    assert result.session.working_directories == ["/realm/project/polylogue"]
    assert result.session.git_branch == "feature/refactor/current-archive-runtime"
    assert not hasattr(result, "bundle")


def test_enrich_bundle_detects_changed_existing_session() -> None:
    session = _session()
    cache = PrepareCache()
    cache.existing["codex-session:conv-1"] = ExistingSession(
        session_id="codex-session:conv-1",
        content_hash=ContentHash("old-hash"),
    )

    result = enrich_bundle_from_db(session, "ignored-source", _transform(session, content_hash="new-hash"), cache)

    assert result.changed is True


def test_enrich_bundle_preserves_attachment_materialization_plan() -> None:
    session = _session()
    plan = AttachmentMaterializationPlan()
    transform = TransformResult(
        session=session,
        materialization_plan=plan,
        content_hash=ContentHash("hash-1"),
        candidate_cid=SessionId("codex-session:conv-1"),
    )

    result = enrich_bundle_from_db(session, "ignored-source", transform, PrepareCache())

    assert result.materialization_plan is plan
