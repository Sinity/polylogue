"""Execution tests for ``near:id:<ref>`` session-seeded similarity (#1842).

The DSL parser/lowerer accepts ``near:id:<ref>`` and threads it into
``SessionQueryPlan.similar_session_id`` (#1899). This module pins the *execution*
of that field: a session-seeded plan reads the seed session's stored embeddings,
KNN-searches them, excludes the seed itself, and aggregates to session-level hits
ranked by similarity. When the request cannot be honored (no vector backend, or a
seed with no stored embeddings) execution fails *typed* — never a silent empty or
unfiltered listing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.archive_execution import list_summaries_archive
from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.search_hits import plan_has_search_hit_evidence, search_hits_for_plan
from polylogue.config import Config, Source
from polylogue.core.enums import Origin, Provider
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.storage_records import SessionBuilder


def _unit_vector(*, axis0: float, axis1: float) -> list[float]:
    vec = [0.0] * EMBEDDING_DIMENSION
    vec[0] = axis0
    vec[1] = axis1
    return vec


def _build_index(db_path: Path) -> None:
    for session_key, text in (
        ("conv-seed", "alpha seed session"),
        ("conv-near", "alpha near neighbor"),
        ("conv-far", "zeta unrelated topic"),
        ("conv-unembedded", "no vectors here"),
    ):
        (
            SessionBuilder(db_path, session_key)
            .provider(Provider.CODEX.value)
            .title(session_key)
            .updated_at("2026-04-22T12:00:00+00:00")
            .add_message("m1", role="user", text=text)
            .save()
        )


def _message_rows(db_path: Path) -> dict[str, tuple[str, str]]:
    """Return ``{session_key_suffix: (session_id, message_id)}`` from the index."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT message_id, session_id FROM messages").fetchall()
    finally:
        conn.close()
    mapping: dict[str, tuple[str, str]] = {}
    for row in rows:
        session_id = str(row["session_id"])
        message_id = str(row["message_id"])
        for suffix in ("seed", "near", "far", "unembedded"):
            if f"conv-{suffix}" in session_id:
                mapping[suffix] = (session_id, message_id)
    return mapping


@pytest.fixture
def seeded_archive(tmp_path: Path) -> tuple[Path, Config, dict[str, tuple[str, str]]]:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    _build_index(db_path)
    mapping = _message_rows(db_path)

    embeddings_db = archive_root / "embeddings.db"
    conn = sqlite3.connect(embeddings_db)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    vectors = {
        "seed": _unit_vector(axis0=1.0, axis1=0.0),
        "near": _unit_vector(axis0=0.99, axis1=0.141),
        "far": _unit_vector(axis0=0.0, axis1=1.0),
    }
    for suffix, vector in vectors.items():
        session_id, message_id = mapping[suffix]
        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_id,
            origin=Origin.CODEX_SESSION,
            embedding=vector,
            model="voyage-4",
            embedded_at_ms=1_767_225_700_000,
            content_hash=b"x" * 32,
        )
    conn.close()

    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        db_path=db_path,
    )
    return archive_root, config, mapping


def _provider(archive_root: Path) -> SqliteVecProvider:
    provider = SqliteVecProvider(
        voyage_key="test-key",
        db_path=archive_root / "embeddings.db",
        model="voyage-4",
    )
    provider.dimension = EMBEDDING_DIMENSION
    provider._vec_available = None
    return provider


async def test_near_id_returns_similar_sessions_excluding_seed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
) -> None:
    archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]
    near_id = mapping["near"][0]
    far_id = mapping["far"][0]

    plan = SessionQueryPlan(similar_session_id=seed_id, vector_provider=_provider(archive_root))
    summaries = await list_summaries_archive(plan, archive_root=archive_root, config=config)

    result_ids = [str(summary.id) for summary in summaries]
    # The seed session is excluded from its own similarity results.
    assert seed_id not in result_ids
    # Both other embedded sessions surface, the near neighbor ahead of the far one.
    assert near_id in result_ids
    assert far_id in result_ids
    assert result_ids.index(near_id) < result_ids.index(far_id)


async def test_near_id_seed_without_embeddings_fails_typed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
) -> None:
    archive_root, config, mapping = seeded_archive
    unembedded_id = mapping["unembedded"][0]

    plan = SessionQueryPlan(similar_session_id=unembedded_id, vector_provider=_provider(archive_root))
    with pytest.raises(ExpressionCompileError) as excinfo:
        await list_summaries_archive(plan, archive_root=archive_root, config=config)
    assert excinfo.value.field == "near"


async def test_near_id_without_vector_backend_fails_typed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
) -> None:
    archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]

    # No vector provider on the plan and no Voyage key in config -> no backend can
    # be constructed. This must fail typed rather than broaden to a full listing.
    plan = SessionQueryPlan(similar_session_id=seed_id)
    with pytest.raises(ExpressionCompileError) as excinfo:
        await list_summaries_archive(plan, archive_root=archive_root, config=config)
    assert excinfo.value.field == "near"


async def test_search_hits_for_plan_resolves_session_seed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
) -> None:
    archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]
    near_id = mapping["near"][0]

    plan = SessionQueryPlan(similar_session_id=seed_id, vector_provider=_provider(archive_root), limit=5)
    hits = await search_hits_for_plan(plan, config)

    result_ids = [hit.session_id for hit in hits]
    assert seed_id not in result_ids
    assert near_id in result_ids
    assert all(hit.retrieval_lane == "semantic" for hit in hits)


async def test_search_hits_for_plan_session_seed_no_backend_fails_typed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
) -> None:
    _archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]

    plan = SessionQueryPlan(similar_session_id=seed_id)
    with pytest.raises(ExpressionCompileError):
        await search_hits_for_plan(plan, config)


def test_session_seed_counts_as_search_hit_evidence() -> None:
    assert plan_has_search_hit_evidence(SessionQueryPlan(similar_session_id="abc123")) is True
    assert plan_has_search_hit_evidence(SessionQueryPlan()) is False


def test_compiled_near_id_threads_session_seed() -> None:
    plan = compile_expression("near:id:abc123").to_plan()
    assert plan.similar_session_id == "abc123"
