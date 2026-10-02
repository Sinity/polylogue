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
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.archive_execution import list_archive, list_summaries_archive
from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.search_hits import plan_has_search_hit_evidence, search_hits_for_plan
from polylogue.config import Config, Source
from polylogue.core.enums import MaterialOrigin, Origin, Provider
from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.core.protocols import ScopedVectorQuery, VectorProvider
from polylogue.storage.embeddings.identity import vector_derivation_hash
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
        ("conv-seed", "alpha seed session with enough prose"),
        ("conv-near", "alpha near neighbor with enough prose"),
        ("conv-near-two", "alpha second near neighbor with enough prose"),
        ("conv-near-three", "alpha third near neighbor with enough prose"),
        ("conv-far", "zeta unrelated topic with enough prose"),
        ("conv-far-two", "zeta second unrelated topic with enough prose"),
        ("conv-unembedded", "no vectors here but enough prose"),
    ):
        (
            SessionBuilder(db_path, session_key)
            .provider(Provider.CODEX.value)
            .title(session_key)
            .updated_at("2026-04-22T12:00:00+00:00")
            .add_message("m1", role="user", text=text, material_origin=MaterialOrigin.HUMAN_AUTHORED)
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
        for suffix in ("seed", "near-three", "near-two", "near", "far-two", "far", "unembedded"):
            if session_id.endswith(f"conv-{suffix}"):
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
        "near-two": _unit_vector(axis0=0.95, axis1=0.312),
        "near-three": _unit_vector(axis0=0.9, axis1=0.436),
        "far": _unit_vector(axis0=0.0, axis1=1.0),
        "far-two": _unit_vector(axis0=-0.2, axis1=0.98),
    }
    text_by_suffix = {
        "seed": "alpha seed session with enough prose",
        "near": "alpha near neighbor with enough prose",
        "near-two": "alpha second near neighbor with enough prose",
        "near-three": "alpha third near neighbor with enough prose",
        "far": "zeta unrelated topic with enough prose",
        "far-two": "zeta second unrelated topic with enough prose",
    }
    for suffix, vector in vectors.items():
        session_id, message_id = mapping[suffix]
        # Content-addressed (polylogue-q88p): distinguish each geometrically
        # distinct fixture vector by its own hash, or they would dedup onto
        # one stored vector under a shared placeholder key.
        upsert_message_embedding(
            conn,
            message_id=message_id,
            session_id=session_id,
            origin=Origin.CODEX_SESSION,
            embedding=vector,
            model="voyage-4",
            embedded_at_ms=1_767_225_700_000,
            vector_derivation_hash=vector_derivation_hash(model="voyage-4", input_text=text_by_suffix[suffix]),
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
        archive_root=archive_root,
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


@pytest.mark.parametrize("offset", (0, 1, 2, 3))
async def test_near_id_applies_one_final_ranked_offset(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]], offset: int
) -> None:
    """Session-seeded list pages advance through ranked summaries exactly once."""
    archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]
    plan = SessionQueryPlan(
        similar_session_id=seed_id,
        vector_provider=_provider(archive_root),
        limit=1,
        offset=offset,
    )

    page = await list_summaries_archive(plan, archive_root=archive_root, config=config)
    unpaged = await list_summaries_archive(
        SessionQueryPlan(similar_session_id=seed_id, vector_provider=_provider(archive_root)),
        archive_root=archive_root,
        config=config,
    )

    expected = [str(summary.id) for summary in unpaged][offset : offset + 1]
    assert [str(summary.id) for summary in page] == expected


@pytest.mark.parametrize("full_session", (False, True))
async def test_text_semantic_pages_apply_the_ranked_offset_once(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]], full_session: bool
) -> None:
    """Semantic pages are disjoint for summaries and fully hydrated sessions."""
    archive_root, config, mapping = seeded_archive
    scored = [
        (mapping[suffix][1], float(index))
        for index, suffix in enumerate(("near", "near-two", "near-three", "far", "far-two"), start=1)
    ]

    class RankedVectors:
        @contextmanager
        def scoped_query(
            self,
            session_ids,
            *,
            text=None,
            seed_session_id=None,
            index_connection,
            configure_connection,
            check_cancelled,
        ):
            del session_ids, text, seed_session_id, index_connection, configure_connection
            check_cancelled()
            yield ScopedVectorQuery(rows=iter(scored))

    vectors = cast(VectorProvider, RankedVectors())
    plan = SessionQueryPlan(similar_text="pagination", vector_provider=vectors, limit=2, offset=2)
    unpaged = SessionQueryPlan(similar_text="pagination", vector_provider=vectors)
    if full_session:
        page_ids = [str(item.id) for item in await list_archive(plan, archive_root=archive_root, config=config)]
        whole_ids = [str(item.id) for item in await list_archive(unpaged, archive_root=archive_root, config=config)]
    else:
        page_ids = [
            str(item.id) for item in await list_summaries_archive(plan, archive_root=archive_root, config=config)
        ]
        whole_ids = [
            str(item.id) for item in await list_summaries_archive(unpaged, archive_root=archive_root, config=config)
        ]

    assert page_ids == whole_ids[2:4]


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
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]

    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda *args, **kwargs: None)
    # An unavailable vector runtime must fail typed rather than broaden the listing.
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
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive_root, config, mapping = seeded_archive
    seed_id = mapping["seed"][0]
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda *args, **kwargs: None)

    plan = SessionQueryPlan(similar_session_id=seed_id)
    with pytest.raises(EmbeddingRetrievalNotReadyError):
        await search_hits_for_plan(plan, config)


async def test_search_hits_for_plan_reports_backend_construction_failure_as_failed(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: mapping every non-``unavailable`` lane failure to ``pending`` tells callers to wait."""
    _archive_root, config, mapping = seeded_archive

    def broken_backend(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("misconfigured embedding backend")

    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", broken_backend)
    plan = SessionQueryPlan(similar_session_id=mapping["seed"][0])
    with pytest.raises(EmbeddingRetrievalNotReadyError) as excinfo:
        await search_hits_for_plan(plan, config)
    assert excinfo.value.readiness_status == "failed"


def test_session_seed_counts_as_search_hit_evidence() -> None:
    assert plan_has_search_hit_evidence(SessionQueryPlan(similar_session_id="abc123")) is True
    assert plan_has_search_hit_evidence(SessionQueryPlan()) is False


def test_compiled_near_id_threads_session_seed() -> None:
    plan = compile_expression("near:id:abc123").to_plan()
    assert plan.similar_session_id == "abc123"


@pytest.mark.parametrize("full_session", (False, True))
async def test_sorted_semantic_pages_concatenate_the_sorted_candidate_relation(
    tmp_path: Path, full_session: bool
) -> None:
    """Date-sorted semantic pages tile one relation: no repeats, no skips.

    Rank order and date order disagree here (rank 1 is the oldest session).
    Anti-vacuity: sizing the candidate pool from ``limit + offset`` admits
    newer, lower-ranked candidates on deeper pages; they sort ahead of rows
    page one already served, so pages repeat those rows and skip others.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    for rank in range(1, 7):
        (
            SessionBuilder(db_path, f"conv-rank-{rank}")
            .provider(Provider.CODEX.value)
            .title(f"rank {rank}")
            .updated_at(f"2026-04-{10 + rank:02d}T12:00:00+00:00")
            .add_message(
                "m1", role="user", text=f"ranked session {rank} prose", material_origin=MaterialOrigin.HUMAN_AUTHORED
            )
            .save()
        )
    with sqlite3.connect(db_path) as conn:
        message_by_session = {
            str(session_id): str(message_id)
            for message_id, session_id in conn.execute("SELECT message_id, session_id FROM messages")
        }
    ranked_sessions = sorted(message_by_session, key=lambda session_id: int(session_id.rsplit("-", 1)[-1]))
    scored = [(message_by_session[session_id], float(index)) for index, session_id in enumerate(ranked_sessions)]

    class RankedVectors:
        @contextmanager
        def scoped_query(
            self,
            session_ids,
            *,
            text=None,
            seed_session_id=None,
            index_connection,
            configure_connection,
            check_cancelled,
        ):
            del session_ids, text, seed_session_id, index_connection, configure_connection
            check_cancelled()
            yield ScopedVectorQuery(rows=iter(scored))

    config = Config(archive_root=archive_root, render_root=tmp_path / "render", sources=[], db_path=db_path)
    vectors = cast(VectorProvider, RankedVectors())

    async def page(offset: int) -> list[str]:
        plan = SessionQueryPlan(similar_text="ranked", vector_provider=vectors, sort="date", limit=1, offset=offset)
        reader = list_archive if full_session else list_summaries_archive
        return [str(item.id) for item in await reader(plan, archive_root=archive_root, config=config)]

    served = [session_id for offset in range(6) for session_id in await page(offset)]

    # Every eligible ranked session belongs to the explicit-sort relation.
    assert served == list(reversed(ranked_sessions))


async def test_near_id_resolves_retained_provider_without_acquisition_key(
    seeded_archive: tuple[Path, Config, dict[str, tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring credentials at either seed resolver breaks both read projections."""
    archive_root, config, mapping = seeded_archive
    config.embedding_model = "voyage-4"
    from polylogue.config import PolylogueConfig

    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: PolylogueConfig())
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    plan = SessionQueryPlan(similar_session_id=mapping["seed"][0], limit=10)
    summaries = await list_summaries_archive(plan, archive_root=archive_root, config=config)
    hits = await search_hits_for_plan(plan, config=config)
    assert mapping["near"][0] in [str(summary.id) for summary in summaries]
    assert mapping["near"][0] in [hit.session_id for hit in hits]
