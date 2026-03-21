"""Site builder publication-manifest contracts."""

from __future__ import annotations

import asyncio

from polylogue.site.builder import SiteBuilder, SiteConfig
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RunRecord
from tests.infra.storage_records import ConversationBuilder, record_run


def _seed_latest_run(db_path) -> None:
    with open_connection(db_path) as conn:
        record_run(
            conn,
            RunRecord(
                run_id="run-001",
                timestamp="2026-03-22T12:00:00+00:00",
                counts={"conversations": 1, "messages": 2},
                drift={"providers": {"chatgpt": 0}},
                indexed=True,
                duration_ms=456,
            ),
        )
        conn.commit()


def test_site_builder_returns_typed_manifest_and_persists_it(db_path, tmp_path) -> None:
    """A site build writes a manifest file and persists the same manifest to SQLite."""
    (
        ConversationBuilder(db_path, "conv-site-1")
        .provider("chatgpt")
        .title("Site Build Test")
        .updated_at("2020-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="hello")
        .add_message("m2", role="assistant", text="world")
        .save()
    )
    _seed_latest_run(db_path)

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(enable_search=True, search_provider="lunr"),
        backend=backend,
        repository=repository,
    )

    try:
        manifest = builder.build()
        persisted = asyncio.run(repository.get_latest_publication("site"))
    finally:
        asyncio.run(backend.close())

    assert manifest.publication_kind == "site"
    assert manifest.archive.total_conversations == 1
    assert manifest.outputs.total_conversation_pages == 1
    assert manifest.outputs.rendered_conversation_pages == 1
    assert manifest.outputs.reused_conversation_pages == 0
    assert manifest.outputs.search_status == "json_index_written"
    assert manifest.latest_run is not None
    assert manifest.latest_run.run_id == "run-001"
    assert manifest.artifact_proof is not None
    assert (tmp_path / "site" / "site-manifest.json").exists()
    assert "site-manifest.json" not in {
        entry.relative_path for entry in manifest.artifacts.entries
    }
    assert persisted is not None
    assert persisted.publication_id == manifest.publication_id
    assert persisted.manifest["outputs"]["rendered_conversation_pages"] == 1


def test_site_builder_reports_reused_pages_on_incremental_rebuild(db_path, tmp_path) -> None:
    """Incremental rebuilds distinguish reused conversation pages from rendered ones."""
    (
        ConversationBuilder(db_path, "conv-site-2")
        .provider("chatgpt")
        .title("Incremental")
        .updated_at("2020-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="hello")
        .save()
    )

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(enable_search=False),
        backend=backend,
        repository=repository,
    )

    try:
        first = builder.build()
        second = builder.build()
    finally:
        asyncio.run(backend.close())

    assert first.outputs.rendered_conversation_pages == 1
    assert second.outputs.rendered_conversation_pages == 0
    assert second.outputs.reused_conversation_pages == 1
    assert second.outputs.failed_conversation_pages == 0
