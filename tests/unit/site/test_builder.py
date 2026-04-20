"""Site builder publication-manifest contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from polylogue.site.builder import SiteBuilder, SiteConfig
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RunRecord
from polylogue.types import SearchProvider
from tests.infra.storage_records import ConversationBuilder, record_run


def _seed_latest_run(db_path: Path) -> None:
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


def test_site_builder_returns_typed_manifest_and_persists_it(db_path: Path, tmp_path: Path) -> None:
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
        config=SiteConfig(enable_search=True, search_provider=SearchProvider.LUNR),
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
    assert manifest.artifact_proof.package_versions == {}
    assert manifest.artifact_proof.element_kinds == {}
    assert manifest.artifact_proof.resolution_reasons == {}
    assert manifest.maintenance is not None
    assert "messages_fts" in manifest.maintenance.derived_models
    assert (tmp_path / "site" / "site-manifest.json").exists()
    assert "site-manifest.json" not in {entry.relative_path for entry in manifest.artifacts.entries}
    assert persisted is not None
    assert persisted.publication_id == manifest.publication_id
    outputs = persisted.manifest.get("outputs")
    assert isinstance(outputs, dict)
    assert outputs["rendered_conversation_pages"] == 1
    assert "maintenance" in persisted.manifest


def test_site_builder_reports_reused_pages_on_incremental_rebuild(db_path: Path, tmp_path: Path) -> None:
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


def test_site_builder_emits_progress_during_scan_and_manifest_write(db_path: Path, tmp_path: Path) -> None:
    (
        ConversationBuilder(db_path, "conv-site-progress")
        .provider("chatgpt")
        .title("Progress")
        .updated_at("2020-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="hello")
        .save()
    )

    progress_events: list[tuple[int, str | None]] = []

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    builder = SiteBuilder(
        output_dir=tmp_path / "site",
        config=SiteConfig(enable_search=False),
        backend=backend,
        repository=repository,
        progress_callback=lambda amount, desc=None: progress_events.append((amount, desc)),
    )

    try:
        builder.build()
    finally:
        asyncio.run(backend.close())

    descriptions = [desc for _, desc in progress_events if desc]
    assert "Building site: preparing output" in descriptions
    assert any(desc.startswith("Building site: scanning archive ") for desc in descriptions)
    assert "Building site: writing manifest" in descriptions
    assert any(amount == 1 for amount, _ in progress_events)
