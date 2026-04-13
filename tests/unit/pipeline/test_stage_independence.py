"""Pipeline stage independence and idempotency tests.

Verifies that each pipeline stage can run independently, that running
a stage twice produces the same result (no duplicates), and that empty
input is handled cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository


def _make_backend(tmp_path: Path) -> tuple[SQLiteBackend, Path]:
    """Create a backend with initialized schema."""
    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass
    return SQLiteBackend(db_path=db_path), db_path


def _write_synthetic_files(
    tmp_path: Path,
    provider: str = "chatgpt",
    count: int = 2,
    seed: int = 42,
) -> list[Path]:
    """Write synthetic corpus files and return their paths."""
    spec = CorpusSpec.for_provider(
        provider,
        count=count,
        messages_min=4,
        messages_max=7,
        seed=seed,
        origin="generated.test-stage-independence",
        tags=("synthetic", "test", "stage-independence"),
    )
    out_dir = tmp_path / "sources" / provider
    written = SyntheticCorpus.write_spec_artifacts(spec, out_dir, prefix="synth")
    return list(written.files)


# =============================================================================
# Acquisition stage independence
# =============================================================================


class TestAcquisitionStageIndependence:
    """The acquire stage stores raw bytes without parsing."""

    @pytest.mark.asyncio
    async def test_acquire_runs_independently(self, tmp_path: Path, workspace_env) -> None:
        """Acquire stage completes without downstream stages."""
        from polylogue.config import Source
        from polylogue.pipeline.services.acquisition import AcquisitionService

        backend, _ = _make_backend(tmp_path)
        service = AcquisitionService(backend)
        files = _write_synthetic_files(tmp_path)

        sources = [Source(name="test", path=files[0])]
        result = await service.acquire_sources(sources)
        assert result.counts["acquired"] >= 0
        assert result.counts["errors"] == 0
        await backend.close()

    @pytest.mark.asyncio
    async def test_acquire_idempotent(self, tmp_path: Path, workspace_env) -> None:
        """Running acquire twice does not create duplicates."""
        from polylogue.config import Source
        from polylogue.pipeline.services.acquisition import AcquisitionService

        backend, _ = _make_backend(tmp_path)
        service = AcquisitionService(backend)
        files = _write_synthetic_files(tmp_path, count=1)

        sources = [Source(name="test", path=files[0])]

        r1 = await service.acquire_sources(sources)
        assert r1.counts["acquired"] == 1
        r2 = await service.acquire_sources(sources)

        # Second run should not acquire new records (mtime-based or hash-based dedup)
        assert r2.counts["acquired"] == 0
        await backend.close()

    @pytest.mark.asyncio
    async def test_acquire_empty_source(self, tmp_path: Path, workspace_env) -> None:
        """Acquire with no source files completes cleanly."""
        from polylogue.pipeline.services.acquisition import AcquisitionService

        backend, _ = _make_backend(tmp_path)
        service = AcquisitionService(backend)

        result = await service.acquire_sources([])
        assert result.counts["acquired"] == 0
        assert result.counts["errors"] == 0
        await backend.close()


# =============================================================================
# Validation stage independence
# =============================================================================


class TestValidationStageIndependence:
    """The validate stage runs against raw records in DB."""

    @pytest.mark.asyncio
    async def test_validate_empty(self, tmp_path: Path, workspace_env) -> None:
        """Validate with no raw IDs completes cleanly."""
        from polylogue.pipeline.services.validation import ValidationService

        backend, _ = _make_backend(tmp_path)
        service = ValidationService(backend)

        result = await service.validate_raw_ids(raw_ids=[])
        assert result.counts["validated"] == 0
        assert result.counts["errors"] == 0
        await backend.close()


# =============================================================================
# Parse stage independence
# =============================================================================


class TestParseStageIndependence:
    """The parse stage converts raw records to conversations."""

    @pytest.mark.asyncio
    async def test_parse_empty(self, tmp_path: Path, workspace_env) -> None:
        """Parse with no sources (empty) completes cleanly."""
        from polylogue.config import Config
        from polylogue.pipeline.services.parsing import ParsingService

        backend, _ = _make_backend(tmp_path)
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        repo = ConversationRepository(backend=backend)

        render_root = tmp_path / "render"
        render_root.mkdir()
        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=render_root,
        )
        service = ParsingService(
            repository=repo,
            archive_root=archive_root,
            config=config,
        )

        # parse_sources with empty sources → zero conversations
        result = await service.parse_sources(sources=[])
        assert result.counts["conversations"] == 0
        await backend.close()

    @pytest.mark.asyncio
    async def test_parse_from_raw_empty(self, tmp_path: Path, workspace_env) -> None:
        """Parse from raw with no raw IDs completes cleanly."""
        from polylogue.config import Config
        from polylogue.pipeline.services.parsing import ParsingService

        backend, _ = _make_backend(tmp_path)
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        render_root = tmp_path / "render"
        render_root.mkdir()
        repo = ConversationRepository(backend=backend)

        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=render_root,
        )
        service = ParsingService(
            repository=repo,
            archive_root=archive_root,
            config=config,
        )

        result = await service.parse_from_raw(raw_ids=[])
        assert result.counts["conversations"] == 0
        await backend.close()


# =============================================================================
# Render stage independence
# =============================================================================


class TestRenderStageIndependence:
    """The render stage generates output from conversations in DB."""

    @pytest.mark.asyncio
    async def test_render_empty_conversation_list(self, tmp_path: Path, workspace_env) -> None:
        """Render with empty conversation ID list completes cleanly."""
        from polylogue.pipeline.services.rendering import RenderService
        from polylogue.rendering.renderers import MarkdownRenderer

        backend, _ = _make_backend(tmp_path)
        render_root = tmp_path / "render"
        render_root.mkdir()
        archive_root = tmp_path / "archive"
        archive_root.mkdir()

        renderer = MarkdownRenderer(archive_root=archive_root)
        service = RenderService(
            renderer=renderer,
            render_root=render_root,
            backend=backend,
        )
        result = await service.render_conversations(conversation_ids=[])
        assert result.rendered_count == 0
        assert len(result.failures) == 0
        await backend.close()


# =============================================================================
# Index stage independence
# =============================================================================


class TestIndexStageIndependence:
    """The index stage builds FTS and other search indices."""

    @pytest.mark.asyncio
    async def test_index_empty_db(self, tmp_path: Path, workspace_env) -> None:
        """Index on empty DB completes cleanly."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        backend, _ = _make_backend(tmp_path)
        archive_root = tmp_path / "archive"
        archive_root.mkdir(exist_ok=True)
        render_root = tmp_path / "render"
        render_root.mkdir(exist_ok=True)
        config = Config(sources=[], archive_root=archive_root, render_root=render_root)
        service = IndexService(config=config, backend=backend)
        success = await service.rebuild_index()
        assert success is True
        await backend.close()

    @pytest.mark.asyncio
    async def test_index_idempotent(self, tmp_path: Path, workspace_env) -> None:
        """Running index twice produces the same result."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        backend, _ = _make_backend(tmp_path)
        archive_root = tmp_path / "archive"
        archive_root.mkdir(exist_ok=True)
        render_root = tmp_path / "render"
        render_root.mkdir(exist_ok=True)
        config = Config(sources=[], archive_root=archive_root, render_root=render_root)
        service = IndexService(config=config, backend=backend)

        r1 = await service.rebuild_index()
        r2 = await service.rebuild_index()

        assert r1 is True
        assert r2 is True
        await backend.close()

    @pytest.mark.asyncio
    async def test_update_index_empty_ids(self, tmp_path: Path, workspace_env) -> None:
        """Updating index with no conversation IDs succeeds."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        backend, _ = _make_backend(tmp_path)
        archive_root = tmp_path / "archive"
        archive_root.mkdir(exist_ok=True)
        render_root = tmp_path / "render"
        render_root.mkdir(exist_ok=True)
        config = Config(sources=[], archive_root=archive_root, render_root=render_root)
        service = IndexService(config=config, backend=backend)

        success = await service.update_index(conversation_ids=[])
        assert success is True
        await backend.close()
