"""Tests for pipeline service classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services import IndexService, IngestionService, RenderService
from polylogue.storage.repository import StorageRepository


class TestIngestionService:
    """Tests for IngestionService."""

    def test_initialization(self, tmp_path: Path):
        """IngestionService should initialize with required dependencies."""
        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
        )
        repository = StorageRepository()

        service = IngestionService(repository, config.archive_root, config)

        assert service.repository is repository
        assert service.archive_root == config.archive_root
        assert service.config is config


class TestRenderService:
    """Tests for RenderService."""

    def test_initialization(self, tmp_path: Path):
        """RenderService should initialize with required paths."""
        template_path = tmp_path / "template.html"
        render_root = tmp_path / "render"
        archive_root = tmp_path / "archive"

        service = RenderService(template_path, render_root, archive_root)

        assert service.template_path == template_path
        assert service.render_root == render_root
        assert service.archive_root == archive_root

    def test_render_conversations_empty_list(self, tmp_path: Path):
        """RenderService should handle empty conversation list."""
        service = RenderService(None, tmp_path / "render", tmp_path / "archive")

        result = service.render_conversations([])

        assert result.rendered_count == 0
        assert result.failures == []

    def test_render_conversations_tracks_failures(self, tmp_path: Path):
        """RenderService should track failures when rendering fails."""
        service = RenderService(None, tmp_path / "render", tmp_path / "archive")

        # Patch at the point where it's imported in the rendering service module
        with patch("polylogue.pipeline.services.rendering.render_conversation") as mock_render:

            def render_side_effect(conversation_id, **kwargs):
                if "fail" in conversation_id:
                    raise ValueError("Test error")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            result = service.render_conversations(["success-1", "fail-1", "success-2"])

            assert result.rendered_count == 2
            assert len(result.failures) == 1
            assert result.failures[0]["conversation_id"] == "fail-1"
            assert "Test error" in result.failures[0]["error"]


class TestIndexService:
    """Tests for IndexService."""

    def test_initialization(self, tmp_path: Path):
        """IndexService should initialize with config."""
        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
        )

        service = IndexService(config)

        assert service.config is config

    def test_update_index_empty_list(self, tmp_path: Path, workspace_env):
        """IndexService should handle empty conversation list."""
        from polylogue.storage.db import connection_context

        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
        )

        with connection_context(None) as conn:
            service = IndexService(config, conn)

            # Empty list should ensure index exists and return True
            result = service.update_index([])

            assert result is True

    def test_get_index_status_when_no_index(self, tmp_path: Path):
        """IndexService should return status when index doesn't exist."""
        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
        )
        service = IndexService(config)

        # Without mocking, this will use actual index_status()
        # which should return exists=False for new database
        status = service.get_index_status()

        assert "exists" in status
        assert "count" in status
