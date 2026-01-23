"""Tests for polylogue.pipeline.runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestRenderFailureTracking:
    """Tests for tracking render failures in pipeline."""

    def test_render_failure_tracked_in_result(self, tmp_path: Path):
        """Render failures should be tracked in RunResult.

        This test SHOULD FAIL until failure tracking is implemented.
        """
        from polylogue.pipeline.runner import run_sources
        from polylogue.config import Config, Source
        from polylogue.pipeline.models import RunResult

        # Create a minimal config
        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
            template_path=None,
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        # Mock render_conversation to fail for specific conversation
        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                if "fail-conv" in conversation_id:
                    raise ValueError("Render failed for testing")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            # Mock _all_conversation_ids to return test data
            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:success-conv", "test:fail-conv"]

                # Run pipeline in render stage
                result = run_sources(
                    config=config,
                    stage="render",
                    source_names=None,
                )

                # Result should be RunResult
                assert isinstance(result, RunResult)

                # Result should track render failures
                assert hasattr(result, "render_failures"), "RunResult should have render_failures attribute"
                assert isinstance(result.render_failures, list)
                assert len(result.render_failures) > 0, "Should have tracked at least one render failure"

                # Check failure details
                failure = result.render_failures[0]
                assert "conversation_id" in failure
                assert failure["conversation_id"] == "test:fail-conv"
                assert "error" in failure

    def test_render_continues_after_failure(self, tmp_path: Path):
        """Pipeline should continue rendering other conversations after one fails."""
        from polylogue.pipeline.runner import run_sources
        from polylogue.config import Config

        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
            template_path=None,
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        render_attempts = []

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                render_attempts.append(conversation_id)
                if "second" in conversation_id:
                    raise ValueError("Failed on purpose")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:first", "test:second", "test:third"]

                result = run_sources(
                    config=config,
                    stage="render",
                )

                # Should attempt all renders even if one fails
                assert len(render_attempts) >= 3, f"Expected 3 render attempts, got {len(render_attempts)}"

                # Verify we didn't stop at the failure
                assert "test:first" in render_attempts
                assert "test:second" in render_attempts
                assert "test:third" in render_attempts

    def test_render_failure_count_in_counts(self, tmp_path: Path):
        """Pipeline should include render_failures count in result.counts."""
        from polylogue.pipeline.runner import run_sources
        from polylogue.config import Config

        config = Config(
            version=2,
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            path=tmp_path / "config.json",
            template_path=None,
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                if conversation_id in ["test:fail1", "test:fail2"]:
                    raise ValueError("Render failed")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:success", "test:fail1", "test:fail2"]

                result = run_sources(
                    config=config,
                    stage="render",
                )

                # Check counts include render_failures
                assert "render_failures" in result.counts
                assert result.counts["render_failures"] == 2
                assert result.counts["rendered"] == 1
