"""Tests for runner-level progress callback propagation."""

from __future__ import annotations

import pytest


class TestRunnerProgressPropagation:
    """Verify run_sources propagates progress_callback to render/index stages."""

    @pytest.mark.parametrize(
        "stage,with_callback,expected_indexed,expected_rendered,expected_first_desc",
        [
            ("render", True, False, 0, None),
            ("render", False, False, 0, None),
            ("index", True, True, 0, "Indexing"),
        ],
    )
    def test_stage_callback_matrix(
        self,
        workspace_env,
        stage,
        with_callback,
        expected_indexed,
        expected_rendered,
        expected_first_desc,
    ):
        """run_sources accepts optional callback and emits stage-appropriate progress."""
        import asyncio

        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        config = Config(
            sources=[],
            archive_root=archive_root,
            render_root=archive_root / "render",
        )

        callback_calls = []

        def track_callback(amount, desc=None):
            callback_calls.append({"amount": amount, "desc": desc})

        callback = track_callback if with_callback else None

        result = asyncio.run(
            run_sources(
                config=config,
                stage=stage,
                progress_callback=callback,
            )
        )

        assert result.indexed is expected_indexed
        assert result.counts.get("rendered", 0) == expected_rendered
        if expected_first_desc is None:
            assert callback_calls == []
        else:
            assert callback_calls
            assert callback_calls[0]["desc"] == expected_first_desc
