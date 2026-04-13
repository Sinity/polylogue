"""Unit tests for shared verification workspace helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from polylogue.config import Config
from polylogue.paths import Source
from polylogue.scenarios import CorpusSpec
from polylogue.showcase.workspace import (
    create_verification_workspace,
    generate_synthetic_fixtures,
    run_pipeline_for_configured_sources,
    run_pipeline_for_fixture_workspace,
)


def test_run_pipeline_for_configured_sources_uses_all_sources(tmp_path):
    workspace = create_verification_workspace(tmp_path / "workspace")
    sources = [
        Source(name="chatgpt", path=tmp_path / "chatgpt"),
        Source(name="codex", path=tmp_path / "codex"),
    ]
    config = Config(
        archive_root=tmp_path / "archive-root",
        render_root=tmp_path / "render-root",
        sources=sources,
    )

    with patch("polylogue.showcase.workspace.get_config", return_value=config):
        with patch("polylogue.pipeline.runner.run_sources", new_callable=AsyncMock) as mock_run:
            run_pipeline_for_configured_sources(workspace)

    selected = mock_run.await_args.kwargs["config"].sources
    assert [source.name for source in selected] == ["chatgpt", "codex"]


def test_run_pipeline_for_configured_sources_filters_named_sources(tmp_path):
    workspace = create_verification_workspace(tmp_path / "workspace")
    sources = [
        Source(name="chatgpt", path=tmp_path / "chatgpt"),
        Source(name="codex", path=tmp_path / "codex"),
    ]
    config = Config(
        archive_root=tmp_path / "archive-root",
        render_root=tmp_path / "render-root",
        sources=sources,
    )

    with patch("polylogue.showcase.workspace.get_config", return_value=config):
        with patch("polylogue.pipeline.runner.run_sources", new_callable=AsyncMock) as mock_run:
            run_pipeline_for_configured_sources(workspace, source_names=["codex"])

    selected = mock_run.await_args.kwargs["config"].sources
    assert [source.name for source in selected] == ["codex"]


def test_run_pipeline_for_fixture_workspace_mirrors_fixtures_to_inbox(tmp_path):
    workspace = create_verification_workspace(tmp_path / "workspace")
    fixture_dir = workspace.fixture_dir / "claude-ai"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "sample.json").write_text("{}")

    with patch("polylogue.pipeline.runner.run_sources", new_callable=AsyncMock) as mock_run:
        run_pipeline_for_fixture_workspace(workspace)

    assert (workspace.inbox_dir / "claude-ai" / "sample.json").exists()
    selected = mock_run.await_args.kwargs["config"].sources
    assert [source.name for source in selected] == ["claude-ai"]
    assert [source.path for source in selected] == [fixture_dir]


def test_generate_synthetic_fixtures_supports_inferred_corpus_specs(tmp_path):
    inferred = (
        CorpusSpec(
            provider="chatgpt",
            package_version="v1",
            count=1,
            messages_min=4,
            messages_max=4,
            seed=3,
            profile_family_ids=("cluster-a",),
        ),
    )

    with patch(
        "polylogue.schemas.operator_inference.list_inferred_corpus_specs",
        return_value=inferred,
    ):
        generate_synthetic_fixtures(
            tmp_path / "fixtures",
            count=1,
            style="showcase",
            corpus_source="inferred",
        )

    assert (Path(tmp_path) / "fixtures" / "chatgpt" / "showcase-00.json").exists()
