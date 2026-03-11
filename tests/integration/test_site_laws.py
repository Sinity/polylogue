"""Generalized contracts for static-site builder and CLI surfaces."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.cli.commands.site import site_command
from polylogue.cli.types import AppEnv
from polylogue.paths import safe_path_component
from polylogue.services import build_runtime_services
from polylogue.site.builder import ConversationIndex, SiteBuilder, SiteConfig
from tests.infra.strategies import (
    build_conversation_summary,
    build_message_counts,
    conversation_summary_batch_strategy,
    conversation_summary_spec_strategy,
    expected_index_pages,
    site_archive_spec_strategy,
)


async def _empty_messages(*args, **kwargs):
    if False:  # pragma: no cover
        yield None


async def _summary_pages(*pages):
    for page in pages:
        if page:
            yield page


def _configure_summary_pages(repo: AsyncMock, summaries):
    repo.iter_summary_pages = MagicMock(
        side_effect=lambda *args, **kwargs: _summary_pages(summaries)
    )
    repo.iter_messages = _empty_messages


def _make_site_env(*, backend: AsyncMock, repository: AsyncMock) -> AppEnv:
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.summary = MagicMock()
    repository.backend = backend
    return AppEnv(ui=ui, services=build_runtime_services(backend=backend, repository=repository))


@settings(max_examples=60, deadline=None)
@given(spec=conversation_summary_spec_strategy())
def test_conversation_index_from_summary_contract(spec) -> None:
    summary = build_conversation_summary(spec)
    index = ConversationIndex.from_summary(summary, spec.message_count)

    assert index.id == spec.conversation_id
    assert index.title == summary.display_title
    assert index.provider == spec.provider
    assert index.message_count == spec.message_count
    assert index.preview == (spec.summary or "")
    assert index.path == (
        f"{safe_path_component(spec.provider, fallback='provider')}/"
        f"{spec.conversation_id[:12]}/conversation.html"
    )
    if summary.created_at is not None:
        assert index.created_at == summary.created_at.strftime("%Y-%m-%d")
    else:
        assert index.created_at is None
    if summary.updated_at is not None:
        assert index.updated_at == summary.updated_at.strftime("%Y-%m-%d %H:%M")
    else:
        assert index.updated_at is None


@settings(max_examples=40, deadline=None)
@given(spec=site_archive_spec_strategy())
def test_site_builder_archive_shape_contract(spec) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summaries = [build_conversation_summary(summary) for summary in spec.summaries]
        backend = AsyncMock()
        backend.get_message_counts_batch.return_value = build_message_counts(spec.summaries)
        repository = AsyncMock()
        _configure_summary_pages(repository, summaries)

        output_dir = tmp_path / "site"
        if spec.precreate_output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        builder = SiteBuilder(
            output_dir=output_dir,
            config=SiteConfig(
                title=spec.custom_title or "Contract Archive",
                enable_search=spec.enable_search,
                search_provider=spec.search_provider,
                include_dashboard=spec.include_dashboard,
            ),
            backend=backend,
            repository=repository,
        )

        result = builder.build()

        providers = {summary.provider for summary in spec.summaries}
        assert result["conversations"] == len(spec.summaries)
        assert result["index_pages"] == expected_index_pages(spec)
        assert (output_dir / "index.html").exists()
        for provider in providers:
            assert (output_dir / safe_path_component(provider, fallback="provider")).exists()
        assert (output_dir / "dashboard.html").exists() == spec.include_dashboard


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    summaries=conversation_summary_batch_strategy(min_size=0, max_size=5),
    mode=st.sampled_from(("disabled", "lunr", "pagefind")),
)
def test_site_builder_search_surface_contract(summaries, mode: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backend = AsyncMock()
        backend.get_message_counts_batch.return_value = build_message_counts(summaries)
        repository = AsyncMock()
        _configure_summary_pages(repository, [build_conversation_summary(summary) for summary in summaries])
        output_dir = tmp_path / "site"

        enable_search = mode != "disabled"
        search_provider = "pagefind" if mode == "pagefind" else "lunr"
        builder = SiteBuilder(
            output_dir=output_dir,
            config=SiteConfig(
                title="Search Contract",
                enable_search=enable_search,
                search_provider=search_provider,
                include_dashboard=False,
            ),
            backend=backend,
            repository=repository,
        )
        builder.build()

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        if mode == "disabled":
            assert "search-index.json" not in index_html
            assert "_pagefind/pagefind-ui.js" not in index_html
            assert not (output_dir / "search-index.json").exists()
            assert not (output_dir / "pagefind.json").exists()
        elif mode == "lunr":
            assert "search-index.json" in index_html
            assert "_pagefind/pagefind-ui.js" not in index_html
            payload = json.loads((output_dir / "search-index.json").read_text(encoding="utf-8"))
            assert len(payload) == len(summaries)
        else:
            assert "_pagefind/pagefind-ui.js" in index_html
            assert not (output_dir / "search-index.json").exists()
            assert (output_dir / "pagefind.json").exists()


@settings(max_examples=30, deadline=None)
@given(spec=site_archive_spec_strategy())
def test_site_command_contract(spec) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runner = CliRunner()
        summaries = [build_conversation_summary(summary) for summary in spec.summaries]
        backend = AsyncMock()
        backend.get_message_counts_batch.return_value = build_message_counts(spec.summaries)
        repository = AsyncMock()
        _configure_summary_pages(repository, summaries)
        env = _make_site_env(backend=backend, repository=repository)
        output_dir = tmp_path / "site"

        args = ["--output", str(output_dir)]
        if spec.custom_title:
            args.extend(["--title", spec.custom_title])
        if not spec.enable_search:
            args.append("--no-search")
        else:
            args.extend(["--search-provider", spec.search_provider])
        if not spec.include_dashboard:
            args.append("--no-dashboard")

        result = runner.invoke(site_command, args, obj=env)

        assert result.exit_code == 0
        assert f"Site generated: {len(spec.summaries)} conversations" in result.output
        assert f"{expected_index_pages(spec)} index pages" in result.output
        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        assert (spec.custom_title or "Polylogue Archive") in index_html
        assert (output_dir / "dashboard.html").exists() == spec.include_dashboard
