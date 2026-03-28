"""Generalized contracts for static-site builder and CLI surfaces."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.cli.commands.site import site_command
from polylogue.cli.types import AppEnv
from polylogue.paths import safe_path_component
from polylogue.services import build_runtime_services
from polylogue.site.builder import ArchiveIndexStats, ConversationIndex, SiteBuilder, SiteConfig
from tests.infra.strategies import (
    ConversationSummarySpec,
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


def _make_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.queries = MagicMock()
    backend.queries.get_message_counts_batch = AsyncMock(return_value={})
    return backend


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
        backend = _make_backend()
        backend.queries.get_message_counts_batch.return_value = build_message_counts(spec.summaries)
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
        assert result.archive.total_conversations == len(spec.summaries)
        assert result.outputs.total_index_pages == expected_index_pages(spec)
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
        backend = _make_backend()
        backend.queries.get_message_counts_batch.return_value = build_message_counts(summaries)
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
        backend = _make_backend()
        backend.queries.get_message_counts_batch.return_value = build_message_counts(spec.summaries)
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


def test_site_builder_scan_archive_streaming_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        specs = [
            ConversationSummarySpec(
                conversation_id="conv-a",
                provider="claude-ai",
                title="A",
                summary="alpha",
                tags=("law",),
                created_at=None,
                updated_at=None,
                message_count=2,
            ),
            ConversationSummarySpec(
                conversation_id="conv-b",
                provider="chatgpt",
                title="B",
                summary="beta",
                tags=(),
                created_at=None,
                updated_at=None,
                message_count=3,
            ),
        ]
        summaries = [build_conversation_summary(spec) for spec in specs]
        backend = _make_backend()
        backend.queries.get_message_counts_batch.return_value = build_message_counts(specs)
        repository = AsyncMock()
        _configure_summary_pages(repository, summaries)

        builder = SiteBuilder(
            output_dir=tmp_path / "site",
            config=SiteConfig(enable_search=True, search_provider="lunr", include_dashboard=False),
            backend=backend,
            repository=repository,
        )
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        builder._generate_conversation_page = AsyncMock(return_value="rendered")  # type: ignore[method-assign]

        stats, generated = asyncio.run(builder._scan_archive(incremental=True))

        assert generated.total == len(specs)
        assert generated.rendered == len(specs)
        assert stats.total_conversations == len(specs)
        assert stats.total_messages == sum(spec.message_count for spec in specs)
        assert stats.provider_counts == {
            summary.provider.value: sum(1 for item in summaries if item.provider == summary.provider)
            for summary in summaries
        }
        payload = json.loads((builder.output_dir / "search-index.json").read_text(encoding="utf-8"))
        assert [entry["id"] for entry in payload] == [spec.conversation_id for spec in specs]


def test_site_builder_root_and_provider_index_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        builder = SiteBuilder(
            output_dir=tmp_path / "site",
            config=SiteConfig(title="Archive", include_dashboard=False),
            backend=AsyncMock(),
            repository=AsyncMock(),
        )
        archive_stats = ArchiveIndexStats(
            root_page_conversations=[
                ConversationIndex(
                    id="conv-1",
                    title="One",
                    provider="claude-ai",
                    created_at="2024-01-01",
                    updated_at="2024-01-01 00:00",
                    message_count=2,
                    preview="hello",
                    path="claude/conv-1/conversation.html",
                )
            ],
            provider_counts={"claude-ai": 1, "chatgpt": 2},
            provider_messages={"claude-ai": 2, "chatgpt": 5},
            provider_order=["claude-ai", "chatgpt"],
            total_conversations=3,
            total_messages=7,
        )
        builder._write_template_stream = AsyncMock()  # type: ignore[method-assign]
        builder._iter_conversation_indexes = MagicMock(
            side_effect=lambda provider=None: _summary_pages(
                [
                    ConversationIndex(
                        id=f"{provider}-1",
                        title=f"{provider} title",
                        provider=provider or "claude-ai",
                        created_at="2024-01-01",
                        updated_at="2024-01-01 00:00",
                        message_count=1,
                        preview="preview",
                        path=f"{provider}/conv/conversation.html",
                    )
                ]
            )
        )

        asyncio.run(builder._generate_root_index(archive_stats, generated_at="2026-03-11 10:00"))
        provider_pages = asyncio.run(
            builder._generate_provider_indexes(archive_stats, generated_at="2026-03-11 10:00")
        )

        assert provider_pages == 2
        calls = builder._write_template_stream.await_args_list
        assert calls[0].args[1] == builder.output_dir / "index.html"
        assert calls[0].kwargs["conversations"] == archive_stats.root_page_conversations
        assert calls[1].args[1] == builder.output_dir / safe_path_component("claude-ai", fallback="provider") / "index.html"
        assert calls[2].args[1] == builder.output_dir / safe_path_component("chatgpt", fallback="provider") / "index.html"


def test_site_builder_pagefind_config_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        builder = SiteBuilder(
            output_dir=tmp_path / "site",
            config=SiteConfig(enable_search=True, search_provider="pagefind"),
            backend=AsyncMock(),
            repository=AsyncMock(),
        )
        builder.output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            builder._generate_pagefind_config()

        assert json.loads((builder.output_dir / "pagefind.json").read_text(encoding="utf-8")) == {
            "site": str(builder.output_dir),
            "output_subdir": "_pagefind",
        }
        mock_run.assert_not_called()
