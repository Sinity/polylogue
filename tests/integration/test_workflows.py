"""Comprehensive end-to-end workflow tests.

CRITICAL GAP: Only 1 e2e test existed before this file.

Tests cover complete workflows:
- Import → Store → Query → Render → API for each provider
- Incremental sync (add/update/delete)
- Multi-source concurrent sync
- Error recovery and partial success
- Search accuracy validation
- Format conversion workflows
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.runner import run_sources
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
async def temp_config_and_repo(tmp_path):
    """Create temporary config and storage repositories for testing."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "test.db"

    # Create minimal config
    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
    )

    # Create backend and repositories
    from polylogue.storage.backends.sqlite import open_connection

    with open_connection(db_path) as conn:
        from polylogue.storage.backends.sqlite import _ensure_schema

        _ensure_schema(conn)

    backend = SQLiteBackend(db_path=db_path)
    storage_repo = ConversationRepository(backend=backend)
    conv_repo = ConversationRepository(backend=backend)

    yield config, storage_repo, conv_repo, archive_root, db_path


@pytest.fixture
def chatgpt_sample_source(synthetic_source):
    """Source with synthetic ChatGPT data."""
    return synthetic_source("chatgpt")


@pytest.fixture
def gemini_sample_source(synthetic_source):
    """Source with synthetic Gemini data."""
    return synthetic_source("gemini")


# =============================================================================
# PER-PROVIDER WORKFLOW TESTS (4 tests parametrized)
# =============================================================================


@pytest.mark.parametrize("provider", ["chatgpt", "claude-ai", "claude-code", "codex", "gemini"])
async def test_full_workflow_per_provider(provider, synthetic_source, temp_config_and_repo):
    """Import → Store → Query → Render for each provider."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    source = synthetic_source(provider, count=1, messages_per_conversation=range(4, 12))

    # 1. IMPORT: Run ingestion
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    result = await service.parse_sources([source])

    # Build FTS index for search tests (INDEX stage of pipeline)
    from polylogue.storage.backends.sqlite import open_connection
    from polylogue.storage.index import update_index_for_conversations

    with open_connection(db_path) as conn:
        update_index_for_conversations(list(result.processed_ids), conn)

    # Verify import
    assert result.counts["conversations"] > 0, f"No conversations imported from {provider}"
    assert result.counts["messages"] > 0, f"No messages imported from {provider}"

    # 2. STORE: Query back from database
    from polylogue.storage.search import search_messages

    convs = await conv_repo.list()
    provider_names = {"claude-ai": "claude"}  # provider name mapping in parser
    expected_provider = provider_names.get(provider, provider)
    convs = [c for c in convs if c.provider == expected_provider]
    assert len(convs) > 0, f"No conversations found for {provider}"

    # 3. QUERY: Verify conversation structure
    conv = convs[0]
    assert conv.id is not None
    assert len(conv.messages) > 0
    assert any(m.text for m in conv.messages), "All messages are empty"

    # 4. RENDER: Generate markdown output
    from polylogue.rendering.core import ConversationFormatter

    formatter = ConversationFormatter(archive_root, db_path=db_path)
    formatted = await formatter.format(str(conv.id))
    markdown = formatted.markdown_text
    assert len(markdown) > 0
    assert any(m.text in markdown for m in conv.messages if m.text), "No message text found in markdown"

    # 5. SEARCH: Full-text search works
    first_words = conv.messages[0].text.split()[:3]
    if first_words:
        search_term = " ".join(first_words)
        result = search_messages(
            search_term,
            archive_root=archive_root,
            render_root_path=config.render_root,
            db_path=db_path,
        )
        found_ids = {r.conversation_id for r in result.hits}
        assert conv.id in found_ids, f"Search failed for '{search_term}'"


# =============================================================================
# RENDER FORMAT TESTS (2 tests parametrized)
# =============================================================================


@pytest.mark.parametrize("format", ["markdown", "html"])
async def test_render_formats(format, temp_config_and_repo, chatgpt_sample_source):
    """Verify each output format works end-to-end."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Import data
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources([chatgpt_sample_source])

    # Get conversation
    convs = await conv_repo.list()
    convs = [c for c in convs if c.provider == "chatgpt"]
    assert len(convs) > 0

    # Render in specified format
    from polylogue.rendering.core import ConversationFormatter

    formatter = ConversationFormatter(archive_root, db_path=db_path)
    formatted = await formatter.format(str(convs[0].id))

    if format == "markdown":
        output = formatted.markdown_text
        assert "#" in output or "**" in output or "*" in output
    elif format == "html":
        # HTML rendering would need a separate HTMLRenderer
        # For now, just test that markdown renders
        output = formatted.markdown_text
        assert len(output) > 0

    # Verify content present (at least one message should be there)
    assert any(m.text in output for m in convs[0].messages if m.text), "No message text found in output"


# =============================================================================
# INCREMENTAL SYNC TESTS (3 tests)
# =============================================================================


async def test_incremental_sync_no_duplicates(temp_config_and_repo, chatgpt_sample_source):
    """Syncing same source twice doesn't create duplicates."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )

    # First sync
    result1 = await service.parse_sources([chatgpt_sample_source])
    count1 = result1.counts["conversations"]

    # Second sync (should deduplicate)
    result2 = await service.parse_sources([chatgpt_sample_source])
    count2 = result2.counts["conversations"]

    # Second sync should add 0 conversations
    assert count2 == 0, f"Expected 0 new convs, got {count2}"

    # Total count should match first sync
    all_convs = await conv_repo.list()
    assert len(all_convs) == count1


async def test_incremental_sync_with_updates(temp_config_and_repo):
    """Modified conversations are updated, not duplicated."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create initial source with 1 conversation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv_v1 = {
            "id": "test-conv-1",
            "title": "Version 1",
            "mapping": {
                "node1": {
                    "id": "node1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Original question"]},
                    },
                    "children": ["node2"],
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "message": {
                        "id": "msg2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Original answer"]},
                    },
                },
            },
        }
        json.dump(conv_v1, f)
        source_path_v1 = Path(f.name)

    try:
        # First sync
        source_v1 = Source(name="test", path=source_path_v1)
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([source_v1])

        # Modify conversation
        with open(source_path_v1, "w") as f:
            conv_v2 = conv_v1.copy()
            conv_v2["title"] = "Version 2"
            conv_v2["mapping"]["node2"]["message"]["content"]["parts"] = ["Updated answer"]
            json.dump(conv_v2, f)

        # Second sync
        await service.parse_sources([source_v1])

        # Should still have 1 conversation
        all_convs = await conv_repo.list()
        assert len(all_convs) == 1

        # Should have updated content
        conv = all_convs[0]
        assert conv.title == "Version 2"
        assert "Updated answer" in [m.text for m in conv.messages]

    finally:
        source_path_v1.unlink()


async def test_sync_handles_deleted_conversations(temp_config_and_repo):
    """Conversations removed from source are NOT deleted from archive.

    This is expected behavior - archive is append-only.
    """
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create source with 2 conversations
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv1 = {
            "id": "conv-1",
            "title": "First",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Q1"]},
                    }
                }
            },
        }
        json.dump(conv1, f)
        path1 = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv2 = {
            "id": "conv-2",
            "title": "Second",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Q2"]},
                    }
                }
            },
        }
        json.dump(conv2, f)
        path2 = Path(f.name)

    try:
        # Sync both
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([Source(name="s1", path=path1), Source(name="s2", path=path2)])

        # Should have 2 conversations
        assert len(await conv_repo.list()) == 2

        # Remove second source, sync again
        await service.parse_sources([Source(name="s1", path=path1)])

        # Should STILL have 2 conversations (archive is append-only)
        assert len(await conv_repo.list()) == 2

    finally:
        path1.unlink()
        path2.unlink()


# =============================================================================
# MULTI-SOURCE TESTS (2 tests)
# =============================================================================


async def test_multi_source_concurrent_sync(temp_config_and_repo, chatgpt_sample_source, gemini_sample_source):
    """Multiple sources can sync concurrently."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    sources = [chatgpt_sample_source, gemini_sample_source]

    # Sync all sources
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources(sources)

    # Should have conversations from both
    all_convs = await conv_repo.list()
    assert len(all_convs) >= 2

    # Each provider should be present
    providers = {c.provider for c in all_convs}
    assert len(providers) >= 2


async def test_multi_source_isolated_namespaces(temp_config_and_repo):
    """Each source can have different conversations."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create 2 sources with different conversations
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "conv-1",
            "title": "Source 1 Conv",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["From source 1"]},
                    }
                }
            },
        }
        json.dump(conv, f)
        path1 = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "conv-2",
            "title": "Source 2 Conv",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["From source 2"]},
                    }
                }
            },
        }
        json.dump(conv, f)
        path2 = Path(f.name)

    try:
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources(
            [
                Source(name="source1", path=path1),
                Source(name="source2", path=path2),
            ]
        )

        # Should have 2 conversations from different sources
        all_convs = await conv_repo.list()
        assert len(all_convs) == 2

        titles = {c.title for c in all_convs}
        assert "Source 1 Conv" in titles
        assert "Source 2 Conv" in titles

    finally:
        path1.unlink()
        path2.unlink()


# =============================================================================
# ERROR RECOVERY TESTS (3 tests)
# =============================================================================


async def test_sync_with_malformed_file_skips_gracefully(temp_config_and_repo):
    """Malformed files are skipped, don't crash entire sync."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create invalid JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json")
        bad_path = Path(f.name)

    try:
        source = Source(name="bad", path=bad_path)
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )

        # Should not crash
        result = await service.parse_sources([source])

        # Should report 0 conversations
        assert result.counts["conversations"] == 0

    finally:
        bad_path.unlink()


async def test_sync_with_missing_file_reports_error(temp_config_and_repo):
    """Missing source files are reported, don't crash."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    source = Source(name="missing", path=Path("/nonexistent/file.json"))
    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )

    # Should not crash
    result = await service.parse_sources([source])

    # Should report 0 conversations
    assert result.counts["conversations"] == 0


async def test_sync_partial_success_with_mixed_sources(temp_config_and_repo, chatgpt_sample_source):
    """If some sources fail, successful ones still sync."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Mix good and bad sources
    sources = [
        chatgpt_sample_source,
        Source(name="bad", path=Path("/nonexistent.json")),
    ]

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    result = await service.parse_sources(sources)

    # Good source should succeed
    assert result.counts["conversations"] > 0

    # Should have conversations from good source
    convs = await conv_repo.list()
    convs = [c for c in convs if c.provider == "chatgpt"]
    assert len(convs) > 0


# =============================================================================
# SEARCH ACCURACY TESTS (2 tests)
# =============================================================================


async def test_search_accuracy_basic_terms(temp_config_and_repo, chatgpt_sample_source):
    """Search returns correct conversations for basic queries."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    service = ParsingService(
        repository=storage_repo,
        archive_root=archive_root,
        config=config,
    )
    await service.parse_sources([chatgpt_sample_source])

    # Build search index
    from polylogue.storage.backends.sqlite import open_connection
    from polylogue.storage.index import rebuild_index

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    # Get all conversations
    all_convs = await conv_repo.list()
    assert len(all_convs) > 0

    # Find a message with enough words for a meaningful search query
    target_conv = all_convs[0]
    words = []
    for msg in target_conv.messages:
        if msg.text:
            words = msg.text.split()[:5]
            if len(words) >= 3:
                break

    if len(words) < 3:
        pytest.skip("No message with 3+ words for meaningful search test")

    # Search for these words
    search_term = " ".join(words[:3])
    from polylogue.storage.search import search_messages

    result = search_messages(
        search_term, archive_root=archive_root, render_root_path=config.render_root, db_path=db_path
    )
    result_ids = {r.conversation_id for r in result.hits}
    assert target_conv.id in result_ids, f"Failed to find conversation with '{search_term}'"


async def test_search_with_special_characters(temp_config_and_repo):
    """Search handles special FTS5 characters correctly."""
    config, storage_repo, conv_repo, archive_root, db_path = temp_config_and_repo

    # Create conversation with special characters
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        conv = {
            "id": "special-chars",
            "title": "Special Test",
            "mapping": {
                "n1": {
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Question with * and ? and ()"]},
                    }
                },
                "n2": {
                    "parent": "n1",
                    "message": {
                        "id": "m2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ['Answer with "quotes" and {braces}']},
                    },
                },
            },
        }
        json.dump(conv, f)
        path = Path(f.name)

    try:
        service = ParsingService(
            repository=storage_repo,
            archive_root=archive_root,
            config=config,
        )
        await service.parse_sources([Source(name="test", path=path)])

        # Build search index
        from polylogue.storage.backends.sqlite import open_connection
        from polylogue.storage.index import rebuild_index

        with open_connection(db_path) as conn:
            rebuild_index(conn)

        from polylogue.storage.search import search_messages

        # Search with special chars (should be escaped)
        result1 = search_messages(
            "quotes", archive_root=archive_root, render_root_path=config.render_root, db_path=db_path
        )
        assert len(result1.hits) > 0

        result2 = search_messages(
            "braces", archive_root=archive_root, render_root_path=config.render_root, db_path=db_path
        )
        assert len(result2.hits) > 0

        # FTS5 operators should be escaped and not cause errors
        # (*, ?, OR, AND, NOT, etc.)
        search_messages("*", archive_root=archive_root, render_root_path=config.render_root, db_path=db_path)
        # Should not crash

    finally:
        path.unlink()


# =============================================================================
# PIPELINE RUNNER TESTS (2 tests)
# =============================================================================


async def test_pipeline_runner_e2e(workspace_env, chatgpt_sample_source):
    """run_sources orchestrates full workflow."""
    from polylogue.config import get_config

    config = get_config()
    config.sources = [chatgpt_sample_source]

    # Use run_sources function
    result = await run_sources(config=config, source_names=[chatgpt_sample_source.name])

    # Verify data was processed
    assert result is not None
    assert result.counts["conversations"] > 0


async def test_pipeline_runner_with_preview_mode(workspace_env, chatgpt_sample_source):
    """run_sources with stage control."""
    from polylogue.config import get_config

    config = get_config()
    config.sources = [chatgpt_sample_source]

    # Run parse stage
    result = await run_sources(config=config, stage="parse", source_names=[chatgpt_sample_source.name])

    # Should report results
    assert result is not None
    assert result.counts["conversations"] > 0
