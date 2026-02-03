"""Tests for ConversationRepository and render_conversation functions.

This module tests:
1. Repository queries (get, list, list with provider filter, search)
2. Repository partial ID resolution
3. Render output formatting (markdown and HTML)
4. Render with attachments
5. Semantic projections through the repository
"""

from __future__ import annotations

import pytest

from polylogue.ingestion import IngestBundle, ingest_bundle
from polylogue.lib.repository import ConversationRepository
from polylogue.rendering.renderers import HTMLRenderer
from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
from tests.factories import DbFactory
from tests.helpers import make_attachment, make_conversation, make_message


# test_db fixture is in conftest.py


# ============================================================================
# REPOSITORY: get() Tests
# ============================================================================


def test_repository_get_conversation(test_db):
    """ConversationRepository.get() returns Conversation with messages loaded."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create a conversation with multiple messages
    factory.create_conversation(
        id="c1",
        provider="chatgpt",
        title="Test Conversation",
        messages=[
            {"id": "m1", "role": "user", "text": "Hello, how are you?"},
            {"id": "m2", "role": "assistant", "text": "I'm doing well, thanks for asking!"},
        ],
    )

    # Get the conversation
    conv = repo.get("c1")

    assert conv is not None
    assert conv.id == "c1"
    assert conv.provider == "chatgpt"
    assert conv.title == "Test Conversation"
    assert len(conv.messages) == 2

    # Verify message order and content
    assert conv.messages[0].id == "m1"
    assert conv.messages[0].role == "user"
    assert conv.messages[0].text == "Hello, how are you?"
    assert conv.messages[1].id == "m2"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].text == "I'm doing well, thanks for asking!"


def test_repository_get_nonexistent_returns_none(test_db):
    """ConversationRepository.get() returns None for nonexistent ID."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    result = repo.get("nonexistent-id")

    assert result is None


def test_repository_get_with_provider_meta(test_db):
    """Repository correctly deserializes provider_meta JSON."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(
        id="c-meta",
        provider="claude",
        messages=[{"id": "m1", "role": "user", "text": "test"}],
    )

    conv = repo.get("c-meta")

    assert conv is not None
    # Conversation should be retrievable even with null provider_meta
    assert conv.id == "c-meta"


# ============================================================================
# REPOSITORY: list() Tests
# ============================================================================


def test_repository_list_conversations(test_db):
    """ConversationRepository.list() returns all conversations."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create multiple conversations
    factory.create_conversation(id="c1", provider="chatgpt", title="Conv 1", messages=[])
    factory.create_conversation(id="c2", provider="claude", title="Conv 2", messages=[])
    factory.create_conversation(id="c3", provider="codex", title="Conv 3", messages=[])

    # List all
    result = repo.list()

    assert len(result) == 3
    ids = {c.id for c in result}
    assert ids == {"c1", "c2", "c3"}


def test_repository_list_with_limit(test_db):
    """ConversationRepository.list() respects limit parameter."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create 5 conversations
    for i in range(5):
        factory.create_conversation(id=f"c{i}", provider="test", messages=[])

    result = repo.list(limit=3)

    assert len(result) == 3


def test_repository_list_with_offset(test_db):
    """ConversationRepository.list() respects offset parameter."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create 5 conversations
    for i in range(5):
        factory.create_conversation(id=f"c{i}", provider="test", title=f"Conv {i}", messages=[])

    # Get first 2
    first_batch = repo.list(limit=2, offset=0)
    assert len(first_batch) == 2

    # Get second 2
    second_batch = repo.list(limit=2, offset=2)
    assert len(second_batch) == 2

    # Batches should be different
    first_ids = {c.id for c in first_batch}
    second_ids = {c.id for c in second_batch}
    assert first_ids != second_ids


def test_repository_list_empty_returns_empty(test_db):
    """ConversationRepository.list() returns empty list when no conversations."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    result = repo.list()

    assert result == []


# ============================================================================
# REPOSITORY: list() with provider filter
# ============================================================================


def test_repository_list_by_provider(test_db):
    """ConversationRepository.list(provider=...) filters by provider."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create conversations with different providers
    factory.create_conversation(id="c1", provider="chatgpt", messages=[])
    factory.create_conversation(id="c2", provider="chatgpt", messages=[])
    factory.create_conversation(id="c3", provider="claude", messages=[])

    # Filter by chatgpt
    result = repo.list(provider="chatgpt")

    assert len(result) == 2
    for conv in result:
        assert conv.provider == "chatgpt"


def test_repository_list_by_provider_excludes_others(test_db):
    """ConversationRepository.list(provider=...) excludes other providers."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(id="c1", provider="chatgpt", messages=[])
    factory.create_conversation(id="c2", provider="claude", messages=[])
    factory.create_conversation(id="c3", provider="codex", messages=[])

    result = repo.list(provider="claude")

    assert len(result) == 1
    assert result[0].provider == "claude"
    assert result[0].id == "c2"


def test_repository_list_by_provider_returns_empty_when_no_match(test_db):
    """ConversationRepository.list(provider=...) returns empty for no matches."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(id="c1", provider="chatgpt", messages=[])

    result = repo.list(provider="nonexistent-provider")

    assert result == []


# ============================================================================
# REPOSITORY: search() Tests
# ============================================================================


def test_repository_search_returns_matching_conversations(test_db, workspace_env):
    """ConversationRepository.search() returns conversations matching query."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    # Create conversations with searchable content
    factory.create_conversation(
        id="c1", provider="test", messages=[{"id": "m1", "role": "user", "text": "Python programming"}]
    )
    factory.create_conversation(
        id="c2", provider="test", messages=[{"id": "m2", "role": "user", "text": "JavaScript tutorials"}]
    )

    # Build the FTS index
    with open_connection(test_db) as conn:
        from polylogue.storage.index import rebuild_index

        rebuild_index(conn)

    # Search for Python
    result = repo.search("Python")

    assert len(result) >= 1
    assert any(c.id == "c1" for c in result)


def test_repository_search_raises_when_index_not_built(test_db, db_without_fts):
    """ConversationRepository.search() raises DatabaseError when FTS table doesn't exist."""
    backend = SQLiteBackend(db_path=db_without_fts)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(db_without_fts)

    # Create conversation without building index
    factory.create_conversation(
        id="c1", provider="test", messages=[{"id": "m1", "role": "user", "text": "Python"}]
    )

    # Search without index should raise DatabaseError
    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        repo.search("Python")
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Search index not built" in str(exc_info.value)


# ============================================================================
# REPOSITORY: resolve_id() Tests
# ============================================================================


def test_repository_resolve_id_exact_match(test_db):
    """ConversationRepository.resolve_id() returns full ID for exact match."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(id="conv:abc123:full", provider="test", messages=[])

    result = repo.resolve_id("conv:abc123:full")

    assert result == "conv:abc123:full"


def test_repository_resolve_id_prefix_match(test_db):
    """ConversationRepository.resolve_id() returns full ID for unique prefix."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(id="conv:abc123:full", provider="test", messages=[])

    result = repo.resolve_id("conv:abc")

    assert result == "conv:abc123:full"


def test_repository_resolve_id_ambiguous_returns_none(test_db):
    """ConversationRepository.resolve_id() returns None for ambiguous prefix."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(id="conv:abc:1", provider="test", messages=[])
    factory.create_conversation(id="conv:abc:2", provider="test", messages=[])

    result = repo.resolve_id("conv:abc")

    assert result is None


# ============================================================================
# REPOSITORY: view() Tests
# ============================================================================


def test_repository_view_with_prefix(test_db):
    """ConversationRepository.view() resolves prefix and returns Conversation."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(
        id="claude:conv123abc", provider="claude", messages=[{"id": "m1", "role": "user", "text": "hello"}]
    )

    # Access by prefix
    conv = repo.view("claude:conv")

    assert conv is not None
    assert conv.id == "claude:conv123abc"


def test_repository_view_returns_none_for_nonexistent(test_db):
    """ConversationRepository.view() returns None for nonexistent ID."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)

    result = repo.view("nonexistent")

    assert result is None


# ============================================================================
# RENDER: Markdown Output Tests
# ============================================================================


def test_render_conversation_markdown_has_structure(workspace_env, storage_repository):
    """render_conversation() produces valid markdown with title and role headers."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-md", title="My Conversation"),
        messages=[
            make_message("m1", "c-md", text="Hello, assistant!"),
            make_message("m2", "c-md", role="assistant", text="Hi there, user!"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-md", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Check structure
    assert "# My Conversation" in markdown
    assert "## user" in markdown
    assert "## assistant" in markdown
    assert "Hello, assistant!" in markdown
    assert "Hi there, user!" in markdown


def test_render_conversation_markdown_includes_provider(workspace_env, storage_repository):
    """render_conversation() markdown includes provider information."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-prov", provider_name="claude"),
        messages=[],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-prov", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    assert "Provider: claude" in markdown
    assert "Conversation ID: c-prov" in markdown


def test_render_conversation_markdown_messages_separated(workspace_env, storage_repository):
    """render_conversation() separates messages with blank lines."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-sep"),
        messages=[
            make_message("m1", "c-sep", text="First message"),
            make_message("m2", "c-sep", role="assistant", text="Second message"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-sep", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Messages should be separated by blank lines (## header, optional timestamp, text, blank line)
    lines = markdown.split("\n")
    # Verify structure: should have multiple sections
    assert len([l for l in lines if l.startswith("## ")]) == 2


def test_render_conversation_markdown_with_timestamp(workspace_env, storage_repository):
    """render_conversation() includes timestamps when present."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-ts"),
        messages=[make_message("m1", "c-ts", text="Hello", timestamp="2024-01-15T10:30:00")],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-ts", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    assert "Timestamp: 2024-01-15T10:30:00" in markdown


# ============================================================================
# RENDER: HTML Output Tests
# ============================================================================


def test_render_conversation_html_valid(workspace_env, storage_repository):
    """render_conversation() produces valid HTML with proper structure."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-html", title="HTML Test"),
        messages=[
            make_message("m1", "c-html", text="Question?"),
            make_message("m2", "c-html", role="assistant", text="Answer!"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-html", output_root)

    html = html_path.read_text(encoding="utf-8")

    # Check HTML structure
    assert "<!doctype html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "<title>HTML Test</title>" in html


def test_render_conversation_html_escapes_content(workspace_env, storage_repository):
    """render_conversation() escapes HTML special characters."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-esc", title="<script>alert('xss')</script>"),
        messages=[make_message("m1", "c-esc", text="<img src=x onerror='alert(1)'>")],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-esc", output_root)

    html = html_path.read_text(encoding="utf-8")

    # Script should be escaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # IMG should be escaped
    assert "<img" not in html
    assert "&lt;img" in html


def test_render_conversation_html_includes_content(workspace_env, storage_repository):
    """render_conversation() HTML includes conversation content."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-con", title="Content Test"),
        messages=[make_message("m1", "c-con", text="Important content here")],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-con", output_root)

    html = html_path.read_text(encoding="utf-8")

    assert "Important content here" in html


# ============================================================================
# RENDER: Attachments Tests
# ============================================================================


def test_render_conversation_with_message_attachments(workspace_env, storage_repository):
    """render_conversation() includes attachments associated with messages."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-att-msg", title="With Attachments"),
        messages=[make_message("m1", "c-att-msg", text="Check this file")],
        attachments=[make_attachment("att-1", "c-att-msg", "m1", mime_type="application/pdf", provider_meta={"name": "document.pdf"})],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-att-msg", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Attachment should be referenced
    assert "Attachment:" in markdown or "attachment" in markdown.lower()


def test_render_conversation_with_orphan_attachments(workspace_env, storage_repository):
    """render_conversation() includes attachments not linked to messages."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-orphan", title="With Orphan Attachments"),
        messages=[make_message("m1", "c-orphan", text="Some text")],
        attachments=[
            # Orphan attachment (no message_id)
            make_attachment("att-orphan", "c-orphan", None, mime_type="text/plain", size_bytes=100, provider_meta={"name": "orphaned_file.txt"}),
        ],
    )

    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-orphan", output_root)
    md_path = html_path.parent / "conversation.md"

    markdown = md_path.read_text(encoding="utf-8")

    # Orphan attachments should appear in "attachments" section
    assert "attachments" in markdown.lower()


# ============================================================================
# RENDER: File Output Tests
# ============================================================================


def test_render_conversation_writes_markdown_file(workspace_env, storage_repository):
    """render_conversation() writes markdown file to expected location."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-file", title="File Test"),
        messages=[],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-file", output_root)
    md_path = html_path.parent / "conversation.md"

    assert md_path.exists()
    assert md_path.suffix == ".md"
    assert md_path.name == "conversation.md"


def test_render_conversation_writes_html_file(workspace_env, storage_repository):
    """render_conversation() writes HTML file to expected location."""
    archive_root = workspace_env["archive_root"]

    bundle = IngestBundle(
        conversation=make_conversation("c-html-file", title="HTML File Test"),
        messages=[],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("c-html-file", output_root)

    assert html_path.exists()
    assert html_path.suffix == ".html"
    assert html_path.name == "conversation.html"


# ============================================================================
# INTEGRATION: Repository + Semantic Projections
# ============================================================================


def test_repository_conversation_supports_projections(test_db):
    """Conversations from repository support semantic projections."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(
        id="c-proj",
        provider="test",
        messages=[
            {"id": "m1", "role": "user", "text": "How do I write Python?"},
            {"id": "m2", "role": "assistant", "text": "Here's how to write Python code: ..."},
            {"id": "m3", "role": "user", "text": "Thanks!"},
        ],
    )

    conv = repo.get("c-proj")

    # Should support projection methods
    assert hasattr(conv, "substantive_only")
    assert hasattr(conv, "user_only")
    assert hasattr(conv, "without_noise")

    # Should be able to use them
    user_conv = conv.user_only()
    assert len(user_conv.messages) == 2
    assert all(m.is_user for m in user_conv.messages)


def test_repository_conversation_supports_iteration(test_db):
    """Conversations from repository support iteration methods."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(
        id="c-iter",
        provider="test",
        messages=[
            {"id": "m1", "role": "user", "text": "Question? This is a long enough message to be substantive."},
            {"id": "m2", "role": "assistant", "text": "Answer! This is also a long enough message to be substantive."},
        ],
    )

    conv = repo.get("c-iter")

    # Should support iter_pairs - note: iter_pairs() only yields pairs from substantive messages
    pairs = list(conv.iter_pairs())
    assert len(pairs) == 1
    assert pairs[0].user.text == "Question? This is a long enough message to be substantive."
    assert pairs[0].assistant.text == "Answer! This is also a long enough message to be substantive."


def test_repository_conversation_supports_statistics(test_db):
    """Conversations from repository support statistics methods."""
    backend = SQLiteBackend(db_path=test_db)
    repo = ConversationRepository(backend=backend)
    factory = DbFactory(test_db)

    factory.create_conversation(
        id="c-stats",
        provider="test",
        messages=[
            {"id": "m1", "role": "user", "text": "Hello world"},
            {"id": "m2", "role": "assistant", "text": "Hi there friend"},
        ],
    )

    conv = repo.get("c-stats")

    assert conv.message_count == 2
    assert conv.user_message_count == 1
    assert conv.assistant_message_count == 1
    assert conv.word_count > 0


# ============================================================================
# STORAGE REPOSITORY TESTS (merged from test_repository_backend.py)
# ============================================================================


from polylogue.storage.repository import StorageRepository


@pytest.fixture
def repository_with_backend(sqlite_backend: SQLiteBackend) -> StorageRepository:
    """Create a repository using the backend."""
    return StorageRepository(backend=sqlite_backend)


def test_repository_save_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test saving conversation via backend abstraction."""
    conv = make_conversation("conv1", title="Test Conversation")
    msg = make_message("msg1", "conv1", text="Hello")

    counts = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["skipped_conversations"] == 0
    assert counts["skipped_messages"] == 0


def test_repository_deduplication_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test that duplicate conversations are skipped when using backend."""
    conv = make_conversation("conv1", title="Test", created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z", content_hash="samehash")
    msg = make_message("msg1", "conv1", text="Hello")

    # First save
    counts1 = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )
    assert counts1["conversations"] == 1
    assert counts1["messages"] == 1

    # Second save with same hash (should skip)
    counts2 = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[],
    )
    assert counts2["conversations"] == 0
    assert counts2["skipped_conversations"] == 1
    assert counts2["messages"] == 0
    assert counts2["skipped_messages"] == 1


def test_repository_with_attachments_via_backend(repository_with_backend: StorageRepository) -> None:
    """Test saving conversation with attachments via backend."""
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="Hello")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    counts = repository_with_backend.save_conversation(
        conversation=conv,
        messages=[msg],
        attachments=[att],
    )

    assert counts["conversations"] == 1
    assert counts["messages"] == 1
    assert counts["attachments"] == 1
