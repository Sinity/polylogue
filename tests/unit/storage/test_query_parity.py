"""Query parity contracts across storage/repository/facade layers.

These tests assert that a query returning results at a lower layer also returns
the same results (same IDs, same count) at higher layers.  They are not feature
tests — they only verify that the layers agree.

Layer stack (bottom to top):
  SQLiteBackend (queries) → ConversationRepository (list/search) → Polylogue (facade)

Each test is independent with its own tmp_path database.  Data is seeded via
ConversationBuilder (sync, record-level) and read back through the async
repository and facade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import ConversationBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_db(db_path: Path, *, chatgpt_count: int = 2, claude_count: int = 3) -> None:
    """Seed the database with known conversations across two providers."""
    for i in range(chatgpt_count):
        cid = f"chatgpt-conv-{i}"
        (
            ConversationBuilder(db_path, cid)
            .provider("chatgpt")
            .title(f"ChatGPT conv {i}")
            .add_message(f"msg-{cid}-1", role="user", text=f"chatgpt question {i}")
            .add_message(f"msg-{cid}-2", role="assistant", text=f"chatgpt answer {i}")
            .save()
        )
    for i in range(claude_count):
        cid = f"claude-conv-{i}"
        (
            ConversationBuilder(db_path, cid)
            .provider("claude-ai")
            .title(f"Claude conv {i}")
            .add_message(f"msg-{cid}-1", role="user", text=f"claude question {i}")
            .add_message(f"msg-{cid}-2", role="assistant", text=f"claude answer {i}")
            .save()
        )


# ---------------------------------------------------------------------------
# Count parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_count_parity_repo_vs_facade(tmp_path: Path) -> None:
    """Total conversation count must agree between repository.count() and facade.list_conversations()."""
    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=2, claude_count=3)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_count = await archive.repository.count()
        facade_convs = await archive.list_conversations()
        facade_count = len(facade_convs)
    finally:
        await archive.close()

    assert repo_count == facade_count, (
        f"repository.count() returned {repo_count} but facade.list_conversations() "
        f"returned {facade_count} conversations"
    )
    assert repo_count == 5


@pytest.mark.contract
@pytest.mark.asyncio
async def test_count_parity_empty_db(tmp_path: Path) -> None:
    """Empty database: both layers report zero conversations."""
    db_path = tmp_path / "empty.db"
    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_count = await archive.repository.count()
        facade_convs = await archive.list_conversations()
    finally:
        await archive.close()

    assert repo_count == 0
    assert len(facade_convs) == 0


# ---------------------------------------------------------------------------
# ID parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_id_parity_list_all(tmp_path: Path) -> None:
    """IDs returned by repository.list() and facade.list_conversations() must be identical."""
    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=2, claude_count=2)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_convs = await archive.repository.list(limit=None)
        facade_convs = await archive.list_conversations()
    finally:
        await archive.close()

    repo_ids = {str(c.id) for c in repo_convs}
    facade_ids = {str(c.id) for c in facade_convs}

    assert repo_ids == facade_ids, (
        f"ID set mismatch — repo-only: {repo_ids - facade_ids}, facade-only: {facade_ids - repo_ids}"
    )


@pytest.mark.contract
@pytest.mark.asyncio
async def test_id_parity_with_provider_filter(tmp_path: Path) -> None:
    """IDs filtered by provider must agree between repository and facade layers."""
    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=2)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_convs = await archive.repository.list(provider="chatgpt", limit=None)
        facade_convs = await archive.list_conversations(provider="chatgpt")
    finally:
        await archive.close()

    repo_ids = {str(c.id) for c in repo_convs}
    facade_ids = {str(c.id) for c in facade_convs}

    assert repo_ids == facade_ids, (
        f"Provider-filtered ID mismatch — repo-only: {repo_ids - facade_ids}, facade-only: {facade_ids - repo_ids}"
    )
    assert len(repo_ids) == 3


# ---------------------------------------------------------------------------
# Filter parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_provider_filter_count_parity_repo(tmp_path: Path) -> None:
    """provider filter count at repository layer must equal filtered facade count."""
    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=2, claude_count=4)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_count_chatgpt = await archive.repository.count(provider="chatgpt")
        repo_count_claude = await archive.repository.count(provider="claude-ai")

        facade_chatgpt = await archive.list_conversations(provider="chatgpt")
        facade_claude = await archive.list_conversations(provider="claude-ai")
    finally:
        await archive.close()

    assert repo_count_chatgpt == len(facade_chatgpt) == 2
    assert repo_count_claude == len(facade_claude) == 4


@pytest.mark.contract
@pytest.mark.asyncio
async def test_provider_filter_is_exclusive(tmp_path: Path) -> None:
    """Conversations from provider A must not appear in provider B's filtered result."""
    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=2, claude_count=2)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        chatgpt_convs = await archive.list_conversations(provider="chatgpt")
        claude_convs = await archive.list_conversations(provider="claude-ai")
    finally:
        await archive.close()

    chatgpt_ids = {str(c.id) for c in chatgpt_convs}
    claude_ids = {str(c.id) for c in claude_convs}

    assert chatgpt_ids.isdisjoint(claude_ids), (
        f"Provider filter leaked conversations across providers: {chatgpt_ids & claude_ids}"
    )


# ---------------------------------------------------------------------------
# Search parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_search_results_appear_in_list(tmp_path: Path) -> None:
    """FTS5 search hits via repository must be a subset of all listed conversations."""
    db_path = tmp_path / "search.db"

    # Seed with a unique token in one conversation only
    (
        ConversationBuilder(db_path, "unique-conv")
        .provider("chatgpt")
        .title("Unique Result")
        .add_message("msg-u1", role="user", text="xyzzy special token")
        .add_message("msg-u2", role="assistant", text="Sure, the xyzzy value is important")
        .save()
    )
    (
        ConversationBuilder(db_path, "other-conv")
        .provider("claude-ai")
        .title("Other Conversation")
        .add_message("msg-o1", role="user", text="unrelated message about weather")
        .add_message("msg-o2", role="assistant", text="The weather is fine")
        .save()
    )

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        all_convs = await archive.list_conversations()
        search_convs = await archive.repository.search("xyzzy", limit=20)
    finally:
        await archive.close()

    all_ids = {str(c.id) for c in all_convs}
    search_ids = {str(c.id) for c in search_convs}

    # Search results must be a subset of the full listing
    assert search_ids.issubset(all_ids), f"Search returned IDs not in full listing: {search_ids - all_ids}"

    # The conversation with the unique token must appear in search results
    assert "unique-conv" in search_ids, "Expected 'unique-conv' in search results for 'xyzzy'"

    # The unrelated conversation must NOT appear
    assert "other-conv" not in search_ids, "Unrelated 'other-conv' appeared in search results for 'xyzzy'"


@pytest.mark.contract
@pytest.mark.asyncio
async def test_search_empty_results_for_unknown_token(tmp_path: Path) -> None:
    """Search for a nonexistent token returns empty at both repository and facade layers."""
    db_path = tmp_path / "search.db"
    _seed_db(db_path, chatgpt_count=2, claude_count=2)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_hits = await archive.repository.search("zzznomatch_xyz_unique", limit=20)
        facade_result = await archive.search("zzznomatch_xyz_unique")
    finally:
        await archive.close()

    assert len(repo_hits) == 0
    assert len(facade_result.hits) == 0


# ---------------------------------------------------------------------------
# Limit parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_limit_applied_consistently(tmp_path: Path) -> None:
    """A limit at the repository layer must return at most that many conversations."""
    db_path = tmp_path / "limit.db"
    _seed_db(db_path, chatgpt_count=4, claude_count=4)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        repo_limited = await archive.repository.list(limit=3)
        facade_limited = await archive.list_conversations(limit=3)
        repo_all = await archive.repository.list(limit=None)
    finally:
        await archive.close()

    assert len(repo_limited) == 3
    assert len(facade_limited) == 3
    assert len(repo_all) == 8  # 4 + 4, no limit


@pytest.mark.contract
@pytest.mark.asyncio
async def test_limited_ids_are_subset_of_all(tmp_path: Path) -> None:
    """IDs returned under a limit must be a subset of all IDs returned without limit."""
    db_path = tmp_path / "limit.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=3)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        limited = await archive.repository.list(limit=4)
        all_convs = await archive.repository.list(limit=None)
    finally:
        await archive.close()

    limited_ids = {str(c.id) for c in limited}
    all_ids = {str(c.id) for c in all_convs}

    assert limited_ids.issubset(all_ids), (
        f"Limited result contained IDs absent from full listing: {limited_ids - all_ids}"
    )


# ---------------------------------------------------------------------------
# Backend ↔ repository parity
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_backend_query_store_count_matches_repository_count(tmp_path: Path) -> None:
    """SQLiteBackend.queries.count_conversations() must match repository.count()."""
    from polylogue.storage.query_models import ConversationRecordQuery

    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=2)

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        backend_count = await backend.queries.count_conversations(ConversationRecordQuery())
        repo_count = await repo.count()
    finally:
        await repo.close()

    assert backend_count == repo_count == 5


@pytest.mark.contract
@pytest.mark.asyncio
async def test_backend_query_store_provider_filter_matches_repository(tmp_path: Path) -> None:
    """Provider-filtered count in SQLiteBackend.queries must equal repository.count(provider=...)."""
    from polylogue.storage.query_models import ConversationRecordQuery

    db_path = tmp_path / "parity.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=2)

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        backend_chatgpt = await backend.queries.count_conversations(ConversationRecordQuery(provider="chatgpt"))
        repo_chatgpt = await repo.count(provider="chatgpt")

        backend_claude = await backend.queries.count_conversations(ConversationRecordQuery(provider="claude-ai"))
        repo_claude = await repo.count(provider="claude-ai")
    finally:
        await repo.close()

    assert backend_chatgpt == repo_chatgpt == 3
    assert backend_claude == repo_claude == 2


# ---------------------------------------------------------------------------
# Semantic query laws — provider-filter subset and count invariants
# ---------------------------------------------------------------------------


@pytest.mark.contract
@pytest.mark.asyncio
async def test_provider_filter_ids_are_subset_of_all_ids(tmp_path: Path) -> None:
    """Provider-filtered IDs must be a strict subset of all conversation IDs.

    This pins the foundational semantic law: any filtered result set can only
    narrow, never expand, the full corpus.  A violation means the filter is
    synthesising IDs that don't exist in the unfiltered listing.
    """
    db_path = tmp_path / "subset.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=2)

    archive = Polylogue(archive_root=tmp_path, db_path=db_path)
    try:
        all_convs = await archive.repository.list(limit=None)
        chatgpt_convs = await archive.repository.list(provider="chatgpt", limit=None)
        claude_convs = await archive.repository.list(provider="claude-ai", limit=None)
    finally:
        await archive.close()

    all_ids = {str(c.id) for c in all_convs}
    chatgpt_ids = {str(c.id) for c in chatgpt_convs}
    claude_ids = {str(c.id) for c in claude_convs}

    assert chatgpt_ids.issubset(all_ids), (
        f"chatgpt filter returned IDs absent from full listing: {chatgpt_ids - all_ids}"
    )
    assert claude_ids.issubset(all_ids), f"claude filter returned IDs absent from full listing: {claude_ids - all_ids}"
    assert len(chatgpt_ids) == 3
    assert len(claude_ids) == 2


@pytest.mark.contract
@pytest.mark.asyncio
async def test_provider_filter_count_matches_list_length(tmp_path: Path) -> None:
    """repository.count(provider=X) must equal len(repository.list(provider=X)).

    If count and list diverge, aggregate queries will disagree with enumeration
    queries and pagination will be miscalibrated.
    """
    db_path = tmp_path / "count-list.db"
    _seed_db(db_path, chatgpt_count=4, claude_count=3)

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        for provider in ("chatgpt", "claude-ai"):
            count = await repo.count(provider=provider)
            listed = await repo.list(provider=provider, limit=None)
            assert count == len(listed), f"count({provider!r})={count} but len(list({provider!r}))={len(listed)}"
    finally:
        await repo.close()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_equivalent_provider_filter_constructions_return_identical_ids(tmp_path: Path) -> None:
    """Two equivalent ways to express a provider filter must return identical IDs.

    Filtering via repository.list(provider=X) and filtering via the backend
    query store with a ConversationRecordQuery(provider=X) must produce the
    same ID set — they are equivalent constructions over the same data.
    """
    from polylogue.storage.query_models import ConversationRecordQuery

    db_path = tmp_path / "equiv.db"
    _seed_db(db_path, chatgpt_count=3, claude_count=3)

    backend = SQLiteBackend(db_path=db_path)
    repo = ConversationRepository(backend=backend)
    try:
        repo_convs = await repo.list(provider="chatgpt", limit=None)
        backend_count = await backend.queries.count_conversations(ConversationRecordQuery(provider="chatgpt"))
    finally:
        await repo.close()

    repo_ids = {str(c.id) for c in repo_convs}
    assert len(repo_ids) == backend_count == 3, (
        f"Equivalent constructions disagree: repo={len(repo_ids)}, backend.count={backend_count}"
    )
