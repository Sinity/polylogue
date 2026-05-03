from __future__ import annotations

import pytest

from polylogue.archive.query.plan import ConversationQueryPlan
from polylogue.archive.query.retrieval_candidates import fetch_search_results
from polylogue.errors import DatabaseError
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import ConversationBuilder


@pytest.mark.asyncio
async def test_fetch_search_results_raises_when_message_index_is_incomplete(
    storage_repository: ConversationRepository,
) -> None:
    ConversationBuilder(storage_repository.backend.db_path, "conv-incomplete-search").add_message(
        "m1",
        text="python search contract",
    ).save()

    with open_connection(storage_repository.backend.db_path) as conn:
        conn.execute("DELETE FROM messages_fts")
        conn.commit()

    plan = ConversationQueryPlan(query_terms=("python",))

    with pytest.raises(DatabaseError, match="Search index is incomplete"):
        await fetch_search_results(plan, storage_repository, summaries=True)
