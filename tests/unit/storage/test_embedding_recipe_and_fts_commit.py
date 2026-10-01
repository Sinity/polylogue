"""Embedding recipe identity and the FTS repair transaction boundary."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingRequestSpec
from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder


def test_equal_provider_requests_have_equal_recipe_identities() -> None:
    """Reordering the same request map used to change recipe_hash."""
    options = (("truncation", False), ("output_dtype", "float"))
    first = EmbeddingRecipe.current(model="voyage-3", dimensions=1024, request_options=options)
    second = EmbeddingRecipe.current(model="voyage-3", dimensions=1024, request_options=tuple(reversed(options)))
    first_request = EmbeddingRequestSpec(first, "same evidence")
    second_request = EmbeddingRequestSpec(second, "same evidence")
    assert first_request.provider_request == second_request.provider_request
    assert first_request.vector_derivation_hash == second_request.vector_derivation_hash
    assert first.recipe_hash == second.recipe_hash
    assert first.request_options == second.request_options
    with pytest.raises(ValueError, match="duplicate keys"):
        replace(first, request_options=(("truncation", True), ("truncation", False)))


@pytest.mark.parametrize("element_type", ["float64", "int8", "unknown"])
def test_embedding_recipe_refuses_unimplemented_output_encodings(element_type: str) -> None:
    """A recipe could claim another representation while storage wrote float32."""
    with pytest.raises(ValueError, match="only float32"):
        EmbeddingRecipe.current(model="voyage-3", dimensions=1024, element_type=element_type)
    recipe = EmbeddingRecipe.current(model="voyage-3", dimensions=1024)
    with pytest.raises(ValueError, match="only float32"):
        replace(recipe, element_type=element_type)


def test_fts_owned_commit_failure_rolls_back_and_allows_retry(test_conn: sqlite3.Connection, test_db: Path) -> None:
    """COMMIT outside the try leaves a live owned transaction after rejection."""
    builder = SessionBuilder(test_db, "w5-fts-commit")
    builder.add_message(role="user", text="FTS commit boundary evidence")
    builder.save()
    session_id = builder.native_session_id()
    test_conn.commit()
    denied = False

    def deny_one_commit(action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None) -> int:
        nonlocal denied
        if action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT" and not denied:
            denied = True
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    test_conn.set_authorizer(deny_one_commit)
    try:
        with write_lease("test.w5.fts-commit", archive_root=test_db.parent):
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                repair_message_fts_index_sync(test_conn, [session_id])
        assert denied, "the failure must be at the real owned COMMIT"
        assert not test_conn.in_transaction
    finally:
        test_conn.set_authorizer(None)
    with write_lease("test.w5.fts-retry", archive_root=test_db.parent):
        repair_message_fts_index_sync(test_conn, [session_id])
    assert not test_conn.in_transaction
