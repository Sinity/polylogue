"""Production evidence for message-grain embedding publication races.

The daemon's common derivation adapter reserves a current message binding,
performs provider work without writer authority, then refuses publication if
canonical message semantics changed.  Session attempt/status rows are telemetry
and cannot turn that refused replacement into a current vector.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from polylogue.storage.sqlite.write_lease import write_lease

_DRIFT_TEXT = "The prose this attempt was computed from, with enough words to embed."
_REPLACEMENT_TEXT = "Different prose was written while the provider request was still in flight."
T = TypeVar("T")


def _write_single_message_session(root: Path, *, native_id: str, text: str) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    with ArchiveStore(root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )


def test_message_publication_refuses_source_mutation_during_provider_call(tmp_path: Path) -> None:
    """A vector computed from superseded message semantics is never published.

    Anti-vacuity: remove the publication-time input-binding revalidation and
    ``publish`` returns true, leaving a vector ref for the source snapshot the
    provider observed before the concurrent full replacement.
    """

    from polylogue.operations.embedding_derivation import make_embedding_frame
    from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    root = tmp_path / "archive"
    session_id = _write_single_message_session(root, native_id="drift-source", text=_DRIFT_TEXT)
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    with sqlite3.connect(embeddings_db) as probe:
        loaded, error = try_load_sqlite_vec(probe)
    if not loaded:
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")

    class _MutatingProvider:
        model = "voyage-4"
        dimension = 1024

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            assert texts == [_DRIFT_TEXT]
            assert input_type == "document"
            assert _write_single_message_session(root, native_id="drift-source", text=_REPLACEMENT_TEXT) == session_id
            return [[0.25] * self.dimension]

    def admit(actor: str, function: Callable[[], T]) -> T:
        with write_lease(actor, archive_root=root):
            return function()

    adapter = EmbeddingDerivationAdapter(index_db, _MutatingProvider(), archive_root=root, reserve=admit)
    frame = make_embedding_frame(index_db, archive_root=root, adapter=adapter, scope=(session_id,))
    (key,), cursor = adapter.required_page(frame, cursor=None, limit=10)
    assert cursor is None
    replacement = adapter.compute(frame, key)

    with write_lease("test.embedding.publish", archive_root=root):
        assert adapter.publish(frame, replacement) is False
    assert adapter.inspect(frame, (key,))[key] == "missing"
