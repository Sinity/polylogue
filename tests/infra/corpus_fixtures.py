"""SyntheticCorpus integration fixture for seeding DBs through the real pipeline.

Provides a ``corpus_seeded_db`` factory fixture that generates wire-format bytes,
writes them to temp files, and runs the pipeline to produce a seeded database.
This is additive infrastructure — it does NOT replace ConversationBuilder tests.

Usage::

    def test_query_on_real_data(corpus_seeded_db):
        db_path = corpus_seeded_db(providers=("chatgpt",), count=2, seed=42)
        # db_path now contains parsed data from the real pipeline
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _seed_db(
    tmp_path: Path,
    providers: Sequence[str] = ("chatgpt", "claude-ai"),
    count: int = 3,
    seed: int = 42,
) -> Path:
    """Generate wire-format bytes, write to temp, run pipeline, return db_path."""
    # Defer all heavy imports to avoid circular import at conftest collection time
    from polylogue.config import Source
    from polylogue.pipeline.prepare import prepare_records
    from polylogue.scenarios import build_default_corpus_specs
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.sources import iter_source_conversations
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.runtime import RawConversationRecord
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.storage.sqlite.connection import open_connection

    db_path = tmp_path / "corpus_seeded.db"
    with open_connection(db_path):
        pass  # Auto-initialize schema

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    async def _do_seed() -> None:
        available = set(SyntheticCorpus.available_providers())
        specs = build_default_corpus_specs(
            providers=(provider for provider in providers if provider in available),
            count=count,
            messages_min=4,
            messages_max=11,
            seed=seed,
        )
        for spec in specs:
            provider = spec.provider
            provider_dir = corpus_dir / provider
            written = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="corpus")

            for file_path, artifact in zip(written.files, written.batch.artifacts, strict=True):
                raw_bytes = artifact.raw_bytes
                raw_id = hashlib.sha256(raw_bytes).hexdigest()
                record = RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=provider,
                    source_name=provider,
                    source_path=str(file_path),
                    blob_size=len(raw_bytes),
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
                await backend.save_raw_conversation(record)

        # Parse + ingest
        archive_root = tmp_path / "archive"
        archive_root.mkdir()

        for provider_dir in sorted(corpus_dir.iterdir()):
            provider = provider_dir.name
            for file_path in sorted(provider_dir.iterdir()):
                source = Source(name=provider, path=file_path)
                raw_id = hashlib.sha256(file_path.read_bytes()).hexdigest()
                for convo in iter_source_conversations(source):
                    await prepare_records(
                        convo,
                        source_name=provider,
                        archive_root=archive_root,
                        backend=backend,
                        repository=repository,
                        raw_id=raw_id,
                    )

        await backend.close()

    asyncio.run(_do_seed())
    return db_path


@pytest.fixture
def corpus_seeded_db(tmp_path: Path, workspace_env: dict[str, Path]) -> Callable[..., Path]:
    """Factory: returns db_path seeded with corpus data through the real pipeline.

    Usage::

        def test_something(corpus_seeded_db):
            db = corpus_seeded_db(providers=("chatgpt",), count=2)
    """

    del workspace_env

    def _seed(
        providers: Sequence[str] = ("chatgpt", "claude-ai"),
        count: int = 3,
        seed: int = 42,
    ) -> Path:
        return _seed_db(tmp_path, providers=providers, count=count, seed=seed)

    return _seed
