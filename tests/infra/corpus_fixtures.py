"""SyntheticCorpus integration fixture for seeding DBs through the real pipeline.

Provides a ``corpus_seeded_db`` factory fixture that generates wire-format bytes,
writes them to temp files, and runs the pipeline to produce a seeded database.
This is additive infrastructure — it does NOT replace SessionBuilder tests.

Usage::

    def test_query_on_real_data(corpus_seeded_db):
        db_path = corpus_seeded_db(providers=("chatgpt",), count=2, seed=42)
        # db_path now contains parsed data from the real pipeline
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest


def _seed_db(
    tmp_path: Path,
    archive_root: Path,
    providers: Sequence[str] = ("chatgpt", "claude-ai"),
    count: int = 3,
    seed: int = 42,
) -> Path:
    """Generate wire-format bytes, write to temp, ingest directly, return index db.

    Writes synthetic corpus artifacts and ingests them through the archive's
    archive ingest path so the data lands in the same store the facade/CLI read.
    """
    # Defer all heavy imports to avoid circular import at conftest collection time
    from polylogue.config import Source
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive
    from polylogue.scenarios import build_default_corpus_specs
    from polylogue.schemas.synthetic import SyntheticCorpus

    archive_root.mkdir(parents=True, exist_ok=True)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    async def _do_seed() -> None:
        available = set(SyntheticCorpus.available_providers())
        specs = build_default_corpus_specs(
            providers=(provider for provider in providers if provider in available),
            count=count,
            messages_min=4,
            messages_max=11,
            seed=seed,
        )
        sources: list[Source] = []
        for spec in specs:
            provider_dir = corpus_dir / spec.provider
            written = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="corpus")
            sources.extend(Source(name=spec.provider, path=file_path) for file_path in written.files)

        if sources:
            await parse_sources_archive(archive_root, sources)

    asyncio.run(_do_seed())
    return archive_root / "index.db"


@pytest.fixture
def corpus_seeded_db(tmp_path: Path, workspace_env: dict[str, Path]) -> Callable[..., Path]:
    """Factory: returns db_path seeded with corpus data through the real pipeline.

    Usage::

        def test_something(corpus_seeded_db):
            db = corpus_seeded_db(providers=("chatgpt",), count=2)
    """

    archive_root = workspace_env["archive_root"]

    def _seed(
        providers: Sequence[str] = ("chatgpt", "claude-ai"),
        count: int = 3,
        seed: int = 42,
    ) -> Path:
        return _seed_db(tmp_path, archive_root, providers=providers, count=count, seed=seed)

    return _seed
