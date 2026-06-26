"""Seed the deterministic demo archive without daemon scheduling."""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import build_demo_corpus_specs, seed_demo_user_overlays
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

from .models import DemoSeedResult

DEMO_SOURCE_DIRNAME = "demo-fixture-world-source"


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    """Temporarily run relative-source ingestion from *path*."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def materialize_demo_source(root: Path, *, force: bool = False) -> Path:
    """Write deterministic demo source artifacts under ``root``."""

    source_root = root / DEMO_SOURCE_DIRNAME
    if force and source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    SyntheticCorpus.write_specs_artifacts(
        build_demo_corpus_specs(),
        source_root,
        prefix="demo",
        index_width=2,
    )
    return source_root


def _materialize_session_insights(archive_root: Path, session_ids: list[str]) -> None:
    """Build the session-profile insight read models for *session_ids*.

    ``parse_sources_archive`` writes the ``sessions``/``messages`` tree but does
    not materialize the derived insight tables (``session_profiles`` and
    siblings); the daemon convergence path normally does that in a separate
    stage. The no-daemon demo seed must run the same rebuild so that the
    postmortem / recovery-digest read surfaces resolve against a populated demo
    archive instead of an empty one. Passing ``session_ids`` makes
    ``rebuild_session_insights_sync`` commit internally and skip the full-table
    delete/rebuild path.
    """

    if not session_ids:
        return
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        rebuild_session_insights_sync(conn, session_ids=session_ids)
    finally:
        conn.close()


def demo_source_specs(source_root: Path) -> list[Source]:
    """Return relative source specs for the materialized demo world."""

    return [
        Source(name="chatgpt", path=Path("chatgpt")),
        Source(name="claude-code", path=Path("claude-code")),
        Source(name="codex", path=Path("codex")),
    ]


async def seed_demo_archive(
    archive_root: Path,
    *,
    force: bool = False,
    with_overlays: bool = False,
) -> DemoSeedResult:
    """Materialize, ingest, and optionally overlay the deterministic demo archive."""

    source_root = materialize_demo_source(archive_root, force=force)
    with _pushd(source_root):
        result = await parse_sources_archive(archive_root, demo_source_specs(source_root))

    session_ids = sorted(result.processed_ids)
    _materialize_session_insights(archive_root, session_ids)

    overlay = seed_demo_user_overlays(archive_root) if with_overlays else None
    return DemoSeedResult(
        archive_root=archive_root,
        source_root=source_root,
        session_count=int(result.counts["sessions"]),
        message_count=int(result.counts["messages"]),
        session_ids=tuple(sorted(result.processed_ids)),
        overlays_seeded=overlay is not None,
        assertion_count=len(overlay.assertion_ids) if overlay else 0,
    )


__all__ = ["DEMO_SOURCE_DIRNAME", "demo_source_specs", "materialize_demo_source", "seed_demo_archive"]
