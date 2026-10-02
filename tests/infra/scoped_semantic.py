"""Synthetic purchased vectors and query transport for scoped ranking controls."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from polylogue.config import Config
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.vector_archive import seed_vector_archive


def axis_vector(value: float) -> list[float]:
    return [value, *([0.0] * 1023)]


def ranking_archive(
    root: Path,
    samples: Sequence[tuple[str, str, str, float | None]],
    *,
    query_axis: float,
    monkeypatch: pytest.MonkeyPatch,
    concurrent_writes: bool = False,
) -> tuple[Config, SqliteVecProvider, dict[tuple[str, str], tuple[str, str]], list[dict[str, object]]]:
    bootstrap_archive_root(root)
    identities = seed_vector_archive(
        root, [(sid, mid, text, axis_vector(value) if value is not None else None) for sid, mid, text, value in samples]
    )
    if concurrent_writes:
        with closing(sqlite3.connect(root / "index.db")) as connection, closing(connection.cursor()) as cursor:
            cursor.execute("PRAGMA journal_mode=WAL")
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["input_type"] == "query"
        requests.append(payload)
        return httpx.Response(200, json={"data": [{"embedding": axis_vector(query_axis)}]})

    original = httpx.Client
    monkeypatch.setattr(
        "polylogue.storage.search_providers.sqlite_vec_embeddings.httpx.Client",
        lambda **kwargs: original(transport=httpx.MockTransport(reply), **kwargs),
    )
    provider = SqliteVecProvider("synthetic-key", db_path=root / "embeddings.db", archive_root=root)
    config = Config(archive_root=root, render_root=root / "render", db_path=root / "index.db", sources=[])
    return config, provider, identities, requests


def declare_ranking_repository(root: Path, session_ids: Sequence[str]) -> None:
    """Project parser-declared repo evidence through the actual edge writer."""
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base_models import ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import _write_repo_edges

    remote = "https://example.invalid/neutral/target.git"
    with closing(sqlite3.connect(root / "index.db")) as connection, closing(connection.cursor()) as cursor:
        for session_id in session_ids:
            cursor.execute("UPDATE sessions SET git_repository_url = ? WHERE session_id = ?", (remote, session_id))
            _write_repo_edges(
                connection,
                session_id,
                ParsedSession(
                    source_name=Provider.CODEX, provider_session_id=session_id, messages=[], git_repository_url=remote
                ),
            )
        connection.commit()
