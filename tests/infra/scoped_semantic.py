"""Synthetic purchased vectors and query transport for scoped ranking controls."""

from __future__ import annotations

import json
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
    samples: list[tuple[str, str, str, float]],
    *,
    query_axis: float,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Config, SqliteVecProvider, dict[tuple[str, str], tuple[str, str]], list[dict[str, object]]]:
    bootstrap_archive_root(root)
    identities = seed_vector_archive(root, [(sid, mid, text, axis_vector(value)) for sid, mid, text, value in samples])
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
