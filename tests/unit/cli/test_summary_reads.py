"""The bare CLI summary requests archive facts through declared operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.cli import operation_kernel
from polylogue.cli.shared.helpers import get_origin_counts, list_archive_coverage_insights
from polylogue.services import RuntimeServices


def test_origin_counts_use_canonical_aggregate(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[object, str, dict[str, object], object]] = []

    def dispatch(config: object, request: Any, *, archive_root: object = None) -> SimpleNamespace:
        calls.append((config, request.operation, request.payload, archive_root))
        return SimpleNamespace(value={"groups": {"codex": 2, "chatgpt": 1}})

    config = object()
    monkeypatch.setattr(operation_kernel, "dispatch", dispatch)
    result = asyncio.run(
        get_origin_counts(
            services=cast(RuntimeServices, SimpleNamespace(get_config=lambda: config)), db_path=tmp_path / "index.db"
        )
    )
    assert result == [("codex", 2), ("chatgpt", 1)]
    assert calls == [(config, "query.aggregate", {"mode": "stats_by", "group_by": "origin"}, tmp_path)]


def test_coverage_summary_uses_the_resident_insight_page(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def dispatch(_config: object, request: Any, *, archive_root: object = None) -> SimpleNamespace:
        assert archive_root is None
        calls.append((request.operation, request.payload))
        return SimpleNamespace(
            value={
                "page": {"insight": "archive_coverage", "items": [], "total": 0},
                "outcome": {"state": "empty"},
            }
        )

    monkeypatch.setattr(operation_kernel, "dispatch", dispatch)
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: object())
    assert asyncio.run(list_archive_coverage_insights()) == []
    assert calls[0][0] == "insights.list"
    page = calls[0][1]["page"]
    assert isinstance(page, dict)
    assert page["insight"] == "archive_coverage"
