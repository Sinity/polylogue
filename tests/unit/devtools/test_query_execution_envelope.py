"""Tests for the live query execution envelope lab command."""

from __future__ import annotations

from pathlib import Path

import pytest

import devtools.query_execution_envelope as envelope_module
from devtools.query_execution_envelope import _proc_memory, _temp_used_bytes, measure_query_envelope


def test_proc_memory_is_nonnegative() -> None:
    rss, pss, swap = _proc_memory()
    assert rss >= 0
    assert pss >= 0
    assert swap >= 0


def test_temp_usage_missing_path_is_zero(tmp_path: Path) -> None:
    assert _temp_used_bytes(tmp_path / "missing") == 0


async def test_measure_query_envelope_runs_the_declared_repetition_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt covers every query round and all four resource dimensions."""

    (tmp_path / "index.db").write_bytes(b"synthetic index")
    calls = 0

    class FakeEnvelope:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"items": [{"group_key": "shell", "count": 1}]}

    class FakePolylogue:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakePolylogue:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def query_units(self, expression: str, *, limit: int) -> FakeEnvelope:
            nonlocal calls
            assert expression == "actions where tool:shell | group by tool | count"
            assert limit == 100
            calls += 1
            return FakeEnvelope()

    monkeypatch.setattr(envelope_module, "Polylogue", FakePolylogue)
    monkeypatch.setattr(envelope_module, "_proc_memory", lambda: (100, 80, 0))
    monkeypatch.setattr(envelope_module, "_temp_used_bytes", lambda _root: 100)

    receipt = await measure_query_envelope(
        tmp_path,
        warmup=0,
        baseline_rounds=2,
        sample_interval_s=0.001,
        max_rss_bytes=100,
        max_pss_bytes=80,
        max_swap_growth_bytes=0,
        max_temp_growth_bytes=0,
    )

    assert calls == 22
    assert receipt["status"] == "succeeded"
    assert len(receipt["samples"]) == 22
    assert len(receipt["final_samples"]) == 3
    assert receipt["return_checks"] == {"rss": True, "pss": True, "swap": True, "temp": True}
    assert receipt["absolute_checks"] == {"rss": True, "pss": True, "swap": True, "temp": True}


async def test_measure_query_envelope_fails_when_declared_rss_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absolute RSS declaration is a real failure condition."""

    (tmp_path / "index.db").write_bytes(b"synthetic index")

    class FakeEnvelope:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"items": []}

    class FakePolylogue:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakePolylogue:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def query_units(self, _expression: str, *, limit: int) -> FakeEnvelope:
            assert limit == 100
            return FakeEnvelope()

    monkeypatch.setattr(envelope_module, "Polylogue", FakePolylogue)
    monkeypatch.setattr(envelope_module, "_proc_memory", lambda: (100, 80, 0))
    monkeypatch.setattr(envelope_module, "_temp_used_bytes", lambda _root: 100)

    receipt = await measure_query_envelope(
        tmp_path,
        warmup=0,
        baseline_rounds=1,
        sample_interval_s=0.001,
        max_rss_bytes=99,
        max_pss_bytes=80,
        max_swap_growth_bytes=0,
        max_temp_growth_bytes=0,
    )

    assert receipt["status"] == "failed"
    assert receipt["absolute_checks"]["rss"] is False
