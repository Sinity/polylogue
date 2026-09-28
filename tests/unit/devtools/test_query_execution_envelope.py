"""Tests for the live query execution envelope lab command."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import devtools.query_execution_envelope as envelope_module
from devtools.query_execution_envelope import (
    ResourceProbeUnavailableError,
    ResourceSample,
    _parse_proc_memory,
    _proc_memory,
    _temp_used_bytes,
    measure_query_envelope,
)


def test_concurrent_peak_observations_keep_every_independent_maximum() -> None:
    """Concurrent updates retain each dimension's maximum, even from different samples."""
    peak = envelope_module._ResourcePeak(ResourceSample(0, 0, 0, 0))
    samples = [
        ResourceSample(100, 10, 1, 4),
        ResourceSample(20, 200, 2, 3),
        ResourceSample(30, 30, 300, 2),
        ResourceSample(40, 40, 4, 400),
    ]
    with ThreadPoolExecutor(max_workers=len(samples)) as pool:
        list(pool.map(peak.observe, samples))

    assert peak.snapshot() == ResourceSample(100, 200, 300, 400), (
        "each dimension must retain its maximum even when another sample peaks elsewhere"
    )


def test_proc_memory_is_nonnegative() -> None:
    rss, pss, swap = _proc_memory()
    assert rss >= 0
    assert pss >= 0
    assert swap >= 0


def test_temp_usage_missing_path_is_unmeasured(tmp_path: Path) -> None:
    with pytest.raises(envelope_module.TempProbeUnavailableError):
        _temp_used_bytes(tmp_path / "missing")


def test_parse_proc_memory_reads_every_declared_field() -> None:
    rss, pss, swap = _parse_proc_memory("VmRSS:\t2048 kB\nVmSwap:\t4 kB\nPss:\t1024 kB\n")

    assert (rss, pss, swap) == (2048 * 1024, 1024 * 1024, 4 * 1024)


def test_parse_proc_memory_refuses_a_missing_field() -> None:
    """A field procfs does not report is refused, never sampled as zero.

    Restoring a zero default makes this green and every declared limit
    satisfiable without measuring anything.
    """
    with pytest.raises(ResourceProbeUnavailableError, match="Pss"):
        _parse_proc_memory("VmRSS:\t2048 kB\nVmSwap:\t4 kB\n")


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


async def test_active_generation_is_pinned_for_open_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opened DB and reported size must both name the promoted generation."""
    active = tmp_path / ".index-generations" / "gen-active" / "index.db"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"active-generation")
    (tmp_path / "index.db").write_bytes(b"stale-shadow")
    monkeypatch.setattr(envelope_module, "resolve_active_index_path", lambda _root: active)
    opened: list[Path] = []

    class FakeEnvelope:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"items": []}

    class FakePolylogue:
        def __init__(self, *, archive_root: Path, db_path: Path) -> None:
            opened.append(db_path)

        async def __aenter__(self) -> FakePolylogue:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def query_units(self, _expression: str, *, limit: int) -> FakeEnvelope:
            return FakeEnvelope()

    monkeypatch.setattr(envelope_module, "Polylogue", FakePolylogue)
    monkeypatch.setattr(envelope_module, "_proc_memory", lambda: (100, 80, 0))
    monkeypatch.setattr(envelope_module, "_temp_used_bytes", lambda _root: 100)
    receipt = await measure_query_envelope(
        tmp_path,
        warmup=0,
        baseline_rounds=1,
        sample_interval_s=0,
        max_rss_bytes=100,
        max_pss_bytes=80,
        max_swap_growth_bytes=0,
        max_temp_growth_bytes=0,
    )
    assert opened == [active.resolve()]
    assert receipt["archive_generation"] == "gen-active"


async def test_background_probe_failure_is_not_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed between-round sample blocks success even if later probes work."""
    import asyncio

    (tmp_path / "index.db").write_bytes(b"index")
    calls = 0

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
            await asyncio.sleep(0.005)
            return FakeEnvelope()

    def probe() -> tuple[int, int, int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ResourceProbeUnavailableError("transient procfs read failure")
        return 100, 80, 0

    monkeypatch.setattr(envelope_module, "Polylogue", FakePolylogue)
    monkeypatch.setattr(envelope_module, "_proc_memory", probe)
    monkeypatch.setattr(envelope_module, "_temp_used_bytes", lambda _root: 100)
    with pytest.raises(ResourceProbeUnavailableError, match="background resource sample failed"):
        await measure_query_envelope(
            tmp_path,
            warmup=0,
            baseline_rounds=1,
            sample_interval_s=0.001,
            max_rss_bytes=100,
            max_pss_bytes=80,
            max_swap_growth_bytes=0,
            max_temp_growth_bytes=0,
        )
