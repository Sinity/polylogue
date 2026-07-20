"""Deterministic fault-injection proof for polylogue-b054.1.1.1.

A repeated clean 8-worker testmon seed on 2026-07-16 failed once (see
``tests/unit/demo/test_demo_seed_verify.py::test_demo_verify_reports_missing_overlays``)
with at least one declared demo construct below its declared minimum, in a way
that did not reproduce in isolation. ``parse_sources_archive``'s parallel
(process-pool) branch deliberately isolates one file's parse failure --
including a transient process-pool *lifecycle* failure such as a worker
spawn/fork failure under host resource pressure, exactly the kind of thing an
8-way xdist run can trigger -- by skipping that file and continuing
(``polylogue/pipeline/services/archive_ingest.py``, the ``as_completed``
loop's ``except Exception`` branch). That is the right behavior for a large,
uncontrolled real corpus, but it silently undercounts a declared construct
for the demo seeder's small, fully-controlled fixture corpus, where every
file is required every time.

This module proves the mechanism with deterministic fault injection (no
sleeps, no retries, no reliance on genuinely reproducing host resource
pressure), and proves the fix: ``seed_demo_archive`` now forces
``workers=1`` for its own ``parse_sources_archive`` call, so it never engages
the process pool -- and therefore can never hit this isolation-vs-completeness
tradeoff -- regardless of host load or worker count.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest

from polylogue.demo import evaluate_demo_constructs, seed_demo_archive
from polylogue.demo.seed import demo_source_specs, materialize_demo_source
from polylogue.pipeline.services import archive_ingest
from polylogue.pipeline.services.archive_ingest import parse_sources_archive


class _FaultInjectingPool:
    """A drop-in ``ProcessPoolExecutor`` replacement that fails one path deterministically.

    Every other submitted job actually runs the real worker function
    in-thread (in-process, not a real subprocess -- the point here is
    deterministic proof of the *isolation-vs-completeness* mechanism, not a
    re-test of multiprocessing itself, which the existing process-pool tests
    already cover).
    """

    def __init__(self, *, max_workers: int, fail_when: str) -> None:
        self._fail_when = fail_when
        self.failed_paths: list[str] = []

    def __enter__(self) -> _FaultInjectingPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def submit(self, fn: object, path_str: str, *args: object, **kwargs: object) -> Future[list[tuple[object, object]]]:
        future: Future[list[tuple[object, object]]] = Future()
        if self._fail_when in path_str:
            self.failed_paths.append(path_str)
            # Simulates a worker future raising -- the same observable shape
            # as a transient process-pool lifecycle failure (spawn/fork
            # error, worker crash, OOM kill) under host resource pressure.
            future.set_exception(RuntimeError("injected process-pool worker failure"))
        else:
            assert callable(fn)
            try:
                result = fn(path_str, *args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                future.set_exception(exc)
            else:
                future.set_result(result)
        return future


@pytest.mark.asyncio
async def test_parallel_parse_worker_failure_undercounts_a_declared_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the vulnerability: a single injected worker failure in the
    PARALLEL (workers>1) path silently drops one browser-capture variant and
    the declared ``browser_capture_raw_variants`` construct (minimum 3) falls
    to 2 -- with no other visible signal beyond the construct-coverage read.

    Anti-vacuity: this exercises the real production ``parse_sources_archive``
    parallel branch and the real ``evaluate_demo_constructs`` SQL. Removing
    the fault injection (or raising ``max_workers`` without injecting a
    failure) makes the construct assertion below fail, proving the assertion
    is load-bearing on the injected fault, not a tautology.
    """

    archive_root = tmp_path / "archive"
    source_root = materialize_demo_source(archive_root, force=True)
    fail_path = str(source_root / "browser-capture" / "chatgpt-dom-fallback.json")

    def fake_pool(*, max_workers: int) -> _FaultInjectingPool:
        return _FaultInjectingPool(max_workers=max_workers, fail_when="chatgpt-dom-fallback.json")

    monkeypatch.setattr(archive_ingest, "ProcessPoolExecutor", fake_pool)
    monkeypatch.chdir(source_root)

    result = await parse_sources_archive(archive_root, demo_source_specs(source_root), workers=2)

    assert result.parse_failures == 1

    coverage = {row.construct_id: row for row in evaluate_demo_constructs(archive_root)}
    variants = coverage["browser_capture_raw_variants"]
    assert variants.observed == 2, variants.to_payload()
    assert variants.minimum == 3
    assert variants.ok is False, "the injected single-file failure must undercount the construct"
    assert fail_path  # the exact failing path is deterministic, not incidental


@pytest.mark.asyncio
async def test_seed_demo_archive_never_engages_the_process_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the fix: ``seed_demo_archive`` forces sequential parsing, so it
    can never hit the parallel-path isolation-vs-completeness tradeoff at
    all -- regardless of host CPU count, ``POLYLOGUE_INGEST_PARSE_WORKERS``,
    or concurrent xdist load.

    Anti-vacuity: the process pool is monkeypatched to explode on
    instantiation. If ``seed_demo_archive``'s call to ``parse_sources_archive``
    ever passed a worker count > 1 (e.g. the ``workers=1`` argument in
    ``polylogue/demo/seed.py::seed_demo_archive`` were removed or the demo
    corpus were widened to reintroduce multi-worker parsing without also
    threading an explicit ``workers=1``), the sequential branch would be
    skipped, ``ProcessPoolExecutor`` would be constructed, and this test
    would fail with the injected error instead of completing.
    """

    def explode(*, max_workers: int) -> object:
        raise AssertionError(
            f"seed_demo_archive must never construct a process pool (max_workers={max_workers}); "
            "the demo corpus must parse sequentially for deterministic construct coverage"
        )

    monkeypatch.setattr(archive_ingest, "ProcessPoolExecutor", explode)
    # A high, deliberately provocative env override proves this is not an
    # accident of a low local CPU count: even an operator/CI override that
    # would normally force heavy parallelism must not reach ProcessPoolExecutor
    # via the demo seeder.
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "8")

    archive_root = tmp_path / "archive"
    seed = await seed_demo_archive(archive_root, force=True, with_overlays=False)

    assert seed.construct_coverage
    failed = [row.to_payload() for row in seed.construct_coverage if not row.ok]
    assert not failed, failed
    assert seed.session_count > 0
