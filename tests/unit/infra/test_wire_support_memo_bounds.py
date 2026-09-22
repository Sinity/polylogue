"""Every wire-support memo holds a declared amount and no more.

The memos in ``tests.infra.wire_support`` are module globals, so each one
lives for the whole of a pytest worker process. Seven of them had no limit and
no eviction, and the largest held nearly a megabyte of generated wire payloads
per memoized corpus.

Each test names the mutation that makes it red: remove the eviction and the
retention passes the declared budget. Each also pins the opposite direction,
because a memo that stores nothing satisfies every ceiling while destroying
the reason the memo exists.

The real budget constants are exercised, not stand-ins. A test that lowered
``_GENERATED_WITNESS_BYTES_LIMIT`` to something cheap would prove the
eviction loop runs and prove nothing about what a worker retains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from polylogue.schemas.synthetic import wire_formats
from tests.infra import wire_support


@dataclass(frozen=True)
class _StubCorpus:
    """The attributes ``_corpus_key`` and ``memo_batch`` read, and nothing else."""

    provider: str
    package_version: str
    element_kind: str
    schema: tuple[str, ...]
    workload_profile: tuple[str, ...]
    _coverage_witness_mode: bool = False


def _corpus(index: int) -> Any:
    """A corpus stand-in typed as ``Any`` on purpose.

    ``_CoverageCorpus`` is the protocol the real generator needs; the memo key
    reads six attributes of it. Implementing the whole protocol here would
    assert that the key reads more than it does.
    """
    return _StubCorpus(
        provider=f"stub-{index}",
        package_version="1",
        element_kind="session",
        schema=("schema", str(index)),
        workload_profile=("workload",),
    )


def _retained_payload_bytes() -> int:
    return sum(len(payload) for entry in wire_support._GENERATED_WITNESSES.values() for payload in entry)


@pytest.fixture
def isolated_memos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty memos for the test, and whatever the process held restored after."""
    for name in ("_GENERATED_WITNESSES", "_KEYWORD_TUPLES", "_CONSTRUCT_COVERAGE", "_VALIDATIONS"):
        current = getattr(wire_support, name)
        monkeypatch.setattr(wire_support, name, type(current)())
    monkeypatch.setattr(wire_support, "_GENERATED_WITNESS_BYTES", 0)


@pytest.mark.usefixtures("isolated_memos")
def test_generated_witness_memo_holds_its_declared_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention stays under the declared budget however many corpora are memoized.

    RED WITHOUT: delete the eviction loop in ``_retain_witnesses``. The memo
    then keeps every corpus and holds the full 128 MiB this test offers,
    against a 96 MiB budget.
    """
    payload_bytes = 4 * 1024 * 1024
    corpora = 32
    generated = 0

    def stub_witnesses(corpus: Any, *, seed: int, max_witnesses: int = 128) -> list[bytes]:
        nonlocal generated
        generated += 1
        return [bytes(payload_bytes)]

    monkeypatch.setattr(wire_formats, "generate_coverage_witnesses", stub_witnesses)
    with wire_support.shared_wire_generation():
        for index in range(corpora):
            wire_formats.generate_coverage_witnesses(_corpus(index), seed=1, max_witnesses=1)

    assert payload_bytes * corpora > wire_support._GENERATED_WITNESS_BYTES_LIMIT, (
        "the fixture must offer more than the budget or the bound is never exercised"
    )
    assert generated == corpora, "every distinct corpus must reach the real generator exactly once"
    retained = _retained_payload_bytes()
    assert retained <= wire_support._GENERATED_WITNESS_BYTES_LIMIT, (
        f"memo retained {retained / 1024 / 1024:.1f} MiB in {len(wire_support._GENERATED_WITNESSES)} entries, "
        f"budget {wire_support._GENERATED_WITNESS_BYTES_LIMIT / 1024 / 1024:.1f} MiB"
    )
    assert _retained_payload_bytes() == wire_support._GENERATED_WITNESS_BYTES, (
        "the running byte count must match what the memo actually holds"
    )


@pytest.mark.usefixtures("isolated_memos")
def test_generated_witness_memo_still_answers_from_its_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound must not degenerate into a memo that never answers.

    RED WITHOUT: make ``_retain_witnesses`` a no-op -- the trivial way to
    satisfy every byte ceiling -- and the second call re-enters the generator.
    """
    calls = 0

    def stub_witnesses(corpus: Any, *, seed: int, max_witnesses: int = 128) -> list[bytes]:
        nonlocal calls
        calls += 1
        return [b"payload"]

    monkeypatch.setattr(wire_formats, "generate_coverage_witnesses", stub_witnesses)
    with wire_support.shared_wire_generation():
        first = wire_formats.generate_coverage_witnesses(_corpus(0), seed=1, max_witnesses=1)
        second = wire_formats.generate_coverage_witnesses(_corpus(0), seed=1, max_witnesses=1)

    assert calls == 1, f"the memo answered nothing: the generator ran {calls} times for one key"
    assert first == second == [b"payload"]


@pytest.mark.usefixtures("isolated_memos")
def test_generated_witness_memo_keeps_the_corpus_it_was_just_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corpus larger than the whole budget is still the one that survives.

    RED WITHOUT: drop the ``len(...) > 1`` guard in ``_retain_witnesses`` and
    an over-budget entry evicts itself, turning the memo into a slower no-op.
    """
    oversized = wire_support._GENERATED_WITNESS_BYTES_LIMIT + 1024

    def stub_witnesses(corpus: Any, *, seed: int, max_witnesses: int = 128) -> list[bytes]:
        return [bytes(oversized)]

    monkeypatch.setattr(wire_formats, "generate_coverage_witnesses", stub_witnesses)
    with wire_support.shared_wire_generation():
        wire_formats.generate_coverage_witnesses(_corpus(0), seed=1, max_witnesses=1)

    assert len(wire_support._GENERATED_WITNESSES) == 1, (
        f"an over-budget corpus evicted itself: {len(wire_support._GENERATED_WITNESSES)} entries retained"
    )
    assert _retained_payload_bytes() == oversized


@pytest.mark.usefixtures("isolated_memos")
def test_keyword_tuple_pool_holds_its_declared_entry_budget() -> None:
    """The interning pool stops growing, and still interns what it holds.

    RED WITHOUT: remove the ``_evict_to`` call in ``_pooled`` and the pool
    keeps one entry per distinct combination -- here 1.5x its limit.
    """
    limit = wire_support._KEYWORD_TUPLE_LIMIT
    offered = limit + limit // 2
    for index in range(offered):
        wire_support._pooled((f"keyword-{index}",))

    assert offered > limit, "the fixture must offer more than the budget"
    assert len(wire_support._KEYWORD_TUPLES) == limit, (
        f"pool holds {len(wire_support._KEYWORD_TUPLES)} entries against a {limit}-entry budget"
    )
    # Still a pool: an equal tuple built independently comes back as the same
    # object, so eviction has not silently disabled the interning.
    recent = f"keyword-{offered - 1}"
    assert wire_support._pooled((recent,)) is wire_support._pooled((recent,))
