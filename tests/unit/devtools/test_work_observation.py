from __future__ import annotations

import threading

import pytest

from devtools.fresh_build_bench.work_ledger import reduce_work_events
from polylogue import logging as plog
from polylogue.core import work_observation as work


@pytest.fixture(autouse=True)
def info_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plog, "_threshold", plog.INFO)


def test_same_key_bodies_are_not_erased_by_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events:
        for _ in range(2):
            with work.body("synthetic", "parse", work_key="same-complete-input") as body:
                assert body is not None
                artifact = body.produced()
            work.consume("synthetic", "parse", artifact_ref=artifact, work_key="same-complete-input", outcome="reused")
    assert not [event for event in events if event["event"] == "log.field_rejected"]
    ledger = reduce_work_events(events + events)
    operation = ledger["operations"][0]
    assert operation["started"] == operation["completed"] == 2
    assert operation["distinct_complete_work_keys"] == 1
    assert operation["execution_amplification"] == 2
    assert ledger["artifact_products"] == 2
    assert ledger["consumer_decisions"] == {"reused": 2}
    assert ledger["coverage"]["transport_duplicates"] == 6


def test_unmatched_entry_and_unknown_key_make_ratio_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events:
        with work.body("synthetic", "validate"):
            pass
        with work.body("synthetic", "validate", work_key="complete"):
            pass
    events.pop()
    ledger = reduce_work_events(events)
    operation = ledger["operations"][0]
    assert operation["started"] == 2
    assert operation["unknown_key_bodies"] == 1
    assert operation["incomplete"] == 1
    assert operation["execution_amplification"] is None


def test_exception_body_survives_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events, pytest.raises(ValueError, match="synthetic"):
        with work.body("synthetic", "parse", work_key="complete"):
            raise ValueError("synthetic")
    operation = reduce_work_events(events)["operations"][0]
    assert operation["failed"] == 1
    assert operation["completed"] == 0


def test_disabled_work_observation_allocates_no_identity_or_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", False)

    def forbidden() -> None:
        raise AssertionError("disabled observation allocated an identity")

    monkeypatch.setattr(work, "_identity", forbidden)
    monkeypatch.setattr(work, "uuid4", forbidden)
    with plog.capture() as events:
        with work.body("synthetic", "parse") as body:
            assert body is None
        work.consume("synthetic", "parse", artifact_ref=None, outcome="consumed")
    assert events == []


def test_worker_generations_are_not_native_tid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    identities: list[tuple[str, str, int]] = []

    def observe() -> None:
        identities.append(work._identity())

    for _ in range(2):
        worker = threading.Thread(target=observe)
        worker.start()
        worker.join()
    assert identities[0][0] == identities[1][0]
    assert identities[0][1] != identities[1][1]


def test_required_field_loss_blocks_semantic_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events:
        with work.body("synthetic", "parse", work_key="complete"):
            pass
    events[-1]["required_measurement_fields_omitted"] = 1
    assert reduce_work_events(events)["operations"][0]["execution_amplification"] is None


def test_internal_event_gap_is_not_a_zero_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events:
        for _ in range(2):
            with work.body("synthetic", "parse", work_key="complete"):
                pass
    events[1:3] = []
    ledger = reduce_work_events(events)
    assert ledger["coverage"]["internal_sequence_gaps"] == 2
    assert ledger["operations"][0]["execution_amplification"] is None


def test_sink_loss_withdraws_work_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools.fresh_build_bench.trace_ledger import reduce_events

    monkeypatch.setattr(work, "ENABLED", True)
    with plog.capture() as events:
        with work.body("synthetic", "parse", work_key="complete"):
            pass
    events.append({"event": "log.stats", "dropped": 1})
    _trace, ledger = reduce_events(events)
    assert ledger["work_observations"]["operations"][0]["execution_amplification"] is None
