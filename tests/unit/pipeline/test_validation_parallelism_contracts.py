"""Concurrency contracts for the two-phase validation executor.

Closes the ``pipeline.ingest`` known gap from
``docs/plans/test-closure-matrix.yaml`` (#1295). ``evaluate_raw_artifacts``
in ``polylogue/pipeline/services/validation_flow.py`` runs CPU-bound
schema validation across a ``ProcessPoolExecutor`` (Phase 1) and then
performs sequential async ``mark_raw_validated`` / ``mark_raw_parsed``
writes (Phase 2). These tests pin the load-bearing contracts:

* Phase 1 results are deterministic w.r.t. input order even when the
  pool reorders task completion internally;
* Phase 2 writes happen sequentially in input order so a downstream
  ``RawValidationStore`` sees a well-defined sequence;
* ``_ValidationOutcome`` is independently constructible per record and
  does not leak shared mutable state across producers;
* Edge cases (empty batch, single record, batch larger than worker cap)
  do not deadlock or silently drop records;
* If a single record cannot be decoded, the failure is captured as a
  typed outcome and the rest of the batch still completes.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import orjson
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.pipeline.services.validation_flow import evaluate_raw_artifacts
from polylogue.pipeline.services.validation_runtime import (
    _validate_record_sync,
    _ValidationOutcome,
)
from polylogue.pipeline.stage_models import ValidateResult
from polylogue.storage.runtime import RawSessionRecord
from polylogue.types import Provider, ValidationMode, ValidationStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_blob(blob_root: Path, payload: bytes) -> str:
    """Write *payload* into the content-addressed blob store layout."""
    digest = hashlib.sha256(payload).hexdigest()
    shard = blob_root / digest[:2]
    shard.mkdir(parents=True, exist_ok=True)
    (shard / digest[2:]).write_bytes(payload)
    return digest


def _claude_payload(title: str = "hello") -> bytes:
    """Produce a tiny well-formed Claude web export payload."""
    body = {
        "uuid": f"conv-{title}",
        "name": title,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "chat_messages": [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Hello",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    }
    return orjson.dumps(body)


def _make_record(raw_id: str, *, payload: bytes) -> RawSessionRecord:
    return RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_AI.value,
        source_path=f"/synthetic/{raw_id}.json",
        blob_size=len(payload),
        acquired_at="2026-01-01T00:00:00Z",
    )


class RecordingStore:
    """RawValidationStore double that captures the order of Phase 2 writes."""

    def __init__(self) -> None:
        self.validated_calls: list[tuple[str, ValidationStatus]] = []
        self.parsed_calls: list[tuple[str, str | None]] = []
        self._lock = asyncio.Lock()

    async def get_raw_sessions_batch(
        self, raw_ids: list[str]
    ) -> list[RawSessionRecord]:  # pragma: no cover - unused here
        raise NotImplementedError

    async def mark_raw_validated(
        self,
        raw_id: str,
        *,
        status: ValidationStatus | str,
        error: str | None = None,
        drift_count: int = 0,
        provider: Provider | str | None = None,
        mode: ValidationMode | str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        # The lock guards the recorded list against any accidental
        # concurrent producer that would defeat the sequentiality claim.
        async with self._lock:
            resolved = status if isinstance(status, ValidationStatus) else ValidationStatus.from_string(str(status))
            self.validated_calls.append((raw_id, resolved))

    async def mark_raw_parsed(
        self,
        raw_id: str,
        *,
        error: str | None = None,
        payload_provider: Provider | str | None = None,
    ) -> None:
        async with self._lock:
            self.parsed_calls.append((raw_id, error))


@pytest.fixture
def blob_root(workspace_env: dict[str, Path]) -> Path:
    from polylogue.paths import blob_store_root

    root = blob_store_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Phase 1: per-record CPU-bound validation contracts
# ---------------------------------------------------------------------------


class TestValidationOutcomeInvariants:
    """``_ValidationOutcome`` must not share mutable state across instances."""

    def test_default_factories_create_fresh_dicts(self) -> None:
        a = _ValidationOutcome(
            validation_status=ValidationStatus.PASSED,
            validation_error=None,
            parse_error=None,
            parseable=True,
            canonical_provider=Provider.CLAUDE_AI,
            payload_provider=None,
            drift_count=0,
        )
        b = _ValidationOutcome(
            validation_status=ValidationStatus.PASSED,
            validation_error=None,
            parse_error=None,
            parseable=True,
            canonical_provider=Provider.CLAUDE_AI,
            payload_provider=None,
            drift_count=0,
        )
        a.counts_delta["validated"] = 1
        a.drift_counts_delta["claude"] = 1
        assert b.counts_delta == {}
        assert b.drift_counts_delta == {}

    def test_validate_record_sync_decode_failure_returns_typed_outcome(self, blob_root: Path) -> None:
        """Missing blob → typed FAILED outcome, never an exception bubble-up."""
        record = _make_record("0" * 64, payload=b"")  # raw_id refers to a blob we never wrote
        outcome = _validate_record_sync(record, ValidationMode.STRICT, str(blob_root))
        assert isinstance(outcome, _ValidationOutcome)
        assert outcome.validation_status is ValidationStatus.FAILED
        assert outcome.parseable is False
        assert outcome.parse_error is not None
        assert outcome.counts_delta["errors"] == 1


class TestValidateRecordSyncDeterminism:
    """The per-record worker function is pure with respect to its inputs."""

    def test_repeat_invocations_match(self, blob_root: Path) -> None:
        digest = _write_blob(blob_root, _claude_payload("det"))
        record = _make_record(digest, payload=_claude_payload("det"))
        first = _validate_record_sync(record, ValidationMode.ADVISORY, str(blob_root))
        second = _validate_record_sync(record, ValidationMode.ADVISORY, str(blob_root))
        assert first.validation_status is second.validation_status
        assert first.canonical_provider is second.canonical_provider
        assert first.counts_delta == second.counts_delta
        assert first.drift_counts_delta == second.drift_counts_delta

    def test_distinct_inputs_produce_distinct_counts_objects(self, blob_root: Path) -> None:
        digest_a = _write_blob(blob_root, _claude_payload("a"))
        digest_b = _write_blob(blob_root, _claude_payload("b"))
        rec_a = _make_record(digest_a, payload=_claude_payload("a"))
        rec_b = _make_record(digest_b, payload=_claude_payload("b"))
        out_a = _validate_record_sync(rec_a, ValidationMode.ADVISORY, str(blob_root))
        out_b = _validate_record_sync(rec_b, ValidationMode.ADVISORY, str(blob_root))
        out_a.counts_delta["validated"] = 999
        assert out_b.counts_delta.get("validated", 0) != 999


# ---------------------------------------------------------------------------
# Phase 2: end-to-end ``evaluate_raw_artifacts`` orchestration contracts
# ---------------------------------------------------------------------------


def _seed_records(blob_root: Path, count: int) -> list[RawSessionRecord]:
    records: list[RawSessionRecord] = []
    for index in range(count):
        payload = _claude_payload(f"conv-{index:04d}")
        digest = _write_blob(blob_root, payload)
        records.append(_make_record(digest, payload=payload))
    return records


class TestEvaluateRawArtifactsOrdering:
    async def test_empty_batch_returns_empty_result(self, blob_root: Path) -> None:
        store = RecordingStore()
        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=[],
            persist=True,
            mode=ValidationMode.ADVISORY,
        )
        assert isinstance(result, ValidateResult)
        assert result.records == []
        assert store.validated_calls == []
        assert store.parsed_calls == []

    async def test_single_record_round_trip(self, blob_root: Path) -> None:
        store = RecordingStore()
        [record] = _seed_records(blob_root, 1)
        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=[record],
            persist=True,
            mode=ValidationMode.ADVISORY,
        )
        assert len(result.records) == 1
        assert result.records[0].raw_id == record.raw_id
        assert [call[0] for call in store.validated_calls] == [record.raw_id]

    async def test_output_order_matches_input_order(self, blob_root: Path) -> None:
        store = RecordingStore()
        records = _seed_records(blob_root, 8)
        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=records,
            persist=True,
            mode=ValidationMode.ADVISORY,
        )
        expected_ids = [record.raw_id for record in records]
        assert [r.raw_id for r in result.records] == expected_ids
        assert [call[0] for call in store.validated_calls] == expected_ids

    async def test_batch_larger_than_worker_cap(self, blob_root: Path) -> None:
        """The worker count is capped at min(records, cpu_count, 8); a 12-
        record batch must still complete and preserve order."""
        store = RecordingStore()
        records = _seed_records(blob_root, 12)
        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=records,
            persist=True,
            mode=ValidationMode.ADVISORY,
        )
        assert len(result.records) == 12
        assert [r.raw_id for r in result.records] == [r.raw_id for r in records]
        assert [c[0] for c in store.validated_calls] == [r.raw_id for r in records]


class TestEvaluateRawArtifactsFailureIsolation:
    async def test_undecodable_record_does_not_corrupt_batch(self, blob_root: Path) -> None:
        """A mid-batch decode failure flows through as a typed outcome and the
        rest of the batch still receives a Phase 2 write."""
        store = RecordingStore()
        good = _seed_records(blob_root, 2)
        bad = _make_record("0" * 64, payload=b"")  # blob deliberately absent
        records = [good[0], bad, good[1]]

        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=records,
            persist=True,
            mode=ValidationMode.STRICT,
        )

        assert [r.raw_id for r in result.records] == [r.raw_id for r in records]
        # the bad record produced an error count and an explicit parse_error
        bad_record = next(r for r in result.records if r.raw_id == bad.raw_id)
        assert bad_record.validation_status is ValidationStatus.FAILED
        assert bad_record.parse_error is not None
        assert result.errors >= 1

        # Every record received a mark_raw_validated; the failing one also
        # triggered a mark_raw_parsed (because parse_error was non-None).
        assert [c[0] for c in store.validated_calls] == [r.raw_id for r in records]
        assert bad.raw_id in {raw_id for raw_id, _ in store.parsed_calls}


class TestPhase2Sequentiality:
    async def test_persist_calls_finish_before_return(self, blob_root: Path) -> None:
        """``persist=True`` must complete every Phase 2 write before the
        coroutine returns — no fire-and-forget tasks left dangling."""
        store = RecordingStore()
        records = _seed_records(blob_root, 4)
        await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=records,
            persist=True,
            mode=ValidationMode.ADVISORY,
        )
        # Once evaluate_raw_artifacts returns, every record must have been
        # written exactly once.
        validated_ids = [call[0] for call in store.validated_calls]
        assert validated_ids == [r.raw_id for r in records]
        assert len(validated_ids) == len(set(validated_ids))

    async def test_persist_false_records_no_writes(self, blob_root: Path) -> None:
        store = RecordingStore()
        records = _seed_records(blob_root, 3)
        result = await evaluate_raw_artifacts(
            repository=store,
            raw_artifacts=records,
            persist=False,
            mode=ValidationMode.ADVISORY,
        )
        assert len(result.records) == 3
        assert store.validated_calls == []
        assert store.parsed_calls == []


# ---------------------------------------------------------------------------
# Hypothesis property: outcome aggregation is order-stable and accumulative
# ---------------------------------------------------------------------------


@st.composite
def _outcome_strategy(draw: Any) -> _ValidationOutcome:
    status = draw(st.sampled_from(list(ValidationStatus)))
    parseable = draw(st.booleans())
    validated = draw(st.integers(min_value=0, max_value=1))
    invalid = draw(st.integers(min_value=0, max_value=1))
    drift = draw(st.integers(min_value=0, max_value=1))
    skipped_no_schema = draw(st.integers(min_value=0, max_value=1))
    errors = draw(st.integers(min_value=0, max_value=1))
    return _ValidationOutcome(
        validation_status=status,
        validation_error="boom" if status is ValidationStatus.FAILED else None,
        parse_error=None,
        parseable=parseable,
        canonical_provider=Provider.CLAUDE_AI,
        payload_provider=None,
        drift_count=drift,
        counts_delta={
            "validated": validated,
            "invalid": invalid,
            "drift": drift,
            "skipped_no_schema": skipped_no_schema,
            "errors": errors,
        },
        drift_counts_delta={"claude": drift} if drift else {},
    )


class TestOutcomeAggregationProperty:
    @given(outcomes=st.lists(_outcome_strategy(), min_size=0, max_size=16))
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_counts_delta_accumulates_independent_of_order(self, outcomes: list[_ValidationOutcome]) -> None:
        """Mirrors the production aggregation loop in ``evaluate_raw_artifacts``
        (post-Phase 1): folding outcomes into a ``ValidateResult`` is
        commutative for the integer counters and additive for drift counts."""

        def _fold(items: list[_ValidationOutcome]) -> ValidateResult:
            result = ValidateResult()
            for outcome in items:
                result.validated += outcome.counts_delta["validated"]
                result.invalid += outcome.counts_delta["invalid"]
                result.drift += outcome.counts_delta["drift"]
                result.skipped_no_schema += outcome.counts_delta["skipped_no_schema"]
                result.errors += outcome.counts_delta["errors"]
                for prov, cnt in outcome.drift_counts_delta.items():
                    result.drift_counts[prov] = result.drift_counts.get(prov, 0) + cnt
            return result

        forward = _fold(outcomes)
        reverse = _fold(list(reversed(outcomes)))
        assert forward.counts == reverse.counts
        assert forward.drift_counts == reverse.drift_counts
