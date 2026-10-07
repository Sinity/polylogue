"""Production-route cost law for exact retained Codex prefix checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import ValidationMode
from polylogue.schemas import retained_validation
from polylogue.sources import revision_backfill
from tests.infra.retained_replay import replay_retained_components
from tests.infra.revision_backfill_benchmark import build_revision_chain_corpus


@pytest.mark.parametrize("capture_count", [5, 51, 804])
@pytest.mark.parametrize("validation_mode", [ValidationMode.ADVISORY, ValidationMode.STRICT])
@pytest.mark.timeout(600)
def test_codex_prefix_checkpoint_cost_is_constant_and_each_capture_gets_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture_count: int,
    validation_mode: ValidationMode,
) -> None:
    """Each chain needs three full preparations and one head-record scan.

    The production replay route still persists validation mode, a verdict, and
    its own parser census for every acquired capture, including interiors
    whose parsed artifacts are derived from the exact byte-prefix proof.
    """
    raw_ids = build_revision_chain_corpus(
        tmp_path,
        superseded_count=capture_count - 1,
        final_payload_bytes=capture_count,
        native_singleton=True,
    )
    prepared_ids: list[str] = []
    observed_records = 0
    original_prepare = revision_backfill.prepare_retained_jsonl_artifact
    original_observe = retained_validation.PrefixValidationState.observe

    def counted_prepare(evidence_reader: Any, raw_id: str, *, directory: Path, **kwargs: Any) -> Any:
        prepared_ids.append(raw_id)
        return original_prepare(evidence_reader, raw_id, directory=directory, **kwargs)

    def counted_observe(state: Any, record: Any) -> None:
        nonlocal observed_records
        observed_records += 1
        original_observe(state, record)

    monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", counted_prepare)
    monkeypatch.setattr(retained_validation.PrefixValidationState, "observe", counted_observe)

    result = replay_retained_components(tmp_path, validation_mode=validation_mode)

    assert result.scanned == capture_count
    assert result.classified_full == capture_count - 1
    assert result.replayed_logical_sources == 1
    assert len(prepared_ids) == 3, prepared_ids
    assert set(prepared_ids) == {raw_ids[0], raw_ids[1], raw_ids[-1]}
    # The generated head has one session_meta record plus one response_item
    # for each later capture, and the checkpoint scan observes each once.
    assert observed_records == capture_count, observed_records

    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            "SELECT raw_id, validation_mode, validation_status, validated_at_ms, "
            "revision_authority FROM raw_sessions ORDER BY acquired_at_ms"
        ).fetchall()
        census_ids = {str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_authority_parser_census").fetchall()}
    assert [row[0] for row in rows] == raw_ids
    assert all(row[1] == validation_mode.value for row in rows)
    assert all(row[2] is not None and row[3] is not None for row in rows)
    assert all(row[4] == "byte_proven" for row in rows[2:-1])
    assert set(raw_ids).issubset(census_ids)
