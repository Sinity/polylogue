"""Production-route cost law for exact retained Codex prefix checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider, ValidationMode
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.schemas import retained_validation
from polylogue.sources import prepared_jsonl
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.storage.derived.raw import (
    _neutral_artifact_key,
    _neutral_cohort_identity,
    _NeutralParserOperand,
)
from tests.infra.retained_replay import replay_retained_components
from tests.infra.revision_backfill_benchmark import build_revision_chain_corpus


def test_neutral_cohort_digest_is_stable_typed_and_shared_by_local_keys() -> None:
    raw_ids = tuple(f"raw-{index:03d}" for index in range(51))
    operands: dict[str, _NeutralParserOperand] = {}
    for index, raw_id in enumerate(raw_ids):
        source_path = f"codex/{raw_id}.jsonl" if index != 25 else "codex/surrogate-\udcff.jsonl"
        coordinate = CapturedZipMemberCoordinate(
            canonical_container="/archive/capture.zip",
            declared_container="/declared/capture.zip",
            member_name=raw_id,
            entry_ordinal=index,
            split_index=0,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
            container_blob_hash="a" * 64,
            decoder_fingerprint="b" * 64,
        )
        operands[raw_id] = _NeutralParserOperand(
            descriptor=(Provider.CODEX, f"{index:064x}", source_path, RawRevisionKind.FULL, index + 1),
            profile_identity=None,
            fallback_timestamp=None,
            native_id=None,
            zip_coordinate=coordinate,
            append_logical_key=None,
        )

    digest = _neutral_cohort_identity(raw_ids, operands)
    assert len(digest) == 64
    assert _neutral_cohort_identity(raw_ids, operands) == digest
    assert _neutral_cohort_identity(tuple(reversed(raw_ids)), operands) != digest
    keys = tuple(_neutral_artifact_key(raw_id, operands[raw_id], ValidationMode.ADVISORY, digest) for raw_id in raw_ids)
    assert all(key[-1] == ("eligible-cohort-sha256", digest) for key in keys)
    assert all(key[-1] != raw_ids for key in keys)


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
    prepared_hashes: list[str | None] = []
    observed_records = 0
    original_prepare = prepared_jsonl.prepare_jsonl_blob
    original_observe = retained_validation.PrefixValidationState.observe

    def counted_prepare(
        blob_path: str,
        source_path: str,
        provider_value: str,
        fallback_id: str,
        *,
        source_sha256: str | None = None,
        **kwargs: Any,
    ) -> Any:
        prepared_hashes.append(source_sha256)
        return original_prepare(
            blob_path,
            source_path,
            provider_value,
            fallback_id,
            source_sha256=source_sha256,
            **kwargs,
        )

    def counted_observe(state: Any, record: Any) -> None:
        nonlocal observed_records
        observed_records += 1
        original_observe(state, record)

    monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", counted_prepare)
    monkeypatch.setattr(retained_validation.PrefixValidationState, "observe", counted_observe)

    result = replay_retained_components(tmp_path, validation_mode=validation_mode)

    assert result.scanned == capture_count
    assert result.classified_full == capture_count - 1
    assert result.replayed_logical_sources == 1
    assert len(prepared_hashes) == 3, prepared_hashes
    with sqlite3.connect(tmp_path / "source.db") as conn:
        raw_hashes = {
            str(raw_id): str(blob_hash).lower()
            for raw_id, blob_hash in conn.execute("SELECT raw_id, hex(blob_hash) FROM raw_sessions")
        }
    assert set(prepared_hashes) == {raw_hashes[raw_ids[0]], raw_hashes[raw_ids[1]], raw_hashes[raw_ids[-1]]}
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


def test_neutral_endpoint_artifacts_are_owned_before_later_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A later endpoint failure closes the earlier parsed endpoint owners."""
    raw_ids = build_revision_chain_corpus(
        tmp_path,
        superseded_count=4,
        final_payload_bytes=5,
        native_singleton=True,
    )
    original_prepare = prepared_jsonl.prepare_jsonl_blob
    original_discard = PreparedJsonl.discard
    created: list[PreparedJsonl] = []
    discarded: list[PreparedJsonl] = []

    def fail_on_head(
        blob_path: str,
        source_path: str,
        provider_value: str,
        fallback_id: str,
        **kwargs: Any,
    ) -> PreparedJsonl:
        if len(created) == 2:
            raise RuntimeError("injected head preparation failure")
        artifact = original_prepare(blob_path, source_path, provider_value, fallback_id, **kwargs)
        created.append(artifact)
        return artifact

    def count_discard(self: PreparedJsonl) -> None:
        discarded.append(self)
        original_discard(self)

    monkeypatch.setattr(prepared_jsonl, "prepare_jsonl_blob", fail_on_head)
    monkeypatch.setattr(PreparedJsonl, "discard", count_discard)

    with pytest.raises(RuntimeError, match="injected head preparation failure"):
        replay_retained_components(tmp_path, selected_raw_ids=raw_ids)

    assert len(created) == 2
    created_sessions = {artifact.sessions_path for artifact in created}
    discarded_sessions = {artifact.sessions_path for artifact in discarded}
    assert None not in created_sessions
    assert created_sessions == discarded_sessions
