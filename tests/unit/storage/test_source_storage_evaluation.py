"""Laws for the post-reindex source-storage comparison lab."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.source_storage_evaluation import (
    WORKLOAD_NAMES,
    CdcCandidate,
    compare_frozen_workload,
    frozen_workload,
)
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _session(case: str, index: int, payload: bytes) -> ParsedSession:
    text = payload.decode("ascii")
    message = ParsedMessage(
        provider_message_id=f"{case}-{index}",
        role=Role.USER,
        text=text,
        position=0,
        variant_index=0,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=f"source-evaluation-{case}-{index}",
        title=case,
        updated_at="2026-01-01T00:00:00Z",
        messages=[message],
    )


def test_workload_is_frozen_and_covers_every_required_transition() -> None:
    cases = frozen_workload()
    assert tuple(case.name for case in cases) == WORKLOAD_NAMES
    assert all(len(case.observations) == len(case.source_paths) for case in cases)


def test_workload_uses_production_admission_as_semantic_oracle(tmp_path: Path) -> None:
    """A candidate must not bypass the real acquire/parse/materialize route."""
    with ArchiveStore(tmp_path / "production-admission") as archive:
        for case in frozen_workload():
            for index, payload in enumerate(case.observations):
                result = archive.write_raw_and_parsed_result(
                    _session(case.name, index, payload),
                    payload=payload,
                    source_path=case.source_paths[index],
                    acquired_at_ms=index + 1,
                )
                assert result.session_id.startswith("codex-session:")


def test_frontier_and_cdc_reconstruct_identical_admitted_bytes() -> None:
    comparison = compare_frozen_workload()
    assert comparison.workload_count == len(WORKLOAD_NAMES)
    assert len(comparison.laws) == comparison.workload_count
    assert comparison.frontier_write_bytes >= comparison.cdc_write_bytes


def test_cdc_detects_corruption_and_missing_chunks() -> None:
    candidate = CdcCandidate()
    candidate.admit(b"record-000\nrecord-001\n", observation_id="case")
    refs, _, _ = candidate.manifests["case"]
    candidate.chunks[refs[0]] = b"tampered"
    with pytest.raises(ValueError, match="integrity"):
        candidate.read("case")

    candidate = CdcCandidate()
    candidate.admit(b"record-000\nrecord-001\n", observation_id="case")
    refs, _, _ = candidate.manifests["case"]
    del candidate.chunks[refs[-1]]
    with pytest.raises(ValueError, match="missing chunk"):
        candidate.read("case")


def test_cdc_retry_is_idempotent_and_partial_publication_is_unreadable() -> None:
    candidate = CdcCandidate()
    payload = b"record-000\nrecord-001\n"
    candidate.admit(payload, observation_id="case")
    stored_chunks = dict(candidate.chunks)
    candidate.admit(payload, observation_id="case")
    assert candidate.chunks == stored_chunks
    assert candidate.read("case") == payload

    refs, expected_hash, expected_length = candidate.manifests["case"]
    candidate.manifests["partial"] = (refs, expected_hash, expected_length + 1)
    with pytest.raises(ValueError, match="integrity"):
        candidate.read("partial")


def test_privacy_erasure_removes_only_unreferenced_cdc_chunks() -> None:
    candidate = CdcCandidate()
    candidate.admit(b"shared-prefix\nprivate-a", observation_id="a")
    candidate.admit(b"shared-prefix\nprivate-b", observation_id="b")
    before = set(candidate.chunks)
    candidate.erase("a")
    assert candidate.read("b") == b"shared-prefix\nprivate-b"
    assert set(candidate.chunks) < before
    candidate.erase("b")
    assert candidate.chunks == {}
