"""Neutral Source baselines for the durable index replacement contract."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind, raw_failure_classification_reason
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def source_baseline(path: Path) -> tuple[sqlite3.Connection, tuple[str, str]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    raw_ids: list[str] = []
    store = BlobStore(path.parent / "blob")
    for ordinal in range(2):
        payload = f'{{"synthetic_record":{ordinal}}}\n'.encode()
        # Source metadata alone is not a retained acquisition. The canonical
        # backup/restart route needs these exact original bytes in this archive.
        store.write_from_bytes(payload)
        raw_ids.append(
            write_source_raw_session(
                conn,
                origin=Origin.CODEX_SESSION,
                capture_mode=Provider.CODEX,
                source_path="/synthetic/windowless.jsonl",
                canonical_source_path="/synthetic/windowless.jsonl",
                source_index=-1,
                payload=payload,
                acquired_at_ms=1,
            )
        )
    return conn, (raw_ids[0], raw_ids[1])


def missing_coordinates_artifact(raw_id: str) -> ArchiveSourceArtifact:
    return ArchiveSourceArtifact(
        artifact_id=f"missing-coordinates:{raw_id}",
        origin=Origin.CODEX_SESSION,
        source_path="/synthetic/windowless.jsonl",
        source_index=-1,
        artifact_kind=RawFailureEvidenceKind.TERMINAL_MISSING_SOURCE_COORDINATES.value,
        support_status=RawFailureEvidenceKind.TERMINAL_MISSING_SOURCE_COORDINATES.support_status,
        classification_reason=raw_failure_classification_reason(
            diagnostic=None,
            evidence_ref="proof:missing-source-coordinates",
            outcome_code=RawFailureEvidenceKind.TERMINAL_MISSING_SOURCE_COORDINATES.value,
            remediation=None,
            retryable=False,
            trusted_validation_failure=False,
        ),
        parse_as_session=False,
        schema_eligible=False,
    )
