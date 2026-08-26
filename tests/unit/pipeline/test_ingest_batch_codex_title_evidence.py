"""Codex title evidence is carried by acquisition, never discovered by replay."""

from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_batch._core import _resolve_codex_sidecar_snapshots
from polylogue.storage.runtime import RawSessionRecord


def test_batch_resolution_marks_optional_evidence_absent_without_reading_path() -> None:
    record = RawSessionRecord(
        raw_id="raw-1",
        source_name="codex",
        source_path="/ambient/.codex/sessions/rollout.jsonl",
        payload_provider=Provider.CODEX,
        blob_size=1,
        acquired_at="2026-01-01T00:00:00+00:00",
    )

    _resolve_codex_sidecar_snapshots([record], archive_root=None)  # type: ignore[arg-type]

    # Anti-vacuity: changing or creating the ambient path cannot affect this
    # replay; only a source-cut-carried bundle may contain title evidence.
    assert record.sidecar_snapshot == {}
