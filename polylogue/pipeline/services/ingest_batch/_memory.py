"""Memory-release helpers for large ingest result payloads."""

from __future__ import annotations

from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionData

INGEST_RELEASE_BLOB_MB_THRESHOLD = 16.0
INGEST_RELEASE_MESSAGE_THRESHOLD = 1_000
INGEST_RELEASE_ROW_THRESHOLD = 10_000


def _session_payload_row_count(cdata: SessionData) -> int:
    return (
        len(cdata.message_tuples)
        + len(cdata.block_tuples)
        + len(cdata.action_event_tuples)
        + len(cdata.provider_event_tuples)
        + len(cdata.attachment_tuples)
        + len(cdata.attachment_ref_tuples)
    )


def ingest_result_needs_memory_release(ir: IngestRecordResult) -> bool:
    if ir.serialized_size_bytes is not None:
        return ir.serialized_size_bytes >= int(INGEST_RELEASE_BLOB_MB_THRESHOLD * 1024 * 1024)
    total_messages = sum(len(cdata.message_tuples) for cdata in ir.sessions)
    if total_messages >= INGEST_RELEASE_MESSAGE_THRESHOLD:
        return True
    total_rows = sum(_session_payload_row_count(cdata) for cdata in ir.sessions)
    return total_rows >= INGEST_RELEASE_ROW_THRESHOLD


def discard_session_data_payload(cdata: SessionData) -> None:
    cdata.message_tuples.clear()
    cdata.block_tuples.clear()
    cdata.action_event_tuples.clear()
    cdata.provider_event_tuples.clear()
    cdata.attachment_tuples.clear()
    cdata.attachment_ref_tuples.clear()


def discard_ingest_result_payload(ir: IngestRecordResult) -> None:
    ir.sessions.clear()
    ir.parsed_sessions.clear()


__all__ = [
    "INGEST_RELEASE_BLOB_MB_THRESHOLD",
    "INGEST_RELEASE_MESSAGE_THRESHOLD",
    "discard_session_data_payload",
    "discard_ingest_result_payload",
    "ingest_result_needs_memory_release",
]
