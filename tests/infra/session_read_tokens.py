"""Canonical message tokens for metadata-only CLI window controls."""

from dataclasses import replace

from polylogue.archive.query.transaction import QueryContinuation
from polylogue.operations.session_contracts import SessionRead
from polylogue.operations.transcript_window import frame_request


def message_window_token(ref: str, *, page_size: int, offset: int) -> str:
    _, frame = frame_request(SessionRead(ref=ref, limit=page_size, offset=offset))
    frame = replace(frame, archive_epoch="synthetic-window-control")
    return QueryContinuation(frame, frame.result_ref).encode()
