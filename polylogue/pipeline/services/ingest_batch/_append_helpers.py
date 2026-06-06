"""Append-ingest tuple helpers."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import cast

from polylogue.pipeline.services.ingest_worker import MessageTuple, ProviderEventTuple, SessionTuple


def append_content_hash(existing_hash: str | None, tail_hash: str) -> str:
    return sha256(f"{existing_hash}\0{tail_hash}".encode()).hexdigest() if existing_hash else tail_hash


def tail_content_hash(changed_messages: Sequence[MessageTuple], fallback_hash: str) -> str:
    if not changed_messages:
        return fallback_hash
    return sha256("\0".join(str(message[6]) for message in changed_messages).encode()).hexdigest()


def session_tuple_without_raw_id(session: SessionTuple) -> SessionTuple:
    return cast(SessionTuple, (*session[:13], None, *session[14:]))


def provider_event_tuple_without_raw_id(event: ProviderEventTuple) -> ProviderEventTuple:
    return cast(ProviderEventTuple, (*event[:10], None, event[11]))


__all__ = [
    "append_content_hash",
    "session_tuple_without_raw_id",
    "provider_event_tuple_without_raw_id",
    "tail_content_hash",
]
