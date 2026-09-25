"""File-backed JSONL preparation transport for source-owned ingest."""

from __future__ import annotations

import hashlib
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from polylogue.storage.sqlite.archive_tiers.write_shard import discard_session_shard


@dataclass(frozen=True, slots=True)
class PreparedJsonl:
    """Private transport fenced to the exact bytes interpreted by a worker."""

    blob_hash: str | None
    sessions_path: Path | None
    shard_path: Path | None
    error: str | None = None

    def discard(self) -> None:
        if self.sessions_path is not None:
            self.sessions_path.unlink(missing_ok=True)
        if self.shard_path is not None:
            discard_session_shard(self.shard_path)

    def load_sessions(self) -> list[ParsedSession]:
        if self.sessions_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no session carrier")
        try:
            with self.sessions_path.open("rb") as handle:
                sessions = pickle.load(handle)
        finally:
            self.sessions_path.unlink(missing_ok=True)
        if not isinstance(sessions, list) or any(not isinstance(item, ParsedSession) for item in sessions):
            raise ValueError("JSONL preparation contains an invalid session carrier")
        return sessions


def prepare_jsonl_blob(
    blob_path: str,
    source_path: str,
    provider_value: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
) -> PreparedJsonl:
    """Parse and seal one blob without transferring a parsed tree over IPC."""
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    sessions_path = directory / f"sessions-{uuid.uuid4().hex}.pickle"
    shard_path: Path | None = None
    try:
        provider = Provider.from_string(provider_value)

        def source_digest() -> str:
            digest = hashlib.sha256()
            with Path(blob_path).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        before_hash = source_digest()
        with Path(blob_path).open("rb") as handle:
            records = _iter_json_stream(
                handle,
                Path(source_path).name,
                fail_on_decode_error=provider is Provider.UNKNOWN,
            )
            if is_stream:
                sessions = parse_stream_payload(provider, records, fallback_id, source_path=source_path)
            else:
                sessions = parse_payload(provider, list(records), fallback_id, source_path=source_path)
        after_hash = source_digest()
        if before_hash != after_hash:
            raise ValueError("blob changed during worker preparation")
        shard_path = prepare_session_shard(directory, sessions).path if sessions else None
        with sessions_path.open("xb") as handle:
            pickle.dump(sessions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return PreparedJsonl(after_hash, sessions_path, shard_path)
    except Exception as exc:
        sessions_path.unlink(missing_ok=True)
        if shard_path is not None:
            discard_session_shard(shard_path)
        return PreparedJsonl(None, None, None, f"{type(exc).__name__}: {exc}"[:500])
