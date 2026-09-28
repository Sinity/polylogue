"""The raw-ingest route computes content identity in the parse worker only.

``ingest_record`` (the parse worker) hashes each session once and builds its
full-replace rows and message content identities. ``_process_ingest_batch_sync``
(the writer) admits that carrier through ``prepare_session_write``'s validated
gate and publishes it without hashing anything.

Anti-vacuity: the writer phase makes every identity function fatal, so the
session is not written if the worker stops building ``prepared_rows`` or the
writer stops passing it to ``prepare_session_write``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import polylogue.pipeline.ids as pipeline_ids
import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
import polylogue.storage.sqlite.archive_tiers.revision_governance as revision_governance
from polylogue.core.enums import Provider
from polylogue.pipeline.services import ingest_worker as ingest_worker_mod
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, ingest_record
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    reset_blob_store()
    yield store
    reset_blob_store()


def _claude_code_session() -> bytes:
    lines = []
    for index, (role, text) in enumerate((("user", "hello"), ("assistant", "hi there"), ("user", "hello"))):
        lines.append(
            b'{"type":"%s","uuid":"u%d","sessionId":"carrier","parentUuid":%s,"cwd":"/tmp",'
            b'"timestamp":"2026-01-01T00:00:0%dZ","message":{"role":"%s","content":"%s"}}\n'
            % (
                role.encode(),
                index,
                b"null" if index == 0 else b'"u%d"' % (index - 1),
                index,
                role.encode(),
                text.encode(),
            )
        )
    return b"".join(lines)


def _record(store: BlobStore, source_path: Path) -> RawSessionRecord:
    content = _claude_code_session()
    source_path.write_bytes(content)
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name="claude-code",
        source_path=str(source_path),
        payload_provider=Provider.CLAUDE_CODE,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


def _counting(counts: dict[str, int], name: str, fn: Callable[..., object]) -> Callable[..., object]:
    def wrapper(*args: object, **kwargs: object) -> object:
        counts[name] = counts.get(name, 0) + 1
        return fn(*args, **kwargs)

    return wrapper


def _fatal(name: str) -> Callable[..., object]:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"writer computed {name} instead of admitting the worker's carrier")

    return boom


def test_worker_computes_identity_once_and_writer_admits_it(
    tmp_path: Path, blob_store: BlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(blob_store, tmp_path / "session.jsonl")
    archive_root = bootstrap_archive_root(tmp_path / "archive")

    worker_counts: dict[str, int] = {}
    with monkeypatch.context() as worker_patch:
        worker_patch.setattr(
            ingest_worker_mod,
            "session_content_hash",
            _counting(worker_counts, "session_content_hash", pipeline_ids.session_content_hash),
        )
        worker_patch.setattr(
            archive_tier_write,
            "message_content_identities",
            _counting(worker_counts, "message_content_identities", pipeline_ids.message_content_identities),
        )
        result = ingest_record(record, str(archive_root), "off", blob_root_str=str(blob_store.root))

    assert result.error is None
    [payload] = result.sessions
    assert worker_counts == {"session_content_hash": 1, "message_content_identities": 1}
    carrier = payload.prepared_rows
    assert carrier is not None
    assert carrier.session_content_hash.hex() == payload.content_hash
    assert len(carrier.content_identities) == 3

    for module, name in (
        (pipeline_ids, "session_content_hash"),
        (revision_governance, "session_content_hash"),
        (ingest_batch_core, "session_content_hash"),
        (archive_tier_write, "message_content_identities"),
        (archive_tier_write, "disk_message_content_identities"),
        (archive_tier_write, "_build_message_rows"),
        (archive_tier_write, "_build_block_rows"),
    ):
        monkeypatch.setattr(module, name, _fatal(name))

    def worker_result(record_arg: RawSessionRecord, *_args: object, **_kwargs: object) -> IngestRecordResult:
        assert record_arg.raw_id == record.raw_id
        return result

    monkeypatch.setattr(ingest_batch_core, "ingest_record", worker_result)
    db_path = archive_root / "index.db"
    summary = _process_ingest_batch_sync(
        [record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_store.root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.changed_session_ids == [payload.session_id]
    with open_connection(db_path) as conn:
        stored_hash = conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?", (payload.session_id,)
        ).fetchone()[0]
        stored_identities = conn.execute(
            "SELECT content_identity, content_occurrence FROM messages WHERE session_id = ? ORDER BY position",
            (payload.session_id,),
        ).fetchall()
    assert bytes(stored_hash).hex() == payload.content_hash
    assert [tuple(row) for row in stored_identities] == list(carrier.content_identities)
