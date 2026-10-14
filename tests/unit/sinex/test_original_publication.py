"""Accepted source output retains separate original custody through its outbox."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.config import load_polylogue_config
from polylogue.core.enums import Provider
from polylogue.sinex.models import PublicationMode, PublicationPayload
from polylogue.sinex.obligations import list_obligations, load_payload
from polylogue.sinex.provenance import require_originals_commitment
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from tests.infra.retained_replay import publish_retained_payload


@pytest.mark.asyncio
async def test_acquired_source_outbox_drains_through_actual_writer_admission(tmp_path: Path) -> None:
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import LocalReferenceTransport
    from tests.infra.sinex_publication import acquired_publication, drain_publication

    obligation, _ = await acquired_publication(tmp_path)
    service = PublicationService(tmp_path / "source.db", PublicationMode.MIRROR, LocalReferenceTransport())
    summary = await drain_publication(service)
    assert summary.attempted == summary.confirmed == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = list_obligations(conn)
        assert len(rows) == 1 and rows[0].request_id == obligation.request_id
        assert rows[0].last_receipt_state is not None
        assert rows[0].last_receipt_state.value == "persisted_confirmed"
    assert (await drain_publication(service, redrive=True)).confirmed == 0


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_large_native_fixture_retains_actual_source_and_complete_record_set(tmp_path: Path) -> None:
    from collections import Counter

    from polylogue.storage.blob_store import BlobStore
    from tests.infra.sinex_publication import acquired_publication

    _, payload = await acquired_publication(tmp_path, extra_grok_responses=1536)
    manifest = json.loads(payload.manifest_bytes)
    counts = Counter(anchor["kind"] for anchor in manifest["anchors"].values())
    assert counts == {"session": 1, "session_event": 3, "message": 1540, "block": 1545, "attachment": 2}
    assert len(manifest["anchors"]) == 3091
    assert [item.role for item in payload.originals] == ["provider_native", "attachment", "attachment"]
    provider = payload.originals[0]
    assert provider.acquisition.kind == "full_snapshot"
    original = json.loads(BlobStore(tmp_path / "blob").read_all(provider.blob_hash))
    native_responses = original["raw_provider_payload"]["responses"]["responses"]
    assert len(native_responses) == 1540
    assert native_responses[4]["parentResponseId"] == "attachment"
    assert native_responses[-1]["responseId"] == "large-01535"
    assert len(original["session"]["attachments"]) == 2
    require_originals_commitment(payload)


@pytest.mark.asyncio
async def test_actual_retained_acceptance_stages_original_custody_with_source_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = load_polylogue_config()
    configured = replace(
        configured,
        _data={**configured._data, "sinex_mode": PublicationMode.MIRROR.value, "archive_root": str(tmp_path)},
    )
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **_kwargs: configured)
    payload = (
        json.dumps(
            {
                "type": "user",
                "sessionId": "s1",
                "uuid": "m1",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "neutral transcript"},
            }
        ).encode()
        + b"\n"
    )
    raw_id, written = await publish_retained_payload(
        tmp_path,
        provider=Provider.CLAUDE_CODE,
        payload=payload,
        source_path="/synthetic/s1.jsonl",
        acquired_at_ms=1000,
    )
    assert written == ("claude-code-session:s1",)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        obligations = list_obligations(conn)
        assert len(obligations) == 1
        staged = load_payload(conn, obligations[0])
        assert staged.originals[0].role == "provider_native"
        assert staged.originals[0].source_ref == f"polylogue.raw:{raw_id}"
        assert staged.originals[0].size_bytes == len(payload)
        require_originals_commitment(staged)
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()[0] is not None
        assert conn.execute("SELECT count(*) FROM sinex_publication_originals").fetchone()[0] == 1
        assert ";parsers=" in staged.normalizer_version


@pytest.mark.asyncio
async def test_two_append_windows_publish_full_output_with_all_contributing_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from functools import partial
    from hashlib import sha256

    from polylogue.archive.revision_authority import (
        RawRevisionAuthority,
        RawRevisionEnvelope,
        RawRevisionKind,
        append_source_revision,
    )
    from polylogue.material_protocol.v1 import RevisionManifest, decode_session_revision
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
    from tests.infra.retained_replay import replay_retained_components_async

    configured = load_polylogue_config()
    configured = replace(configured, _data={**configured._data, "sinex_mode": "mirror", "archive_root": str(tmp_path)})
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **_kwargs: configured)
    parts = [
        json.dumps(
            {
                "type": "user",
                "sessionId": "s1",
                "uuid": f"m{i}",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": f"neutral message {i}"},
            }
        ).encode()
        + b"\n"
        for i in range(3)
    ]

    ids: list[str] = []
    revision = sha256(parts[0]).hexdigest()
    offset = 0
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))

    def acquire(index: int) -> str:
        nonlocal revision, offset
        part = parts[index]
        previous = revision
        if index:
            revision = append_source_revision(previous, sha256(part).hexdigest())
        envelope = RawRevisionEnvelope(
            "claude-code-session:s1",
            RawRevisionKind.FULL if index == 0 else RawRevisionKind.APPEND,
            revision,
            index,
            predecessor_source_revision=previous if index else None,
            predecessor_raw_id=ids[-1] if index else None,
            baseline_raw_id=ids[0] if index else None,
            append_start_offset=offset if index else None,
            append_end_offset=offset + len(part) if index else None,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        )
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=part,
                source_path="/synthetic/s1.jsonl",
                canonical_source_path="/synthetic/s1.jsonl",
                acquired_at_ms=index + 1,
                native_id="s1",
                revision=envelope,
            )
        offset += len(part)
        return raw_id

    for index in range(3):
        raw_id = await run_archive_fixture_write(tmp_path, partial(acquire, index))
        ids.append(raw_id)
        # Each ancestor has already been acknowledged and published before
        # the next append arrives; only that new raw seeds convergence.
        await replay_retained_components_async(tmp_path, selected_raw_ids=(raw_id,))
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.row_factory = sqlite3.Row
            staged = [load_payload(conn, row) for row in list_obligations(conn)]
            assert len(staged) == index + 1
            assert any(
                [item.source_ref for item in payload.originals] == [f"polylogue.raw:{value}" for value in ids]
                for payload in staged
            )
            assert (
                conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()[0]
                is not None
            )
    raw_ids = tuple(ids)
    assert not Path("/synthetic/s1.jsonl").exists()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        obligations = list_obligations(conn)
        assert len(obligations) == 3
        candidates = [load_payload(conn, row) for row in obligations]
        payload = next(value for value in candidates if len(value.originals) == 3)
        assert [item.source_ref for item in payload.originals] == [f"polylogue.raw:{raw_id}" for raw_id in raw_ids]
        assert [BlobStore(tmp_path / "blob").read_all(item.blob_hash) for item in payload.originals] == parts
        manifest = RevisionManifest.from_dict(json.loads(payload.manifest_bytes))
        segments = dict(payload.segments)
        decoded = decode_session_revision(
            manifest, {d.index: segments[d.filename] for d in (*manifest.segments, manifest.head_segment)}
        )
        assert [message.text for message in decoded.messages] == [f"neutral message {i}" for i in range(3)]
        require_originals_commitment(payload)


@pytest.mark.asyncio
async def test_unpublished_source_plan_cannot_leave_outbox_original_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.sinex import prepared

    configured = replace(
        load_polylogue_config(),
        _data={**load_polylogue_config()._data, "sinex_mode": "mirror", "archive_root": str(tmp_path)},
    )
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **_kwargs: configured)
    original = prepared.prepare_publication_payload

    def fail_after_capture(
        seal: PreparedIndexMutation, payload: PublicationPayload, *, mode: PublicationMode, now_ms: int
    ) -> None:
        original(seal, payload, mode=mode, now_ms=now_ms)
        raise RuntimeError("synthetic source preparation cancellation")

    monkeypatch.setattr(prepared, "prepare_publication_payload", fail_after_capture)
    body = b'{"type":"user","sessionId":"s1","uuid":"m1","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"neutral"}}\n'
    with pytest.raises(RuntimeError, match="synthetic source preparation cancellation"):
        await publish_retained_payload(
            tmp_path, provider=Provider.CLAUDE_CODE, payload=body, source_path="/synthetic/s1.jsonl", acquired_at_ms=1
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT count(*) FROM sinex_publication_obligations").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM sinex_publication_originals").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_acquired_attachment_original_is_distinct_and_outbox_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.blob_liveness import inspect_blob_liveness
    from polylogue.storage.blob_store import BlobStore
    from tests.infra.live_provider_proof import native_proof_artifact

    configured = replace(
        load_polylogue_config(),
        _data={**load_polylogue_config()._data, "sinex_mode": "mirror", "archive_root": str(tmp_path)},
    )
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda **_kwargs: configured)
    envelope, _, _ = native_proof_artifact(tmp_path, "native-inline-attachment-v1.json", Provider.GROK)
    original_bytes = json.dumps(envelope).encode()
    raw_id, written = await publish_retained_payload(
        tmp_path,
        provider=Provider.GROK,
        payload=original_bytes,
        source_path="/synthetic/grok-capture.json",
        acquired_at_ms=1,
    )
    assert written
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        payload = load_payload(conn, list_obligations(conn)[0])
        providers = [item for item in payload.originals if item.role == "provider_native"]
        attachments = [item for item in payload.originals if item.role == "attachment"]
        assert len(providers) == 1 and attachments
        assert BlobStore(tmp_path / "blob").read_all(providers[0].blob_hash) == original_bytes
        assert {item.blob_hash for item in attachments}.isdisjoint({providers[0].blob_hash})
        assert all(BlobStore(tmp_path / "blob").read_all(item.blob_hash) for item in attachments)
        for item in payload.originals:
            assert "source.db.sinex_publication_originals" in inspect_blob_liveness(conn, item.blob_hash).surfaces
        # Remove only the synthetic acquisition's older ref surfaces. The
        # publication's own durable custody remains an authoritative GC root.
        conn.execute("DELETE FROM blob_refs WHERE ref_id=?", (raw_id,))
        conn.execute("DELETE FROM raw_sessions WHERE raw_id=?", (raw_id,))
        for item in payload.originals:
            assert "source.db.sinex_publication_originals" in inspect_blob_liveness(conn, item.blob_hash).surfaces
        require_originals_commitment(load_payload(conn, list_obligations(conn)[0]))
        conn.rollback()


@pytest.mark.asyncio
async def test_logical_sqlite_original_keeps_exact_export_scope_after_live_database_removal(tmp_path: Path) -> None:
    from polylogue.sinex.originals import originals_for_raw_ids
    from polylogue.sources.sqlite_export import MemberExportScope, logical_export_bytes
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write

    external = tmp_path / "neutral-provider.sqlite"
    with sqlite3.connect(external) as database:
        database.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT)")
        database.execute("CREATE TABLE excluded (body TEXT)")
        database.execute("INSERT INTO messages VALUES ('m1','neutral message')")
        database.execute("INSERT INTO excluded VALUES ('outside acquisition scope')")
    payload = logical_export_bytes(external, scope=MemberExportScope(tables=("messages",), member="neutral-provider"))
    physical = external.read_bytes()
    archive_root = tmp_path / "archive"

    def acquire() -> str:
        bootstrap_archive_root(archive_root)
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.UNKNOWN,
                payload=payload,
                source_path=str(external),
                canonical_source_path=str(external),
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(archive_root, acquire)
    external.unlink()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        cas = BlobStore(archive_root / "blob")
        originals = originals_for_raw_ids(conn, (raw_id,), blob_store=cas)
    assert len(originals) == 1
    original = originals[0]
    assert original.acquisition.kind == "logical_snapshot"
    assert original.acquisition.tables == ("messages",)
    assert original.acquisition.addressing_mode is None
    assert original.acquisition.byte_start is None and original.acquisition.byte_end is None
    assert cas.read_all(original.blob_hash) == payload and payload != physical
    assert b"outside acquisition scope" not in payload
    assert not external.exists()
