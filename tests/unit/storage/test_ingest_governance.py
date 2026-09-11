"""Prepared membership governance keeps parse work outside the writer hold."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.ingest_governance import (
    prepare_ingest_cohort,
    prepare_raw_census,
    publish_ingest_cohort,
    publish_raw_census,
)
from polylogue.storage.sqlite.archive_tiers import revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _session(
    *texts: str,
    session_id: str = "prepared-membership",
    attachment_bytes: bytes | None = None,
    precomputed_blob: tuple[str, int] | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        messages=[
            ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text=text)
            for index, text in enumerate(texts)
        ],
        attachments=(
            [
                ParsedAttachment(
                    provider_attachment_id="prepared-attachment",
                    message_provider_id="m0",
                    name="prepared.bin",
                    mime_type="application/octet-stream",
                    size_bytes=len(attachment_bytes) if attachment_bytes is not None else precomputed_blob[1],
                    inline_bytes=attachment_bytes,
                    precomputed_blob=precomputed_blob,
                )
            ]
            if attachment_bytes is not None or precomputed_blob is not None
            else []
        ),
    )


def _parse_from(sessions_by_raw_id: dict[str, ParsedSession]):
    def parse(_archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
        return [sessions_by_raw_id[raw_id]]

    return parse


def _write_raws(archive: ArchiveStore, count: int) -> tuple[str, ...]:
    return tuple(
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=f'{{"raw":{index}}}'.encode(),
            source_path=f"prepared-{index}.jsonl",
            acquired_at_ms=index + 1,
        )
        for index in range(count)
    )


def _publish_census(archive: ArchiveStore, raw_id: str, parse, *, at_ms: int) -> None:
    prepared = prepare_raw_census(
        archive,
        raw_id,
        parser_fingerprint="prepared-test-parser",
        parse_retained_raw=parse,
        censused_at_ms=at_ms,
    )
    assert publish_raw_census(archive, prepared).published


def test_prepared_census_reuses_projection_and_does_not_terminalize_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publishing must use the compute projection and remain source-census-only.

    Anti-vacuity: if publication reprojects the session or marks a raw parsed,
    the patched projector or the final source-row assertion makes this fail.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)
    sessions = {raw_id: _session("one", "two")}
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        assert reader.blob_path_for_hash(blob_hash) == tmp_path / "blob" / blob_hash[:2] / blob_hash[2:]
        prepared = prepare_raw_census(
            reader,
            raw_id,
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=_parse_from(sessions),
            censused_at_ms=7,
        )

    monkeypatch.setattr(
        revision_governance,
        "session_revision_projection",
        lambda _session: (_ for _ in ()).throw(AssertionError("writer recomputed prepared projection")),
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        assert publish_raw_census(archive, prepared).published

        source = archive._ensure_source_conn()
        assert source.execute(
            "SELECT status, member_count FROM raw_membership_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete", 1)
        assert source.execute(
            "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (
            None,
            None,
        )


def test_read_only_census_uses_canonical_retained_raw_parser(tmp_path: Path) -> None:
    """Shared compute reads retained material without a blob publisher.

    Anti-vacuity: a publisher-only retained-material helper fails on the real
    read-only archive before the canonical backfill parser can return Codex's
    one retained session.
    """
    bootstrap_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"read-only-retained"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m0",'
        b'"role":"user","content":[{"type":"input_text","text":"retained"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="read-only-retained.jsonl",
            acquired_at_ms=1,
        )
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        prepared = prepare_raw_census(
            reader,
            raw_id,
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse_retained_raw_sessions,
            censused_at_ms=2,
        )
    assert prepared.status.value == "complete"
    assert prepared.logical_keys == ("codex-session:read-only-retained",)


def test_prepared_cohort_revalidates_head_and_descriptor_before_writer_mutation(tmp_path: Path) -> None:
    """A later eligible head and an equal-count descriptor change both defer.

    Anti-vacuity: removing either frontier/descriptor binding lets the stale
    prepared cohort call the index writer after durable source evidence moved.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, second_raw, third_raw = _write_raws(archive, 3)
        sessions = {
            first_raw: _session("one"),
            second_raw: _session("one", "two"),
            third_raw: _session("one", "two", "three"),
        }
        parse = _parse_from(sessions)
        _publish_census(archive, first_raw, parse, at_ms=1)
        _publish_census(archive, second_raw, parse, at_ms=2)
        initial = prepare_ingest_cohort(
            archive,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(first_raw, second_raw),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=3,
        )
        assert publish_ingest_cohort(archive, initial).published

        stale_head = prepare_ingest_cohort(
            archive,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(first_raw, second_raw),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=4,
        )
        _publish_census(archive, third_raw, parse, at_ms=5)
        advancing = prepare_ingest_cohort(
            archive,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(third_raw,),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=6,
        )
        assert publish_ingest_cohort(archive, advancing).published
        head_result = publish_ingest_cohort(archive, stale_head)
        assert not head_result.published
        assert head_result.reprepare_required
        assert head_result.reason in {
            "eligible membership selector changed",
            "accepted head or persisted session frontier changed",
        }

        stale_descriptor = prepare_ingest_cohort(
            archive,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(third_raw,),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=7,
        )
        with archive._ensure_source_conn():
            archive._ensure_source_conn().execute(
                "UPDATE raw_sessions SET source_path = ? WHERE raw_id = ?",
                ("equal-count-but-new-descriptor.jsonl", third_raw),
            )
        descriptor_result = publish_ingest_cohort(archive, stale_descriptor)
        assert not descriptor_result.published
        assert descriptor_result.reprepare_required
        assert descriptor_result.reason == "raw descriptor, membership, or census changed"


def test_read_only_compute_defers_attachment_publication_until_writer_revalidation(tmp_path: Path) -> None:
    """Read-only compute carries attachment evidence; only the writer publishes it.

    Anti-vacuity: a compute-side publisher rejects under a read-only archive,
    while a stale prepared attachment must leave both index attachments and
    source blob references absent before a fresh writer-admitted publication.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, second_raw = _write_raws(archive, 2)
        sessions = {
            first_raw: _session("one"),
            second_raw: _session("one", "two", attachment_bytes=b"prepared-attachment-bytes"),
        }
        parse = _parse_from(sessions)
        _publish_census(archive, first_raw, parse, at_ms=1)
        _publish_census(archive, second_raw, parse, at_ms=2)

    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        prepared = prepare_ingest_cohort(
            reader,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(first_raw, second_raw),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=3,
        )
        assert prepared.prepared_rows_by_raw_id
        assert prepared.prepared_attachment_blobs
        stale_staged_path = prepared.prepared_attachment_blobs[0].prepared_blob.temporary_path
        assert stale_staged_path.is_file()

    with ArchiveStore.open_existing(tmp_path, read_only=False) as writer:
        with writer._ensure_source_conn():
            writer._ensure_source_conn().execute(
                "UPDATE raw_sessions SET source_path = ? WHERE raw_id = ?",
                ("changed-before-writer-admission.jsonl", second_raw),
            )
        stale = publish_ingest_cohort(writer, prepared)
        assert not stale.published
        assert stale.reprepare_required
        assert not stale_staged_path.exists()
        assert writer._conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 0
        assert writer._ensure_source_conn().execute(
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'"
        ).fetchone() == (0,)

    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        fresh = prepare_ingest_cohort(
            reader,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(first_raw, second_raw),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=4,
        )
        fresh_staged_path = fresh.prepared_attachment_blobs[0].prepared_blob.temporary_path
        assert fresh_staged_path.is_file()
    with ArchiveStore.open_existing(tmp_path, read_only=False) as writer:
        result = publish_ingest_cohort(writer, fresh)
        assert result.published
        assert result.session_id == "codex-session:prepared-membership"
        assert not fresh_staged_path.exists()
        assert writer._conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 1
        assert writer._ensure_source_conn().execute(
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'"
        ).fetchone() == (1,)


def test_convertible_multi_session_retirement_preserves_complete_census(tmp_path: Path) -> None:
    """Retiring one full export must retain every logical member it contained.

    Anti-vacuity: passing only the target session to replacement would delete
    the sibling membership, so the complete two-key assertion becomes red.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, second_raw = _write_raws(archive, 2)
        archive.bind_raw_revision(
            first_raw,
            RawRevisionEnvelope(
                logical_source_key="codex-session:prepared-membership",
                kind=RawRevisionKind.FULL,
                source_revision="prepared-full-v1",
                acquisition_generation=0,
            ),
        )
        archive.bind_raw_revision(
            second_raw,
            RawRevisionEnvelope(
                logical_source_key="codex-session:prepared-membership",
                kind=RawRevisionKind.FULL,
                source_revision="prepared-full-v2",
                acquisition_generation=1,
                predecessor_raw_id=first_raw,
                baseline_raw_id=first_raw,
            ),
        )
        sessions = {
            first_raw: _session("one", session_id="prepared-membership").model_copy(
                update={"messages": [ParsedMessage(provider_message_id="m0", role=Role.USER, text="one")]}
            ),
            second_raw: _session("two", session_id="prepared-membership"),
        }

        def parse(_archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
            return [sessions[raw_id], _session("other", session_id="prepared-sibling")]

        prepared = prepare_ingest_cohort(
            archive,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=3,
        )
        result = publish_ingest_cohort(archive, prepared)
        assert not result.published
        assert result.reprepare_required
        assert set(result.retired_raw_ids) == {first_raw, second_raw}
        assert result.reprepare_logical_source_keys == (
            "codex-session:prepared-membership",
            "codex-session:prepared-sibling",
        )
        for raw_id in (first_raw, second_raw):
            assert archive._ensure_source_conn().execute(
                "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key",
                (raw_id,),
            ).fetchall() == [
                ("codex-session:prepared-membership",),
                ("codex-session:prepared-sibling",),
            ]


def test_precomputed_attachment_is_preserved_without_compute_publication(tmp_path: Path) -> None:
    """A pre-existing attachment blob is carried into the writer map unchanged.

    Anti-vacuity: omitting the precomputed entry from the map makes the shared
    attachment writer raise its explicit missing-preacquisition ValueError.
    """
    bootstrap_archive_root(tmp_path)
    hash_hex, size = BlobStore(tmp_path / "blob").write_from_bytes(b"precomputed attachment")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)
        sessions = {raw_id: _session("one", precomputed_blob=(hash_hex, size))}
        parse = _parse_from(sessions)
        _publish_census(archive, raw_id, parse, at_ms=1)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        prepared = prepare_ingest_cohort(
            reader,
            logical_source_key="codex-session:prepared-membership",
            accepted_raw_ids=(raw_id,),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=2,
        )
        assert prepared.prepared_attachment_blobs[0].prepared_blob is None
    with ArchiveStore.open_existing(tmp_path, read_only=False) as writer:
        assert publish_ingest_cohort(writer, prepared).published
        assert writer._conn.execute("SELECT lower(hex(blob_hash)) FROM attachments").fetchone()[0] == hash_hex
