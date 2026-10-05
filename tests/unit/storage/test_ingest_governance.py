"""Prepared membership governance keeps parse work outside the writer hold."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TypeAlias

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import BlockType, Provider
from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError
from polylogue.pipeline.ids import bound_session_content_hash, session_content_hash
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.ingest_governance import (
    _census_binding,
    prepare_ingest_cohort,
    prepare_raw_census,
    publish_ingest_cohort,
    publish_raw_census,
)
from polylogue.storage.sqlite.archive_tiers import revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

ParseFunction: TypeAlias = Callable[[ArchiveStore, str], list[ParsedSession]]


def test_prepared_census_binding_includes_revision_authority(tmp_path: Path) -> None:
    """Authority-only census changes stale prepared descriptors.

    Anti-vacuity: remove ``revision_authority`` from ``_census_binding``'s
    SELECT/value and these two prepared-work snapshots become equal.
    """
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)
        conn = archive._ensure_source_conn()
        conn.execute(
            "INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail, revision_authority) "
            "VALUES (?, 'parser', 'complete', 0, 1, '', 'semantic')",
            (raw_id,),
        )
        before = _census_binding(archive, raw_id)
        conn.execute("UPDATE raw_membership_census SET revision_authority='byte' WHERE raw_id=?", (raw_id,))
        after = _census_binding(archive, raw_id)
        assert before != after


def _session(
    *texts: str,
    session_id: str = "prepared-membership",
    attachment_bytes: bytes | None = None,
    precomputed_blob: tuple[str, int] | None = None,
) -> ParsedSession:
    if attachment_bytes is not None:
        attachments = [
            ParsedAttachment(
                provider_attachment_id="prepared-attachment",
                message_provider_id="m0",
                name="prepared.bin",
                mime_type="application/octet-stream",
                size_bytes=len(attachment_bytes),
                inline_bytes=attachment_bytes,
                precomputed_blob=precomputed_blob,
            )
        ]
    elif precomputed_blob is not None:
        attachments = [
            ParsedAttachment(
                provider_attachment_id="prepared-attachment",
                message_provider_id="m0",
                name="prepared.bin",
                mime_type="application/octet-stream",
                size_bytes=precomputed_blob[1],
                inline_bytes=None,
                precomputed_blob=precomputed_blob,
            )
        ]
    else:
        attachments = []
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        messages=[
            ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text=text)
            for index, text in enumerate(texts)
        ],
        attachments=attachments,
    )


def _parse_from(sessions_by_raw_id: dict[str, ParsedSession]) -> ParseFunction:
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


def _publish_census(archive: ArchiveStore, raw_id: str, parse: ParseFunction, *, at_ms: int) -> None:
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


@pytest.mark.asyncio
async def test_read_only_census_uses_canonical_retained_raw_parser(tmp_path: Path, monkeypatch) -> None:
    """The original retained worker reads and censuses bytes before Index publication."""
    from polylogue.storage.raw_authority import iter_parser_census_logical_keys

    payload = (
        b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m0",'
        b'"role":"user","content":[{"type":"input_text","text":"retained"}]}}\n'
    )

    def check(_unbound, carried, replacement, publish):
        assert carried.source_name is Provider.CODEX
        assert carried.provider_session_id == "prepared-membership"
        assert [message.text for message in carried.messages] == ["retained"]
        with closing(sqlite3.connect(tmp_path / "source.db")) as source:
            census = source.execute("SELECT status, logical_keys_json FROM raw_authority_parser_census").fetchall()
            assert len(census) == 1
            assert census[0][0] == "complete"
            assert tuple(iter_parser_census_logical_keys(census[0][1])) == ("codex-session:prepared-membership",)
            assert source.execute("SELECT parsed_at_ms, parse_error FROM raw_sessions").fetchall() == [(None, None)]
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert publish()
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT native_id FROM sessions").fetchall() == [("prepared-membership",)]

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, check, use_production_parser=True, raw_payload=payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", ["descriptor", "authority"])
async def test_canonical_preparation_rejects_changed_source_binding(tmp_path: Path, monkeypatch, binding) -> None:
    """The original writer refuses stale Source evidence before any Index effects."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    def check(_unbound, _carried, replacement, publish):
        raw_id = replacement.key

        def move_source():
            with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
                source = archive._ensure_source_conn()
                if binding == "descriptor":
                    original = source.execute(
                        "SELECT source_path FROM raw_sessions WHERE raw_id=?", (raw_id,)
                    ).fetchone()
                    assert original == ("prepared-membership.jsonl",)
                    source.execute(
                        "UPDATE raw_sessions SET source_path=? WHERE raw_id=?",
                        ("equal-count-new-descriptor.jsonl", raw_id),
                    )
                else:
                    original = source.execute(
                        "SELECT revision_authority FROM raw_sessions WHERE raw_id=?", (raw_id,)
                    ).fetchone()
                    assert original == (RawRevisionAuthority.BYTE_PROVEN.value,)
                    changed = RawRevisionAuthority.QUARANTINED.value
                    source.execute("UPDATE raw_sessions SET revision_authority=? WHERE raw_id=?", (changed, raw_id))
                    assert (
                        source.execute(
                            "SELECT revision_authority FROM raw_sessions WHERE raw_id=?", (raw_id,)
                        ).fetchone()
                        != original
                    )
                source.commit()

        admit_stage_write("fixture.raw.changed-source-binding", move_source)
        with pytest.raises(ReferenceSealStaleError):
            publish()
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
            assert index.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, check)


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


def test_membership_that_becomes_eligible_after_preparation_defers_publication(tmp_path: Path) -> None:
    """Revalidation rereads current membership, and request ownership still bounds it.

    Two facts at once: a request-owned raw whose complete census lands
    between preparation and publication must be seen by revalidation, and a
    complete, undecided member of the same logical key that the request never
    accepted must stay quarantined in both reads.

    Anti-vacuity: answer revalidation from the prepared cohort's own selector
    snapshot instead of rereading current membership and the first assertion
    goes green while a stale cohort publishes over evidence that moved; drop
    request ownership from the membership read and the unaccepted member is
    pulled into the selector by preparation itself.
    """
    bootstrap_archive_root(tmp_path)
    logical_source_key = "codex-session:prepared-membership"
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, second_raw, late_raw, unaccepted_raw = _write_raws(archive, 4)
        sessions = {
            first_raw: _session("one"),
            second_raw: _session("one", "two"),
            late_raw: _session("one", "two", "three"),
            unaccepted_raw: _session("one", "two", "three", "four"),
        }
        parse = _parse_from(sessions)
        _publish_census(archive, first_raw, parse, at_ms=1)
        _publish_census(archive, second_raw, parse, at_ms=2)
        # Complete and undecided on the same logical key, but never accepted by
        # this request: an unrelated quarantined candidate.
        _publish_census(archive, unaccepted_raw, parse, at_ms=3)

        accepted = (first_raw, second_raw, late_raw)
        stale = prepare_ingest_cohort(
            archive,
            logical_source_key=logical_source_key,
            accepted_raw_ids=accepted,
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=4,
        )
        # late_raw carries no census yet, so it holds no membership row to select.
        assert late_raw not in stale.selector_raw_ids
        assert unaccepted_raw not in stale.selector_raw_ids
        assert {first_raw, second_raw} <= set(stale.selector_raw_ids)

        # The accepted raw becomes an eligible member after preparation.
        _publish_census(archive, late_raw, parse, at_ms=5)

        result = publish_ingest_cohort(archive, stale)
        assert not result.published
        assert result.reprepare_required
        assert result.reason == "eligible membership selector changed"

        fresh = prepare_ingest_cohort(
            archive,
            logical_source_key=logical_source_key,
            accepted_raw_ids=accepted,
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=6,
        )
        assert late_raw in fresh.selector_raw_ids
        assert unaccepted_raw not in fresh.selector_raw_ids
        assert publish_ingest_cohort(archive, fresh).published


@pytest.mark.parametrize("retired", [False, True])
def test_only_explicit_census_retirement_selects_an_unaccepted_sibling(tmp_path: Path, retired: bool) -> None:
    """Default raw quarantine is not request authority; actual retirement is."""
    bootstrap_archive_root(tmp_path)
    key = "codex-session:prepared-membership"
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        accepted, sibling = _write_raws(archive, 2)
        parse = _parse_from({accepted: _session("one"), sibling: _session("other")})
        _publish_census(archive, accepted, parse, at_ms=1)
        _publish_census(archive, sibling, parse, at_ms=2)
        source = archive._ensure_source_conn()
        assert source.execute(
            "SELECT revision_authority FROM raw_sessions WHERE raw_id = ?", (sibling,)
        ).fetchone() == ("quarantined",)
        if retired:
            # This is the typed producer's durable retirement result: the
            # complete census keeps the parsed identity after byte governance
            # relinquishes its raw logical-source key.
            source.execute(
                "UPDATE raw_membership_census SET revision_authority = 'quarantined' WHERE raw_id = ?", (sibling,)
            )
            source.execute("UPDATE raw_sessions SET logical_source_key = NULL WHERE raw_id = ?", (sibling,))
            source.commit()
        prepared = prepare_ingest_cohort(
            archive,
            logical_source_key=key,
            accepted_raw_ids=(accepted,),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=3,
        )
        assert accepted in prepared.selector_raw_ids
        assert (sibling in prepared.selector_raw_ids) is retired


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
        assert prepared.prepared_artifact is not None
        prepared_claim = next(prepared.prepared_artifact.iter_attachment_claims())[2]
        stale_staged_path = prepared_claim.prepared_path
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
        assert fresh.prepared_artifact is not None
        fresh_claim = next(fresh.prepared_artifact.iter_attachment_claims())[2]
        fresh_staged_path = fresh_claim.prepared_path
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


async def _run_precomputed_raw_case(root, monkeypatch, assertion, *, before_publication=None):
    """Use a captured CAS attachment and the original retained Raw publication."""
    precomputed = None

    def setup():
        nonlocal precomputed
        precomputed = BlobStore(root / "blob").write_from_bytes(b"precomputed attachment")

    def session():
        assert precomputed is not None
        return _session("one", precomputed_blob=precomputed)

    captures = []

    def observe(artifact):
        from contextlib import closing

        assert precomputed is not None
        hash_hex, _size = precomputed
        with closing(artifact.iter_attachment_claims()) as claims:
            claim = next(claims)[2]
            assert next(claims, None) is None
        assert claim.receipt.blob_hash == hash_hex
        assert claim.prepared_path.is_file()
        with sqlite3.connect(root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone() == (0,)
        # The original admitted Source publisher consumes this private file
        # before final Index publication. Exercise conservation/refusal before
        # that actual publication, then check its durable receipt afterwards.
        if not captures and before_publication is not None:
            before_publication(hash_hex, claim)
        captures.append(claim)

    def check(_unbound, _carried, _replacement, publish):
        assert precomputed is not None
        assert captures
        assertion(precomputed[0], captures[-1], publish)

    await _run_original_raw_carrier_case(
        root,
        monkeypatch,
        check,
        session_factory=session,
        seed_setup=setup,
        artifact_observer=observe,
    )


def _assert_precomputed_raw_refs(root, hash_hex):
    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT lower(hex(blob_hash)) FROM attachments").fetchone() == (hash_hex,)
    with sqlite3.connect(root / "source.db") as conn:
        blob = bytes.fromhex(hash_hex)
        assert conn.execute("SELECT ref_type FROM blob_refs WHERE blob_hash=?", (blob,)).fetchall() == [("attachment",)]
        assert conn.execute(
            "SELECT COUNT(*) FROM blob_publication_reservations WHERE blob_hash=?",
            (blob,),
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_precomputed_attachment_is_reserved_and_referenced_by_the_writer(tmp_path: Path, monkeypatch) -> None:
    """The actual Raw writer references its captured claim and consumes its reservation."""

    def check(hash_hex, claim, publish):
        assert publish()
        assert not claim.prepared_path.exists()
        _assert_precomputed_raw_refs(tmp_path, hash_hex)

    await _run_precomputed_raw_case(tmp_path, monkeypatch, check)


@pytest.mark.asyncio
async def test_precomputed_attachment_captured_bytes_survive_public_blob_collection(
    tmp_path: Path, monkeypatch
) -> None:
    """Collecting the public copy cannot erase the original prepared private capture."""

    def collect(hash_hex, claim):
        blob_path = BlobStore(tmp_path / "blob").blob_path(hash_hex)
        blob_path.unlink()
        assert claim.prepared_path.read_bytes() == b"precomputed attachment"

    def check(hash_hex, claim, publish):
        assert publish()
        assert not claim.prepared_path.exists()
        _assert_precomputed_raw_refs(tmp_path, hash_hex)
        assert BlobStore(tmp_path / "blob").blob_path(hash_hex).read_bytes() == b"precomputed attachment"

    await _run_precomputed_raw_case(tmp_path, monkeypatch, check, before_publication=collect)


@pytest.mark.asyncio
async def test_missing_sealed_attachment_capture_refuses_before_index_publication(tmp_path: Path, monkeypatch) -> None:
    """Missing original private capture refuses without publishing an Index attachment."""

    from polylogue.core.storage_faults import ArchiveStorageFaultError, StorageFaultKind

    def remove_capture(_hash_hex, claim):
        claim.prepared_path.unlink()

    def unexpected_publication(*_args):
        raise AssertionError("a missing private capture must refuse before destination publication")

    with pytest.raises(ArchiveStorageFaultError) as refusal:
        await _run_precomputed_raw_case(
            tmp_path, monkeypatch, unexpected_publication, before_publication=remove_capture
        )
    assert refusal.value.kind is StorageFaultKind.EVICTED
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
async def test_canonical_raising_parser_publishes_only_failed_source_census(tmp_path: Path, monkeypatch) -> None:
    """A genuine parser exception yields a publishable Source receipt, not Index work."""
    calls = []

    def raising_parser(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("synthetic retained parser failure")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_stream", raising_parser)

    def check(raw_id, receipts):
        assert len(calls) == 1
        assert len(receipts) == 1
        kind, result = receipts[0]
        assert kind == "census"
        assert result.scanned == 1
        assert result.quarantined == 1
        with closing(sqlite3.connect(tmp_path / "source.db")) as source:
            status, count, detail = source.execute(
                "SELECT status, member_count, detail FROM raw_membership_census WHERE raw_id=?", (raw_id,)
            ).fetchone()
            assert status == "failed"
            assert count == 0
            assert "synthetic retained parser failure" in detail
            assert source.execute(
                "SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
            ).fetchone() == ("failed",)
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    await _run_original_raw_carrier_case(
        tmp_path,
        monkeypatch,
        None,
        use_production_parser=True,
        raw_payload=(
            b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n'
            b'{"type":"response_item","payload":{"type":"message","id":"m0",'
            b'"role":"user","content":[{"type":"input_text","text":"retained"}]}}\n'
        ),
        source_receipt_assertion=check,
    )


def test_raising_production_parser_records_a_failed_census_instead_of_escaping(tmp_path: Path) -> None:
    """One unparseable retained raw is a FAILED census, not a refused generation.

    Anti-vacuity: the FAILED arm of prepare_raw_census models parse failure as
    ``parsed is None``, but the sole production callable is typed
    ``-> list[ParsedSession]`` and raises. Remove the census-boundary catch and
    this test raises instead of returning, which is what fenced the whole source
    generation at operations/daemon_ingest.py. Every other test in this file
    uses a ``_parse_from`` double that can only return, so none reaches here.
    """
    bootstrap_archive_root(tmp_path)

    def raising_parse(_archive: ArchiveStore, _raw_id: str) -> list[ParsedSession]:
        raise ValueError("synthetic unparseable retained payload")

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"census-failure"}}\n',
            source_path="census-failure.jsonl",
            acquired_at_ms=1,
        )

    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        prepared = prepare_raw_census(
            reader,
            raw_id,
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=raising_parse,
            censused_at_ms=2,
        )

    assert prepared.status.value == "failed"
    assert prepared.sessions is None
    assert "synthetic unparseable retained payload" in prepared.detail

    # The recorded disposition is publishable, so the generation continues.
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        publication = publish_raw_census(archive, prepared)
    assert publication.published is True


def test_transient_sqlite_contention_is_not_recorded_as_a_parse_failure(tmp_path: Path) -> None:
    """Contention must stay retryable rather than quarantining the raw.

    Anti-vacuity: catching bare ``Exception`` at the census boundary turns a
    SQLITE_BUSY during retained-material reads into a permanent FAILED census;
    this test then returns a census instead of propagating.
    """
    import sqlite3

    bootstrap_archive_root(tmp_path)

    def busy_parse(_archive: ArchiveStore, _raw_id: str) -> list[ParsedSession]:
        exc = sqlite3.OperationalError("database is locked")
        exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
        exc.sqlite_errorname = "SQLITE_BUSY"
        raise exc

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"census-busy"}}\n',
            source_path="census-busy.jsonl",
            acquired_at_ms=1,
        )

    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        with pytest.raises(sqlite3.OperationalError):
            prepare_raw_census(
                reader,
                raw_id,
                parser_fingerprint="prepared-test-parser",
                parse_retained_raw=busy_parse,
                censused_at_ms=2,
            )


def test_unparseable_selector_member_is_a_typed_per_key_refusal(tmp_path: Path) -> None:
    """A selector member that no longer parses refuses its key, not the generation.

    Anti-vacuity: ``prepare_ingest_cohort`` reaches ``_session_for_key`` for
    every selector member, including the unconditionally-admitted
    ``raw_revision_head_raw_id``. Before the fix that helper raised a bare
    ``RuntimeError("... no longer parses uniquely")``, which escaped to
    ``operations/daemon_ingest.py``'s ``except Exception: fence("refused")``
    and cost every other logical key in the source generation. Delete the
    ``CohortMembershipRefusalError`` raises (or the parse-boundary catch) and
    this test fails: ``pytest.raises`` sees a plain ``RuntimeError`` /
    ``ValueError`` that no caller can distinguish from a genuine
    generation-level fault, and the ``reason``/``raw_id`` attributes the daemon
    counts are absent.

    It also pins message honesty: the zero-match case never established "no
    longer parses uniquely", so the refusal must not claim it did.
    """
    bootstrap_archive_root(tmp_path)
    key = "codex-session:prepared-membership"
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, second_raw = _write_raws(archive, 2)
        parse = _parse_from({first_raw: _session("one"), second_raw: _session("one", "two")})
        _publish_census(archive, first_raw, parse, at_ms=1)
        _publish_census(archive, second_raw, parse, at_ms=2)
        published = prepare_ingest_cohort(
            archive,
            logical_source_key=key,
            accepted_raw_ids=(first_raw, second_raw),
            parser_fingerprint="prepared-test-parser",
            parse_retained_raw=parse,
            acquired_at_ms=3,
        )
        assert publish_ingest_cohort(archive, published).published

        # The accepted head is now in the selector. Re-preparing with a parser
        # that raises for it is exactly the production shape: the real
        # ``parse_retained_raw_sessions`` raises rather than returning None.
        def raising_for_head(archive_: ArchiveStore, raw_id: str) -> list[ParsedSession]:
            if raw_id == second_raw:
                raise ValueError("synthetic unparseable retained payload")
            return parse(archive_, raw_id)

        with pytest.raises(CohortMembershipRefusalError) as raised:
            prepare_ingest_cohort(
                archive,
                logical_source_key=key,
                accepted_raw_ids=(first_raw, second_raw),
                parser_fingerprint="prepared-test-parser",
                parse_retained_raw=raising_for_head,
                acquired_at_ms=4,
            )
        assert raised.value.logical_source_key == key
        assert raised.value.raw_id == second_raw
        assert "synthetic unparseable retained payload" in raised.value.reason

        # A member that parses but contributes nothing to this key is the same
        # typed refusal, with a reason that does not overclaim.
        def foreign_for_head(archive_: ArchiveStore, raw_id: str) -> list[ParsedSession]:
            if raw_id == second_raw:
                return [_session("elsewhere", session_id="a-different-logical-session")]
            return parse(archive_, raw_id)

        with pytest.raises(CohortMembershipRefusalError) as empty:
            prepare_ingest_cohort(
                archive,
                logical_source_key=key,
                accepted_raw_ids=(first_raw, second_raw),
                parser_fingerprint="prepared-test-parser",
                parse_retained_raw=foreign_for_head,
                acquired_at_ms=5,
            )
        assert "none for this logical key" in empty.value.reason
        assert "no longer parses uniquely" not in str(empty.value)


def _adversarial_session(*, session_id: str = "prepared-membership") -> ParsedSession:
    """One session covering every canonical-byte axis the digest must survive.

    Numeric exponents and floats, NFC/NFD-distinguishable text, non-ASCII
    object keys, nested tool_input mappings, a session event payload, an
    attachment, and two byte-identical messages that can only be separated by
    their content occurrence. A carrier derived from anything other than the
    declared payload partition diverges on one of these.
    """
    repeated = ParsedMessage(provider_message_id="", role=Role.USER, text="repeat")
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id,
        title="café ： \U0001f9ea",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:05:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="café — naïve",
                timestamp="2026-01-01T00:00:00Z",
            ),
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                text=None,
                timestamp="2026-01-01T00:01:00Z",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id="t1",
                        tool_input={
                            "exponent": 1e-7,
                            "big": 1e22,
                            "negative_zero": -0.0,
                            "integral_float": 2.0,
                            "ékey": {"nested": [1, 2.5, True, None, "ż"]},
                        },
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="t1",
                        text="ok",
                        outcome_unknown_reason="not_reported",
                    ),
                ],
            ),
            repeated,
            repeated.model_copy(),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m0",
                name="für.txt",
                mime_type="text/plain",
                size_bytes=3,
            )
        ],
        session_events=[
            ParsedSessionEvent(
                event_type="turn_context",
                timestamp="2026-01-01T00:00:30Z",
                payload={"ü": {"depth": [{"x": 1e-7}]}},
            )
        ],
    )


async def _run_original_raw_carrier_case(
    root,
    monkeypatch,
    assertion,
    *,
    refuse_rehash=False,
    session_factory=None,
    seed_setup=None,
    artifact_observer=None,
    use_production_parser=False,
    raw_payload=None,
    source_receipt_assertion=None,
):
    """Keep original Source evidence and the canonical paged carrier in one creator."""
    from contextlib import closing

    from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    parsed = None

    def seed():
        bootstrap_archive_root(root)
        if seed_setup is not None:
            seed_setup()
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=(
                    b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n'
                    if raw_payload is None
                    else raw_payload
                ),
                source_path="prepared-membership.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, seed)
    unbound = _adversarial_session() if session_factory is None else session_factory()

    def worker(reader, selected_raw_id, *, directory):
        nonlocal parsed
        if parsed is None:
            parsed = unbound.model_copy(update={"content_hash": session_content_hash(unbound)})
        assert selected_raw_id == raw_id
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        # The parser-output fixture uses the original admitted worker seam;
        # it does not supply another Source snapshot or publication permit.
        if refuse_rehash:

            def refuse(*_args, **_kwargs):
                raise AssertionError("canonical Raw preparation must carry the parse-worker digest")

            monkeypatch.setattr("polylogue.pipeline.ids.session_content_hash", refuse)
        artifact = PreparedJsonl.from_sessions(
            [parsed],
            blob_hash=blob_hash,
            artifact_directory=directory,
            publication_publisher=ArchiveBlobPublisher(root / "source.db", root / "blob"),
            publication_source_read=reader,
        )
        if artifact_observer is not None:
            artifact_observer(artifact)
        return artifact

    if not use_production_parser:
        monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_retained_jsonl_artifact", worker)
    async with prepared_live_convergence_owner(root) as owner:

        def exercise():
            from polylogue.core.stage_admission import admit_stage_write

            adapter = make_raw_observation_derivation(root, compute_adapter=owner._compute_adapter)
            frame = raw_observation_frame(root)
            replacement = adapter.compute(frame, raw_id)
            completed_preparatory_phases = set()
            try:
                # The replacement first owns canonical Source receipts, not
                # destination writes. Preserve that separation and then read
                # the real committed census/classification in a fresh frame.
                while not replacement.prepared_writes:
                    phase = (replacement.needs_source_census, replacement.needs_source_classification)
                    assert any(phase), "Raw returned neither a preparatory receipt nor a destination write"
                    assert phase not in completed_preparatory_phases, "Raw preparatory receipt did not advance"
                    completed_preparatory_phases.add(phase)
                    from polylogue.core.stage_admission import admit_stage_write

                    receipts = []
                    expected_receipt = (
                        replacement.prepared_source_census
                        if replacement.needs_source_census
                        else replacement.prepared_source_classification
                    )
                    assert expected_receipt is not None
                    destination_changed = admit_stage_write(
                        "fixture.raw.preparatory-receipt",
                        lambda frame=frame, replacement=replacement, receipts=receipts: adapter.publish(
                            frame, replacement, phase_receipt=lambda kind, receipt: receipts.append((kind, receipt))
                        ),
                    )
                    assert destination_changed is False
                    assert receipts == [
                        ("census" if replacement.needs_source_census else "classification", expected_receipt.result)
                    ]
                    if source_receipt_assertion is not None:
                        source_receipt_assertion(raw_id, receipts)
                        return
                    replacement.close()
                    frame = raw_observation_frame(root)
                    replacement = adapter.compute(frame, raw_id)
                retained = replacement.prepared_inputs[raw_id]
                assert retained.prepared_artifact is not None
                with closing(retained.prepared_artifact.iter_sessions()) as sessions:
                    carried = next(sessions)
                    assert next(sessions, None) is None
                assertion(
                    unbound,
                    carried,
                    replacement,
                    lambda: admit_stage_write(
                        "fixture.raw.final-publication",
                        lambda frame=frame, replacement=replacement: adapter.publish(frame, replacement),
                    ),
                )
            finally:
                replacement.close()

        await owner.run_convergence_sync("fixture.raw.original-digest", exercise)


@pytest.mark.asyncio
async def test_cohort_carries_byte_identical_session_digest(tmp_path: Path, monkeypatch) -> None:
    """The actual retained Raw carrier preserves every original digest axis."""

    def assert_carrier(unbound, carried, replacement, _publish):
        expected_hex = str(session_content_hash(unbound))
        assert bound_session_content_hash(unbound) is None
        assert bound_session_content_hash(carried) == expected_hex
        excluded = {"messages", "attachments", "session_events", "content_hash"}
        assert carried.model_dump(mode="json", exclude=excluded) == unbound.model_dump(mode="json", exclude=excluded)
        assert [message.model_dump(mode="json") for message in carried.messages] == [
            message.model_dump(mode="json") for message in unbound.messages
        ]
        assert [event.model_dump(mode="json") for event in carried.session_events] == [
            event.model_dump(mode="json") for event in unbound.session_events
        ]
        assert [attachment.model_dump(mode="json") for attachment in carried.attachments] == [
            attachment.model_dump(mode="json") for attachment in unbound.attachments
        ]
        writes = tuple(replacement.prepared_writes.values())
        assert len(writes) == 1
        rows = writes[0].rows
        assert writes[0].input_content_hash == bytes.fromhex(expected_hex)
        assert rows.session_content_hash == bytes.fromhex(expected_hex)
        assert len(rows.content_identities) == len(unbound.messages)
        digests = [digest for digest, _occurrence in rows.content_identities]
        occurrences = [occurrence for _digest, occurrence in rows.content_identities]
        assert digests[2] == digests[3]
        assert occurrences[2:4] == [0, 1]

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, assert_carrier)


@pytest.mark.asyncio
async def test_cohort_does_not_rehash_projected_session(tmp_path: Path, monkeypatch) -> None:
    """After the worker binds the digest, the genuine Raw lowering must reuse it."""
    expected_hex = str(session_content_hash(_adversarial_session()))

    def assert_carrier(_unbound, carried, replacement, _publish):
        assert bound_session_content_hash(carried) == expected_hex
        writes = tuple(replacement.prepared_writes.values())
        assert len(writes) == 1
        assert writes[0].rows.session_content_hash == bytes.fromhex(expected_hex)

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, assert_carrier, refuse_rehash=True)


@pytest.mark.asyncio
async def test_canonical_accepted_head_foreign_parse_is_a_typed_per_key_refusal(tmp_path: Path, monkeypatch) -> None:
    """The actual accepted Raw head refuses a valid parse missing its selected key."""
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed():
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n',
                source_path="prepared-membership.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)
    session = _session("one")

    def worker(reader, selected_raw_id, *, directory):
        assert selected_raw_id == raw_id
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        parsed = session.model_copy(update={"content_hash": session_content_hash(session)})
        return PreparedJsonl.from_sessions(
            [parsed],
            blob_hash=blob_hash,
            artifact_directory=directory,
            publication_publisher=ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob"),
            publication_source_read=reader,
        )

    monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_retained_jsonl_artifact", worker)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        results = await owner.ingest_retained_raw_ids((raw_id,))
        assert sum(len(result.written_session_ids) for result in results) == 1
        with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
            assert reader.index_connection is not None
            row = reader.index_connection.execute("SELECT raw_id FROM sessions").fetchone()
            assert row is not None
            assert row[0] == raw_id
        session = _session("elsewhere", session_id="a-different-logical-session")
        with pytest.raises(CohortMembershipRefusalError) as empty:
            await owner.ingest_retained_raw_ids((raw_id,))
        assert empty.value.logical_source_key == "codex-session:prepared-membership"
        assert empty.value.raw_id == raw_id
        assert "none for this logical key" in empty.value.reason
        assert "no longer parses uniquely" not in str(empty.value)


@pytest.mark.asyncio
async def test_canonical_retained_sqlite_busy_stays_retryable(tmp_path: Path, monkeypatch) -> None:
    """SQLITE_BUSY through the actual worker is never a terminal parse census."""
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed():
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"census-busy"}}\n',
                source_path="census-busy.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)
    failure = sqlite3.OperationalError("database is locked")
    failure.sqlite_errorcode = sqlite3.SQLITE_BUSY
    failure.sqlite_errorname = "SQLITE_BUSY"
    calls = []

    def busy_parse(*args, **kwargs):
        calls.append((args, kwargs))
        raise failure

    monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_jsonl_blob", busy_parse)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        with pytest.raises(RetainedPreparationRetryableError) as retryable:
            await owner.ingest_retained_raw_ids((raw_id,))
        assert retryable.value.__cause__ is failure
    assert len(calls) == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as source, source:
        assert (
            source.execute("SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)).fetchone()
            is None
        )
        assert source.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone() == (1,)
    with closing(sqlite3.connect(tmp_path / "index.db")) as index, index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.asyncio
async def test_original_raw_selects_each_sessions_own_attachment_claim(tmp_path: Path, monkeypatch) -> None:
    """One actual Raw never reuses its first session's attachment lookup."""
    from polylogue.pipeline.ids import session_id as make_session_id
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    sessions = []
    expected = {}

    def seed():
        bootstrap_archive_root(tmp_path)
        store = BlobStore(tmp_path / "blob")
        for name in ("first", "second"):
            blob = store.write_from_bytes(name.encode())
            session = _session(name, session_id=f"prepared-{name}", precomputed_blob=blob)
            session = session.model_copy(
                update={
                    "messages": [ParsedMessage(provider_message_id=f"{name}-message", role=Role.USER, text=name)],
                    "attachments": [
                        session.attachments[0].model_copy(
                            update={
                                "provider_attachment_id": f"{name}-attachment",
                                "message_provider_id": f"{name}-message",
                            }
                        )
                    ],
                }
            )
            sessions.append(session)
            expected[str(make_session_id(session.source_name, session.provider_session_id))] = blob[0]
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"prepared-multiple"}}\n',
                source_path="prepared-multiple.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)

    def worker(reader, selected_raw_id, *, directory):
        assert selected_raw_id == raw_id
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        return PreparedJsonl.from_sessions(
            [session.model_copy(update={"content_hash": session_content_hash(session)}) for session in sessions],
            blob_hash=blob_hash,
            artifact_directory=directory,
            publication_publisher=ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob"),
            publication_source_read=reader,
        )

    monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_retained_jsonl_artifact", worker)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        results = await owner.ingest_retained_raw_ids((raw_id,))
        assert sum(len(result.written_session_ids) for result in results) == 2
    with closing(sqlite3.connect(tmp_path / "index.db")) as index, index:
        rows = index.execute(
            "SELECT r.session_id,lower(hex(a.blob_hash)) FROM attachment_refs r "
            "JOIN attachments a ON a.attachment_id=r.attachment_id"
        ).fetchall()
        assert len(rows) == 2
        assert dict(rows) == expected
    with closing(sqlite3.connect(tmp_path / "source.db")) as source, source:
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
        assert {
            row[0] for row in source.execute("SELECT lower(hex(blob_hash)) FROM blob_refs WHERE ref_type='attachment'")
        } == set(expected.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("has_file_id", [True, False])
async def test_canonical_retained_attachment_claims_keep_equal_content_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_file_id: bool
) -> None:
    """Raw publication must retain both original provider coordinates for equal bytes."""
    precomputed: tuple[str, int] | None = None

    def seed() -> None:
        nonlocal precomputed
        precomputed = BlobStore(tmp_path / "blob").write_from_bytes(b"equal attachment bytes")

    def session() -> ParsedSession:
        assert precomputed is not None
        parsed = _session("neutral", precomputed_blob=precomputed)
        original = parsed.attachments[0]
        parsed.attachments = [
            original.model_copy(
                update={
                    "provider_attachment_id": f"reference-{suffix}",
                    "provider_file_id": f"file-{suffix}" if has_file_id else None,
                }
            )
            for suffix in ("a", "b")
        ]
        return parsed

    def check(_unbound, _carried, replacement, publish) -> None:
        assert precomputed is not None
        assert publish()
        blob_hash = bytes.fromhex(precomputed[0])
        expected = {
            (replacement.key, f"attachment:file-{suffix}" if has_file_id else f"attachment-ref:reference-{suffix}")
            for suffix in ("a", "b")
        }
        with closing(sqlite3.connect(tmp_path / "source.db")) as source:
            assert (
                set(
                    source.execute(
                        "SELECT ref_id,source_path FROM blob_refs WHERE ref_type='attachment' AND blob_hash=?",
                        (blob_hash,),
                    )
                )
                == expected
            )
            assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash=?", (blob_hash,)).fetchone() == (2,)

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, check, session_factory=session, seed_setup=seed)
