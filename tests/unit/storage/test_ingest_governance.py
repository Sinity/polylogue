"""Prepared membership governance keeps parse work outside the writer hold."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    raw_authority_parser_fingerprint,
)
from polylogue.core.enums import BlockType, Provider
from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError, RawFailureEvidenceKind
from polylogue.pipeline.ids import bound_session_content_hash, session_content_hash
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import (
    PreparedRevisionReplayResult,
    RetainedSessionRead,
    RevisionCensusResult,
)
from polylogue.storage.blob_publication import PreparedBlobPublicationClaim
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers import revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

_Publish: TypeAlias = Callable[[], bool]
_CarrierAssertion: TypeAlias = Callable[[ParsedSession, ParsedSession, RawObservationReplacement, _Publish], None]
_PhaseReceipt: TypeAlias = tuple[
    Literal["census", "classification", "replay"], RevisionCensusResult | PreparedRevisionReplayResult
]


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


def _write_raws(archive: ArchiveStore, count: int) -> tuple[str, ...]:
    return tuple(
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=f'{{"raw":{index}}}'.encode(),
            source_path=f"prepared-{index}.jsonl",
            canonical_source_path=f"prepared-{index}.jsonl",
            acquired_at_ms=index + 1,
        )
        for index in range(count)
    )


@pytest.mark.asyncio
async def test_read_only_census_uses_canonical_retained_raw_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original retained worker reads and censuses bytes before Index publication."""
    from polylogue.storage.raw_authority import iter_parser_census_logical_keys

    payload = (
        b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m0",'
        b'"role":"user","content":[{"type":"input_text","text":"retained"}]}}\n'
    )

    def check(
        _unbound: ParsedSession,
        carried: ParsedSession,
        replacement: RawObservationReplacement,
        publish: _Publish,
    ) -> None:
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
async def test_canonical_preparation_rejects_changed_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    """The original writer refuses stale Source evidence before any Index effects."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    def check(
        _unbound: ParsedSession,
        _carried: ParsedSession,
        replacement: RawObservationReplacement,
        publish: _Publish,
    ) -> None:
        raw_id = replacement.key

        def move_source() -> None:
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
                    # An unbound admitted Raw starts quarantined; any authority
                    # move is a changed Source binding.
                    assert original == (RawRevisionAuthority.QUARANTINED.value,)
                    changed = RawRevisionAuthority.BYTE_PROVEN.value
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


async def _run_precomputed_raw_case(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    assertion: Callable[[str, PreparedBlobPublicationClaim, _Publish], None],
    *,
    before_publication: Callable[[str, PreparedBlobPublicationClaim], None] | None = None,
) -> None:
    """Use a captured CAS attachment and the original retained Raw publication."""
    precomputed: tuple[str, int] | None = None

    def setup() -> None:
        nonlocal precomputed
        precomputed = BlobStore(root / "blob").write_from_bytes(b"precomputed attachment")

    def session() -> ParsedSession:
        assert precomputed is not None
        return _session("one", precomputed_blob=precomputed)

    captures: list[PreparedBlobPublicationClaim] = []

    def observe(artifact: PreparedJsonl) -> None:
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

    def check(
        _unbound: ParsedSession,
        _carried: ParsedSession,
        _replacement: RawObservationReplacement,
        publish: _Publish,
    ) -> None:
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


def _assert_precomputed_raw_refs(root: Path, hash_hex: str) -> None:
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
async def test_precomputed_attachment_is_reserved_and_referenced_by_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual Raw writer references its captured claim and consumes its reservation."""

    def check(hash_hex: str, claim: PreparedBlobPublicationClaim, publish: _Publish) -> None:
        assert publish()
        assert not claim.prepared_path.exists()
        _assert_precomputed_raw_refs(tmp_path, hash_hex)

    await _run_precomputed_raw_case(tmp_path, monkeypatch, check)


@pytest.mark.asyncio
async def test_precomputed_attachment_captured_bytes_survive_public_blob_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collecting the public copy cannot erase the original prepared private capture."""

    def collect(hash_hex: str, claim: PreparedBlobPublicationClaim) -> None:
        blob_path = BlobStore(tmp_path / "blob").blob_path(hash_hex)
        blob_path.unlink()
        assert claim.prepared_path.read_bytes() == b"precomputed attachment"

    def check(hash_hex: str, claim: PreparedBlobPublicationClaim, publish: _Publish) -> None:
        assert publish()
        assert not claim.prepared_path.exists()
        _assert_precomputed_raw_refs(tmp_path, hash_hex)
        assert BlobStore(tmp_path / "blob").blob_path(hash_hex).read_bytes() == b"precomputed attachment"

    await _run_precomputed_raw_case(tmp_path, monkeypatch, check, before_publication=collect)


@pytest.mark.asyncio
async def test_missing_sealed_attachment_capture_refuses_before_index_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing original private capture refuses without publishing an Index attachment."""

    from polylogue.core.storage_faults import ArchiveStorageFaultError, StorageFaultKind

    def remove_capture(_hash_hex: str, claim: PreparedBlobPublicationClaim) -> None:
        claim.prepared_path.unlink()

    def unexpected_publication(*_args: object) -> None:
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
async def test_canonical_raising_parser_settles_terminal_source_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine parser exception yields a settled Source receipt, not Index work.

    A deterministic parser failure on unchanged bytes is the parser's settled
    answer (bcaec252a4): a non-session census with typed terminal evidence and
    a complete parser census, never a ``failed`` census re-run every pass.
    """
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def raising_parser(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise ValueError("synthetic retained parser failure")

    monkeypatch.setattr("polylogue.sources.prepared_jsonl.iter_parsed_stream", raising_parser)

    def check(raw_id: str, receipts: list[_PhaseReceipt]) -> None:
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
            assert status == "non_session"
            assert count == 0
            assert "synthetic retained parser failure" in detail
            assert source.execute(
                "SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
            ).fetchone() == ("complete",)
            assert source.execute(
                "SELECT artifact_kind FROM raw_artifacts WHERE raw_id=? AND artifact_id LIKE 'raw-failure:%'",
                (raw_id,),
            ).fetchone() == (RawFailureEvidenceKind.TERMINAL_UNSUPPORTED_SHAPE.value,)
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
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    assertion: _CarrierAssertion | None,
    *,
    refuse_rehash: bool = False,
    session_factory: Callable[[], ParsedSession] | None = None,
    seed_setup: Callable[[], None] | None = None,
    artifact_observer: Callable[[PreparedJsonl], None] | None = None,
    use_production_parser: bool = False,
    raw_payload: bytes | None = None,
    source_receipt_assertion: Callable[[str, list[_PhaseReceipt]], None] | None = None,
) -> None:
    """Keep original Source evidence and the canonical paged carrier in one creator."""
    from contextlib import closing

    from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    parsed: ParsedSession | None = None

    def seed() -> str:
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
                canonical_source_path="prepared-membership.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, seed)
    unbound = _adversarial_session() if session_factory is None else session_factory()

    def worker(reader: RetainedSessionRead, selected_raw_id: str, *, directory: Path) -> PreparedJsonl:
        nonlocal parsed
        if parsed is None:
            parsed = unbound.model_copy(update={"content_hash": session_content_hash(unbound)})
        assert selected_raw_id == raw_id
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        # The parser-output fixture uses the original admitted worker seam;
        # it does not supply another Source snapshot or publication permit.
        if refuse_rehash:

            def refuse(*_args: object, **_kwargs: object) -> str:
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

        def exercise() -> None:
            from polylogue.core.stage_admission import admit_stage_write

            adapter = make_raw_observation_derivation(root, compute_adapter=owner._compute_adapter)
            frame = raw_observation_frame(root)
            replacement = adapter.compute(frame, raw_id)
            completed_preparatory_phases: set[tuple[bool, bool]] = set()
            try:
                # The replacement first owns canonical Source receipts, not
                # destination writes. Preserve that separation and then read
                # the real committed census/classification in a fresh frame.
                while not replacement.prepared_writes:
                    phase = (replacement.needs_source_census, replacement.needs_source_classification)
                    if not any(phase):
                        # Single-pass convergence: compute committed its Source
                        # phases in place and publication only reports them.
                        committed = list(replacement.committed_phase_receipts)
                        assert committed, "Raw returned neither a preparatory receipt nor a destination write"
                        assert source_receipt_assertion is not None, "an in-place Source receipt has a receipt check"
                        reported: list[_PhaseReceipt] = []

                        def report(
                            kind: Literal["census", "classification", "replay"],
                            receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                            reported: list[_PhaseReceipt] = reported,
                        ) -> None:
                            reported.append((kind, receipt))

                        admit_stage_write(
                            "fixture.raw.committed-receipt",
                            partial(adapter.publish, frame, replacement, phase_receipt=report),
                        )
                        assert reported == committed
                        source_receipt_assertion(raw_id, reported)
                        return
                    assert phase not in completed_preparatory_phases, "Raw preparatory receipt did not advance"
                    completed_preparatory_phases.add(phase)
                    from polylogue.core.stage_admission import admit_stage_write

                    receipts: list[_PhaseReceipt] = []

                    def record_receipt(
                        kind: Literal["census", "classification", "replay"],
                        receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                        receipts: list[_PhaseReceipt] = receipts,
                    ) -> None:
                        receipts.append((kind, receipt))

                    expected_receipt = (
                        replacement.prepared_source_census
                        if replacement.needs_source_census
                        else replacement.prepared_source_classification
                    )
                    assert expected_receipt is not None
                    destination_changed = admit_stage_write(
                        "fixture.raw.preparatory-receipt",
                        partial(adapter.publish, frame, replacement, phase_receipt=record_receipt),
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
                assert replacement.prepared_inputs is not None
                retained = replacement.prepared_inputs[raw_id]
                assert retained.prepared_artifact is not None
                with closing(retained.prepared_artifact.iter_sessions()) as sessions:
                    carried = next(sessions)
                    assert next(sessions, None) is None
                assert assertion is not None, "a carrier case without a Source receipt check asserts its carrier"
                assertion(
                    unbound,
                    carried,
                    replacement,
                    partial(
                        admit_stage_write,
                        "fixture.raw.final-publication",
                        partial(adapter.publish, frame, replacement),
                    ),
                )
            finally:
                replacement.close()

        await owner.run_convergence_sync("fixture.raw.original-digest", exercise)


@pytest.mark.asyncio
async def test_cohort_carries_byte_identical_session_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual retained Raw carrier preserves every original digest axis."""

    def assert_carrier(
        unbound: ParsedSession,
        carried: ParsedSession,
        replacement: RawObservationReplacement,
        _publish: _Publish,
    ) -> None:
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
        assert replacement.prepared_writes is not None
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
async def test_cohort_does_not_rehash_projected_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the worker binds the digest, the genuine Raw lowering must reuse it."""
    expected_hex = str(session_content_hash(_adversarial_session()))

    def assert_carrier(
        _unbound: ParsedSession,
        carried: ParsedSession,
        replacement: RawObservationReplacement,
        _publish: _Publish,
    ) -> None:
        assert bound_session_content_hash(carried) == expected_hex
        assert replacement.prepared_writes is not None
        writes = tuple(replacement.prepared_writes.values())
        assert len(writes) == 1
        assert writes[0].rows.session_content_hash == bytes.fromhex(expected_hex)

    await _run_original_raw_carrier_case(tmp_path, monkeypatch, assert_carrier, refuse_rehash=True)


@pytest.mark.asyncio
async def test_canonical_accepted_head_foreign_parse_is_a_typed_per_key_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual accepted Raw head refuses a valid parse missing its selected key."""
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"prepared-membership"}}\n',
                source_path="prepared-membership.jsonl",
                canonical_source_path="prepared-membership.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)
    session = _session("one")

    def worker(reader: RetainedSessionRead, selected_raw_id: str, *, directory: Path) -> PreparedJsonl:
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
        results = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(result.written_session_ids) for result in results) == 1
        with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
            assert reader.index_connection is not None
            row = reader.index_connection.execute("SELECT raw_id FROM sessions").fetchone()
            assert row is not None
            assert row[0] == raw_id
        session = _session("elsewhere", session_id="a-different-logical-session")
        with pytest.raises(CohortMembershipRefusalError) as empty:
            (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert empty.value.logical_source_key == "codex-session:prepared-membership"
        assert empty.value.raw_id == raw_id
        assert "none for this logical key" in empty.value.reason
        assert "no longer parses uniquely" not in str(empty.value)


@pytest.mark.asyncio
async def test_canonical_retained_sqlite_busy_stays_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLITE_BUSY through the actual worker is never a terminal parse census."""
    from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=b'{"type":"session_meta","payload":{"id":"census-busy"}}\n',
                source_path="census-busy.jsonl",
                canonical_source_path="census-busy.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)
    failure = sqlite3.OperationalError("database is locked")
    failure.sqlite_errorcode = sqlite3.SQLITE_BUSY
    failure.sqlite_errorname = "SQLITE_BUSY"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def busy_parse(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise failure

    monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_jsonl_blob", busy_parse)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        with pytest.raises(RetainedPreparationRetryableError) as retryable:
            (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
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
async def test_original_raw_selects_each_sessions_own_attachment_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One actual Raw never reuses its first session's attachment lookup."""
    from polylogue.pipeline.ids import session_id as make_session_id
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    sessions: list[ParsedSession] = []
    expected: dict[str, str] = {}

    def seed() -> str:
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
                canonical_source_path="prepared-multiple.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(tmp_path, seed)

    def worker(reader: RetainedSessionRead, selected_raw_id: str, *, directory: Path) -> PreparedJsonl:
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
        results = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
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

    def check(
        _unbound: ParsedSession,
        _carried: ParsedSession,
        replacement: RawObservationReplacement,
        publish: _Publish,
    ) -> None:
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


def _move_source(root: Path, sql: str, parameters: tuple[object, ...]) -> Callable[[], None]:
    # An independent Source process moves durable evidence; it is not a
    # participant in the prepared publication's admitted writer custody.
    def move() -> None:
        with closing(sqlite3.connect(root / "source.db")) as source:
            source.execute(sql, parameters)
            source.commit()

    return move


def _membership_census(raw_id: str, sessions: list[ParsedSession], **census: Any) -> Callable[[Any], None]:
    def prepare(seal: Any) -> None:
        revision_governance.replace_raw_membership_census(
            seal,
            raw_id,
            sessions,
            parser_fingerprint=census.pop("parser_fingerprint", raw_authority_parser_fingerprint()),
            censused_at_ms=census.pop("censused_at_ms", 1),
            revision_authority=census.pop("revision_authority", None),
            **census,
        )

    return prepare


def test_prepared_census_binding_includes_revision_authority(tmp_path: Path) -> None:
    """An authority-only census change makes a prepared parser census stale.

    The canonical parser census reads the membership census authority on its
    original Source seal. Anti-vacuity: drop ``revision_authority`` from that
    read and the intervening authority change publishes over moved evidence.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
    from tests.infra.prepared_replay import publish_prepared_source

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)
    _move_source(
        tmp_path,
        "INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms, "
        "detail, revision_authority) VALUES (?, 'parser', 'complete', 0, 1, '', 'semantic')",
        (raw_id,),
    )()
    with pytest.raises(ReferenceSealStaleError):
        publish_prepared_source(
            tmp_path,
            "test.census.authority-binding",
            lambda seal: revision_governance.record_current_parser_source_census(seal, raw_id),
            after_prepare=_move_source(
                tmp_path, "UPDATE raw_membership_census SET revision_authority='byte' WHERE raw_id=?", (raw_id,)
            ),
        )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
        ).fetchone() == (0,)


def test_prepared_census_reuses_projection_and_does_not_terminalize_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publication applies the prepared membership tape and stays census-only.

    Anti-vacuity: if publication reprojects the session the patched projector
    raises; if it marks the raw parsed the final source-row assertion fails.
    """
    from tests.infra.prepared_replay import publish_prepared_source

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)

    def forbid_projection() -> None:
        monkeypatch.setattr(
            revision_governance,
            "session_revision_projection",
            lambda _session: (_ for _ in ()).throw(AssertionError("writer recomputed prepared projection")),
        )

    publish_prepared_source(
        tmp_path,
        "test.census.prepared-projection",
        _membership_census(raw_id, [_session("one", "two")], censused_at_ms=7),
        after_prepare=forbid_projection,
    )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT status, member_count FROM raw_membership_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("complete", 1)
        assert source.execute(
            "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (None, None)


def test_membership_that_becomes_eligible_after_preparation_defers_publication(tmp_path: Path) -> None:
    """A membership census landing after preparation makes the prepared census stale.

    Revalidation must reread current membership for the logical key rather
    than trust its preparation snapshot. Anti-vacuity: answer publication from
    the preparation snapshot and the stale census publishes over moved evidence.
    """
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
    from tests.infra.prepared_replay import publish_prepared_source

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        first_raw, late_raw = _write_raws(archive, 2)
    publish_prepared_source(tmp_path, "test.membership.first", _membership_census(first_raw, [_session("one")]))

    late_membership = _move_source(
        tmp_path,
        "INSERT INTO raw_session_memberships(raw_id, logical_source_key, provider_session_id, "
        "source_revision, normalized_content_hash, message_count) VALUES (?, ?, ?, ?, ?, ?)",
        (late_raw, "codex-session:prepared-membership", "prepared-membership", "late", bytes(32), 2),
    )

    with pytest.raises(ReferenceSealStaleError):
        publish_prepared_source(
            tmp_path,
            "test.membership.stale",
            _membership_census(first_raw, [_session("one", "two")], censused_at_ms=2),
            after_prepare=late_membership,
        )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT censused_at_ms FROM raw_membership_census WHERE raw_id=?", (first_raw,)
        ).fetchone() == (1,)


@pytest.mark.parametrize("retired", [False, True])
def test_only_explicit_census_retirement_selects_an_unaccepted_sibling(tmp_path: Path, retired: bool) -> None:
    """Default raw quarantine is not request authority; explicit retirement is.

    A sibling raw of the same logical key keeps its byte logical key under
    ordinary quarantine; only the typed retirement census relinquishes it while
    retaining the parsed identity as membership.
    """
    from tests.infra.prepared_replay import publish_prepared_source

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        accepted, sibling = _write_raws(archive, 2)
        archive.bind_raw_revision(
            sibling,
            RawRevisionEnvelope(
                logical_source_key="codex-session:prepared-membership",
                kind=RawRevisionKind.FULL,
                source_revision="prepared-sibling-v1",
                acquisition_generation=0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        archive.commit()
    publish_prepared_source(tmp_path, "test.retirement.accepted", _membership_census(accepted, [_session("one")]))
    publish_prepared_source(
        tmp_path,
        "test.retirement.sibling",
        _membership_census(
            sibling,
            [_session("other")],
            revision_authority=RawRevisionAuthority.QUARANTINED,
            retire_full_revision_governance=retired,
        ),
    )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT revision_authority FROM raw_sessions WHERE raw_id = ?", (sibling,)
        ).fetchone() == ("quarantined",)
        membership = source.execute(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id = ?", (sibling,)
        ).fetchall()
        assert membership == [("codex-session:prepared-membership",)]
        raw_key = source.execute("SELECT logical_source_key FROM raw_sessions WHERE raw_id = ?", (sibling,)).fetchone()
        assert (raw_key == (None,)) is retired


@pytest.mark.asyncio
async def test_read_only_compute_defers_attachment_publication_until_writer_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale prepared attachment publishes no Index attachments.

    Compute prepares the attachment off-writer. Its acquired claim is Source
    acquisition evidence, settled as an exact blob reference before derived
    replay can fail or skip (the acquired attachment reference phase); a
    descriptor change before writer admission refuses the Index publication
    and leaves no attachment row. Anti-vacuity: publish Index attachments
    during compute and the stale refusal leaves them behind.
    """
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    def check(
        _unbound: ParsedSession,
        _carried: ParsedSession,
        replacement: RawObservationReplacement,
        publish: _Publish,
    ) -> None:
        with closing(sqlite3.connect(tmp_path / "source.db")) as source:
            acquired_refs = source.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment' AND ref_id = ?", (replacement.key,)
            ).fetchone()
        assert acquired_refs == (1,)
        admit_stage_write(
            "fixture.raw.attachment-descriptor",
            _move_source(
                tmp_path,
                "UPDATE raw_sessions SET source_path=? WHERE raw_id=?",
                ("changed-before-writer-admission.jsonl", replacement.key),
            ),
        )
        with pytest.raises(ReferenceSealStaleError):
            publish()
        with closing(sqlite3.connect(tmp_path / "index.db")) as index:
            assert index.execute("SELECT COUNT(*) FROM attachments").fetchone() == (0,)
        with closing(sqlite3.connect(tmp_path / "source.db")) as source:
            assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone() == (
                acquired_refs
            )

    await _run_original_raw_carrier_case(
        tmp_path,
        monkeypatch,
        check,
        session_factory=lambda: _session("one", "two", attachment_bytes=b"prepared-attachment-bytes"),
    )


def test_convertible_multi_session_retirement_preserves_complete_census(tmp_path: Path) -> None:
    """Retiring one full export retains every logical member it contained.

    Anti-vacuity: replacing with only the target session would delete the
    sibling membership, so the complete two-key assertion becomes red.
    """
    from tests.infra.prepared_replay import publish_prepared_source

    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        (raw_id,) = _write_raws(archive, 1)
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key="codex-session:prepared-membership",
                kind=RawRevisionKind.FULL,
                source_revision="prepared-full-v1",
                acquisition_generation=0,
            ),
        )
        archive.commit()
    publish_prepared_source(
        tmp_path,
        "test.retirement.multi-session",
        _membership_census(
            raw_id,
            [_session("one"), _session("other", session_id="prepared-sibling")],
            revision_authority=RawRevisionAuthority.QUARANTINED,
            retire_full_revision_governance=True,
        ),
    )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key",
            (raw_id,),
        ).fetchall() == [
            ("codex-session:prepared-membership",),
            ("codex-session:prepared-sibling",),
        ]
