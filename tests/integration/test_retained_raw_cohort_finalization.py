"""Real retained records use the supplied owner and canonical cohort finalizer."""

from __future__ import annotations

import base64
import hashlib
import json
from builtins import BaseExceptionGroup
from collections import Counter
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.live_provider_proof import native_proof_artifact
from tests.infra.source_builders import ChatGPTExportBuilder


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", ["json", "jsonl"])
async def test_resident_raw_owner_finalizes_each_real_retained_cohort_member(tmp_path: Path, wire_format: str) -> None:
    root = tmp_path / "archive"
    records = [ChatGPTExportBuilder(f"cohort-{ordinal}").add_node("user", "neutral").build() for ordinal in range(3)]
    payload = (
        json.dumps(records).encode()
        if wire_format == "json"
        else b"\n".join(json.dumps(record).encode() for record in records) + b"\n"
    )

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path=f"cohort.{wire_format}",
                canonical_source_path=f"cohort.{wire_format}",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    expected = {f"chatgpt-export:cohort-{ordinal}" for ordinal in range(3)}
    async with prepared_live_convergence_owner(root) as owner:
        first = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert {key for receipt in first for key in receipt.written_session_ids} == expected
        assert {key for receipt in first for key in receipt.changed_session_ids} == expected
        assert sum(receipt.written_message_count for receipt in first) == 3
        repeated = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert not any(receipt.changed_session_ids for receipt in repeated)
        assert all(before == after for receipt in repeated for _key, before, after, _count in receipt.session_outputs)
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert {row[0] for row in index.execute("SELECT session_id FROM sessions")} == expected
        assert index.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        assert (
            index.execute("SELECT COUNT(*) FROM blocks WHERE block_type='text' AND text='neutral'").fetchone()[0] == 3
        )
        source = archive.source_connection
        assert source is not None
        assert {
            row[0]
            for row in source.execute(
                "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=?", (raw_id,)
            )
        } == expected


@pytest.mark.asyncio
async def test_resident_raw_owner_preserves_published_attachment_receipts_after_cohort_copy(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    envelope, expected_messages, expected_attachments = native_proof_artifact(
        tmp_path, "native-inline-attachment-v1.json", Provider.GROK
    )
    assert expected_attachments > 0
    attachment_bytes = [base64.b64decode(entry["content_base64"]) for entry in envelope["session"]["attachments"]]
    expected_blobs = {hashlib.sha256(content).digest(): len(content) for content in attachment_bytes}
    payload = json.dumps(envelope).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.GROK,
                payload=payload,
                source_path="neutral-grok-capture.json",
                canonical_source_path="neutral-grok-capture.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        first = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(receipt.written_message_count for receipt in first) == expected_messages
        assert len({session_id for receipt in first for session_id in receipt.written_session_ids}) == 1
        repeated = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert not any(receipt.changed_session_ids for receipt in repeated)
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        source = archive.source_connection
        assert index is not None and source is not None
        attachments = index.execute("SELECT blob_hash,byte_count,acquisition_status FROM attachments").fetchall()
        assert len(attachments) == expected_attachments
        assert all(row[2] == "acquired" for row in attachments)
        assert {bytes(row[0]): int(row[1]) for row in attachments} == expected_blobs
        retained = source.execute(
            "SELECT blob_hash,size_bytes FROM blob_refs WHERE ref_type='attachment' AND ref_id=?", (raw_id,)
        ).fetchall()
        assert {bytes(row[0]): int(row[1]) for row in retained} == expected_blobs
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_resident_byte_aggregate_consumes_the_original_full_and_append_carriers(tmp_path: Path) -> None:
    from polylogue.archive.revision_authority import (
        RawRevisionAuthority,
        RawRevisionEnvelope,
        RawRevisionKind,
        append_source_revision,
    )
    from polylogue.sources.acquisition_boundary import bound_profile_identity, bound_source_observation, open_bound_path
    from tests.infra.authoritative_replay_payloads import codex_single_message_bytes

    root = tmp_path / "archive"
    source_path = tmp_path / "sessions" / "append.jsonl"
    baseline = codex_single_message_bytes("append-owner", "neutral baseline")
    delta = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "neutral append"}],
                },
            }
        ).encode()
        + b"\n"
    )
    baseline_revision = hashlib.sha256(baseline).hexdigest()
    key = "codex-session:append-owner"

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(baseline)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            with open_bound_path(source_path, None) as original:
                profile = bound_profile_identity(original)
                canonical, observation = bound_source_observation(original)
                assert canonical is not None and observation is not None
                acquired = original.read()
                assert acquired == baseline
                baseline_raw_id = archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=acquired,
                    source_path=str(source_path),
                    canonical_source_path=canonical,
                    captured_profile_key=profile.key if profile else None,
                    native_id="append-owner",
                    acquired_at_ms=1,
                    file_mtime_ms=observation[3] // 1_000_000,
                    revision=RawRevisionEnvelope(
                        key,
                        RawRevisionKind.FULL,
                        baseline_revision,
                        0,
                        authority=RawRevisionAuthority.BYTE_PROVEN,
                    ),
                )
            with source_path.open("ab") as writer:
                writer.write(delta)
            with open_bound_path(source_path, None) as original:
                next_profile = bound_profile_identity(original)
                next_canonical, observation = bound_source_observation(original)
                assert next_canonical == canonical and observation is not None
                assert (next_profile.key if next_profile else None) == (profile.key if profile else None)
                acquired = original.read()
                assert acquired == baseline + delta
                actual_delta = acquired[len(baseline) :]
                assert actual_delta == delta
                append_raw_id = archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=actual_delta,
                    source_path=str(source_path),
                    canonical_source_path=next_canonical,
                    captured_profile_key=next_profile.key if next_profile else None,
                    native_id="append-owner",
                    acquired_at_ms=2,
                    file_mtime_ms=observation[3] // 1_000_000,
                    revision=RawRevisionEnvelope(
                        key,
                        RawRevisionKind.APPEND,
                        append_source_revision(baseline_revision, hashlib.sha256(actual_delta).hexdigest()),
                        1,
                        predecessor_source_revision=baseline_revision,
                        predecessor_raw_id=baseline_raw_id,
                        baseline_raw_id=baseline_raw_id,
                        append_start_offset=len(baseline),
                        append_end_offset=len(acquired),
                        authority=RawRevisionAuthority.BYTE_PROVEN,
                    ),
                )
            return baseline_raw_id, append_raw_id

    baseline_raw_id, append_raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        first = (await owner.ingest_retained_raw_ids((append_raw_id,))).require_complete()
        assert {value for receipt in first for value in receipt.written_session_ids} == {key}
        assert sum(receipt.written_message_count for receipt in first) == 2
        repeated = (await owner.ingest_retained_raw_ids((append_raw_id,))).require_complete()
        assert not any(receipt.changed_session_ids for receipt in repeated)
        assert sum(receipt.written_message_count for receipt in repeated) == 0
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        source = archive.source_connection
        assert index is not None and source is not None
        rows = index.execute(
            "SELECT b.text FROM messages m JOIN blocks b ON b.message_id=m.message_id "
            "WHERE m.session_id=? AND b.text IS NOT NULL ORDER BY m.position,b.position",
            (key,),
        ).fetchall()
        assert [row[0] for row in rows] == ["neutral baseline", "neutral append"]
        payloads = source.execute(
            "SELECT ref_id,blob_hash FROM blob_refs WHERE ref_type='raw_payload' AND ref_id IN (?,?)",
            (baseline_raw_id, append_raw_id),
        ).fetchall()
        assert {row[0]: bytes(row[1]) for row in payloads} == {
            baseline_raw_id: hashlib.sha256(baseline).digest(),
            append_raw_id: hashlib.sha256(delta).digest(),
        }
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement", ["repeat", "equivalent", "move", "arrival_forward", "arrival_reverse", "insertion"]
)
async def test_resident_byte_aggregate_preserves_acquired_claims_from_both_original_carriers(
    tmp_path: Path, replacement: str
) -> None:
    from polylogue.core.identity_law import message_id
    from polylogue.pipeline.ids import message_content_identities
    from polylogue.sources.parsers.browser_capture import parse as parse_capture
    from tests.infra.retained_jsonl import retained_append_fixture

    root = tmp_path / "archive"
    original, original_messages, original_attachments = native_proof_artifact(
        tmp_path,
        "native-inline-attachment-unrelated-v1.json"
        if replacement == "insertion"
        else "native-inline-attachment-v1.json",
        Provider.GROK,
    )
    cumulative, cumulative_messages, cumulative_attachments = native_proof_artifact(
        tmp_path,
        "native-inline-attachment-append-unrelated-v1.json"
        if replacement == "insertion"
        else "native-inline-attachment-append-v1.json",
        Provider.GROK,
    )
    assert cumulative["session"]["turns"][:original_messages] == original["session"]["turns"]
    assert cumulative["session"]["attachments"] == original["session"]["attachments"]
    assert original_attachments == cumulative_attachments > 0
    native_id = original["session"]["provider_session_id"]
    assert cumulative["session"]["provider_session_id"] == native_id
    key = f"grok-export:{native_id}"
    baseline = json.dumps(original).encode() + b"\n"
    delta = json.dumps(cumulative).encode() + b"\n"
    expected_blobs = {
        hashlib.sha256(base64.b64decode(entry["content_base64"])).digest(): len(
            base64.b64decode(entry["content_base64"])
        )
        for entry in original["session"]["attachments"]
    }
    expected_occurrences = Counter(
        turn["provider_turn_id"] for envelope in (original, cumulative) for turn in envelope["session"]["turns"]
    )
    from polylogue.core.enums import Origin
    from polylogue.sources.parsers.base_models import ParsedMessage
    from polylogue.sources.prepared_message_sink import normalize_active_branch
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    def canonical_expected_messages(envelope: object) -> list[ParsedMessage]:
        session = parse_capture(envelope, "unused")
        original_fields = [message.model_dump(mode="json") for message in session.messages]
        expected_leaf = (
            "followup"
            if any(message.provider_message_id == "followup" for message in session.messages)
            else "attachment"
        )
        assert session.active_leaf_message_provider_id == expected_leaf, (
            session.active_leaf_message_provider_id,
            [message.provider_message_id for message in session.messages],
        )
        normalized = derive_tool_outcomes(
            normalize_active_branch(list(session.messages)), session.session_events, origin=Origin.GROK_EXPORT
        )
        for original_fields_row, message in zip(original_fields, normalized, strict=True):
            current = message.model_dump(mode="json")
            original_blocks = original_fields_row.pop("blocks")
            current_blocks = current.pop("blocks")
            assert current["is_active_leaf"] is (message.provider_message_id == expected_leaf)
            assert original_fields_row == current
            for before, after in zip(original_blocks, current_blocks, strict=True):
                before.pop("tool_outcome")
                after.pop("tool_outcome")
                assert before == after
            outcomes = [
                block.tool_outcome.value if block.tool_outcome is not None else None for block in message.blocks
            ]
            reasons = [block.outcome_unknown_reason for block in message.blocks]
            if message.provider_message_id == "reasoning":
                assert outcomes == [None, "ok", "unknown"]
                assert reasons == [None, None, "not_reported"]
            elif message.provider_message_id == "search":
                assert outcomes == ["unknown", "unknown", "error", "error"]
                assert reasons == [None, "not_reported", None, None]
        return list(normalized)

    original_messages_in_order = [
        message for envelope in (original, cumulative) for message in canonical_expected_messages(envelope)
    ]
    assert Counter(message.provider_message_id for message in original_messages_in_order) == expected_occurrences
    semantic_occurrences = message_content_identities(original_messages_in_order)
    expected_rows = []
    for message, (digest, occurrence) in zip(original_messages_in_order, semantic_occurrences, strict=True):
        native = message.provider_message_id if expected_occurrences[message.provider_message_id] == 1 else None
        expected_rows.append(
            (
                message_id(key, native, content_identity=digest, content_occurrence=occurrence),
                digest,
                occurrence,
                native,
            )
        )

    if replacement == "insertion":
        previous_documents = [
            native_proof_artifact(tmp_path, filename, Provider.GROK)[0]
            for filename in ("native-inline-attachment-v1.json", "native-inline-attachment-append-v1.json")
        ]
        previous_messages = [
            message for document in previous_documents for message in canonical_expected_messages(document)
        ]
        # The insertion law is preservation: every previously present message
        # (by provider ID and text, with multiplicity) survives, plus exactly
        # the two inserted ones. Content digests are not compared because an
        # insertion may legitimately re-parent a later message.
        previous_preserved = Counter((message.provider_message_id, message.text) for message in previous_messages)
        current_preserved = Counter(
            (message.provider_message_id, message.text) for message in original_messages_in_order
        )
        assert not previous_preserved - current_preserved, previous_preserved - current_preserved
        assert len(expected_rows) == len(previous_messages) + 2

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with retained_append_fixture(
            root=root,
            provider=Provider.GROK,
            source_path=tmp_path / "native" / "grok.jsonl",
            native_id=native_id,
            logical_source_key=key,
            baseline=baseline,
            delta=delta,
        ) as (_reader, baseline_raw_id, append_raw_id, _baseline_revision, _append_revision, _start, _end):
            return baseline_raw_id, append_raw_id

    baseline_raw_id, append_raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        selected = (
            (baseline_raw_id, append_raw_id)
            if replacement == "arrival_forward"
            else (append_raw_id, baseline_raw_id)
            if replacement == "arrival_reverse"
            else (append_raw_id,)
        )
        first = (await owner.ingest_retained_raw_ids(selected)).require_complete()
        assert {value for receipt in first for value in receipt.written_session_ids} == {key}
        # APPEND contains a second complete provider document. The canonical
        # byte cohort retains each declared occurrence from both acquired ranges.
        assert sum(receipt.written_message_count for receipt in first) == original_messages + cumulative_messages
        repeated = (await owner.ingest_retained_raw_ids((append_raw_id,))).require_complete()
        assert not any(receipt.changed_session_ids for receipt in repeated)
        assert sum(receipt.written_message_count for receipt in repeated) == 0
        if replacement in {"equivalent", "move"}:
            replacement_baseline = json.dumps(original, separators=(", ", ":  ")).encode() + b"\n"
            replacement_delta = json.dumps(cumulative, separators=(", ", ":  ")).encode() + b"\n"
            assert replacement_baseline != baseline and replacement_delta != delta
            replacement_path = tmp_path / "native" / ("moved-grok.jsonl" if replacement == "move" else "grok.jsonl")

            def acquire_replacement() -> tuple[str, str]:
                with retained_append_fixture(
                    root=root,
                    provider=Provider.GROK,
                    source_path=replacement_path,
                    native_id=native_id,
                    logical_source_key=key,
                    baseline=replacement_baseline,
                    delta=replacement_delta,
                    baseline_generation=2,
                    acquired_at_ms=3,
                ) as (_reader, full_id, append_id, *_evidence):
                    return full_id, append_id

            replacement_full_id, replacement_append_id = await run_archive_fixture_write(root, acquire_replacement)
            assert replacement_full_id != baseline_raw_id and replacement_append_id != append_raw_id
            replaced = (await owner.ingest_retained_raw_ids((replacement_append_id,))).require_complete()
            assert not any(receipt.changed_session_ids for receipt in replaced)
            assert sum(receipt.written_message_count for receipt in replaced) == 0
            assert all(
                before == after for receipt in replaced for _key, before, after, _count in receipt.session_outputs
            )
            replayed = (await owner.ingest_retained_raw_ids((replacement_append_id,))).require_complete()
            assert not any(receipt.changed_session_ids for receipt in replayed)
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        source = archive.source_connection
        assert index is not None and source is not None
        assert [
            tuple(row)
            for row in index.execute(
                "SELECT message_id,content_identity,content_occurrence,native_id FROM messages WHERE session_id=? ORDER BY position,variant_index",
                (key,),
            )
        ] == expected_rows
        assert len(expected_rows) == original_messages + cumulative_messages
        attachments = index.execute("SELECT blob_hash,byte_count,acquisition_status FROM attachments").fetchall()
        assert {bytes(row[0]): int(row[1]) for row in attachments} == expected_blobs
        assert all(row[2] == "acquired" for row in attachments)
        assert (
            index.execute("SELECT COUNT(*) FROM attachment_refs WHERE session_id=?", (key,)).fetchone()[0]
            == original_attachments + cumulative_attachments
        )
        for raw_id, payload in ((baseline_raw_id, baseline), (append_raw_id, delta)):
            assert (
                source.execute(
                    "SELECT blob_hash FROM blob_refs WHERE ref_type='raw_payload' AND ref_id=?", (raw_id,)
                ).fetchone()[0]
                == hashlib.sha256(payload).digest()
            )
            retained = source.execute(
                "SELECT blob_hash,size_bytes FROM blob_refs WHERE ref_type='attachment' AND ref_id=?", (raw_id,)
            ).fetchall()
            assert {bytes(row[0]): int(row[1]) for row in retained} == expected_blobs
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_failed_real_cohort_preparation_releases_original_scratch_after_seal_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources import prepared_merge
    from polylogue.sources.parsers.base_models import ParsedSession
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from tests.infra.retained_jsonl import retained_append_fixture

    root = tmp_path / "archive"
    original, original_messages, _ = native_proof_artifact(tmp_path, "native-inline-attachment-v1.json", Provider.GROK)
    cumulative, cumulative_messages, _ = native_proof_artifact(
        tmp_path, "native-inline-attachment-append-v1.json", Provider.GROK
    )
    native_id = original["session"]["provider_session_id"]
    fault = RuntimeError("synthetic reached aggregate preparation failure")
    reached = False
    from polylogue.pipeline.ids import session_content_hash

    original_hash = session_content_hash

    def fail_reached_aggregate(session: ParsedSession) -> str:
        nonlocal reached
        if session.source_name is Provider.GROK and len(session.messages) == original_messages + cumulative_messages:
            reached = True
            raise fault
        return original_hash(session)

    def acquire() -> str:
        bootstrap_archive_root(root)
        with retained_append_fixture(
            root=root,
            provider=Provider.GROK,
            source_path=tmp_path / "native" / "grok.jsonl",
            native_id=native_id,
            logical_source_key=f"grok-export:{native_id}",
            baseline=json.dumps(original).encode() + b"\n",
            delta=json.dumps(cumulative).encode() + b"\n",
        ) as (_reader, _full_id, append_id, *_evidence):
            return append_id

    raw_id = await run_archive_fixture_write(root, acquire)
    monkeypatch.setattr(prepared_merge, "session_content_hash", fail_reached_aggregate)
    async with prepared_live_convergence_owner(root) as owner:
        with pytest.raises(BaseException) as failure:
            (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        leaves = [failure.value]
        while any(isinstance(item, BaseExceptionGroup) for item in leaves):
            leaves = [
                child
                for item in leaves
                for child in (item.exceptions if isinstance(item, BaseExceptionGroup) else (item,))
            ]
        assert reached and fault in leaves
        assert not any(isinstance(item, NativeConnectionSettlementError) for item in leaves)
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive.source_connection
        index = archive.index_connection
        assert index is not None
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
