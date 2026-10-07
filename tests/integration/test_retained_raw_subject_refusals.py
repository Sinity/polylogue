"""Actual Source decode refusal remains distinct from a dependent subject."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError, RetainedRawDependencyRefusalError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


def _payload(session_id: str) -> bytes:
    return json.dumps(
        [
            {
                "id": session_id,
                "title": session_id,
                "create_time": 1,
                "current_node": "m",
                "mapping": {
                    "m": {
                        "id": "m",
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "m",
                            "author": {"role": "user"},
                            "create_time": 1,
                            "content": {"content_type": "text", "parts": ["neutral"]},
                        },
                    }
                },
            }
        ]
    ).encode()


@pytest.mark.asyncio
async def test_original_refused_dependency_does_not_claim_subject_decode_failure(tmp_path: Path) -> None:
    root = tmp_path / "archive"

    def acquire_bad() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.UNKNOWN,
                payload=b"not json",
                source_path="shared.json",
                canonical_source_path="shared.json",
                acquired_at_ms=1,
            )

    bad = await run_archive_fixture_write(root, acquire_bad)
    async with prepared_live_convergence_owner(root) as owner:
        with pytest.raises(RetainedRawDecodeRefusalError) as strict:
            (await owner.ingest_retained_raw_ids((bad,))).require_complete()
        assert strict.value.raw_id == bad

        def acquire_good() -> tuple[str, str]:
            with ArchiveStore.open_existing(root, read_only=False) as archive:
                dependent = archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_payload("dependent"),
                    source_path="shared.json",
                    canonical_source_path="shared.json",
                    acquired_at_ms=2,
                )
                independent = archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_payload("independent"),
                    source_path="independent.json",
                    canonical_source_path="independent.json",
                    acquired_at_ms=3,
                )
                return dependent, independent

        dependent, independent = await run_archive_fixture_write(root, acquire_good)
        refusals: list[RetainedRawDecodeRefusalError] = []
        dependencies: list[RetainedRawDependencyRefusalError] = []
        results = (
            await owner.ingest_retained_raw_ids(
                (bad, dependent, independent),
                on_terminal_refusal=lambda _keys, refusal: refusals.append(refusal),
                on_dependency_refusal=dependencies.append,
            )
        ).require_complete()
        assert [refusal.raw_id for refusal in refusals] == [bad]
        assert len(dependencies) == 1
        assert dependencies[0].subject_raw_id == dependent
        assert dependencies[0].dependency.raw_id == bad
        assert dependencies[0].dependency.kind == strict.value.kind
        assert sum(len(result.written_session_ids) for result in results) == 1
        assert sum(result.written_message_count for result in results) == 1
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert [row[0] for row in index.execute("SELECT title FROM sessions")] == ["independent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_output", ["foreign", "duplicate"])
@pytest.mark.parametrize("healthy_location", ["independent", "same-raw"])
async def test_original_per_key_output_refusal_preserves_independent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_output: str, healthy_location: str
) -> None:
    from polylogue.core.enums import Role
    from polylogue.core.raw_failure_evidence import CohortMembershipRefusalError
    from polylogue.pipeline.ids import session_content_hash
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

    root = tmp_path / "archive"

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            acquired = [
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=(json.dumps({"type": "session_meta", "payload": {"id": key}}) + "\n").encode(),
                    source_path=f"{key}.jsonl",
                    canonical_source_path=f"{key}.jsonl",
                    acquired_at_ms=1,
                )
                for key in ("selected", "independent")
            ]
            return acquired[0], acquired[1]

    selected, independent = await run_archive_fixture_write(root, acquire)

    def session(key: str, text: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=key,
            messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text=text)],
        )

    outputs = {selected: [session("selected", "original")], independent: [session("independent", "original")]}
    if healthy_location == "same-raw":
        outputs[selected].append(session("shared-healthy", "original"))

    def worker(reader: PreparedSessionSourceRead, raw_id: str, *, directory: Path) -> PreparedJsonl:
        _provider, blob_hash, _path, _kind, _size = reader.raw_revision_descriptor(raw_id)
        parsed = [value.model_copy(update={"content_hash": session_content_hash(value)}) for value in outputs[raw_id]]
        artifact = PreparedJsonl.from_sessions(
            parsed,
            blob_hash=blob_hash,
            artifact_directory=directory,
            publication_publisher=ArchiveBlobPublisher(root / "source.db", root / "blob"),
            publication_source_read=reader,
        )
        if len(parsed) >= 2 and parsed[0].provider_session_id == parsed[1].provider_session_id:
            sequence = artifact.session_sequence()
            assert len(sequence) == len(parsed)
            assert list(sequence.iter_session_ids())[:2] == ["codex-session:selected", "codex-session:selected"]
            assert [sequence[position].messages[0].text for position in range(2)] == ["one", "two"]
            assert [value.messages[0].text for value in sequence][:2] == ["one", "two"]
            with pytest.raises(KeyError):
                sequence.by_session_id("codex-session:selected")
        return artifact

    monkeypatch.setattr("polylogue.sources.revision_backfill.prepare_retained_jsonl_artifact", worker)
    async with prepared_live_convergence_owner(root) as owner:
        original = (await owner.ingest_retained_raw_ids((selected, independent))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in original) == (
            3 if healthy_location == "same-raw" else 2
        )
        outputs[selected] = (
            [session("foreign", "foreign")]
            if failed_output == "foreign"
            else [session("selected", "one"), session("selected", "two")]
        )
        if healthy_location == "same-raw":
            outputs[selected].append(session("shared-healthy", "changed"))
        else:
            outputs[independent] = [session("independent", "changed")]
        with pytest.raises(CohortMembershipRefusalError) as strict:
            (await owner.ingest_retained_raw_ids((selected, independent))).require_complete()
        assert strict.value.raw_id == selected
        assert strict.value.logical_source_key == "codex-session:selected"
        refusals: list[CohortMembershipRefusalError] = []
        receipts = (
            await owner.ingest_retained_raw_ids(
                (selected, independent),
                on_membership_refusal=refusals.append,
            )
        ).require_complete()
        assert len(refusals) == 1
        assert refusals[0].raw_id == selected
        assert refusals[0].logical_source_key == "codex-session:selected"
        assert (
            "none for this logical key" in refusals[0].reason
            if failed_output == "foreign"
            else "not one" in refusals[0].reason
        )
        assert sum(len(receipt.changed_session_ids) for receipt in receipts) == 1, [
            (receipt.changed_session_ids, receipt.session_outputs) for receipt in receipts
        ]
        expected_written = {"codex-session:independent"}
        expected_changed = {"codex-session:independent"}
        if healthy_location == "same-raw":
            # Both Raw IDs receive publication acknowledgements. The unchanged
            # independent body skips physical lowering; the changed shared member
            # still publishes its own rows despite the selected-member refusal.
            expected_written.add("codex-session:shared-healthy")
            expected_changed = {"codex-session:shared-healthy"}
        assert {key for receipt in receipts for key in receipt.written_session_ids} == expected_written
        assert {key for receipt in receipts for key in receipt.changed_session_ids} == expected_changed
        assert sum(receipt.written_message_count for receipt in receipts) == 1
        assert sum(receipt.written_counts["messages"] for receipt in receipts) == 1
        assert sum(receipt.written_counts.get("skipped_sessions", 0) for receipt in receipts) == (
            1 if healthy_location == "same-raw" else 0
        )
        if healthy_location == "same-raw":
            independent_outputs = [
                output
                for receipt in receipts
                for output in receipt.session_outputs
                if output[0] == "codex-session:independent"
            ]
            assert len(independent_outputs) == 1
            assert independent_outputs[0][1] == independent_outputs[0][2]
            assert independent_outputs[0][3] == 1
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        rows = index.execute(
            "SELECT s.session_id,b.text FROM sessions s JOIN blocks b ON b.session_id=s.session_id "
            "WHERE b.block_type='text' ORDER BY s.session_id,b.position"
        ).fetchall()
        expected = [
            ("codex-session:independent", "original" if healthy_location == "same-raw" else "changed"),
            ("codex-session:selected", "original"),
        ]
        if healthy_location == "same-raw":
            expected.append(("codex-session:shared-healthy", "changed"))
        assert [tuple(row) for row in rows] == expected
