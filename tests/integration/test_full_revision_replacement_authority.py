"""Selected original Source FULL proof authorizes only a fresh exact replacement."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority
from polylogue.archive.revision_replay import ApplicationDecision, plan_revision_replay
from polylogue.core.enums import Provider
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    _authorize_selected_full_replacement,
    prepared_raw_revision_candidates,
)
from polylogue.storage.sqlite.write_lease import async_write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.retained_jsonl import acquire_full_revision, prepared_source_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival", [(0, 1), (1, 0)])
async def test_full_replacement_authority_requires_selected_original_source_freshness(
    tmp_path: Path, arrival: tuple[int, int]
) -> None:
    root = tmp_path / "archive"
    native_file = tmp_path / "native" / "conversation.json"
    payloads = (b'{"id":"session","neutral":"longer original evidence"}', b'{"id":"session"}')

    def acquire() -> dict[int, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return {
                generation: acquire_full_revision(
                    archive,
                    provider=Provider.CLAUDE_AI,
                    source_path=native_file,
                    payload=payloads[generation],
                    native_id="session",
                    generation=generation,
                    acquired_at_ms=position + 1,
                )
                for position, generation in enumerate(arrival)
            }

    raw_ids = await run_archive_fixture_write(root, acquire)
    async with async_write_lease("test.runtime.async-fixture", archive_root=root):
        with prepared_source_fixture(root) as reader:
            key = "claude-ai-export:session"
            candidates = {item.raw_id: item for item in prepared_raw_revision_candidates(reader._seal, key)}
            plan = plan_revision_replay(list(candidates.values()))
            assert plan.accepted_raw_ids == (raw_ids[1],)
            assert any(
                item.raw_id == raw_ids[1] and item.decision is ApplicationDecision.SELECTED_BASELINE
                for item in plan.applications
            )
            previous = candidates[raw_ids[0]]
            old_head = (
                key,
                previous.raw_id,
                previous.source_revision,
                b"o" * 32,
                "byte",
                previous.blob_size,
                previous.acquisition_generation,
                None,
            )
            authorization = _authorize_selected_full_replacement(
                reader._seal, plan, candidates, existing_head=old_head, session_id=key, content_hash=b"n" * 32
            )
            if arrival == (0, 1):
                assert authorization is not None
                assert authorization.full_raw_id == raw_ids[1]
                assert authorization.previous_head == old_head
                assert authorization.byte_length < previous.blob_size
            else:
                assert authorization is None
            changed_head = (*old_head[:2], "not-the-original-source-revision", *old_head[3:])
            assert (
                _authorize_selected_full_replacement(
                    reader._seal, plan, candidates, existing_head=changed_head, session_id=key, content_hash=b"n" * 32
                )
                is None
            )

            unproven = dict(candidates)
            unproven[raw_ids[1]] = replace(candidates[raw_ids[1]], authority=RawRevisionAuthority.ASSERTED)
            assert (
                _authorize_selected_full_replacement(
                    reader._seal, plan, unproven, existing_head=old_head, session_id=key, content_hash=b"n" * 32
                )
                is None
            )
            not_selected = replace(
                plan,
                applications=tuple(
                    replace(item, decision=ApplicationDecision.AMBIGUOUS) if item.raw_id == raw_ids[1] else item
                    for item in plan.applications
                ),
            )
            assert (
                _authorize_selected_full_replacement(
                    reader._seal,
                    not_selected,
                    candidates,
                    existing_head=old_head,
                    session_id=key,
                    content_hash=b"n" * 32,
                )
                is None
            )


@pytest.mark.asyncio
async def test_valid_full_membership_conversion_publishes_each_original_member(tmp_path: Path) -> None:
    import hashlib
    import json
    from contextlib import ExitStack
    from functools import partial

    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.sources.revision_backfill import (
        PreparedRetainedInput,
        prepare_retained_jsonl_artifact,
        prepare_revision_source_membership_conversion,
    )
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import publish_prepared_revision_source
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.source_builders import ChatGPTExportBuilder

    root = tmp_path / "archive"
    source_path = tmp_path / "native" / "conversation.json"
    payloads = tuple(
        json.dumps(ChatGPTExportBuilder("session").add_node("user", text).build()).encode()
        for text in ("first original", "changed original")
    )

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            ids = tuple(
                acquire_full_revision(
                    archive,
                    provider=Provider.CHATGPT,
                    source_path=source_path,
                    payload=payload,
                    native_id="session",
                    generation=generation,
                    acquired_at_ms=generation + 1,
                )
                for generation, payload in enumerate(payloads)
            )
            return ids[0], ids[1]

    raw_ids = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:

        def prepare_and_publish() -> None:
            with PreparedIndexMutation.source_only(archive_root=root) as seal, ExitStack() as reads:
                cleanup = ExitStack()
                seal.retain_preparation_payload(cleanup.close)
                reads.enter_context(seal.original_read_snapshot())
                reads.enter_context(seal.source_producer())
                reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
                inputs: dict[str, PreparedRetainedInput] = {}
                for raw_id, payload in zip(raw_ids, payloads, strict=True):
                    descriptor = reader.raw_revision_descriptor(raw_id)
                    path = reader.raw_revision_blob_path(raw_id)
                    assert path is not None
                    artifact = prepare_retained_jsonl_artifact(
                        reader,
                        raw_id,
                        directory=BlobStore(root / "blob")._ensure_private_staging_root() / "conversion-control",
                    )
                    cleanup.callback(artifact.discard)
                    assert artifact.error is None and len(artifact.session_sequence()) == 1
                    info = path.stat()
                    inputs[raw_id] = PreparedRetainedInput(
                        raw_id,
                        Provider.CHATGPT,
                        hashlib.sha256(payload).hexdigest(),
                        descriptor[2],
                        RawRevisionKind.FULL,
                        len(payload),
                        None,
                        raw_authority_parser_fingerprint(),
                        reader.raw_revision_file_mtime(raw_id),
                        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
                        captured_profile_key=reader.raw_profile_identity(raw_id),
                        prepared_artifact=artifact,
                    )
                converted = prepare_revision_source_membership_conversion(
                    seal,
                    reader,
                    logical_keys=("chatgpt-export:session",),
                    prepared_inputs=inputs,
                    prepared_replay_plans={"chatgpt-export:session": ()},
                )
                assert converted == 2
                assert all(reader.raw_has_membership_authority(raw_id) for raw_id in raw_ids)
                reads.close()
                permit = seal.prepare_source_mutation()
                admit_stage_write(
                    "test.full-membership-conversion", partial(publish_prepared_revision_source, seal, permit)
                )

        await owner.run_prepared_sync(
            "test.full-membership-conversion",
            prepare_and_publish,
            settlement_owners=lambda: (),
            estimated_bytes=sum(map(len, payloads)),
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        source = archive.source_connection
        assert source is not None
        assert {
            tuple(row) for row in source.execute("SELECT raw_id,logical_source_key FROM raw_session_memberships")
        } == {(raw_id, "chatgpt-export:session") for raw_id in raw_ids}
