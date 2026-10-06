"""Durable annotation policy is enforced through actual retained reconvergence."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.pipeline.ids import message_content_identity
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.parsers.claude.ai_parser import parse_ai
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session
from tests.infra.retained_identity_payloads import IDENTITY_REFERENCE_CASES, identity_export, identity_export_bytes
from tests.infra.retained_replay import replay_retained_components


@pytest.mark.parametrize(
    "case,prior_preimage,current_arguments",
    IDENTITY_REFERENCE_CASES,
    ids=[case[0] for case in IDENTITY_REFERENCE_CASES],
)
def test_retained_replay_and_promotion_preserve_or_refuse_prior_annotated_identity(
    tmp_path: Path, case: str, prior_preimage: JSONDocument, current_arguments: JSONDocument
) -> None:
    # The old canonical preimage is represented by an ordinary admitted input
    # to the sole current encoder. No predecessor encoder or lookup is added.
    prior = parse_ai(identity_export(prior_preimage), "retained-identity")
    current = parse_ai(identity_export(current_arguments), "retained-identity")
    assert len(prior.messages) == len(current.messages) == 1
    assert prior.messages[0].provider_message_id == current.messages[0].provider_message_id == ""
    changed_identity = message_content_identity(prior.messages[0]) != message_content_identity(current.messages[0])
    assert changed_identity == (not case.startswith("ordinary-"))
    with write_lease("test.retained-identity-seed", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_AI,
                payload=identity_export_bytes(current_arguments),
                source_path="retained-identity.json",
                canonical_source_path="retained-identity.json",
                acquired_at_ms=1,
                revision=RawRevisionEnvelope(
                    logical_source_key="claude-ai-export:retained-identity",
                    kind=RawRevisionKind.FULL,
                    source_revision="retained-v1",
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )
            session_id = write_index_session(archive, prior)
            old_message = str(
                archive._conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[
                    0
                ]
            )
            old_blocks = tuple(
                str(row[0])
                for row in archive._conn.execute(
                    "SELECT block_id FROM blocks WHERE message_id = ? ORDER BY position", (old_message,)
                )
            )
            assert old_blocks
            archive.save_annotation(
                "retained-message-note",
                "message",
                old_message,
                "Preserve this content identity",
                owner_session_id=session_id,
            )
            archive.save_annotation(
                "retained-block-note",
                "block",
                old_blocks[0],
                "Preserve this block identity",
                owner_session_id=session_id,
            )
            archive.commit()
    generations = IndexGenerationStore.for_archive_root(tmp_path)
    before = Path(generations.active_pointer).resolve(strict=True)
    # Retained replay publishes only into the registered cold-build
    # destination, so the candidate is begun and registered as one.
    with write_lease("test.retained-identity-candidate", archive_root=tmp_path):
        cold_build = ColdBuildGeneration.begin(
            tmp_path,
            reason="test-retained-identity",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent"),)),
        )
    candidate = cold_build.generation
    register_cold_build_generation(cold_build)
    try:
        # The daemon owner establishes the cold writer profile before any
        # preparation records the destination's file identity.
        with write_lease("test.retained-identity-destination", archive_root=tmp_path), cold_build.open_writer():
            pass
        result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id], owned_generation=candidate)
        # The readiness pass builds the candidate's deferred indexes and FTS.
        with write_lease("test.retained-identity-readiness", archive_root=tmp_path):
            cold_build.prepare_promotion_candidate()
    finally:
        clear_cold_build_generation()
    assert result.replayed_logical_sources == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        census = source.execute(
            "SELECT parser_fingerprint, status FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        assert census == (raw_authority_parser_fingerprint(), "complete")
    with ArchiveStore.open_existing(tmp_path, read_only=True, index_path=Path(candidate.index_path)) as archive:
        new_message = str(
            archive._conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
        )
        assert (new_message != old_message) is changed_identity

    if changed_identity:
        with pytest.raises(ReferenceSealError):
            generations.prepare_promotion(candidate)
        assert Path(generations.active_pointer).resolve(strict=True) == before
        assert generations.load(candidate.generation_id).state == "inactive"
    else:
        with generations.prepare_promotion(candidate) as prepared:
            with write_lease("test.retained-identity-promote", archive_root=tmp_path):
                generations.promote(candidate, prepared)
        assert Path(generations.active_pointer).resolve(strict=True) == Path(candidate.index_path).resolve(strict=True)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert archive.get_annotation("retained-message-note") is not None
        assert archive.get_annotation("retained-block-note") is not None
        assert (
            archive._conn.execute("SELECT message_id FROM messages WHERE message_id = ?", (old_message,)).fetchone()
            is not None
        )
        assert (
            tuple(
                str(row[0])
                for row in archive._conn.execute(
                    "SELECT block_id FROM blocks WHERE message_id = ? ORDER BY position", (old_message,)
                )
            )
            == old_blocks
        )
