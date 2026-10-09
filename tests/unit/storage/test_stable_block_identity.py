"""Block identity is Source content, while position only states display order."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.claude.ai_parser import parse_ai
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session


def _export(texts: list[str]) -> JSONDocument:
    return {
        "uuid": "stable-blocks",
        "name": "Stable blocks",
        "chat_messages": [
            {"uuid": "m", "sender": "assistant", "content": [{"type": "text", "text": text} for text in texts]}
        ],
    }


def _session(texts: list[str]) -> ParsedSession:
    return parse_ai(_export(texts), "stable-blocks")


@pytest.mark.parametrize(
    "before,after",
    [
        (["A", "B"], ["X", "A", "B"]),
        (["A", "B"], ["B", "A"]),
        (["B", "B"], ["X", "B", "B"]),
    ],
)
def test_asserted_block_survives_insertion_and_reorder(tmp_path: Path, before: list[str], after: list[str]) -> None:
    with write_lease("test.stable-blocks", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            session_id = write_index_session(archive, _session(before))
            block_id = str(
                archive._conn.execute(
                    "SELECT block_id FROM blocks WHERE session_id=? AND text='B' ORDER BY position DESC LIMIT 1",
                    (session_id,),
                ).fetchone()[0]
            )
            archive.save_annotation("stable-note", "block", block_id, "Keep B", owner_session_id=session_id)
            archive.commit()
            with closing(sqlite3.connect(tmp_path / "user.db")) as user:
                prior_assertion = tuple(user.execute("SELECT * FROM assertions").fetchone())
            write_index_session(archive, _session(after))
            archive.commit()
            assert archive._conn.execute("SELECT text FROM blocks WHERE block_id=?", (block_id,)).fetchone()[0] == "B"
            with closing(sqlite3.connect(tmp_path / "user.db")) as user:
                assert tuple(user.execute("SELECT * FROM assertions").fetchone()) == prior_assertion


def test_source_identity_is_exact_and_occurrences_ignore_unrelated_blocks() -> None:
    from polylogue.core.enums import BlockType
    from polylogue.pipeline.ids import block_content_identities, block_content_identity
    from polylogue.sources.parsers.base import ParsedContentBlock

    a = ParsedContentBlock(type=BlockType.TEXT, text="caf\u00e9")
    equivalent = a.model_copy(update={"text": "cafe\u0301"})
    assert block_content_identity(a) == block_content_identity(equivalent)
    assert block_content_identity(a.model_copy(update={"text": None})) != block_content_identity(
        a.model_copy(update={"text": ""})
    )
    for field in ("tool_id", "tool_name"):
        assert block_content_identity(a.model_copy(update={field: "caf\u00e9"})) != block_content_identity(
            a.model_copy(update={field: "cafe\u0301"})
        )
    assert block_content_identity(a.model_copy(update={"tool_input": {"caf\u00e9": 1}})) != block_content_identity(
        a.model_copy(update={"tool_input": {"cafe\u0301": 1}})
    )
    before = block_content_identities([a, a])
    after = block_content_identities([ParsedContentBlock(type=BlockType.TEXT, text="X"), a, a])
    assert before == after[1:]
    assert [row.content_occurrence for row in before] == [0, 1]


def test_outcome_association_and_preparation_sink_preserve_source_identity(tmp_path: Path) -> None:
    from polylogue.core.enums import BlockType, Origin, Role, ToolOutcome
    from polylogue.pipeline.ids import block_content_identity, validate_semantic_hash_partition
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage
    from polylogue.sources.prepared_message_sink import SqliteMessageStore
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    validate_semantic_hash_partition()
    use = ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="call", tool_name="run", tool_input={"command": "true"})
    message = ParsedMessage(provider_message_id="m", role=Role.ASSISTANT, blocks=[use])
    original = block_content_identity(use)
    normalized = derive_tool_outcomes([message], [], origin=Origin.CLAUDE_AI_EXPORT)[0]
    assert normalized.blocks[0].tool_outcome == ToolOutcome.NO_RESULT
    assert block_content_identity(normalized.blocks[0]) == original
    store = SqliteMessageStore(tmp_path / "messages.db")
    try:
        sink = store.new_sink()
        sink.append(normalized)
        assert block_content_identity(sink[0].blocks[0]) == original
    finally:
        store.close()


def test_cross_export_union_preserves_identity_without_grafting_new_occupants(tmp_path: Path) -> None:
    from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session

    with fixture_index_connection(tmp_path / "index.db") as conn:
        session_id = write_fixture_index_session(conn, _session(["A", "B"]), raw_id="raw-one")
        before = {
            str(row["text"]): str(row["block_id"])
            for row in conn.execute(
                "SELECT text,block_id FROM blocks WHERE session_id=? ORDER BY position", (session_id,)
            )
        }
        write_fixture_index_session(conn, _session(["X", "A"]), raw_id="raw-two")
        after = [
            (str(row["text"]), str(row["block_id"]))
            for row in conn.execute(
                "SELECT text,block_id FROM blocks WHERE session_id=? ORDER BY position", (session_id,)
            )
        ]
        assert [text for text, _identity in after] == ["X", "A", "B"]
        assert dict(after)["A"] == before["A"]
        assert dict(after)["B"] == before["B"]


@pytest.mark.parametrize("source_change", [None, "replace", "delete"])
def test_native_message_block_reference_survives_retained_rebuild_and_promotion(
    tmp_path: Path,
    source_change: str | None,
) -> None:
    import json

    from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.sources.live import WatchSource
    from polylogue.sources.live.cold_build import (
        ColdBuildGeneration,
        clear_cold_build_generation,
        register_cold_build_generation,
    )
    from polylogue.storage.index_generation import IndexGenerationStore
    from tests.infra.retained_replay import replay_retained_components

    with write_lease("test.stable-block-retained-seed", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_AI,
                payload=json.dumps(_export(["X", "A", "B"])).encode(),
                source_path="stable-blocks.json",
                canonical_source_path="stable-blocks.json",
                acquired_at_ms=1,
                revision=RawRevisionEnvelope(
                    logical_source_key="claude-ai-export:stable-blocks",
                    kind=RawRevisionKind.FULL,
                    source_revision="retained-v1",
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )
            session_id = write_index_session(archive, _session(["A", "B"]))
            block_id = str(
                archive._conn.execute(
                    "SELECT block_id FROM blocks WHERE session_id=? AND text='B'", (session_id,)
                ).fetchone()[0]
            )
            archive.save_annotation("retained-stable-note", "block", block_id, "Keep B", owner_session_id=session_id)
            archive.commit()
    with closing(sqlite3.connect(tmp_path / "user.db")) as user:
        before_user = tuple(user.execute("SELECT * FROM assertions").fetchone())
    generations = IndexGenerationStore.for_archive_root(tmp_path)
    with write_lease("test.stable-block-candidate", archive_root=tmp_path):
        cold_build = ColdBuildGeneration.begin(
            tmp_path,
            reason="test-stable-block",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent"),)),
        )
    candidate = cold_build.generation
    register_cold_build_generation(cold_build)
    try:
        with write_lease("test.stable-block-destination", archive_root=tmp_path), cold_build.open_writer():
            pass
        result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id], owned_generation=candidate)
        with write_lease("test.stable-block-readiness", archive_root=tmp_path):
            cold_build.prepare_promotion_candidate()
    finally:
        clear_cold_build_generation()
    assert result.replayed_logical_sources == 1
    with generations.prepare_promotion(candidate) as prepared:
        with write_lease("test.stable-block-promote", archive_root=tmp_path):
            generations.promote(candidate, prepared)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert tuple(
            archive._conn.execute("SELECT text,position FROM blocks WHERE block_id=?", (block_id,)).fetchone()
        ) == (
            "B",
            2,
        )
    with closing(sqlite3.connect(tmp_path / "user.db")) as user:
        assert tuple(user.execute("SELECT * FROM assertions").fetchone()) == before_user
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.operations.source_target_read import (
        SourceTargetChangedError,
        bind_source_block,
        revalidate_source_block,
    )

    with open_operation_read(tmp_path) as snapshot:
        bind_source_block(snapshot, session_id=session_id, block_id=block_id)
        if source_change is not None:
            # Sabotage only this synthetic fixture after the actual reader pin.
            with closing(sqlite3.connect(tmp_path / "source.db")) as source:
                if source_change == "replace":
                    source.execute("UPDATE raw_sessions SET blob_hash=? WHERE raw_id=?", (bytes(32), raw_id))
                else:
                    source.execute("DELETE FROM raw_sessions WHERE raw_id=?", (raw_id,))
                source.commit()
        with write_lease("test.source-target-current", archive_root=tmp_path):
            with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
                if source_change is None:
                    revalidate_source_block(snapshot, archive, session_id=session_id, block_id=block_id)
                else:
                    with pytest.raises(SourceTargetChangedError):
                        revalidate_source_block(snapshot, archive, session_id=session_id, block_id=block_id)
    with closing(sqlite3.connect(tmp_path / "user.db")) as user:
        assert tuple(user.execute("SELECT * FROM assertions").fetchone()) == before_user
