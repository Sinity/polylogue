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


def test_stable_mark_can_be_retracted_after_block_disappears(tmp_path: Path) -> None:
    from polylogue.operations.mutation_actuators import MarkArgs, MarkRemoveActuator
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.operations.user_overlay_mutations import _target

    with write_lease("test.stable-block-retraction", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            session_id = write_index_session(archive, _session(["A", "B"]))
            block_id = str(archive._conn.execute("SELECT block_id FROM blocks WHERE text='B'").fetchone()[0])
            archive.add_mark("block", block_id, "star", owner_session_id=session_id)
            archive._conn.execute("DELETE FROM blocks WHERE block_id=?", (block_id,))
            archive.commit()
    with open_operation_read(tmp_path) as snapshot:
        target = _target(
            snapshot,
            tmp_path,
            {"session_id": session_id, "target_type": "block", "target_id": block_id},
            require_present=False,
        )
        with write_lease("test.stable-block-retraction-apply", archive_root=tmp_path):
            with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
                actuator = MarkRemoveActuator()
                args = MarkArgs(archive, target[0], target[1], "star", target[2])
                plan = actuator.prepare(args)
                assert plan.target_refs == (f"block:{block_id}",)
                assert actuator.apply(plan, args).affected_count == 1
                archive.commit()
                assert not list(archive.list_marks(target_type="block", target_id=block_id))


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
            with closing(archive._open_user_write_connection(initialize=True)) as user:
                assertion_count = int(user.execute("SELECT COUNT(*) FROM assertions").fetchone()[0])
            with pytest.raises(ValueError, match="generated stable block_id"):
                archive.save_annotation(
                    "new-opaque-block-ref", "block", "legacy-opaque", "Must not persist", owner_session_id=session_id
                )
            with closing(archive._open_user_write_connection(initialize=True)) as user:
                assert int(user.execute("SELECT COUNT(*) FROM assertions").fetchone()[0]) == assertion_count
                user.execute(
                    "INSERT INTO assertions(assertion_id,target_ref,kind,body_text,created_at_ms,updated_at_ms) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        "historical-positional-assertion",
                        "block:historical-opaque-token",
                        "annotation",
                        "Retain historical bytes without rebound",
                        1,
                        1,
                    ),
                )
                historical_assertion = tuple(
                    user.execute(
                        "SELECT * FROM assertions WHERE assertion_id='historical-positional-assertion'"
                    ).fetchone()
                )
                user.commit()
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
        assert (
            tuple(
                user.execute("SELECT * FROM assertions WHERE assertion_id='historical-positional-assertion'").fetchone()
            )
            == historical_assertion
        )
    from polylogue.operations.mutation_actuators import AnnotationSaveActuator, AnnotationSaveArgs
    from polylogue.operations.operation_context import open_operation_read
    from polylogue.operations.source_target_read import (
        SourceTargetChangedError,
    )
    from polylogue.operations.user_overlay_mutations import _source_guard, _target

    with open_operation_read(tmp_path) as snapshot:
        message_id = str(
            snapshot.archive._conn.execute("SELECT message_id FROM blocks WHERE block_id=?", (block_id,)).fetchone()[0]
        )
        target = _target(
            snapshot,
            tmp_path,
            {"session_id": session_id, "target_type": "block", "target_id": f"{message_id}:2"},
        )
        assert target == ("block", block_id, session_id, message_id)
        with write_lease("test.source-target-current", archive_root=tmp_path):
            with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
                actuator = AnnotationSaveActuator()
                args = AnnotationSaveArgs(
                    archive,
                    "new-source-note",
                    "block",
                    block_id,
                    "Still B",
                    session_id,
                    source_guard=_source_guard(snapshot, archive, target),
                )
                plan = actuator.prepare(args)
                if source_change is not None:
                    # Sabotage only this synthetic fixture after real PREPARE.
                    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
                        if source_change == "replace":
                            source.execute("UPDATE raw_sessions SET blob_hash=? WHERE raw_id=?", (bytes(32), raw_id))
                        else:
                            source.execute("DELETE FROM raw_sessions WHERE raw_id=?", (raw_id,))
                        source.commit()
                if source_change is None:
                    receipt = actuator.apply(plan, args)
                    assert receipt.affected_count == 1
                    archive.commit()
                    assert archive.get_annotation("new-source-note")["target_id"] == block_id
                else:
                    with pytest.raises(SourceTargetChangedError):
                        actuator.apply(plan, args)
                    assert archive.get_annotation("new-source-note") is None
    with closing(sqlite3.connect(tmp_path / "user.db")) as user:
        assert (
            tuple(user.execute("SELECT * FROM assertions WHERE assertion_id=?", (before_user[0],)).fetchone())
            == before_user
        )
