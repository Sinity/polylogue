"""Source-native occurrence claims remain separate from unique row identity."""

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    AssertedBranchPointAmbiguousError,
    read_archive_session_envelope,
)
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


def _session(
    name: str, pairs: list[tuple[str, str]], *, parent: str | None = None, anchor: str | None = None
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=name,
        title=name,
        parent_session_provider_id=parent,
        branch_type=BranchType.FORK if parent else None,
        branch_point_provider_message_id=anchor,
        messages=[
            ParsedMessage(
                provider_message_id=native,
                role=Role.USER,
                text=text,
                position=i,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            )
            for i, (native, text) in enumerate(pairs)
        ],
    )


def test_asserted_native_name_refuses_two_distinct_rows_in_same_parent(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [("dup", "first"), ("dup", "second")]))
        rows = conn.execute(
            "SELECT native_id, message_id FROM messages WHERE session_id=? ORDER BY position", (parent,)
        ).fetchall()
        assert len(rows) == 2 and all(row[0] is None and ":c:" in row[1] for row in rows)
        envelope = read_archive_session_envelope(conn, parent)
        assert envelope.lineage_complete and [row.blocks[0].text for row in envelope.messages] == ["first", "second"]
        with pytest.raises(AssertedBranchPointAmbiguousError) as refused:
            write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor="dup"))
        assert refused.value.code == "asserted_branch_point_ambiguous"


def test_duplicate_native_outside_composed_cut_does_not_refuse_unique_source_occurrence(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = _session("p", [("dup", "first"), ("anchor", "middle"), ("dup", "outside")])
        write_fixture_index_session(conn, parent)
        write_fixture_index_session(
            conn, _session("cut", [("dup", "first"), ("anchor", "middle"), ("own", "own")], parent="p")
        )
        child = write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="cut", anchor="dup"))
        envelope = read_archive_session_envelope(conn, child)
        assert envelope.lineage_complete
        assert [row.blocks[0].text for row in envelope.messages] == ["first", "tail"]


def test_exact_source_names_survive_injective_native_storage(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [("dup\ud800", "first"), ("dup\ud801", "second")]))
        from polylogue.core.identity_law import message_id

        assert {row[0] for row in conn.execute("SELECT message_id FROM messages WHERE session_id=?", (parent,))} == {
            message_id(parent, "dup\ud800"),
            message_id(parent, "dup\ud801"),
        }
        child = write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor="dup\ud800"))
        envelope = read_archive_session_envelope(conn, child)
        assert envelope.lineage_complete
        assert [row.blocks[0].text for row in envelope.messages] == ["first", "tail"]


def test_generated_content_local_id_is_not_source_native_evidence(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [("", "first"), ("", "second")]))
        local = str(
            conn.execute("SELECT message_id FROM messages WHERE session_id=? ORDER BY position", (parent,)).fetchone()[
                0
            ]
        ).removeprefix(parent + ":")
        write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor=local))
        assert (
            conn.execute(
                "SELECT branch_point_message_id FROM session_links WHERE src_session_id='codex-session:child'"
            ).fetchone()[0]
            is None
        )


def test_duplicate_source_occurrences_remain_after_append_and_full_replay(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = _session("p", [("dup", "first"), ("dup", "second")])
        write_fixture_index_session(conn, parent)
        write_fixture_index_session(conn, _session("p", [("tail", "later")]), merge_append=True)
        replay = _session("p", [("dup", "first"), ("dup", "second"), ("tail", "later")])
        write_fixture_index_session(conn, replay, force_replace=True)
        with pytest.raises(AssertedBranchPointAmbiguousError):
            write_fixture_index_session(conn, _session("child", [("own", "own")], parent="p", anchor="dup"))


@pytest.mark.parametrize("native", ["dup", "dup\ud800"])
def test_publication_roundtrip_preserves_duplicate_source_assertion_refusal(tmp_path: Path, native: str) -> None:
    from polylogue.material_protocol.v1 import RevisionManifest, decode_session_revision, verify_revision
    from polylogue.sinex.material_adapter import encode_parsed_session_publication

    original = _session("p", [(native, "first"), (native, "second")])
    payload = encode_parsed_session_publication(original, session_id="codex-session:p")
    manifest = RevisionManifest.from_dict(json.loads(payload.manifest_bytes))
    names = dict(payload.segments)
    segments = {d.index: names[d.filename] for d in (*manifest.segments, manifest.head_segment)}
    verify_revision(manifest, segments)
    decoded = decode_session_revision(manifest, segments)
    assert [m.native_id for m in decoded.messages] == [None, None]
    assert [m.source_native_id for m in decoded.messages] == [native, native]
    # No Index importer for material records exists: exercise the actual writer
    # with the decoded Source carrier rather than canonical content row IDs.
    restored = _session("p", [(m.source_native_id or "", m.text or "") for m in decoded.messages])
    with fixture_index_connection(tmp_path / "index.db") as conn:
        write_fixture_index_session(conn, restored)
        with pytest.raises(AssertedBranchPointAmbiguousError):
            write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor=native))


@pytest.mark.parametrize("native", ["dup", "dup\ud800\udc00"])
def test_partial_acquisition_cannot_erase_original_duplicate_native_ambiguity(tmp_path: Path, native: str) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(
            conn, _session("p", [(native, "first"), (native, "second")]), raw_id="raw-first"
        )
        from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
        from polylogue.storage.sqlite.archive_tiers.write import (
            _message_row_id,
            _union_with_existing_rows,
            prepare_session_rows,
        )

        incoming = _session("p", [(native, "first")])
        incoming_rows = prepare_session_rows(incoming)
        inline_messages, inline_blocks, carry = _union_with_existing_rows(
            conn,
            parent,
            list(incoming_rows.message_rows),
            list(incoming_rows.block_rows),
            raw_id="raw-partial",
            existing_raw_id="raw-first",
        )
        columns = [c.name for c in archive_tiers_specs.MESSAGES_SPEC.writable_columns if c.extract_placeholder == "?"]
        indexes = {name: i for i, name in enumerate(columns)}
        inline_ids = [_message_row_id(parent, row, indexes) for row in inline_messages]
        assert carry is not None and len(inline_blocks) == 2
        write_fixture_index_session(conn, incoming, raw_id="raw-partial")
        assert [
            row[0]
            for row in conn.execute("SELECT message_id FROM messages WHERE session_id=? ORDER BY position", (parent,))
        ] == inline_ids
        envelope = read_archive_session_envelope(conn, parent)
        assert envelope.lineage_complete
        assert [m.blocks[0].text for m in envelope.messages] == ["first", "second"]
        with pytest.raises(AssertedBranchPointAmbiguousError):
            write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor=native))


@pytest.mark.parametrize("previous_duplicates", [True, False])
def test_normalization_change_keeps_original_owner_for_incoming_projections(
    tmp_path: Path, previous_duplicates: bool
) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        previous = [("dup", "first"), ("dup", "second")] if previous_duplicates else [("dup", "first")]
        parent = write_fixture_index_session(conn, _session("p", previous), raw_id="raw-first")
        original_owner = conn.execute(
            "SELECT message_id FROM messages WHERE session_id=? ORDER BY position LIMIT 1", (parent,)
        ).fetchone()[0]
        incoming = _session("p", [("dup", "first")] if previous_duplicates else [("dup", "first"), ("dup", "second")])
        incoming.messages[0].is_active_leaf = True
        incoming.attachments = [
            ParsedAttachment(
                provider_attachment_id="artifact", message_provider_id="dup", message_position=0, name="neutral.txt"
            )
        ]
        incoming.session_events = [
            ParsedSessionEvent(
                event_type="compaction",
                owner_coordinate=MessageOwnerCoordinate(position=0),
                payload={"summary": "neutral"},
            )
        ]
        write_fixture_index_session(conn, incoming, raw_id="raw-second")
        assert (
            conn.execute("SELECT active_leaf_message_id FROM sessions WHERE session_id=?", (parent,)).fetchone()[0]
            == original_owner
        )
        assert (
            conn.execute("SELECT message_id FROM attachment_refs WHERE session_id=?", (parent,)).fetchone()[0]
            == original_owner
        )
        assert (
            conn.execute("SELECT source_message_id FROM session_events WHERE session_id=?", (parent,)).fetchone()[0]
            == original_owner
        )
        assert [m.blocks[0].text for m in read_archive_session_envelope(conn, parent).messages] == ["first", "second"]


def test_same_content_with_different_source_names_remains_two_occurrences(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [("a", "same")]), raw_id="raw-a")
        write_fixture_index_session(conn, _session("p", [("b", "same")]), raw_id="raw-b")
        write_fixture_index_session(conn, _session("p", [("a", "same")]), raw_id="raw-a-again")
        assert [m.message_id for m in read_archive_session_envelope(conn, parent).messages] == [
            parent + ":n:a",
            parent + ":n:b",
        ]
        assert conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (parent,)).fetchone()[0] == 2


@pytest.mark.parametrize("native, expected", [(" dup ", ["first", "tail"]), ("dup", ["first", "second", "tail"])])
def test_original_source_spelling_stays_distinct_in_native_identity_and_anchor(
    tmp_path: Path, native: str, expected: list[str]
) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [(" dup ", "first"), ("dup", "second")]))
        assert [
            row[0]
            for row in conn.execute("SELECT native_id FROM messages WHERE session_id=? ORDER BY position", (parent,))
        ] == [" dup ", "dup"]
        child = write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor=native))
        envelope = read_archive_session_envelope(conn, child)
        assert envelope.lineage_complete
        assert [m.blocks[0].text for m in envelope.messages] == expected


def test_unbacked_asserted_source_name_stays_incomplete_until_parent_supplies_it(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        write_fixture_index_session(conn, _session("p", [("before", "before")]))
        child = write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor="later"))
        pending = read_archive_session_envelope(conn, child)
        assert not pending.lineage_complete
        assert pending.lineage_truncation_reason == "dangling_branch_point"
        assert [m.blocks[0].text for m in pending.messages] == ["tail"]
        write_fixture_index_session(conn, _session("p", [("before", "before"), ("later", "later")]))
        settled = read_archive_session_envelope(conn, child)
        assert settled.lineage_complete
        assert [m.blocks[0].text for m in settled.messages] == ["before", "later", "tail"]


@pytest.mark.parametrize("anchor,expected", [("\ud800\udc00", 1), ("\U00010000", 2), ("eda080edb080", 3)])
def test_partial_union_preserves_exact_native_namespaces_and_assertion_cut(
    tmp_path: Path, anchor: str, expected: int
) -> None:
    from polylogue.core.identity_law import message_id

    names = ["\ud800\udc00", "\U00010000", "eda080edb080"]
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [(names[0], "same")]), raw_id="first")
        for i, native in enumerate(names[1:], 1):
            write_fixture_index_session(conn, _session("p", [(native, "same")]), raw_id=f"partial-{i}")
        assert {row[0] for row in conn.execute("SELECT message_id FROM messages WHERE session_id=?", (parent,))} == {
            message_id(parent, name) for name in names
        }
        child = write_fixture_index_session(conn, _session("child", [("tail", "tail")], parent="p", anchor=anchor))
        envelope = read_archive_session_envelope(conn, child)
        assert envelope.lineage_complete
        assert len(envelope.messages) == expected + 1


@pytest.mark.parametrize("native", ["\ud800", "\ud800\udc00", "\U00010000", "eda080edb080"])
def test_stored_message_rehash_preserves_original_native_framing(tmp_path: Path, native: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.write import _rehash_session_messages

    with fixture_index_connection(tmp_path / "index.db") as conn:
        sid = write_fixture_index_session(conn, _session("p", [(native, "same")]))
        before = conn.execute("SELECT content_hash FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
        _rehash_session_messages(conn, sid)
        assert conn.execute("SELECT content_hash FROM messages WHERE session_id=?", (sid,)).fetchone()[0] == before
