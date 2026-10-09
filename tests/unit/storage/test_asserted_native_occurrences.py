"""Source-native occurrence claims remain separate from unique row identity."""

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
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


def test_exact_source_names_survive_unique_identity_surrogate_substitution(tmp_path: Path) -> None:
    with fixture_index_connection(tmp_path / "index.db") as conn:
        parent = write_fixture_index_session(conn, _session("p", [("dup\ud800", "first"), ("dup\ud801", "second")]))
        assert all(
            row[0] is None for row in conn.execute("SELECT native_id FROM messages WHERE session_id=?", (parent,))
        )
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
