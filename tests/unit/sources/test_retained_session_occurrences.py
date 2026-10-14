"""Repeated parser occurrences retain evidence independently of logical membership."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.accepted_marker_inputs import read_accepted_marker_inputs_sync, verified_marker_payload_from_blob
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_parser_payloads import _chatgpt_session
from tests.infra.retained_replay import replay_retained_components


def test_identical_export_occurrences_converge_one_member_and_keep_request_evidence(tmp_path: Path) -> None:
    """An identity-keyed census collision must not refuse repeated identical export members."""
    bootstrap_archive_root(tmp_path)
    session = _chatgpt_session("repeated", "::note: retained occurrence")
    payload = json.dumps([session, session, session], sort_keys=True).encode()
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="synthetic/repeated-conversations.json",
            canonical_source_path="synthetic/repeated-conversations.json",
            acquired_at_ms=1,
        )
    result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    assert result.quarantined == 0
    assert result.replayed_logical_sources == 1
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.count_sessions() == 1
        members = archive.source_connection.execute(
            "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=?", (raw_id,)
        ).fetchall()
        assert [tuple(row) for row in members] == [("chatgpt-export:repeated",)]
        accepted = read_accepted_marker_inputs_sync(archive.source_connection, limit=10)
        assert len(accepted) == 1
        receipt = accepted[0].batch
        carrier = verified_marker_payload_from_blob(archive.source_connection, accepted[0])
        try:
            requests = list(carrier.iter_items("request_sessions.item"))
            assert len(requests) == 3
            assert requests[0] == requests[1] == requests[2]
        finally:
            carrier.close()
    replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert [entry.batch for entry in read_accepted_marker_inputs_sync(archive.source_connection, limit=10)] == [
            receipt
        ]


@pytest.mark.parametrize("reverse", [False, True])
def test_contained_occurrences_keep_each_marker_source_and_select_full_content(tmp_path: Path, reverse: bool) -> None:
    """Array order cannot select the shorter member or rebind its marker body."""
    bootstrap_archive_root(tmp_path)
    small = _chatgpt_session("growth", "::note: original")
    full = _chatgpt_session("growth", "::note: original", "::note: added")
    originals = [small, full, full]
    if reverse:
        originals.reverse()
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(originals, sort_keys=True).encode(),
            source_path="synthetic/growth.json",
            canonical_source_path="synthetic/growth.json",
            acquired_at_ms=1,
        )
    result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    assert result.quarantined == 0
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.count_sessions() == 1
        assert archive._conn.execute("SELECT message_count FROM sessions").fetchone()[0] == 2
        accepted = read_accepted_marker_inputs_sync(archive.source_connection, limit=10)
        assert len(accepted) == 1
        carrier = verified_marker_payload_from_blob(archive.source_connection, accepted[0])
        try:
            requests = list(carrier.iter_items("request_sessions.item"))
            frames = cast(list[dict[str, Any]], list(carrier.iter_items("sessions.item")))
            assert len(requests) == len(frames) == 3
            bodies = [[candidate["match"]["body"] for candidate in frame["candidates"]] for frame in frames]
            assert bodies == (
                [["original", "added"], [], ["original"]] if reverse else [["original"], ["original", "added"], []]
            )
        finally:
            carrier.close()


def test_conflicting_occurrences_keep_material_presence_and_visible_source_debt(tmp_path: Path) -> None:
    """A single logical membership must not hide its internal conflicting occurrence."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(
                [
                    _chatgpt_session("fork", "::note: first"),
                    _chatgpt_session("fork", "::note: second"),
                    _chatgpt_session("fork", "::note: second"),
                ],
                sort_keys=True,
            ).encode(),
            source_path="synthetic/fork.json",
            canonical_source_path="synthetic/fork.json",
            acquired_at_ms=1,
        )
    result = replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    assert result.quarantined == 1
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.count_sessions() == 1
        assert (
            archive.source_connection.execute(
                "SELECT decision FROM raw_session_memberships WHERE raw_id=?", (raw_id,)
            ).fetchone()[0]
            == "ambiguous"
        )
        accepted = read_accepted_marker_inputs_sync(archive.source_connection, limit=10)
        assert len(accepted) == 1
        receipt = accepted[0].batch
        carrier = verified_marker_payload_from_blob(archive.source_connection, accepted[0])
        try:
            frames = cast(list[dict[str, Any]], list(carrier.iter_items("sessions.item")))
            assert [[candidate["match"]["body"] for candidate in frame["candidates"]] for frame in frames] == [
                ["first"],
                ["second"],
                [],
            ]
        finally:
            carrier.close()
    replay_retained_components(tmp_path, selected_raw_ids=[raw_id])
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.count_sessions() == 1
        assert [entry.batch for entry in read_accepted_marker_inputs_sync(archive.source_connection, limit=10)] == [
            receipt
        ]
