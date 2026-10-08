"""The retained marker producer seals every accepted raw's source history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputExcisedError,
    marker_input_excision_targets_sync,
    read_accepted_marker_inputs_sync,
)
from polylogue.storage.accepted_marker_producer import (
    accepted_marker_input_is_durable,
    prepare_accepted_marker_carrier,
    stage_accepted_marker_carrier,
)
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_replay import publish_prepared_source

if TYPE_CHECKING:
    from polylogue.storage.accepted_marker_producer import PreparedAcceptedMarkerCarrier
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


def _facts(revision: str) -> dict[str, object]:
    return {
        "blob_hash": revision * 64,
        "provider": "codex",
        "revision_kind": "full",
        "source_path": "sessions.jsonl",
        "parser_fingerprint": "p" * 64,
        "marker_recipe": "m" * 64,
    }


def _carrier(raw_id: str, revision: str) -> PreparedAcceptedMarkerCarrier:
    request: tuple[Mapping[str, object], ...] = (
        {"session_id": f"CODEX_SESSION:{raw_id}", "provider_session_id": raw_id},
    )
    return prepare_accepted_marker_carrier(
        raw_id=raw_id,
        request_facts=_facts(revision),
        request_sessions=lambda: iter(request),
        prepared_sessions=iter(()),
    )


def _publish(root: Path, raw_id: str, revision: str) -> None:
    def prepare(seal: PreparedIndexMutation) -> None:
        stage_accepted_marker_carrier(seal, _carrier(raw_id, revision))

    publish_prepared_source(root, "test.accepted-marker", prepare)


def _publish_many(root: Path, revisions: tuple[tuple[str, str], ...]) -> None:
    def prepare(seal: PreparedIndexMutation) -> None:
        for raw_id, revision in revisions:
            stage_accepted_marker_carrier(seal, _carrier(raw_id, revision))

    publish_prepared_source(root, "test.accepted-marker.batch", prepare)


def test_existing_accepted_identity_can_be_reused_without_preparing_writer_rows(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _publish(tmp_path, "revision-1", "1")
    observed: list[bool] = []

    def inspect(seal: PreparedIndexMutation) -> None:
        observed.append(
            accepted_marker_input_is_durable(
                seal,
                raw_id="revision-1",
                request_facts=_facts("1"),
                request_sessions=lambda: iter(
                    ({"session_id": "CODEX_SESSION:revision-1", "provider_session_id": "revision-1"},)
                ),
            )
        )
        observed.append(
            accepted_marker_input_is_durable(
                seal,
                raw_id="revision-2",
                request_facts=_facts("2"),
                request_sessions=lambda: iter(({"session_id": "CODEX_SESSION:revision-2"},)),
            )
        )

    publish_prepared_source(tmp_path, "test.accepted-marker.identity-check", inspect)

    assert observed == [True, False]


def test_accepted_revision_carriers_survive_reordered_replay_and_restart(tmp_path: Path) -> None:
    """Earlier source marker evidence remains deliverable after a newer revision."""
    bootstrap_archive_root(tmp_path)
    _publish(tmp_path, "revision-2", "2")
    _publish(tmp_path, "revision-1", "1")
    _publish(tmp_path, "revision-2", "2")

    with sqlite3.connect(tmp_path / "source.db") as source:
        page = read_accepted_marker_inputs_sync(source, limit=10)
        assert [entry.batch.raw_id for entry in page] == ["revision-2", "revision-1"]
        assert [entry.sequence for entry in page] == [1, 2]
        assert all(entry.batch.payload_sha256 for entry in page)
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        stream_id = source.execute("SELECT stream_id FROM accepted_marker_stream WHERE singleton=1").fetchone()[0]

    # A new seal/process sees the same immutable stream and the same sequence.
    _publish(tmp_path, "revision-1", "1")
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT stream_id FROM accepted_marker_stream WHERE singleton=1").fetchone() == (
            stream_id,
        )
        assert source.execute("SELECT raw_id,sequence FROM accepted_marker_inputs ORDER BY sequence").fetchall() == [
            ("revision-2", 1),
            ("revision-1", 2),
        ]


def test_excised_accepted_revision_refuses_marker_reproduction(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _publish(tmp_path, "revision-1", "1")
    with write_lease("test.accepted-marker.excision", archive_root=tmp_path):
        initialize_active_archive_root(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as source:
            targets = marker_input_excision_targets_sync(
                source,
                target_session_ids=frozenset({"CODEX_SESSION:revision-1"}),
                target_raw_ids=frozenset({"revision-1"}),
            )
            assert len(targets) == 1
            from polylogue.storage.accepted_marker_inputs import excise_marker_input_targets_sync

            excise_marker_input_targets_sync(source, targets, excised_at_ms=1)

    with pytest.raises(AcceptedMarkerInputExcisedError):
        _publish(tmp_path, "revision-1", "1")
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM excised_marker_inputs").fetchone() == (1,)


def test_multiple_carriers_share_one_stream_root_in_a_source_seal(tmp_path: Path) -> None:
    """One source producer can stage multiple raw histories in one permit."""
    bootstrap_archive_root(tmp_path)
    _publish_many(tmp_path, (("revision-1", "1"), ("revision-2", "2")))
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_stream").fetchone() == (1,)
        assert source.execute("SELECT raw_id,sequence FROM accepted_marker_inputs ORDER BY sequence").fetchall() == [
            ("revision-1", 1),
            ("revision-2", 2),
        ]


def test_carrier_candidates_come_from_the_canonical_prepared_write(tmp_path: Path) -> None:
    """The source candidate coordinates match the rows the writer will publish."""
    bootstrap_archive_root(tmp_path)
    session_id = "codex-session:marker-producer"
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="marker-producer",
        created_at="2026-07-19T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="marker-message",
                role=Role.ASSISTANT,
                text="::note: accepted marker body",
            )
        ],
    )
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.row_factory = sqlite3.Row
        prepared = prepare_session_write(index, session, merge_append=False)
        carrier = prepare_accepted_marker_carrier(
            raw_id="revision-1",
            request_facts=_facts("1"),
            request_sessions=lambda: iter(({"session_id": session_id},)),
            prepared_sessions=((session_id, prepared, ()),),
        )
        try:
            payload = json.loads(b"".join(carrier.chunks()))
            candidates = payload["sessions"][0]["candidates"]
            columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
            prepared_block = dict(zip(columns, prepared.rows.block_rows[0], strict=True))
            assert len(candidates) == 1
            assert candidates[0]["match"]["body"] == "accepted marker body"
            assert candidates[0]["provenance"]["message_id"] == prepared_block["message_id"]
        finally:
            carrier.close()
            prepared.close()
