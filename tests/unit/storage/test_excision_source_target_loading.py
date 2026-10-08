"""Canonical begun-preview Source target reads retain inputs without effects."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from polylogue.core.enums import IngestOutcome
from polylogue.security.excision import (
    _load_excision_source_target,
    _PreparedExcisionBlobSourceRead,
    _stage_excision_source_target,
)
from polylogue.storage.accepted_marker_inputs import (
    append_accepted_marker_input,
    persist_pending_marker_input_sync,
    prepare_accepted_marker_input,
)
from polylogue.storage.blob_liveness import (
    BLOB_OWNERS,
    LivenessState,
    inspect_blob_liveness,
    inspect_session_blob_references,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor, settle_connection_cursors
from polylogue.storage.sqlite.archive_tiers.source_items import (
    AcquisitionDisposition,
    SourceItemMemberDisposition,
    publish_source_generation,
    record_source_item_member_disposition,
    record_source_item_raw_member,
    transition_source_item,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, write_source_hook_event
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision_execution import begin_excision_control
from tests.infra.storage_records import SessionBuilder
from tests.infra.sync_as_async import AsyncConnectionView


@dataclass(frozen=True)
class SourceFixture:
    root: Path
    session_id: str
    item_id: str
    hashes: dict[str, bytes]
    rowids: dict[tuple[str, str], int]
    original_rows: dict[str, tuple[tuple[object, ...], ...]]
    marker_payloads: dict[str, bytes]
    nullable_item_id: str | None
    source_columns: dict[str, tuple[str, ...]]


@pytest.fixture
def source_fixture(tmp_path: Path, request: pytest.FixtureRequest) -> SourceFixture:
    """Real canonical producers, current Source DDL and neutral retained bytes."""
    nullable = bool(getattr(request, "param", False))
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.source-target.fixture", archive_root=root):
        bootstrap_archive_root(root)
        builder = SessionBuilder(root / "index.db", "source-load").provider("codex").add_message(text="Neutral")
        builder.save()
        session_id = builder.native_session_id()
        origin, _, native = session_id.partition(":")
        payloads = {
            "raw": b"Neutral raw",
            "hook": b"Neutral hook",
            "container": b"Neutral container",
            "material": b"Neutral material",
            "foreign-hook": b"Foreign hook",
            "sidecar": b"Neutral sidecar",
        }
        hashes = {
            key: bytes.fromhex(BlobStore(root / "blob").write_from_bytes(value)[0]) for key, value in payloads.items()
        }
        with closing(connect_measured(root / "source.db")) as source:
            with connection_cursor(source, "PRAGMA foreign_keys=ON"):
                pass
            with connection_cursor(
                source,
                "INSERT INTO raw_sessions(raw_id,origin,native_id,source_path,blob_hash,blob_size,acquired_at_ms) VALUES ('raw-main',?,?,?,?,?,1)",
                (origin, native, "synthetic/raw", hashes["raw"], len(payloads["raw"])),
            ):
                pass
            with connection_cursor(
                source,
                "INSERT INTO blob_refs(blob_hash,ref_type,ref_id,source_path,size_bytes,acquired_at_ms) VALUES (?,'raw_payload','raw-main','synthetic/raw',?,1)",
                (hashes["raw"], len(payloads["raw"])),
            ):
                pass
            source.commit()
            for event_id, event_native, key in (
                ("hook-main", native, "hook"),
                ("raw-main", "another-native", "foreign-hook"),
            ):
                event = ArchiveHookEvent(
                    event_id,
                    origin,
                    f"synthetic/{key}",
                    "PostToolUse",
                    {"neutral": True},
                    1,
                    session_native_id=event_native,
                )
                write_source_hook_event(
                    source,
                    origin=origin,
                    source_path=event.source_path,
                    payload=payloads[key],
                    acquired_at_ms=1,
                    raw_id=f"{key}-observation",
                    hook_event=event,
                    carrier_source_id=f"synthetic-{key}",
                    carrier_relative_path=f"{key}.json",
                )
            [item_id] = publish_source_generation(
                source,
                source_generation_id="generation-main",
                manifest_digest="a" * 64,
                addressing_mode="physical-file-v1",
                coordinates=("container.json",),
                observed_at_ms=1,
                input_blob_hashes={"container.json": hashes["container"]},
                enumeration_fingerprint="b" * 64,
            )
            nullable_item_id: str | None = None
            if nullable:
                [nullable_item_id] = publish_source_generation(
                    source,
                    source_generation_id="generation-null",
                    manifest_digest="c" * 64,
                    addressing_mode="physical-file-v1",
                    coordinates=("nullable-container.json",),
                    observed_at_ms=1,
                    input_blob_hashes=None,
                    enumeration_fingerprint=None,
                )
                transition_source_item(
                    source,
                    source_generation_id="generation-null",
                    source_item_id=nullable_item_id,
                    request_id="nullable-admission",
                    disposition=AcquisitionDisposition.ADMITTED,
                    outcome_code=IngestOutcome.SUCCESS,
                    stage="acquire",
                    observed_at_ms=1,
                    raw_id="raw-main",
                    blob_hash=None,
                )
            with connection_cursor(source, "BEGIN IMMEDIATE"):
                pass
            record_source_item_raw_member(
                source,
                source_generation_id="generation-main",
                source_item_id=item_id,
                record_coordinate="record:0",
                raw_id="raw-main",
                raw_blob_hash=hashes["raw"],
            )
            record_source_item_member_disposition(
                source,
                source_generation_id="generation-main",
                source_item_id=item_id,
                entry_ordinal=1,
                member_name="unselected",
                disposition=SourceItemMemberDisposition.UNSELECTED,
                diagnostic="synthetic",
                observed_at_ms=1,
            )
            with connection_cursor(
                source,
                "INSERT INTO material_observations(material_id,referrer_ref,source_uri,acquisition_state,retryable,blob_hash,byte_size,custody,privacy_classification,acquired_at_ms,created_at_ms) VALUES ('material-main',?,'synthetic/material',?,0,?,?,?,?,1,1)",
                (
                    session_id,
                    "claimed" if nullable else "acquired",
                    None if nullable else hashes["material"],
                    None if nullable else len(payloads["material"]),
                    "claimed" if nullable else "retained",
                    "synthetic" if nullable else "private",
                ),
            ):
                pass
            with connection_cursor(
                source,
                "INSERT INTO material_evidence_links(material_id,evidence_ref,relation,authority,confidence,observed_at_ms) VALUES ('material-main',?,'refers_to','provider',1.0,1)",
                (session_id,),
            ):
                pass
            with connection_cursor(
                source,
                "INSERT INTO material_observations(material_id,referrer_ref,source_uri,acquisition_state,retryable,custody,privacy_classification,acquired_at_ms,created_at_ms,supersedes_material_id) VALUES ('incoming-material','another-session','synthetic/incoming','claimed',0,'claimed','synthetic',1,1,'material-main')",
            ):
                pass
            with connection_cursor(
                source,
                "INSERT INTO history_sidecars VALUES ('raw-main',?,'synthetic/sidecar','{}',1,?)",
                (origin, hashes["sidecar"]),
            ):
                pass
            with connection_cursor(
                source,
                "INSERT INTO blob_refs(blob_hash,ref_type,ref_id,source_path,size_bytes,acquired_at_ms) VALUES (?,'sidecar','raw-main','synthetic/sidecar',?,1)",
                (hashes["sidecar"], len(payloads["sidecar"])),
            ):
                pass
            if nullable:
                with connection_cursor(
                    source, "UPDATE raw_hook_events SET blob_hash=NULL WHERE hook_event_id='hook-main'"
                ):
                    pass
            pending = prepare_accepted_marker_input(
                "raw-main", [{"session_id": session_id, "candidates": [{"body": "Neutral pending"}]}]
            )
            accepted = prepare_accepted_marker_input(
                "raw-main",
                [{"session_id": session_id, "candidates": [{"body": "Neutral accepted"}]}],
                request_facts={"revision": "accepted"},
            )
            with closing(sqlite3.connect(root / "index.db")) as index:
                incarnation = str(uuid.uuid4())
                with connection_cursor(
                    index, "UPDATE sessions SET raw_id='raw-main' WHERE session_id=?", (session_id,)
                ):
                    pass
                index.commit()
            persist_pending_marker_input_sync(source, pending, expected_incarnation_id=incarnation)
            asyncio.run(
                append_accepted_marker_input(AsyncConnectionView(source), accepted, index_incarnation_id=incarnation)
            )
            settle_connection_cursors(source)
            source.commit()
            rowids = {}
            for table, key, identity in (
                ("raw_sessions", "raw_id", "raw-main"),
                ("raw_hook_events", "hook_event_id", "hook-main"),
                ("material_observations", "material_id", "material-main"),
                ("material_observations", "material_id", "incoming-material"),
            ):
                with connection_cursor(source, f"SELECT rowid FROM {table} WHERE {key}=?", (identity,)) as rows:
                    row = rows.fetchone()
                assert row is not None
                rowids[table, identity] = int(row[0])
            selections = {
                "raw_sessions": "raw_id='raw-main'",
                "raw_hook_events": "hook_event_id IN ('hook-main','raw-main')",
                "hook_event_carriers": "hook_event_id IN ('hook-main','raw-main')",
                "blob_refs": "ref_id IN ('hook-main','raw-main')",
                "history_sidecars": "sidecar_id='raw-main'",
                "source_items": "source_generation_id IN ('generation-main','generation-null')",
                "source_item_raw_members": "source_generation_id IN ('generation-main','generation-null')",
                "source_item_member_dispositions": "source_generation_id IN ('generation-main','generation-null')",
                "material_observations": "material_id IN ('material-main','incoming-material')",
                "material_evidence_links": "material_id='material-main'",
                "pending_accepted_marker_inputs": "raw_id='raw-main'",
                "accepted_marker_inputs": "raw_id='raw-main'",
            }
            originals = {}
            source_columns = {}
            for table, predicate in selections.items():
                with connection_cursor(source, f"SELECT rowid,* FROM {table} WHERE {predicate} ORDER BY rowid") as rows:
                    assert rows.description is not None
                    source_columns[table] = tuple(column[0] for column in rows.description)
                    originals[table] = tuple(tuple(row) for row in rows)
                assert originals[table]
    return SourceFixture(
        root,
        session_id,
        item_id,
        hashes,
        rowids,
        originals,
        {pending.identity: pending.payload, accepted.identity: accepted.payload},
        nullable_item_id,
        source_columns,
    )


def _assert_no_effects(seal: PreparedIndexMutation) -> None:
    for table in ("known_tier_effects", "known_tier_statements"):
        with connection_cursor(seal._scratch, f"SELECT count(*) FROM temp.{table}") as rows:
            assert rows.fetchone()[0] == 0
    assert seal._pending_tier_permits == {}


@pytest.mark.parametrize("source_fixture", [False, True], indirect=True)
def test_full_target_load_retains_canonical_families_labels_and_nullable_owners(source_fixture: SourceFixture) -> None:
    fixture = source_fixture
    started, _ = begin_excision_control(fixture.root, fixture.session_id, reason="synthetic")
    assert started.operation_id is not None
    with PreparedIndexMutation(fixture.root / "index.db", archive_root=fixture.root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (fixture.session_id,))
        with pytest.raises(ReferenceSealError, match="readable Source dependency is outside declared writable keys"):
            with seal.original_read_snapshot(), seal.source_producer():
                target = seal.original_excision_target(fixture.session_id)
                assert tuple(raw.raw_id for raw in target.raw_targets) == ("raw-main",)
                assert target.hook_event_ids == ("hook-main",)
                assert target.material_ids == ("material-main",)
                assert len(target.containers.members) == 1
                assert target.containers.members[0].source_item_id == fixture.item_id
                assert {
                    (item.source_generation_id, item.source_item_id) for item in target.containers.removable_items
                } == (
                    {("generation-main", fixture.item_id)}
                    | (
                        {("generation-null", fixture.nullable_item_id)}
                        if fixture.nullable_item_id is not None
                        else set()
                    )
                )
                assert {marker.state for marker in target.marker_input_targets} == {"pending", "accepted"}
                _load_excision_source_target(seal, target)
                for marker in target.marker_input_targets:
                    table, key = (
                        ("pending_accepted_marker_inputs", "request_key")
                        if marker.state == "pending"
                        else ("accepted_marker_inputs", "identity")
                    )
                    with seal.source_rows(f"SELECT payload FROM {table} WHERE {key}=?", (marker.identity,)) as rows:
                        assert bytes(rows.fetchone()[0]) == fixture.marker_payloads[marker.identity]
                _assert_no_effects(seal)
                with connection_cursor(
                    seal._scratch, "SELECT DISTINCT table_name FROM temp.original_input_rows WHERE tier='source'"
                ) as rows:
                    retained_tables = {str(row[0]) for row in rows}
                assert {
                    "raw_sessions",
                    "raw_hook_events",
                    "hook_event_carriers",
                    "blob_refs",
                    "source_items",
                    "source_item_raw_members",
                    "source_item_member_dispositions",
                    "material_observations",
                    "material_evidence_links",
                    "pending_accepted_marker_inputs",
                    "accepted_marker_inputs",
                } <= retained_tables
                candidates = dict(seal.excision_source_blob_page())
                assert fixture.hashes["raw"] in candidates
                assert seal._literal_scalar_equal(candidates[fixture.hashes["raw"]], f"{fixture.item_id}:record:0")
                assert fixture.hashes["hook"] in candidates
                assert seal._literal_scalar_equal(candidates[fixture.hashes["hook"]], "hook-main")
                main_item = next(
                    item for item in target.containers.removable_items if item.source_item_id == fixture.item_id
                )
                assert main_item.blob_hash == fixture.hashes["container"]
                assert seal._literal_scalar_equal(
                    candidates[fixture.hashes["container"]], f"generation-main:{fixture.item_id}"
                )
                if fixture.nullable_item_id is not None:
                    nullable_item = next(
                        item
                        for item in target.containers.removable_items
                        if item.source_item_id == fixture.nullable_item_id
                    )
                    assert nullable_item.blob_hash is None
                    assert fixture.hashes["material"] not in candidates
                else:
                    assert fixture.hashes["material"] in candidates
                assert set(candidates) == (
                    {fixture.hashes["raw"], fixture.hashes["hook"], fixture.hashes["container"]}
                    | ({fixture.hashes["material"]} if fixture.nullable_item_id is None else set())
                )
                assert fixture.hashes["foreign-hook"] not in candidates
                assert fixture.hashes["sidecar"] not in candidates
                with connection_cursor(
                    seal._scratch,
                    "SELECT row_address FROM temp.original_input_rows WHERE tier='source' AND table_name='material_observations'",
                ) as rows:
                    assert {int(row[0]) for row in rows} == {
                        fixture.rowids["material_observations", "material-main"],
                        fixture.rowids["material_observations", "incoming-material"],
                    }
                with seal.source_statement(
                    "UPDATE material_observations SET diagnostic='undeclared' WHERE material_id='incoming-material'",
                    table="material_observations",
                    writable_targets=(),
                ):
                    pass
    with closing(sqlite3.connect(f"file:{fixture.root / 'source.db'}?mode=ro", uri=True)) as source:
        for table, original_rows in fixture.original_rows.items():
            for original in original_rows:
                with connection_cursor(source, f"SELECT rowid,* FROM {table} WHERE rowid=?", (original[0],)) as rows:
                    assert tuple(rows.fetchone()) == original
        with connection_cursor(
            source, "SELECT rowid,diagnostic FROM material_observations WHERE material_id='incoming-material'"
        ) as rows:
            assert tuple(rows.fetchone()) == (fixture.rowids["material_observations", "incoming-material"], "")
        with connection_cursor(
            source, "SELECT rowid,source_path,blob_hash FROM raw_sessions WHERE raw_id='raw-main'"
        ) as rows:
            assert tuple(rows.fetchone()) == (
                fixture.rowids["raw_sessions", "raw-main"],
                "synthetic/raw",
                fixture.hashes["raw"],
            )


def test_equal_ref_ids_do_not_enroll_or_delete_independent_typed_owners(source_fixture: SourceFixture) -> None:
    fixture = source_fixture
    started, _ = begin_excision_control(fixture.root, fixture.session_id, reason="synthetic")
    assert started.operation_id is not None
    with PreparedIndexMutation(fixture.root / "index.db", archive_root=fixture.root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (fixture.session_id,))
        with seal.original_read_snapshot(), seal.source_producer():
            target = seal.original_excision_target(fixture.session_id)
            assert "raw-main" not in target.hook_event_ids
            _load_excision_source_target(seal, target)
            candidates = dict(seal.excision_source_blob_page())
            assert fixture.hashes["raw"] in candidates
            assert fixture.hashes["foreign-hook"] not in candidates
            assert fixture.hashes["sidecar"] not in candidates
            counts = _stage_excision_source_target(seal, target, excised_at_ms=1)
            assert counts["source_raw_rows"] == 1
            foreign = (fixture.hashes["foreign-hook"], fixture.hashes["sidecar"])
            decisions = inspect_session_blob_references(
                _PreparedExcisionBlobSourceRead(seal),
                foreign,
                index_conn=seal.observer("index"),
                excluding_session_ids=frozenset({fixture.session_id}),
            )
            assert all(decisions[blob_hash].state is LivenessState.LIVE for blob_hash in foreign)
            assert all("source.db.blob_refs" in decisions[blob_hash].surfaces for blob_hash in foreign)
            with seal.source_rows("SELECT rowid,* FROM blob_refs WHERE ref_id='raw-main' ORDER BY rowid") as rows:
                assert tuple(tuple(row) for row in rows) == tuple(
                    row
                    for row in fixture.original_rows["blob_refs"]
                    if row[2] == "raw-main" and row[3] in {"hook_payload", "sidecar"}
                )
            with seal.source_rows("SELECT rowid,* FROM raw_hook_events WHERE hook_event_id='raw-main'") as rows:
                assert tuple(rows.fetchone()) == next(
                    row for row in fixture.original_rows["raw_hook_events"] if row[1] == "raw-main"
                )
            with seal.source_rows("SELECT rowid,* FROM history_sidecars WHERE sidecar_id='raw-main'") as rows:
                assert tuple(rows.fetchone()) == fixture.original_rows["history_sidecars"][0]
            with seal.source_rows("SELECT ref_type FROM blob_refs WHERE ref_id='raw-main' ORDER BY ref_type") as rows:
                assert tuple(row[0] for row in rows) == ("hook_payload", "sidecar")
            with seal.source_rows(
                "SELECT session_native_id FROM raw_hook_events WHERE hook_event_id='raw-main'"
            ) as rows:
                assert rows.fetchone()[0] == "another-native"
            with seal.source_rows("SELECT sidecar_id FROM history_sidecars WHERE sidecar_id='raw-main'") as rows:
                assert rows.fetchone()[0] == "raw-main"
    with closing(sqlite3.connect(fixture.root / "source.db")) as source:
        for key in ("foreign-hook", "sidecar"):
            decision = inspect_blob_liveness(source, fixture.hashes[key].hex())
            assert decision.state is LivenessState.LIVE
            assert "source.db.blob_refs" in decision.surfaces


@pytest.mark.parametrize(
    "mismatch",
    [
        "hook-owner",
        "hook-origin",
        "item-hash",
        "member-hash",
        "member-owner",
        "material-referrer",
        "material-hash",
        "raw-path",
        "raw-hash",
        "marker-target",
        "hook-arrival",
        "material-arrival",
    ],
)
def test_original_identity_mismatch_refuses_before_source_effects(source_fixture: SourceFixture, mismatch: str) -> None:
    fixture = source_fixture
    started, _ = begin_excision_control(fixture.root, fixture.session_id, reason="synthetic")
    assert started.operation_id is not None
    mutations: dict[str, tuple[str, tuple[object, ...]]] = {
        "hook-owner": (
            "UPDATE raw_hook_events SET session_native_id='another-native' WHERE hook_event_id='hook-main'",
            (),
        ),
        "hook-origin": ("UPDATE raw_hook_events SET origin='claude-code' WHERE hook_event_id='hook-main'", ()),
        "member-owner": ("UPDATE source_item_raw_members SET raw_id=NULL WHERE source_item_id=?", (fixture.item_id,)),
        "item-hash": (
            "UPDATE source_items SET blob_hash=? WHERE source_item_id=?",
            (fixture.hashes["sidecar"], fixture.item_id),
        ),
        "member-hash": (
            "UPDATE source_item_raw_members SET raw_blob_hash=? WHERE source_item_id=?",
            (fixture.hashes["sidecar"], fixture.item_id),
        ),
        "material-referrer": (
            "UPDATE material_observations SET referrer_ref='another-session' WHERE material_id='material-main'",
            (),
        ),
        "material-hash": (
            "UPDATE material_observations SET blob_hash=? WHERE material_id='material-main'",
            (fixture.hashes["sidecar"],),
        ),
        "raw-hash": (
            "UPDATE raw_sessions SET blob_hash=? WHERE raw_id='raw-main'",
            (fixture.hashes["sidecar"],),
        ),
        "raw-path": ("UPDATE raw_sessions SET source_path='synthetic/changed' WHERE raw_id='raw-main'", ()),
    }
    if mismatch != "marker-target":
        with write_lease("test.source-target.mismatch", archive_root=fixture.root):
            with closing(connect_measured(fixture.root / "source.db")) as source:
                if mismatch == "hook-arrival":
                    origin, _, native = fixture.session_id.partition(":")
                    payload = b"Neutral additional hook"
                    BlobStore(fixture.root / "blob").write_from_bytes(payload)
                    event = ArchiveHookEvent(
                        "hook-arrival",
                        origin,
                        "synthetic/arrival-hook",
                        "PostToolUse",
                        {"neutral": True},
                        2,
                        session_native_id=native,
                    )
                    write_source_hook_event(
                        source,
                        origin=origin,
                        source_path=event.source_path,
                        payload=payload,
                        acquired_at_ms=2,
                        raw_id="arrival-observation",
                        hook_event=event,
                        carrier_source_id="synthetic-arrival",
                        carrier_relative_path="arrival.json",
                    )
                elif mismatch == "material-arrival":
                    with connection_cursor(
                        source,
                        "INSERT INTO material_observations(material_id,referrer_ref,source_uri,acquisition_state,retryable,custody,privacy_classification,acquired_at_ms,created_at_ms) VALUES ('material-arrival',?,'synthetic/arrival-material','claimed',0,'claimed','synthetic',2,2)",
                        (fixture.session_id,),
                    ):
                        pass
                else:
                    with connection_cursor(source, *mutations[mismatch]):
                        pass
                settle_connection_cursors(source)
                source.commit()
                if mismatch in {"hook-arrival", "material-arrival"}:
                    for table, originals in fixture.original_rows.items():
                        for original in originals:
                            with connection_cursor(
                                source, f"SELECT rowid,* FROM {table} WHERE rowid=?", (original[0],)
                            ) as rows:
                                assert tuple(rows.fetchone()) == original
    with PreparedIndexMutation(fixture.root / "index.db", archive_root=fixture.root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (fixture.session_id,))
        with seal.original_read_snapshot(), seal.source_producer():
            target = seal.original_excision_target(fixture.session_id)
            if mismatch == "marker-target":
                assert target.marker_input_targets
                marker = replace(target.marker_input_targets[0], carrier_digest="c" * 64)
                target = replace(target, marker_input_targets=(marker, *target.marker_input_targets[1:]))
            with pytest.raises(ReferenceSealError):
                _load_excision_source_target(seal, target)
            _assert_no_effects(seal)


@pytest.mark.parametrize("source_fixture", [False, True], indirect=True)
def test_canonical_target_stage_deletes_only_owned_rows_and_preserves_original_source(
    source_fixture: SourceFixture,
) -> None:
    """Actual staged cleanup consumes every family without gaining live authority."""
    fixture = source_fixture
    started, _ = begin_excision_control(fixture.root, fixture.session_id, reason="synthetic")
    assert started.operation_id is not None
    with PreparedIndexMutation(fixture.root / "index.db", archive_root=fixture.root) as seal:
        seal.bind_begun_excision(started.operation_id, started.plan.plan_hash, (fixture.session_id,))
        with seal.original_read_snapshot(), seal.source_producer():
            target = seal.original_excision_target(fixture.session_id)
            assert len(target.raw_targets) == len(target.hook_event_ids) == len(target.material_ids) == 1
            assert len(target.containers.members) == 1
            assert {marker.state for marker in target.marker_input_targets} == {"pending", "accepted"}
            assert {(item.source_generation_id, item.source_item_id) for item in target.containers.removable_items} == (
                {("generation-main", fixture.item_id)}
                | ({("generation-null", fixture.nullable_item_id)} if fixture.nullable_item_id is not None else set())
            )
            _load_excision_source_target(seal, target)
            # Read independent typed owners through the existing selected reader.
            # These observations load dependencies without classifying candidates
            # or granting either independent owner a mutation role.
            reader = _PreparedExcisionBlobSourceRead(seal)
            for ref_type, hash_key in (("hook_payload", "foreign-hook"), ("sidecar", "sidecar")):
                owner = next(owner for owner in BLOB_OWNERS if owner.ref_type == ref_type)
                assert reader.session_blob_ledger_hashes(owner, (fixture.hashes[hash_key],)) == (
                    fixture.hashes[hash_key],
                )
            _assert_no_effects(seal)
            counts = _stage_excision_source_target(seal, target, excised_at_ms=2)
            for key in (
                "source_marker_inputs_pending",
                "source_marker_inputs_accepted",
                "source_container_members",
                "source_raw_rows",
                "source_hook_events",
                "source_materials",
            ):
                assert counts[key] == 1
            assert counts["source_container_items"] == (2 if fixture.nullable_item_id is not None else 1)
            assert seal._pending_tier_permits == {}
            emptied = {
                "raw_sessions": "raw_id='raw-main'",
                "raw_hook_events": "hook_event_id='hook-main'",
                "hook_event_carriers": "hook_event_id='hook-main'",
                "blob_refs": "ref_id='hook-main' OR (ref_id='raw-main' AND ref_type IN ('raw_payload','attachment'))",
                "source_items": "source_generation_id IN ('generation-main','generation-null')",
                "source_item_raw_members": "source_generation_id='generation-main'",
                "source_item_member_dispositions": "source_generation_id='generation-main'",
                "material_observations": "material_id='material-main'",
                "material_evidence_links": "material_id='material-main'",
                "pending_accepted_marker_inputs": "raw_id='raw-main'",
                "accepted_marker_inputs": "raw_id='raw-main'",
            }
            for table, predicate in emptied.items():
                with seal.source_rows(f"SELECT count(*) FROM {table} WHERE {predicate}") as rows:
                    assert rows.fetchone()[0] == 0
            incoming = list(
                next(row for row in fixture.original_rows["material_observations"] if row[1] == "incoming-material")
            )
            incoming[fixture.source_columns["material_observations"].index("supersedes_material_id")] = None
            with seal.source_rows(
                "SELECT rowid,* FROM material_observations WHERE material_id='incoming-material'"
            ) as rows:
                assert tuple(rows.fetchone()) == tuple(incoming)
            with seal.source_rows("SELECT rowid,* FROM blob_refs WHERE ref_id='raw-main' ORDER BY rowid") as rows:
                assert tuple(tuple(row) for row in rows) == tuple(
                    row
                    for row in fixture.original_rows["blob_refs"]
                    if row[2] == "raw-main" and row[3] in {"hook_payload", "sidecar"}
                )
            with seal.source_rows("SELECT rowid,* FROM raw_hook_events WHERE hook_event_id='raw-main'") as rows:
                assert tuple(rows.fetchone()) == next(
                    row for row in fixture.original_rows["raw_hook_events"] if row[1] == "raw-main"
                )
            with seal.source_rows("SELECT rowid,* FROM history_sidecars WHERE sidecar_id='raw-main'") as rows:
                assert tuple(rows.fetchone()) == fixture.original_rows["history_sidecars"][0]
    with closing(sqlite3.connect(fixture.root / "source.db")) as source:
        for table, originals in fixture.original_rows.items():
            for original in originals:
                with connection_cursor(source, f"SELECT rowid,* FROM {table} WHERE rowid=?", (original[0],)) as rows:
                    assert tuple(rows.fetchone()) == original
        with connection_cursor(source, "SELECT count(*) FROM excised_marker_inputs") as rows:
            assert rows.fetchone()[0] == 0
