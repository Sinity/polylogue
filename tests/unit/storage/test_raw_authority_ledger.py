from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, Outcome, converge
from polylogue.operations.raw_observation_derivation import (
    converge_raw_observations,
    raw_observation_frame,
)
from polylogue.storage import raw_authority as raw_authority_mod
from polylogue.storage import raw_reconciler as raw_reconciler_mod
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.raw_authority import (
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    RawReplayPlan,
    build_raw_replay_plans,
    validate_raw_replay_plan,
)
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierItem,
    RawAuthorityFrontierState,
    inspect_raw_authority_frontier,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.revision_application import RevisionApplicationReceipt
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _derive_raw_observations(root: Path, *, limit: int = 128) -> DerivationReport:
    """Run the bounded canonical raw-observation derivation for this fixture."""
    return converge_raw_observations(
        root,
        source_roots=(),
        limit=limit,
        max_payload_bytes=64 * 1024 * 1024,
    )


def _derived_success(report: DerivationReport) -> bool:
    return report.failed == 0 and report.pending == 0


def _derived_count(report: DerivationReport, outcome: Outcome = Outcome.DONE) -> int:
    assert report.failed == 0
    assert report.pending == 0
    return report.count(outcome)


def _raw_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id"))


def _frontier_test_item(
    *,
    raw_id: str,
    plan_id: str,
    input_raw_ids: tuple[str, ...],
    strategy_witness: JSONDocument,
) -> RawAuthorityFrontierItem:
    return RawAuthorityFrontierItem(
        state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
        actuator=RawAuthorityActuator.COPY_FORWARD_ORIGIN,
        raw_id=raw_id,
        logical_source_key="chatgpt:test",
        session_id="chatgpt-export:test",
        reason="test frontier item",
        evidence_digest="e" * 64,
        input_raw_ids=input_raw_ids,
        source_preconditions=json_document({"raw_id": raw_id}),
        index_preconditions=json_document({"raw_id": raw_id}),
        strategy_witness=strategy_witness,
        plan_id=plan_id,
    )


def test_frontier_plan_witness_carries_no_top_level_raw_id() -> None:
    """A frontier plan's authority witness never hoists a single raw id.

    The witness describes a whole component, so a top-level ``raw_id`` would
    make two plans over the same component disagree depending on which raw
    happened to be named. The blocker row stores this witness verbatim in
    ``expected_json``, so the shape is what a resolution later reads back.

    Anti-vacuity: re-adding ``raw_id`` at the top level of the witness that
    ``_plan`` builds fails this; the nested ``strategy_witness.item.raw_id``
    asserted below proves the raw id is still reachable where it belongs.
    """
    strategy_witness = json_document(
        {
            "schema": "polylogue.raw-authority-strategy-witness.v1",
            "kind": "browser_origin",
            "item": {"raw_id": "primary-raw"},
        }
    )
    item = _frontier_test_item(
        raw_id="primary-raw",
        plan_id="raw-authority-frontier:witness-shape",
        input_raw_ids=("primary-raw", "auxiliary-raw"),
        strategy_witness=strategy_witness,
    )

    plan = raw_reconciler_mod._plan(item)

    assert "raw_id" not in plan.authority_witness
    stored_witness = cast(dict[str, object], plan.authority_witness["strategy_witness"])
    assert cast(dict[str, object], stored_witness["item"])["raw_id"] == "primary-raw"
    assert plan.input_raw_ids == ("primary-raw", "auxiliary-raw")


def _write_codex_raw(
    root: Path,
    *,
    native_id: str,
    source_path: str,
    acquired_at_ms: int,
    text: str = "authored content",
) -> str:
    payload = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
        f'{{"type":"response_item","payload":{{"type":"message","id":"m-{acquired_at_ms}",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
        )


def test_parsed_timestamp_without_exact_application_receipt_fails_closed(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="receipt", source_path="receipt.jsonl", acquired_at_ms=1)
    real_receipt = raw_authority_mod.raw_replay_application_receipt

    def incomplete_receipt(
        root: Path,
        plan: RawReplayPlan,
        *,
        index_db_path: Path | None = None,
    ) -> JSONDocument:
        payload = dict(real_receipt(root, plan, index_db_path=index_db_path))
        payload["head_rows"] = []
        return json_document(payload)

    assert _derived_count(_derive_raw_observations(tmp_path)) == 1
    (raw_id,) = _raw_ids(tmp_path)
    (plan,) = build_raw_replay_plans(tmp_path, ((raw_id,),))
    invalid = incomplete_receipt(tmp_path, plan)
    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, invalid)
    assert valid is False
    assert problems


def test_application_receipt_reads_the_active_generation_not_shadow_index(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="active-receipt", source_path="active.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    receipt = raw_authority_mod.raw_replay_application_receipt(tmp_path, plan)

    assert receipt["index_db_path"] == str(active_index)
    assert receipt["application_rows"] == []


def test_replay_plan_build_and_validation_read_the_active_generation(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="active-plan", source_path="active-plan.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    shadow_plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    assert shadow_plan.index_preconditions["sessions"]

    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    active_plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    valid, observed = validate_raw_replay_plan(tmp_path, shadow_plan)

    assert active_plan.index_preconditions["sessions"] == []
    assert valid is False
    assert observed == active_plan.to_dict()


def test_frontier_census_reads_the_active_generation_not_shadow_index(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    (tmp_path / "index.db").write_bytes(b"not a sqlite database")

    census = inspect_raw_authority_frontier(_config(tmp_path))

    assert census.accepted_head_count == 0
    assert census.plan_count == 0


@pytest.mark.parametrize("field", ["session_id", "accepted_raw_id", "accepted_content_hash"])
def test_application_receipt_requires_exact_application_authority(tmp_path: Path, field: str) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id=f"exact-{field}", source_path=f"{field}.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    application_rows = cast(list[dict[str, object]], receipt["application_rows"])
    assert application_rows
    application_rows[0][field] = f"wrong-{field}"

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)

    assert valid is False
    assert any("no application accepted authority matches" in problem for problem in problems)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_revision", "wrong-source-revision"),
        ("accepted_source_revision", "wrong-accepted-source-revision"),
        ("accepted_frontier_kind", "semantic"),
        ("accepted_frontier", 999),
        ("acquisition_generation", 999),
        ("append_end_offset", 999),
        ("baseline_raw_id", "wrong-baseline"),
        ("predecessor_raw_id", "wrong-predecessor"),
        ("decision_id", "0" * 64),
    ],
)
def test_application_receipt_recovery_rejects_malformed_authority_evidence(
    tmp_path: Path, field: str, replacement: object
) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id=f"malformed-{field}", source_path=f"{field}.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    application_rows = cast(list[dict[str, object]], receipt["application_rows"])
    assert application_rows
    application_rows[0][field] = replacement

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)

    assert valid is False
    assert problems


def test_application_receipt_recovery_rejects_source_revision_from_another_shared_membership(tmp_path: Path) -> None:
    """A grouped raw cannot lend key B's revision evidence to key A's application."""
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="shared-memberships", source_path="shared.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    membership_rows = cast(list[dict[str, object]], receipt["membership_rows"])
    application_rows = cast(list[dict[str, object]], receipt["application_rows"])
    assert len(application_rows) == 1

    shared_key = "codex:shared-memberships-other"
    shared_revision = "shared-membership-other-revision"
    application = application_rows[0]
    application_key = str(application["logical_source_key"])
    membership_rows.extend(
        [
            {
                "raw_id": raw_id,
                "logical_source_key": application_key,
                "source_revision": application["source_revision"],
                "decision": "applied",
            },
            {
                "raw_id": raw_id,
                "logical_source_key": shared_key,
                "source_revision": shared_revision,
                "decision": "applied",
            },
        ]
    )
    application["source_revision"] = shared_revision
    replay_plan = replace(
        plan,
        authority_witness={
            **plan.authority_witness,
            "memberships": [
                {"raw_id": raw_id, "logical_source_key": application_key},
                {"raw_id": raw_id, "logical_source_key": shared_key},
            ],
        },
    )
    application["decision_id"] = RevisionApplicationReceipt(
        raw_id=str(application["raw_id"]),
        session_id=str(application["session_id"]),
        logical_source_key=str(application["logical_source_key"]),
        source_revision=shared_revision,
        acquisition_generation=cast(int, application["acquisition_generation"]),
        decision=ApplicationDecision(str(application["decision"])),
        accepted_raw_id=cast(str | None, application["accepted_raw_id"]),
        accepted_source_revision=cast(str | None, application["accepted_source_revision"]),
        accepted_content_hash=bytes.fromhex(cast(str, application["accepted_content_hash"])),
        accepted_frontier_kind=cast(str | None, application["accepted_frontier_kind"]),
        accepted_frontier=cast(int | None, application["accepted_frontier"]),
        baseline_raw_id=cast(str | None, application["baseline_raw_id"]),
        predecessor_raw_id=cast(str | None, application["predecessor_raw_id"]),
        append_end_offset=cast(int | None, application["append_end_offset"]),
    ).decision_id

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(replay_plan, receipt)

    assert valid is False
    assert any("source revision does not match membership evidence" in problem for problem in problems)


def test_stale_per_raw_parser_fingerprint_is_recensused_before_planning(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="parser-drift", source_path="parser-drift.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    first = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_authority_parser_census SET parser_fingerprint = 'old-parser' WHERE raw_id = ?",
            (raw_id,),
        )
        conn.commit()

    assert _derived_success(_derive_raw_observations(tmp_path))
    second = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    assert first.plan_id == second.plan_id
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT parser_fingerprint FROM raw_authority_parser_census WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()[0]
            == RAW_AUTHORITY_PARSER_FINGERPRINT
        )


def _seed_ambiguous_membership_component(
    tmp_path: Path,
    *,
    native_id: str,
    parser_fingerprint: str | None,
) -> tuple[str, str]:
    """Seed one raw whose membership decision is durably 'ambiguous'.

    ``parser_fingerprint`` controls what (if anything) the per-raw
    ``raw_authority_parser_census`` row records: the CURRENT fingerprint (the
    ambiguous verdict should still be terminal), a fingerprint listed in
    ``SUPERSEDED_MEMBERSHIP_FINGERPRINTS`` (the verdict is stale and must be
    replayable), or ``None`` (no census row at all -- absent evidence must
    stay conservative and remain terminal).
    """
    raw_id = _write_codex_raw(tmp_path, native_id=native_id, source_path=f"{native_id}.jsonl", acquired_at_ms=1)
    logical_source_key = f"codex-session:{native_id}"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 1, 'ambiguous', 1)
            """,
            (raw_id, logical_source_key, native_id, "rev-1", bytes(32)),
        )
        if parser_fingerprint is not None:
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
                ) VALUES (?, ?, 'complete', ?, 'test-seeded', 0)
                """,
                (raw_id, parser_fingerprint, json.dumps([logical_source_key])),
            )
        conn.commit()
    observation_status = RawObservationDerivation(tmp_path).inspect(
        raw_observation_frame(tmp_path, raw_ids=(raw_id,)),
        (raw_id,),
    )[raw_id]
    return raw_id, str(observation_status)


def test_ambiguous_verdict_under_current_fingerprint_stays_terminal(tmp_path: Path) -> None:
    """polylogue-9dxn: an 'ambiguous' decision recorded under the CURRENT
    classifier fingerprint is still authoritative -- it must not be
    replayed without new evidence.
    """
    bootstrap_archive_root(tmp_path)
    _raw_id, observation_status = _seed_ambiguous_membership_component(
        tmp_path, native_id="current-ambiguous", parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT
    )
    assert observation_status == "valid"


@pytest.mark.parametrize("superseded_fingerprint", ["revision-membership-v1", "revision-membership-v2"])
def test_ambiguous_verdict_under_superseded_fingerprint_is_replayable(
    tmp_path: Path, superseded_fingerprint: str
) -> None:
    """polylogue-9dxn: an 'ambiguous' decision recorded under a fingerprint
    listed in SUPERSEDED_MEMBERSHIP_FINGERPRINTS is stale -- a corrected
    classifier deserves a chance to re-derive it, so it must not be
    terminal.

    Anti-vacuity: this exercises the canonical ``RawObservationDerivation``
    inspection route. Reverting its fingerprint-gating clause makes this test
    fail by re-classifying the plan as terminal.
    """
    bootstrap_archive_root(tmp_path)
    assert superseded_fingerprint in raw_authority_mod.SUPERSEDED_MEMBERSHIP_FINGERPRINTS
    _raw_id, observation_status = _seed_ambiguous_membership_component(
        tmp_path, native_id="superseded-ambiguous", parser_fingerprint=superseded_fingerprint
    )
    assert observation_status != "valid"


def test_ambiguous_verdict_with_no_census_row_stays_terminal(tmp_path: Path) -> None:
    """polylogue-9dxn: absent census evidence must default to conservative
    (terminal), not to "assume the classifier fix already applies".
    """
    bootstrap_archive_root(tmp_path)
    _raw_id, observation_status = _seed_ambiguous_membership_component(
        tmp_path, native_id="uncensused-ambiguous", parser_fingerprint=None
    )
    assert observation_status == "valid"


def test_raw_observation_report_bounds_retained_outcomes_without_losing_counts(tmp_path: Path) -> None:
    """Canonical derivation keeps totals authoritative and samples bounded."""
    bootstrap_archive_root(tmp_path)
    for index in range(10):
        _write_codex_raw(
            tmp_path,
            native_id=f"bounded-{index}",
            source_path=f"bounded-{index}.jsonl",
            acquired_at_ms=index,
        )
    adapter = RawObservationDerivation(tmp_path)
    report = converge(
        DerivationRegistry((adapter,)),
        raw_observation_frame(tmp_path),
        budget=Budget(
            page=10,
            discovery=10,
            inspection=20,
            compute=10,
            publication=10,
            retained_outcomes=8,
        ),
    )
    assert report.done == 10
    assert report.failed == 0
    assert len(report.outcomes) == 8
    assert report.truncated is True


def test_frontier_classifies_dangling_head_session_as_corrupt(tmp_path: Path) -> None:
    """polylogue-lkrc AC1/AC6/AC7: the CORRUPT terminal state had zero
    regression coverage anywhere in the suite even though it is one of the
    eight mutually exclusive frontier states the reconciler declares and
    persists as a durable blocker.

    This reproduces the first of ``_classify_frontier``'s three CORRUPT
    triggers (``polylogue/storage/raw_reconciler.py``): ``raw_revision_heads``
    still names a ``session_id`` but the materialized session row it points at
    is gone (a torn write, an interrupted rebuild, or manual tampering with
    the rebuildable index tier). Proven-current accepted heads must never
    silently read as healthy in this shape.
    """
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="dangling-session", source_path="dangling.jsonl", acquired_at_ms=1)
    assert _derived_count(_derive_raw_observations(tmp_path)) == 1

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        session_id = index_conn.execute(
            "SELECT session_id FROM raw_revision_heads WHERE accepted_raw_id = ?", (raw_id,)
        ).fetchone()[0]
        # Simulate the accepted head surviving while its materialized session
        # vanishes underneath it -- the index tier is rebuildable and this is
        # exactly the kind of partial state a crash mid-rebuild can leave.
        index_conn.execute("PRAGMA foreign_keys = OFF")
        index_conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        index_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.CORRUPT
    assert item.reason == "accepted head has no matching materialized session"
    assert item.executable is False
    assert census.state_counts[RawAuthorityFrontierState.CORRUPT.value] == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blocker = source_conn.execute(
            "SELECT reason, resolved_at_ms FROM raw_authority_blockers WHERE json_extract(expected_json, '$.plan_id') = ?",
            (item.plan_id,),
        ).fetchone()
    assert blocker is not None
    assert blocker[1] is None

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = cast(list[dict[str, object]], readiness["raw_authority_frontier_remediation_refs"])
    assert item.plan_id in {ref["plan_id"] for ref in refs}


def test_frontier_classifies_head_session_raw_mismatch_as_corrupt(tmp_path: Path) -> None:
    """polylogue-lkrc AC1/AC6/AC7: reproduces the second reachable CORRUPT
    trigger -- the accepted head names one raw as authoritative
    (``accepted_raw_id``) while the materialized session it points at was
    actually built from a *different* raw. This is the torn-write shape the
    reconciler's own comment describes ("accepted revision head and
    materialized session select different raw authority"): a genuine
    disagreement between two derived-tier tables that a read-only census
    must surface as a durable blocker rather than silently trust the head.
    """
    bootstrap_archive_root(tmp_path)
    accepted_raw_id = _write_codex_raw(
        tmp_path, native_id="mismatch-one", source_path="mismatch.jsonl", acquired_at_ms=1
    )
    assert _derived_count(_derive_raw_observations(tmp_path)) == 1
    # An independent, never-materialized raw acquisition -- stands in for the
    # "wrong" raw a corrupted head could point at.
    phantom_raw_id = _write_codex_raw(tmp_path, native_id="phantom-only", source_path="phantom.jsonl", acquired_at_ms=2)

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        logical_source_key = index_conn.execute(
            "SELECT logical_source_key FROM raw_revision_heads WHERE accepted_raw_id = ?", (accepted_raw_id,)
        ).fetchone()[0]
        index_conn.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = ? WHERE logical_source_key = ?",
            (phantom_raw_id, logical_source_key),
        )
        index_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    # The session itself was materialized from accepted_raw_id, so the
    # classifier resolves the raw row by the SESSION's own raw_id, not the
    # (now wrong) value stashed on the head row -- the surfaced item is still
    # keyed by the real session raw, with the head's disagreement in the
    # reason/evidence.
    item = next(entry for entry in census.items if entry.raw_id == accepted_raw_id)
    assert item.state is RawAuthorityFrontierState.CORRUPT
    assert item.reason == "accepted revision head and materialized session select different raw authority"
    assert item.executable is False
    assert item.index_preconditions["head_accepted_raw_id"] == phantom_raw_id
    assert item.index_preconditions["accepted_raw_id"] == accepted_raw_id
    assert census.state_counts[RawAuthorityFrontierState.CORRUPT.value] == 1

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False


def test_verified_blob_receipt_invalidates_when_blob_bytes_change_underneath_it(tmp_path: Path) -> None:
    """polylogue-byw3y: the safety-critical half of the receipt cache.

    A verification receipt is a durable HINT, never an authority: it may only
    ever be trusted for the exact on-disk fingerprint (dev/inode/size/mtime/
    ctime) it was recorded against. If a blob's bytes are corrupted/mutated
    in place -- the file at the content-addressed path no longer matches its
    own filename hash -- the next census MUST re-verify from scratch and
    reclassify the frontier item as unproven, never silently keep trusting a
    stale receipt. This is the regression this bead's whole design exists to
    prevent: a performance win here would be worthless (and actively unsafe)
    if it could paper over real corruption.
    """
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="tamper-target", source_path="tamper.jsonl", acquired_at_ms=1, text="hello"
    )
    assert _derived_count(_derive_raw_observations(tmp_path)) == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blob_hash_hex = str(
            source_conn.execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        ).lower()

    # First census: proves the blob, records a durable receipt.
    census = inspect_raw_authority_frontier(_config(tmp_path))
    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.PROVEN_CURRENT

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        receipt = source_conn.execute(
            "SELECT st_size FROM verified_blob_receipts WHERE blob_hash = ?",
            (bytes.fromhex(blob_hash_hex),),
        ).fetchone()
    assert receipt is not None, "first census must persist a verification receipt"

    # Corrupt the blob bytes IN PLACE -- same content-addressed filename,
    # different content. This is the exact shape a stale-but-trusted receipt
    # would silently paper over: the fingerprint's st_size differs, so the
    # census must force a real re-hash rather than trust the old receipt.
    blob_path = BlobStore(tmp_path / "blob").blob_path(blob_hash_hex)
    blob_path.write_bytes(blob_path.read_bytes() + b"tampered-bytes")

    census2 = inspect_raw_authority_frontier(_config(tmp_path))
    item2 = next(entry for entry in census2.items if entry.raw_id == raw_id)
    assert item2.state is RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE
    assert item2.reason == "accepted head raw bytes do not prove the expected content-addressed digest"
    assert census2.state_counts[RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE.value] == 1


def test_verified_blob_receipt_skips_rehash_on_unchanged_blob_across_census_passes(tmp_path: Path) -> None:
    """polylogue-byw3y: the performance half -- a blob verified once and left
    untouched must not be re-hashed by a second census pass. Counts actual
    ``BlobStore.verify`` invocations (the real content re-hash) rather than
    trusting a wall-clock or state proxy, so the assertion fails honestly if
    the receipt cache regresses back to re-verifying every restart.
    """
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="unchanged-target", source_path="unchanged.jsonl", acquired_at_ms=1, text="hello"
    )
    assert _derived_count(_derive_raw_observations(tmp_path)) == 1

    verify_calls: list[str] = []
    real_verify = BlobStore.verify

    def _counting_verify(self: BlobStore, hash_hex: str) -> bool:
        verify_calls.append(hash_hex)
        return real_verify(self, hash_hex)

    with patch.object(BlobStore, "verify", _counting_verify):
        census1 = inspect_raw_authority_frontier(_config(tmp_path))
        assert len(verify_calls) == 1, "first census must hash the blob at least once"
        item1 = next(entry for entry in census1.items if entry.raw_id == raw_id)
        assert item1.state is RawAuthorityFrontierState.PROVEN_CURRENT

        verify_calls.clear()
        census2 = inspect_raw_authority_frontier(_config(tmp_path))
        assert verify_calls == [], "second census over an unchanged blob must reuse the durable receipt"
        item2 = next(entry for entry in census2.items if entry.raw_id == raw_id)
        assert item2.state is RawAuthorityFrontierState.PROVEN_CURRENT


def test_ineligible_quarantined_raw_gets_a_terminal_actuator_not_refine_quarantine(tmp_path: Path) -> None:
    """polylogue-u19l: reproduces the absorbing-state defect end to end.

    Live evidence (source.db, read-only, 2026-07-31): 4,147 open
    ``raw_authority_blockers`` rows all read "accepted raw authority remains
    quarantined pending exact refinement proof" with actuator
    REFINE_QUARANTINE, 15,205/17,384 frontier plans residual, fixed_point=0
    on all 256 retained censuses, and the gap count only ever grew
    (16,874 -> 17,384). Root cause: REFINE_QUARANTINE has a real apply()
    dispatch branch (``raw_reconciler.py``, the ``item.actuator is
    RawAuthorityActuator.REFINE_QUARANTINE`` block), but the executability
    gate (``_EXECUTABLE_STATES``) only ever admits SAFELY_REKEYABLE /
    DUPLICATE_ALIAS states -- so once ``inspect_quarantined_accepted_raws``
    proves a quarantined raw's refinement is "ineligible" (a permanent
    structural fact, not a transient one -- see the reasons enumerated in
    ``_inspect_quarantined_accepted_raw``), the census silently promised an
    actuator that neither the daemon nor the operator break-glass path
    could ever select.

    Force a raw into exactly that "ineligible" shape by accepting it
    normally, then flipping only its ``revision_authority`` to
    'quarantined' out from under an otherwise byte-proven envelope -- this
    fails ``_inspect_quarantined_accepted_raw``'s typed-envelope check
    (source/predecessor/baseline columns don't match any of the three
    admitted envelopes), which is exactly the "source raw has an
    incompatible typed authority envelope" ineligibility reason observed
    live. The frontier item must come back non-executable AND with an
    honest NONE actuator -- never REFINE_QUARANTINE -- while remaining
    countable (state_counts) and operator-visible (raw_authority_blockers).
    """
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="quarantine-ineligible", source_path="quarantine.jsonl", acquired_at_ms=1
    )
    assert _derived_count(_derive_raw_observations(tmp_path)) == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET revision_authority = 'quarantined' WHERE raw_id = ?",
            (raw_id,),
        )
        source_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.UNRESOLVED_PROVENANCE
    assert item.actuator is raw_reconciler_mod.RawAuthorityActuator.NONE
    assert item.executable is False
    assert "ineligible" in item.reason
    assert census.state_counts[RawAuthorityFrontierState.UNRESOLVED_PROVENANCE.value] == 1

    # Terminal, countable, operator-visible: still tracked as an open
    # blocker (an operator can find it), just no longer misrepresented as
    # "an automatic actuator will resolve this".
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blocker = source_conn.execute(
            "SELECT reason, resolved_at_ms FROM raw_authority_blockers WHERE json_extract(expected_json, '$.plan_id') = ?",
            (item.plan_id,),
        ).fetchone()
    assert blocker is not None
    assert blocker[1] is None
    assert "ineligible" in blocker[0]

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1


def test_frontier_item_construction_rejects_unreachable_actuator_state_pairs() -> None:
    """polylogue-w32w: the invariant made structurally enforceable.

    ``RawAuthorityFrontierItem.__post_init__`` must reject any combination
    of a dispatch-handled actuator (one with a real apply() branch) paired
    with a state the executability gate does not admit -- the exact defect
    class polylogue-u19l fixed for REFINE_QUARANTINE. This proves the guard
    fires at construction, not merely that today's call sites happen to
    comply.
    """
    row: dict[str, object] = {
        "accepted_raw_id": "raw-1",
        "logical_source_key": "codex:native-1",
        "session_id": "codex-session:native-1",
    }
    with pytest.raises(ValueError, match="unreachable"):
        raw_reconciler_mod._item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=raw_reconciler_mod.RawAuthorityActuator.REFINE_QUARANTINE,
            row=row,
            reason="an actuator with a real apply() handler must only pair with an executable state",
        )
    # The dual is fine: an executable state may pair with a dispatched actuator.
    executable_item = raw_reconciler_mod._item(
        state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
        actuator=raw_reconciler_mod.RawAuthorityActuator.REFINE_QUARANTINE,
        row=row,
        reason="eligible",
    )
    assert executable_item.executable is True
    # And a non-dispatched actuator (REQUEST_JUDGMENT, REACQUIRE, NONE) may
    # legitimately pair with a non-executable state -- those resolve
    # out-of-band (operator judgment, ordinary re-acquisition), not through
    # this apply dispatcher.
    judgment_item = raw_reconciler_mod._item(
        state=RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT,
        actuator=raw_reconciler_mod.RawAuthorityActuator.REQUEST_JUDGMENT,
        row=row,
        reason="needs an operator disposition",
    )
    assert judgment_item.executable is False
