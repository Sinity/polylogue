from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, converge
from polylogue.operations import raw_observation_derivation as raw_observation_derivation_mod
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage import raw_authority as raw_authority_mod
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.derived import raw as raw_derivation_mod
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationInspection
from polylogue.storage.raw_authority import (
    RawReplayPlan,
    build_raw_replay_plans,
    raw_authority_parser_fingerprint,
    validate_raw_replay_plan,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.revision_application import RevisionApplicationReceipt
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _derive_raw_observations(root: Path, *, limit: int = 128) -> DerivationReport:
    """Settle original retained inputs, then return the genuine bounded pass."""
    import asyncio

    from tests.infra.live_ingest import prepared_live_convergence_owner
    from tests.infra.raw_owner_routes import converge_pending_raws_async

    async def exercise() -> DerivationReport:
        async with prepared_live_convergence_owner(root) as owner:
            (await owner.replay_retained_raw_ids(_raw_ids(root))).require_complete()
            return await converge_pending_raws_async(owner, root, limit=limit)

    return asyncio.run(exercise())


def _derived_success(report: DerivationReport) -> bool:
    return report.failed == 0 and report.pending == 0


def _raw_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id"))


def _write_codex_raw(
    root: Path,
    *,
    native_id: str,
    source_path: str,
    acquired_at_ms: int,
    text: str = "authored content",
    byte_proven: bool = False,
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
            canonical_source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            revision=(
                RawRevisionEnvelope(
                    logical_source_key=f"codex-session:{native_id}",
                    kind=RawRevisionKind.FULL,
                    source_revision=f"{native_id}-v1",
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                )
                if byte_proven
                else None
            ),
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

    assert _derived_success(_derive_raw_observations(tmp_path))
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.resolve_exact_session_ids(("codex-session:receipt",)) == {
            "codex-session:receipt": "codex-session:receipt"
        }
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


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_replay_plan_build_and_validation_read_the_active_generation(tmp_path: Path) -> None:
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def acquire() -> str:
        bootstrap_archive_root(tmp_path)
        return _write_codex_raw(tmp_path, native_id="active-plan", source_path="active-plan.jsonl", acquired_at_ms=1)

    raw_id = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 1
    shadow_plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    assert shadow_plan.index_preconditions["sessions"]

    def select_empty_generation() -> None:
        active_index = tmp_path / "generations" / "active" / "index.db"
        initialize_archive_database(active_index, ArchiveTier.INDEX)
        (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    await run_archive_fixture_write(tmp_path, select_empty_generation)
    active_plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    valid, observed = validate_raw_replay_plan(tmp_path, shadow_plan)

    assert active_plan.index_preconditions["sessions"] == []
    assert valid is False
    assert observed == active_plan.to_dict()


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_census_reads_the_active_generation_not_shadow_index(tmp_path: Path) -> None:
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def prepare() -> None:
        bootstrap_archive_root(tmp_path)
        active_index = tmp_path / "generations" / "active" / "index.db"
        initialize_archive_database(active_index, ArchiveTier.INDEX)
        (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
        (tmp_path / "index.db").write_bytes(b"not a sqlite database")

    await run_archive_fixture_write(tmp_path, prepare)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        measured = await owner.run_convergence_sync(
            "fixture.frontier.active",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert measured.healthy and measured.accepted_head_checks == measured.blocking_head_checks == 0


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


def test_historical_application_does_not_override_exact_current_generation(tmp_path: Path) -> None:
    """An older receipt for the same raw/content remains valid history after promotion."""
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="generation-history", source_path="history.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    applications = cast(list[dict[str, object]], receipt["application_rows"])
    heads = cast(list[dict[str, object]], receipt["head_rows"])
    assert len(applications) == len(heads) == 1
    historical = dict(applications[0])
    current = applications[0]
    current["acquisition_generation"] = 1
    heads[0]["acquisition_generation"] = 1
    current_receipt = RevisionApplicationReceipt(
        raw_id=cast(str, current["raw_id"]),
        session_id=cast(str, current["session_id"]),
        logical_source_key=cast(str, current["logical_source_key"]),
        source_revision=cast(str, current["source_revision"]),
        acquisition_generation=1,
        decision=ApplicationDecision(cast(str, current["decision"])),
        accepted_raw_id=cast(str | None, current["accepted_raw_id"]),
        accepted_source_revision=cast(str | None, current["accepted_source_revision"]),
        accepted_content_hash=bytes.fromhex(cast(str, current["accepted_content_hash"])),
        accepted_frontier_kind=cast(str | None, current["accepted_frontier_kind"]),
        accepted_frontier=cast(int | None, current["accepted_frontier"]),
        baseline_raw_id=cast(str | None, current["baseline_raw_id"]),
        predecessor_raw_id=cast(str | None, current["predecessor_raw_id"]),
        append_end_offset=cast(int | None, current["append_end_offset"]),
    )
    current["decision_id"] = current_receipt.decision_id
    applications.append(historical)

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)
    assert valid, problems
    historical_source_revision = historical["accepted_source_revision"]
    historical_decision_id = historical["decision_id"]
    historical["accepted_source_revision"] = "unwitnessed-history"
    historical["decision_id"] = replace(
        current_receipt,
        acquisition_generation=0,
        accepted_source_revision="unwitnessed-history",
    ).decision_id
    valid_forged_history, forged_problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)
    assert not valid_forged_history
    assert any("accepted source revision has no source evidence" in problem for problem in forged_problems)
    historical["accepted_source_revision"] = historical_source_revision
    historical["decision_id"] = historical_decision_id
    applications.pop(0)
    valid_without_current, problems_without_current = raw_authority_mod.validate_raw_replay_application_receipt(
        plan, receipt
    )
    assert not valid_without_current
    assert any("no application accepted authority matches" in problem for problem in problems_without_current)


@pytest.mark.parametrize("membership_decision", ["applied", "superseded_by_winner"])
def test_terminal_membership_cannot_replace_missing_head_application(tmp_path: Path, membership_decision: str) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="missing-application", source_path="missing.jsonl", acquired_at_ms=1)
    assert _derived_success(_derive_raw_observations(tmp_path))
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    application_rows = cast(list[dict[str, object]], receipt["application_rows"])
    assert len(application_rows) == 1
    # A single-session full raw is governed by its full-revision binding and
    # records no membership row; the law needs a terminal membership covering
    # the key, so the receipt and its immutable witness carry one.
    application = application_rows[0]
    application_key = str(application["logical_source_key"])
    application_revision = str(application["source_revision"])
    receipt["membership_rows"] = [
        {
            "raw_id": raw_id,
            "logical_source_key": application_key,
            "source_revision": application_revision,
            "decision": membership_decision,
        }
    ]
    plan = replace(
        plan,
        authority_witness={
            **plan.authority_witness,
            "memberships": [{"raw_id": raw_id, "logical_source_key": application_key}],
        },
    )
    receipt["application_rows"] = []

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)

    assert not valid
    assert not any("non-terminal decision" in problem for problem in problems)
    assert any("no application accepted authority matches" in problem for problem in problems)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_revision", "wrong-source-revision"),
        ("accepted_source_revision", "wrong-accepted-source-revision"),
        # A first head takes the publisher's byte frontier (revision_replay_frontier).
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
    assert application_rows[0][field] != replacement
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
            == raw_authority_parser_fingerprint()
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
    ambiguous verdict should still be terminal), an older semantic fingerprint
    (the verdict is stale and must be replayable), or ``None`` (no census row
    at all -- absent evidence must stay conservative and remain terminal).
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
                    raw_id, parser_fingerprint, status, logical_keys_json, detail
                ) VALUES (?, ?, 'complete', ?, 'test-seeded')
                """,
                (raw_id, parser_fingerprint, json.dumps([logical_source_key])),
            )
        conn.commit()
    observation_status = RawObservationInspection(tmp_path).inspect(
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
        tmp_path, native_id="current-ambiguous", parser_fingerprint=raw_authority_parser_fingerprint()
    )
    assert observation_status == "valid"


def test_ambiguous_verdict_under_previous_dynamic_fingerprint_is_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed executable fingerprint invalidates a former terminal verdict.

    Anti-vacuity: this exercises the canonical ``RawObservationDerivation``
    inspection route. Reverting its fingerprint-gating clause makes this test
    fail by re-classifying the plan as terminal.
    """
    bootstrap_archive_root(tmp_path)
    previous_fingerprint = raw_authority_parser_fingerprint()
    _raw_id, observation_status = _seed_ambiguous_membership_component(
        tmp_path, native_id="previous-dynamic-ambiguous", parser_fingerprint=previous_fingerprint
    )
    assert observation_status == "valid"
    changed_fingerprint = previous_fingerprint[:-1] + ("0" if previous_fingerprint[-1] != "0" else "1")
    monkeypatch.setattr(raw_derivation_mod, "raw_authority_parser_fingerprint", lambda: changed_fingerprint)
    monkeypatch.setattr(
        raw_observation_derivation_mod,
        "raw_authority_parser_fingerprint",
        lambda: changed_fingerprint,
    )
    observation_status = RawObservationInspection(tmp_path).inspect(
        raw_observation_frame(tmp_path, raw_ids=(_raw_id,)),
        (_raw_id,),
    )[_raw_id]
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
    import asyncio

    from polylogue.core.stage_admission import admit_stage_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def exercise() -> DerivationReport:
        async with prepared_live_convergence_owner(tmp_path) as owner:

            def run_phase() -> DerivationReport:
                adapter = RawObservationDerivation(tmp_path, compute_adapter=owner._compute_adapter)
                return converge(
                    DerivationRegistry((adapter,)),
                    raw_observation_frame(tmp_path),
                    budget=Budget(
                        page=10, discovery=10, inspection=20, compute=10, publication=10, retained_outcomes=8
                    ),
                    publisher=admit_stage_write,
                )

            report = await owner.run_prepared_sync(
                "test.ledger.bounded-phase", run_phase, settlement_owners=lambda: (), estimated_bytes=0
            )
            # Single-pass convergence: each raw censuses and replays in one
            # compute, so the bounded outcome list still counts all ten done.
            assert report.done == 10
            assert report.pending == 0
            assert report.failed == 0
            assert len(report.outcomes) == 8
            assert report.truncated is True
            return report

    asyncio.run(exercise())
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.index_connection is not None
        assert archive.index_connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 10


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_classifies_dangling_head_session_as_corrupt(tmp_path: Path) -> None:
    from contextlib import closing

    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def acquire() -> tuple[str, str | None]:
        bootstrap_archive_root(tmp_path)
        raw_id = _write_codex_raw(tmp_path, native_id="dangling-session", source_path="neutral.jsonl", acquired_at_ms=1)
        return raw_id, None

    raw_id, phantom = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(item.written_session_ids) for item in receipts) == 1

        def corrupt() -> None:
            with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
                sid = conn.execute(
                    "SELECT session_id FROM raw_revision_heads WHERE accepted_raw_id=?", (raw_id,)
                ).fetchone()[0]
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))

        await run_archive_fixture_write(tmp_path, corrupt)
        measured = await owner.run_convergence_sync(
            "fixture.frontier.refusal",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert not measured.healthy and measured.blocking_head_checks == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        rows = conn.execute(
            "SELECT expected_json,observed_json,resolved_at_ms,reason FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchall()
    assert len(rows) == 1 and rows[0][2] is None
    expected, observed = json.loads(rows[0][0]), json.loads(rows[0][1])
    assert expected["input_raw_ids"] == [raw_id]
    assert observed["state"] == "corrupt"
    assert rows[0][3] == "accepted head has no matching materialized session"
    assert "actuator" not in observed and "actuator" not in expected["authority_witness"]
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = cast(list[dict[str, object]], readiness["raw_authority_frontier_remediation_refs"])
    assert expected["plan_id"] in {ref["plan_id"] for ref in refs}


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_classifies_head_session_raw_mismatch_as_corrupt(tmp_path: Path) -> None:
    from contextlib import closing

    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def acquire() -> tuple[str, str | None]:
        bootstrap_archive_root(tmp_path)
        raw_id = _write_codex_raw(tmp_path, native_id="mismatch-one", source_path="neutral.jsonl", acquired_at_ms=1)
        phantom = _write_codex_raw(tmp_path, native_id="phantom-only", source_path="phantom.jsonl", acquired_at_ms=2)
        return raw_id, phantom

    raw_id, phantom = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(item.written_session_ids) for item in receipts) == 1

        def corrupt() -> None:
            with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
                conn.execute(
                    "UPDATE raw_revision_heads SET accepted_raw_id=? WHERE accepted_raw_id=?", (phantom, raw_id)
                )

        await run_archive_fixture_write(tmp_path, corrupt)
        measured = await owner.run_convergence_sync(
            "fixture.frontier.refusal",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert not measured.healthy and measured.blocking_head_checks == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        rows = conn.execute(
            "SELECT expected_json,observed_json,resolved_at_ms,reason FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchall()
    assert len(rows) == 1 and rows[0][2] is None
    expected, observed = json.loads(rows[0][0]), json.loads(rows[0][1])
    assert expected["input_raw_ids"] == [raw_id]
    assert observed["state"] == "corrupt"
    assert rows[0][3] == "accepted revision head and materialized session select different raw authority"
    assert expected["index_preconditions"]["head_accepted_raw_id"] == phantom
    assert expected["index_preconditions"]["accepted_raw_id"] == raw_id
    assert "actuator" not in observed and "actuator" not in expected["authority_witness"]
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = cast(list[dict[str, object]], readiness["raw_authority_frontier_remediation_refs"])
    assert expected["plan_id"] in {ref["plan_id"] for ref in refs}


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_quarantined_accepted_head_is_a_terminal_obligation_not_a_promise(tmp_path: Path) -> None:
    from contextlib import closing

    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def acquire() -> tuple[str, str | None]:
        bootstrap_archive_root(tmp_path)
        raw_id = _write_codex_raw(
            tmp_path, native_id="quarantine-ineligible", source_path="neutral.jsonl", acquired_at_ms=1
        )
        return raw_id, None

    raw_id, phantom = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(item.written_session_ids) for item in receipts) == 1

        def corrupt() -> None:
            with closing(sqlite3.connect(tmp_path / "source.db")) as conn, conn:
                conn.execute("UPDATE raw_sessions SET revision_authority='quarantined' WHERE raw_id=?", (raw_id,))

        await run_archive_fixture_write(tmp_path, corrupt)
        measured = await owner.run_convergence_sync(
            "fixture.frontier.refusal",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert not measured.healthy and measured.blocking_head_checks == 1
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        rows = conn.execute(
            "SELECT expected_json,observed_json,resolved_at_ms,reason FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchall()
    assert len(rows) == 1 and rows[0][2] is None
    expected, observed = json.loads(rows[0][0]), json.loads(rows[0][1])
    assert expected["input_raw_ids"] == [raw_id]
    assert observed["state"] == "unresolved_provenance"
    assert rows[0][3] == "accepted raw authority remains quarantined"
    assert "actuator" not in observed and "actuator" not in expected["authority_witness"]
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = cast(list[dict[str, object]], readiness["raw_authority_frontier_remediation_refs"])
    assert expected["plan_id"] in {ref["plan_id"] for ref in refs}


@pytest.mark.parametrize("prior_decision", ["ambiguous", "deferred"])
def test_v5_semantic_refusal_is_recensused_and_replayed_from_retained_bytes(
    tmp_path: Path, prior_decision: str
) -> None:
    bootstrap_archive_root(tmp_path)
    raw_id, _status = _seed_ambiguous_membership_component(
        tmp_path, native_id="semantic-receipt", parser_fingerprint="revision-membership-v5"
    )
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("UPDATE raw_session_memberships SET decision = ? WHERE raw_id = ?", (prior_decision, raw_id))
    derivation = RawObservationInspection(tmp_path)
    assert derivation.inspect(raw_observation_frame(tmp_path, raw_ids=(raw_id,)), (raw_id,))[raw_id] == "stale"

    report = _derive_raw_observations(tmp_path)
    assert _derived_success(report)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT parser_fingerprint, status FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (raw_authority_parser_fingerprint(), "complete")
        # Single-pass convergence: the recensused membership is accepted and
        # replayed in the same pass, so it ends terminally applied.
        assert (
            source.execute("SELECT decision FROM raw_session_memberships WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            == "applied"
        )
    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive.resolve_exact_session_ids(("codex-session:semantic-receipt",)) == {
            "codex-session:semantic-receipt": "codex-session:semantic-receipt",
        }
        assert (
            archive._conn.execute(
                "SELECT text FROM blocks WHERE session_id = ?", ("codex-session:semantic-receipt",)
            ).fetchone()[0]
            == "authored content"
        )
    assert derivation.inspect(raw_observation_frame(tmp_path, raw_ids=(raw_id,)), (raw_id,))[raw_id] == "valid"
