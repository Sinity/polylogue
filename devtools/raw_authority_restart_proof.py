"""Prove raw-authority crash recovery against the production ledger and writer.

The proof builds only fresh synthetic archives below the requested workdir.  It
uses ``repair_raw_materialization`` without lowering its production limits,
injects process crashes at the real outcome/finalization call boundaries, then
reads the source-tier census ledger and validates executed application receipts
with the production postcondition validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO, TypeVar, cast
from unittest.mock import patch

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Provider, Role
from polylogue.core.json import require_json_document
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage import raw_authority, repair
from polylogue.storage.raw_authority import RawReplayPlan, RawReplayPlanOutcome, RawReplayPlanStatus
from polylogue.storage.sqlite.archive_tiers import revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)

# The accepted-head reparse evidence is part of the report and proof identity.
# Keep that expanded shape and identity explicitly versioned instead of
# presenting it as a legacy v1 receipt.
_PROOF_SCHEMA = "polylogue.raw-authority-restart-proof.v2"
_PROOF_IDENTITY_PREFIX = "raw-authority-restart-proof:v2:"
_CASE_SCHEMA = "polylogue.raw-authority-restart-proof-case.v1"
_TERMINAL_STATUSES = {
    RawReplayPlanStatus.EXECUTED.value,
    RawReplayPlanStatus.DEFERRED.value,
    RawReplayPlanStatus.TERMINAL.value,
}
_SELECTED_CONSERVED_STATUSES = _TERMINAL_STATUSES | {
    RawReplayPlanStatus.RETRYABLE.value,
    RawReplayPlanStatus.REJECTED_STALE.value,
}
_EXPECTED_TERMINAL_STATUSES = {
    "solo-one": RawReplayPlanStatus.EXECUTED.value,
    "solo-two": RawReplayPlanStatus.EXECUTED.value,
    "membership-terminal": RawReplayPlanStatus.TERMINAL.value,
    "membership-deferred": RawReplayPlanStatus.DEFERRED.value,
}
_MAX_APPLY_PASSES = 8
_T = TypeVar("_T")


class FaultBoundary(StrEnum):
    """Real durability boundaries exercised by the proof matrix."""

    BEFORE_OUTCOME_COMMIT = "before_outcome_commit"
    AFTER_OUTCOME_COMMIT_BEFORE_CENSUS_FINALIZATION = "after_outcome_commit_before_census_finalization"
    DURING_RESUMED_BATCH = "during_resumed_batch"


class RawAuthorityRestartProofError(RuntimeError):
    """Raised when production evidence does not satisfy the proof contract."""


class _InjectedCrashError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedTopology:
    archive_root: Path
    preview_census_id: str
    preview_sequence_no: int
    plan_ids_by_role: dict[str, str]
    raw_ids_by_role: dict[str, tuple[str, ...]]
    deferred_raw_id: str
    raw_count: int
    membership_row_count: int
    component_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CrashEvidence:
    label: str
    boundary: str
    census_id: str
    sequence_no: int
    plan_id: str | None
    selected_plan_count: int
    recorded_selected_plan_count: int
    validated_executed_receipt_count: int
    interrupted_census_count_before_crash: int


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    crashes: tuple[CrashEvidence, ...]
    apply_passes: tuple[dict[str, object], ...]
    fixed_point_census_ids: tuple[str, str]
    fixed_point_inventory_digest: str
    fixed_point_residual_digest: str


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RawAuthorityRestartProofError(detail)


def _require_not_none(value: _T | None, detail: str) -> _T:
    if value is None:
        raise RawAuthorityRestartProofError(detail)
    return value


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _conversation(session_id: str, *, text: str, update_time: int) -> dict[str, object]:
    return {
        "id": session_id,
        "title": session_id,
        "create_time": 1,
        "update_time": update_time,
        "mapping": {
            "message-1": {
                "id": "message-1",
                "parent": None,
                "children": [],
                "message": {
                    "id": "message-1",
                    "author": {"role": "user"},
                    "create_time": update_time,
                    "content": {"content_type": "text", "parts": [text]},
                },
            }
        },
        "current_node": "message-1",
    }


def _write_bundle(
    root: Path,
    *,
    session_ids: tuple[str, str],
    source_path: str,
    acquired_at_ms: int,
) -> str:
    payload = json.dumps(
        [
            _conversation(
                session_id,
                text=f"restart proof {session_id} revision {acquired_at_ms}",
                update_time=acquired_at_ms,
            )
            for session_id in session_ids
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
        )


def _reparse_browser_payload(native_id: str) -> bytes:
    return json.dumps(
        [_conversation(native_id, text="current parser result", update_time=1)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _application_receipt_rows(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    logical_source_key: str,
) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT decision, decided_at_ms, decision_id
        FROM raw_revision_applications
        WHERE raw_id = ? AND logical_source_key = ?
        ORDER BY decided_at_ms, decision_id
        """,
        (raw_id, logical_source_key),
    ).fetchall()


def _exercise_accepted_head_reparse(case_root: Path) -> dict[str, object]:
    """Prove reparse reaffirmation ordering and its repair refusal semantics."""
    if case_root.exists():
        shutil.rmtree(case_root)
    initialize_active_archive_root(case_root)
    source_path = "browser-capture/chatgpt/reparse.json"
    native_id = "reparse-browser"
    payload = _reparse_browser_payload(native_id)
    current_session = _parse_one(Provider.CHATGPT, payload, source_path)[0]
    historical_session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=native_id,
        messages=[ParsedMessage(provider_message_id="message-1", role=Role.USER, text="historical parser result")],
    )
    current_hash = bytes.fromhex(session_content_hash(current_session))
    historical_hash = bytes.fromhex(session_content_hash(historical_session))

    with ArchiveStore.open_existing(case_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.UNKNOWN,
            capture_mode=Provider.UNKNOWN,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=1,
            native_id=native_id,
        )
        archive.write_parsed_for_retained_raw(
            historical_session,
            raw_id=raw_id,
            source_path=source_path,
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        blob_hash = (
            archive._ensure_source_conn()
            .execute("SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()[0]
        )
        baseline_receipt = RevisionApplicationReceipt(
            raw_id=raw_id,
            session_id="chatgpt-export:reparse-browser",
            logical_source_key="unknown:reparse-browser",
            source_revision=str(blob_hash),
            acquisition_generation=0,
            decision=ApplicationDecision.SELECTED_BASELINE,
            accepted_raw_id=raw_id,
            accepted_source_revision=str(blob_hash),
            accepted_content_hash=historical_hash,
            accepted_frontier_kind="byte",
            accepted_frontier=len(payload),
            baseline_raw_id=raw_id,
            detail="restart-proof:historical-browser-head",
        )
        record_revision_application_sync(archive._conn, baseline_receipt, decided_at_ms=1)
        archive.commit()

    ordering: list[str] = []
    real_reissue = revision_governance._reissue_accepted_head_reparse_receipt
    real_write = cast(Callable[..., str], revision_governance.__dict__["write_parsed_session_to_archive"])

    def record_reissue(*args: object, **kwargs: object) -> None:
        ordering.append("reissue")
        # Deliberately create a timestamp tie so this proof exercises the
        # decision_id tie-break rather than only the primary chronology key.
        kwargs["decided_at_ms"] = 1
        real_reissue(*args, **kwargs)  # type: ignore[arg-type]

    def record_write(*args: object, **kwargs: object) -> str:
        ordering.append("write")
        return real_write(*args, **kwargs)

    with (
        patch.object(revision_governance, "_reissue_accepted_head_reparse_receipt", record_reissue),
        patch.object(revision_governance, "write_parsed_session_to_archive", record_write),
        ArchiveStore.open_existing(case_root, read_only=False) as archive,
    ):
        archive.write_parsed_for_retained_raw(
            current_session,
            raw_id=raw_id,
            source_path=source_path,
            acquired_at_ms=2,
        )

    with sqlite3.connect(case_root / "index.db") as index_conn:
        applications = _application_receipt_rows(
            index_conn,
            raw_id=raw_id,
            logical_source_key="unknown:reparse-browser",
        )
        head_hash, session_hash = index_conn.execute(
            """
            SELECT h.accepted_content_hash, s.content_hash
            FROM raw_revision_heads AS h
            JOIN sessions AS s ON s.session_id = h.session_id
            WHERE h.logical_source_key = 'unknown:reparse-browser'
            """
        ).fetchone()

    with sqlite3.connect(case_root / "source.db") as source_conn:
        source_conn.execute(
            """
            UPDATE raw_sessions
            SET origin = 'unknown-export', capture_mode = 'unknown', logical_source_key = ?,
                revision_kind = 'full', source_revision = lower(hex(blob_hash)),
                revision_authority = 'quarantined', acquisition_generation = 0
            WHERE raw_id = ?
            """,
            ("unknown:reparse-browser", raw_id),
        )
        source_conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, 'chatgpt:reparse-browser', ?, ?, ?, ?, 0,
                      'quarantined', NULL, NULL)
            """,
            (raw_id, native_id, current_hash.hex(), current_hash, len(current_session.messages)),
        )
        source_conn.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, 'restart-proof-parser', 'complete', 1, 2, 'reparse reaffirmation proof')
            """,
            (raw_id,),
        )
        source_conn.commit()

    reparse_receipt = RevisionApplicationReceipt(
        raw_id=raw_id,
        session_id="chatgpt-export:reparse-browser",
        logical_source_key="unknown:reparse-browser",
        source_revision=str(blob_hash),
        acquisition_generation=0,
        decision=ApplicationDecision.REPARSE_REAFFIRMATION,
        accepted_raw_id=raw_id,
        accepted_source_revision=str(blob_hash),
        accepted_content_hash=current_hash,
        accepted_frontier_kind="byte",
        accepted_frontier=len(payload),
        detail="reparse:accepted_head_content_correction",
    )

    with sqlite3.connect(case_root / "index.db") as index_conn:
        index_conn.row_factory = sqlite3.Row
        index_conn.execute("ATTACH DATABASE ? AS source", (str(case_root / "source.db"),))
        repair_item = repair._inspect_browser_capture_origin_strategy(
            case_root,
            raw_id,
            conn=index_conn,
        )

    _require(ordering == ["reissue", "write"], f"accepted-head reparse ordering changed: {ordering}")
    _require(len(applications) == 2, "accepted-head reparse receipt count is not durable")
    application_decisions = [str(row[0]) for row in applications]
    application_timestamps = [int(row[1]) for row in applications]
    application_decision_ids = [str(row[2]) for row in applications]
    expected_application_decisions_by_id = {
        baseline_receipt.decision_id: baseline_receipt.decision.value,
        reparse_receipt.decision_id: reparse_receipt.decision.value,
    }
    expected_application_decision_pairs = sorted(expected_application_decisions_by_id.items())
    expected_application_decision_ids = [decision_id for decision_id, _decision in expected_application_decision_pairs]
    application_decision_pairs = sorted((str(row[2]), str(row[0])) for row in applications)
    _require(
        application_decision_pairs == expected_application_decision_pairs
        and application_timestamps == [1, 1]
        and application_decision_ids == expected_application_decision_ids,
        "accepted-head reparse receipts are not ordered by timestamp and decision id",
    )
    with sqlite3.connect(":memory:") as tie_conn:
        tie_conn.execute(
            """
            CREATE TABLE raw_revision_applications (
                raw_id TEXT NOT NULL,
                logical_source_key TEXT NOT NULL,
                decision TEXT NOT NULL,
                decided_at_ms INTEGER NOT NULL,
                decision_id TEXT NOT NULL
            )
            """
        )
        tie_conn.executemany(
            """
            INSERT INTO raw_revision_applications(
                raw_id, logical_source_key, decision, decided_at_ms, decision_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    raw_id,
                    "unknown:reparse-browser",
                    expected_application_decisions_by_id[decision_id],
                    1,
                    decision_id,
                )
                for decision_id in reversed(expected_application_decision_ids)
            ],
        )
        stable_tie_rows = _application_receipt_rows(
            tie_conn,
            raw_id=raw_id,
            logical_source_key="unknown:reparse-browser",
        )
        timestamp_only_tie_rows = tie_conn.execute(
            """
            SELECT decision, decided_at_ms, decision_id
            FROM raw_revision_applications
            WHERE raw_id = ? AND logical_source_key = ?
            ORDER BY decided_at_ms
            """,
            (raw_id, "unknown:reparse-browser"),
        ).fetchall()
        stable_tie_ids = [str(row[2]) for row in stable_tie_rows]
        timestamp_only_tie_ids = [str(row[2]) for row in timestamp_only_tie_rows]
    _require(
        stable_tie_ids == expected_application_decision_ids
        and sorted(timestamp_only_tie_ids) == expected_application_decision_ids,
        "equal-timestamp composite ordering did not preserve both receipt identities",
    )
    _require(bytes(head_hash) == current_hash == bytes(session_hash), "reparse head and session hashes diverged")
    _require(repair_item.status == "ineligible", "reparse-then-repair did not fail closed")
    _require(
        repair_item.reason == "current accepted head does not exactly prove the normalized session",
        f"reparse-then-repair refusal changed: {repair_item.reason}",
    )
    negative_control_root = case_root.with_name(f"{case_root.name}-negative-control")
    shutil.copytree(case_root, negative_control_root)
    try:
        with sqlite3.connect(negative_control_root / "index.db") as index_conn:
            index_conn.execute(
                """
                DELETE FROM raw_revision_applications
                WHERE raw_id = ? AND logical_source_key = ? AND decision = ?
                """,
                (raw_id, "unknown:reparse-browser", ApplicationDecision.REPARSE_REAFFIRMATION.value),
            )
            record_revision_application_sync(
                index_conn,
                RevisionApplicationReceipt(
                    raw_id=raw_id,
                    session_id="chatgpt-export:reparse-browser",
                    logical_source_key="unknown:reparse-browser",
                    source_revision=str(blob_hash),
                    acquisition_generation=0,
                    decision=ApplicationDecision.SELECTED_BASELINE,
                    accepted_raw_id=raw_id,
                    accepted_source_revision=str(blob_hash),
                    accepted_content_hash=current_hash,
                    accepted_frontier_kind="byte",
                    accepted_frontier=len(payload),
                    baseline_raw_id=raw_id,
                    detail="restart-proof:anti-vacuity-single-receipt",
                ),
                decided_at_ms=1,
            )
            index_conn.commit()

        with sqlite3.connect(negative_control_root / "source.db") as source_conn:
            source_conn.execute(
                """
                UPDATE raw_session_memberships
                SET normalized_content_hash = ?
                WHERE raw_id = ? AND logical_source_key = 'chatgpt:reparse-browser'
                """,
                (bytes(32), raw_id),
            )
            source_conn.commit()

        with sqlite3.connect(negative_control_root / "index.db") as index_conn:
            index_conn.row_factory = sqlite3.Row
            index_conn.execute("ATTACH DATABASE ? AS source", (str(negative_control_root / "source.db"),))
            wrong_hash_repair_item = repair._inspect_browser_capture_origin_strategy(
                negative_control_root,
                raw_id,
                conn=index_conn,
            )
        _require(
            wrong_hash_repair_item.reason == "membership census does not exactly reproduce the accepted session",
            "wrong normalized hash was not distinguished from the two-receipt refusal",
        )
        with sqlite3.connect(negative_control_root / "source.db") as source_conn:
            raw_count = int(source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
    finally:
        shutil.rmtree(negative_control_root)

    with sqlite3.connect(case_root / "index.db") as index_conn:
        retained_applications = _application_receipt_rows(
            index_conn,
            raw_id=raw_id,
            logical_source_key="unknown:reparse-browser",
        )
        retained_head_hash, retained_session_hash = index_conn.execute(
            """
            SELECT h.accepted_content_hash, s.content_hash
            FROM raw_revision_heads AS h
            JOIN sessions AS s ON s.session_id = h.session_id
            WHERE h.logical_source_key = 'unknown:reparse-browser'
            """
        ).fetchone()
    with sqlite3.connect(case_root / "source.db") as source_conn:
        retained_membership_hash = source_conn.execute(
            """
            SELECT normalized_content_hash
            FROM raw_session_memberships
            WHERE raw_id = ? AND logical_source_key = 'chatgpt:reparse-browser'
            """,
            (raw_id,),
        ).fetchone()[0]
    _require(
        sorted((str(row[2]), str(row[0])) for row in retained_applications) == expected_application_decision_pairs
        and bytes(retained_head_hash) == current_hash == bytes(retained_session_hash)
        and bytes(retained_membership_hash) == current_hash,
        "negative control mutated the retained accepted-head reparse archive",
    )
    _require(raw_count == 1, "reparse-then-repair mutated the source raw set while refusing")
    return {
        "ordering": ordering,
        "application_decisions": application_decisions,
        "application_count": len(applications),
        "head_content_hash": current_hash.hex(),
        "session_content_hash": bytes(session_hash).hex(),
        "repair_status": repair_item.status,
        "repair_reason": repair_item.reason,
        "source_raw_count_after_refusal": raw_count,
    }


def _census_plan_rows(root: Path, census_id: str) -> list[sqlite3.Row]:
    with sqlite3.connect(root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT cp.ordinal, cp.plan_id, cp.selected, cp.outcome_status,
                   cp.outcome_recorded, p.input_raw_ids_json
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE cp.census_id = ?
            ORDER BY cp.ordinal
            """,
            (census_id,),
        ).fetchall()


def _prepare_case(case_root: Path) -> PreparedTopology:
    """Create one fresh compact topology and publish its durable preview census."""
    if case_root.exists():
        shutil.rmtree(case_root)
    initialize_active_archive_root(case_root)

    raw_ids_by_role: dict[str, tuple[str, ...]] = {}
    raw_ids_by_role["solo-one"] = (
        _write_bundle(
            case_root,
            session_ids=("solo-one", "solo-one-companion"),
            source_path="solo-one.json",
            acquired_at_ms=1,
        ),
    )
    raw_ids_by_role["solo-two"] = (
        _write_bundle(
            case_root,
            session_ids=("solo-two", "solo-two-companion"),
            source_path="solo-two.json",
            acquired_at_ms=2,
        ),
    )
    raw_ids_by_role["membership-terminal"] = (
        _write_bundle(
            case_root,
            session_ids=("terminal-shared", "terminal-old-only"),
            source_path="terminal-old.json",
            acquired_at_ms=3,
        ),
        _write_bundle(
            case_root,
            session_ids=("terminal-shared", "terminal-new-only"),
            source_path="terminal-new.json",
            acquired_at_ms=4,
        ),
    )
    raw_ids_by_role["membership-deferred"] = (
        _write_bundle(
            case_root,
            session_ids=("deferred-shared", "deferred-old-only"),
            source_path="deferred-old.json",
            acquired_at_ms=5,
        ),
        _write_bundle(
            case_root,
            session_ids=("deferred-shared", "deferred-new-only"),
            source_path="deferred-new.json",
            acquired_at_ms=6,
        ),
    )

    config = _config(case_root)
    preparatory = repair.repair_raw_materialization(config, dry_run=True)
    preparatory_receipt = _require_not_none(
        preparatory.census_receipt,
        "production parser census did not publish a durable receipt",
    )
    _require(preparatory_receipt.quiescent, "compact topology did not complete the production parser census")
    _require(preparatory_receipt.plan_count == 4, "compact topology did not produce four replay components")

    deferred_raw_id = raw_ids_by_role["membership-deferred"][1]
    with sqlite3.connect(case_root / "source.db") as source_conn:
        membership = source_conn.execute(
            """
            SELECT provider_session_id, logical_source_key, source_revision
            FROM raw_session_memberships
            WHERE raw_id = ? AND logical_source_key = 'chatgpt:deferred-shared'
            """,
            (deferred_raw_id,),
        ).fetchone()
    membership = _require_not_none(
        membership,
        "production parser census omitted the deferred membership witness",
    )
    provider_session_id, logical_source_key, source_revision = (str(value) for value in membership)
    with sqlite3.connect(case_root / "index.db") as index_conn:
        index_conn.execute(
            """
            INSERT INTO raw_revision_applications (
                decision_id, raw_id, session_id, logical_source_key,
                source_revision, acquisition_generation, decision, detail, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 0, 'deferred',
                      'ordinary_replay:incomparable_existing_index_state', 0)
            """,
            (
                f"restart-proof-deferred:{deferred_raw_id}",
                deferred_raw_id,
                f"chatgpt-export:{provider_session_id}",
                logical_source_key,
                source_revision,
            ),
        )
        index_conn.commit()

    preview = repair.repair_raw_materialization(config, dry_run=True)
    preview_receipt = _require_not_none(
        preview.census_receipt,
        "production reconciler did not publish the topology preview census",
    )
    _require(preview_receipt.mode == "dry_run", "topology preview did not use the production dry-run census mode")
    _require(preview_receipt.lifecycle_status == "completed", "topology preview census was not finalized")
    _require(preview_receipt.quiescent, "topology preview was not quiescent after parser census completion")
    _require(preview_receipt.plan_count == 4, "topology preview lost a replay component")
    _require(preview_receipt.executable_plan_count == 4, "topology preview did not expose four executable plans")

    candidates = repair._raw_materialization_candidate_ids(config)
    components = repair._raw_materialization_ordered_components(candidates, archive_root=case_root)
    component_sizes = tuple(len(component) for component in components)
    _require(component_sizes == (1, 1, 2, 2), f"unexpected compact topology component sizes: {component_sizes}")
    _require(candidates.adoption_deferred == 1, "synthetic topology did not retain one durable deferred receipt")
    _require(len(candidates.raw_ids) == 5, "deferred membership sibling was not excluded from direct scheduling")
    _require(len(candidates.expanded_raw_ids) == 6, "membership expansion did not conserve all six raw rows")

    with sqlite3.connect(case_root / "source.db") as source_conn:
        raw_count = int(source_conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
        membership_row_count = int(source_conn.execute("SELECT COUNT(*) FROM raw_session_memberships").fetchone()[0])
        complete_census_count = int(
            source_conn.execute("SELECT COUNT(*) FROM raw_membership_census WHERE status = 'complete'").fetchone()[0]
        )
        sibling_keys = {
            str(row[0]): int(row[1])
            for row in source_conn.execute(
                """
                SELECT logical_source_key, COUNT(DISTINCT raw_id)
                FROM raw_session_memberships
                WHERE logical_source_key IN ('chatgpt:terminal-shared', 'chatgpt:deferred-shared')
                GROUP BY logical_source_key
                """
            )
        }
    _require(raw_count == 6, "compact topology did not retain exactly six raw rows")
    _require(membership_row_count == 12, "compact topology did not retain exactly twelve membership rows")
    _require(complete_census_count == 6, "not every synthetic raw has a complete durable membership census")
    _require(
        sibling_keys == {"chatgpt:deferred-shared": 2, "chatgpt:terminal-shared": 2},
        "membership sibling edges were not persisted by the production parser census",
    )

    preview_rows = _census_plan_rows(case_root, preview_receipt.census_id)
    _require(len(preview_rows) == 4, "topology preview ledger row count disagrees with its census receipt")
    _require(
        all(int(row["selected"]) == 0 and str(row["outcome_status"]) == "carried_forward" for row in preview_rows),
        "dry-run preview did not conserve every immutable plan as carried-forward",
    )
    role_by_component = {frozenset(raw_ids): role for role, raw_ids in raw_ids_by_role.items()}
    plan_ids_by_role: dict[str, str] = {}
    ordered_roles: list[str] = []
    for row in preview_rows:
        input_raw_ids = frozenset(cast(list[str], json.loads(str(row["input_raw_ids_json"]))))
        role = _require_not_none(
            role_by_component.get(input_raw_ids),
            "production plan inventory contains an unknown synthetic component",
        )
        ordered_roles.append(role)
        plan_ids_by_role[role] = str(row["plan_id"])
    _require(
        ordered_roles == ["solo-one", "solo-two", "membership-terminal", "membership-deferred"],
        f"production fairness order changed the fault targets: {ordered_roles}",
    )

    return PreparedTopology(
        archive_root=case_root,
        preview_census_id=preview_receipt.census_id,
        preview_sequence_no=preview_receipt.sequence_no,
        plan_ids_by_role=plan_ids_by_role,
        raw_ids_by_role=raw_ids_by_role,
        deferred_raw_id=deferred_raw_id,
        raw_count=raw_count,
        membership_row_count=membership_row_count,
        component_sizes=component_sizes,
    )


def _crash_snapshot(
    root: Path,
    *,
    label: str,
    boundary: FaultBoundary,
    census_id: str,
    plan_id: str | None,
    validated_executed_receipt_count: int,
) -> CrashEvidence:
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT sequence_no,
                   (SELECT COUNT(*) FROM raw_authority_census_plans
                    WHERE census_id = c.census_id AND selected = 1) AS selected_count,
                   (SELECT COUNT(*) FROM raw_authority_census_plans
                    WHERE census_id = c.census_id AND selected = 1 AND outcome_recorded = 1) AS recorded_count
            FROM raw_authority_censuses AS c
            WHERE census_id = ? AND lifecycle_status = 'planned'
            """,
            (census_id,),
        ).fetchone()
        interrupted_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'interrupted'"
            ).fetchone()[0]
        )
    _require(row is not None, f"{label} did not leave the production census in planned state")
    return CrashEvidence(
        label=label,
        boundary=boundary.value,
        census_id=census_id,
        sequence_no=int(row[0]),
        plan_id=plan_id,
        selected_plan_count=int(row[1]),
        recorded_selected_plan_count=int(row[2]),
        validated_executed_receipt_count=validated_executed_receipt_count,
        interrupted_census_count_before_crash=interrupted_count,
    )


def _inject_before_outcome_commit(
    topology: PreparedTopology,
    *,
    expected_role: str,
    label: str,
    boundary: FaultBoundary,
    require_interrupted_census: bool = False,
) -> CrashEvidence:
    expected_plan_id = topology.plan_ids_by_role[expected_role]
    captured: dict[str, object] = {}

    def crash_before_commit(
        archive_root: Path,
        census_id: str,
        outcome: raw_authority.RawReplayPlanOutcome,
    ) -> None:
        _require(Path(archive_root) == topology.archive_root, "fault injection escaped the synthetic archive")
        _require(outcome.plan_id == expected_plan_id, f"{label} reached {outcome.plan_id}, expected {expected_plan_id}")
        _require(
            outcome.status is RawReplayPlanStatus.EXECUTED,
            f"{label} must target an exact-postcondition executable plan, got {outcome.status.value}",
        )
        with sqlite3.connect(topology.archive_root / "source.db") as conn:
            conn.row_factory = sqlite3.Row
            ledger_row = conn.execute(
                """
                SELECT cp.outcome_recorded, p.plan_id, p.input_digest,
                       p.input_raw_ids_json, p.logical_keys_json,
                       p.authority_witness_json, p.source_preconditions_json,
                       p.index_preconditions_json
                FROM raw_authority_census_plans AS cp
                JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
                WHERE cp.census_id = ? AND cp.plan_id = ? AND cp.selected = 1
                """,
                (census_id, outcome.plan_id),
            ).fetchone()
            interrupted_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'interrupted'"
                ).fetchone()[0]
            )
        _require(ledger_row is not None, f"{label} could not read the selected immutable plan")
        _require(int(ledger_row["outcome_recorded"]) == 0, f"{label} was not immediately before the outcome commit")
        application_receipt = _require_not_none(
            outcome.application_receipt,
            f"{label} executable outcome lacked an exact application receipt",
        )
        receipt_valid, receipt_problems = raw_authority.validate_raw_replay_application_receipt(
            _plan_from_row(ledger_row),
            application_receipt,
        )
        _require(
            receipt_valid,
            f"{label} observed non-durable application postconditions: {'; '.join(receipt_problems)}",
        )
        if require_interrupted_census:
            _require(interrupted_count >= 1, f"{label} did not occur during a resumed batch")
        captured["census_id"] = census_id
        captured["plan_id"] = outcome.plan_id
        raise _InjectedCrashError(label)

    with patch.object(repair, "record_raw_replay_outcome", side_effect=crash_before_commit):
        try:
            repair.repair_raw_materialization(_config(topology.archive_root))
        except _InjectedCrashError as exc:
            _require(str(exc) == label, f"unexpected injected crash: {exc}")
        else:
            raise RawAuthorityRestartProofError(f"{label} fault was not reached")
    census_id = cast(str, captured.get("census_id"))
    plan_id = cast(str, captured.get("plan_id"))
    _require(bool(census_id and plan_id), f"{label} did not capture its durable boundary")
    evidence = _crash_snapshot(
        topology.archive_root,
        label=label,
        boundary=boundary,
        census_id=census_id,
        plan_id=plan_id,
        validated_executed_receipt_count=1,
    )
    _require(evidence.recorded_selected_plan_count == 0, f"{label} committed an outcome before crashing")
    return evidence


def _inject_after_outcomes_before_finalization(topology: PreparedTopology) -> CrashEvidence:
    label = "after all outcome commits before census finalization"
    captured: dict[str, object] = {}

    def crash_before_finalization(
        archive_root: Path,
        census_id: str,
        **_kwargs: object,
    ) -> None:
        _require(Path(archive_root) == topology.archive_root, "fault injection escaped the synthetic archive")
        with sqlite3.connect(topology.archive_root / "source.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT cp.plan_id, cp.outcome_status, cp.outcome_recorded,
                       cp.application_receipt_json, p.input_digest,
                       p.input_raw_ids_json, p.logical_keys_json,
                       p.authority_witness_json, p.source_preconditions_json,
                       p.index_preconditions_json
                FROM raw_authority_census_plans AS cp
                JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
                WHERE cp.census_id = ? AND cp.selected = 1
                ORDER BY cp.ordinal
                """,
                (census_id,),
            ).fetchall()
        _require(len(rows) == 4, "finalization fault did not observe all four selected production plans")
        _require(
            all(int(row["outcome_recorded"]) == 1 for row in rows),
            "finalization fault ran before every outcome commit",
        )
        _require(
            {str(row["outcome_status"]) for row in rows}
            == {
                RawReplayPlanStatus.EXECUTED.value,
                RawReplayPlanStatus.TERMINAL.value,
                RawReplayPlanStatus.DEFERRED.value,
            },
            "finalization fault did not observe executed, terminal, and deferred production outcomes",
        )
        executed_rows = [row for row in rows if str(row["outcome_status"]) == RawReplayPlanStatus.EXECUTED.value]
        _require(len(executed_rows) == 2, "finalization fault did not retain two executed application receipts")
        for row in executed_rows:
            receipt = require_json_document(
                json.loads(str(row["application_receipt_json"])),
                context="committed raw replay application receipt",
            )
            receipt_valid, receipt_problems = raw_authority.validate_raw_replay_application_receipt(
                _plan_from_row(row),
                receipt,
            )
            _require(
                receipt_valid,
                "committed executed outcome failed exact postcondition validation before finalization: "
                + "; ".join(receipt_problems),
            )
        captured["census_id"] = census_id
        raise _InjectedCrashError(label)

    with patch.object(repair, "finalize_raw_authority_census", side_effect=crash_before_finalization):
        try:
            repair.repair_raw_materialization(_config(topology.archive_root))
        except _InjectedCrashError as exc:
            _require(str(exc) == label, f"unexpected injected crash: {exc}")
        else:
            raise RawAuthorityRestartProofError("after-outcome finalization fault was not reached")
    census_id = cast(str, captured.get("census_id"))
    _require(bool(census_id), "after-outcome finalization fault did not capture its census")
    evidence = _crash_snapshot(
        topology.archive_root,
        label=label,
        boundary=FaultBoundary.AFTER_OUTCOME_COMMIT_BEFORE_CENSUS_FINALIZATION,
        census_id=census_id,
        plan_id=None,
        validated_executed_receipt_count=2,
    )
    _require(
        evidence.recorded_selected_plan_count == evidence.selected_plan_count == 4,
        "after-outcome crash did not leave four committed selected outcomes",
    )
    return evidence


def _result_summary(result: repair.RepairResult) -> dict[str, object]:
    receipt = result.census_receipt
    return {
        "success": result.success,
        "repaired_count": result.repaired_count,
        "detail": result.detail,
        "census_id": receipt.census_id if receipt is not None else None,
        "census_sequence": receipt.sequence_no if receipt is not None else None,
        "census_mode": receipt.mode if receipt is not None else None,
        "census_lifecycle": receipt.lifecycle_status if receipt is not None else None,
        "candidate_count": result.metrics.get("raw_materialization_candidate_count"),
        "remaining_candidate_count": result.metrics.get("raw_materialization_remaining_candidate_count"),
        "recovered_census_count": result.metrics.get("raw_materialization_recovered_census_count", 0.0),
        "conservation_error_count": result.metrics.get("raw_materialization_plan_conservation_error_count", 0.0),
        "plan_status_counts": {
            status.value: sum(outcome.status is status for outcome in result.plan_outcomes)
            for status in RawReplayPlanStatus
        },
    }


def _outcome_has_application_decision(outcome: RawReplayPlanOutcome, expected: str) -> bool:
    if outcome.application_receipt is None:
        return False
    rows = outcome.application_receipt.get("application_rows")
    return isinstance(rows, list) and any(isinstance(row, dict) and row.get("decision") == expected for row in rows)


def _resume_and_drain(topology: PreparedTopology) -> tuple[dict[str, object], ...]:
    results: list[dict[str, object]] = []
    config = _config(topology.archive_root)
    for _attempt in range(_MAX_APPLY_PASSES):
        result = repair.repair_raw_materialization(config)
        summary = _result_summary(result)
        results.append(summary)
        conservation_error = result.metrics.get("raw_materialization_plan_conservation_error_count", 0.0)
        _require(conservation_error == 0.0, "production repair reported a plan conservation error")
        unresolved = result.metrics.get("raw_materialization_unresolved_blocker_count", 0.0)
        _require(unresolved == 0.0, "restart became fail-closed behind an unresolved durable blocker")
        candidates = repair._raw_materialization_candidate_ids(config)
        with sqlite3.connect(topology.archive_root / "source.db") as conn:
            planned = int(
                conn.execute(
                    "SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'planned'"
                ).fetchone()[0]
            )
        if not candidates.raw_ids and planned == 0:
            if not result.success:
                # The scenario deliberately injects these two non-success
                # outcomes. Accept only their typed application evidence;
                # production continues to report the archive unhealthy.
                expected = {
                    topology.plan_ids_by_role["membership-terminal"]: (
                        RawReplayPlanStatus.TERMINAL,
                        "ambiguous",
                    ),
                    topology.plan_ids_by_role["membership-deferred"]: (
                        RawReplayPlanStatus.DEFERRED,
                        "deferred",
                    ),
                }
                exceptional = tuple(
                    outcome
                    for outcome in result.plan_outcomes
                    if outcome.status not in {RawReplayPlanStatus.EXECUTED, RawReplayPlanStatus.CARRIED_FORWARD}
                )
                expected_failure = {outcome.plan_id for outcome in exceptional} == set(expected)
                for outcome in exceptional:
                    expected_outcome = expected.get(outcome.plan_id)
                    expected_failure = (
                        expected_failure
                        and expected_outcome is not None
                        and outcome.status is expected_outcome[0]
                        and _outcome_has_application_decision(outcome, expected_outcome[1])
                    )
                blocker_metrics = (
                    result.metrics.get("raw_materialization_plan_conservation_error_count", 0),
                    result.metrics.get("raw_materialization_unresolved_blocker_count", 0),
                    result.metrics.get("raw_materialization_missing_blob_count", 0),
                    result.metrics.get("raw_materialization_resource_blocked_count", 0),
                    result.metrics.get("raw_materialization_no_progress_count", 0),
                    result.metrics.get("raw_materialization_remaining_byte_authority_pending_count", 0),
                )
                _require(
                    expected_failure and not any(blocker_metrics),
                    f"production repair drained candidates but reported unexplained failure: {result.detail}",
                )
            return tuple(results)
    raise RawAuthorityRestartProofError("production repair did not drain the compact topology after restart")


def _confirm_fixed_point(topology: PreparedTopology) -> tuple[tuple[str, str], str, str]:
    receipts = []
    for _pass in range(2):
        result = repair.repair_raw_materialization(_config(topology.archive_root), dry_run=True)
        receipt = _require_not_none(
            result.census_receipt,
            "quiescent dry-run did not publish a durable census receipt",
        )
        _require(result.success, f"quiescent dry-run failed: {result.detail}")
        _require(receipt.mode == "dry_run", "fixed-point proof used a non-dry-run census")
        _require(receipt.lifecycle_status == "completed", "fixed-point census was not finalized")
        _require(receipt.quiescent, "fixed-point census was not quiescent")
        _require(receipt.plan_count == 0, "fixed-point census retained replay plans")
        _require(receipt.executable_plan_count == 0, "fixed-point census retained executable plans")
        _require(receipt.residual_plan_count == 0, "fixed-point census retained residual plans")
        _require(receipt.post_plan_count == 0, "fixed-point postflight retained replay plans")
        receipts.append(receipt)
    first, second = receipts
    _require(first.inventory_digest == second.inventory_digest, "fixed-point inventory digests diverged")
    _require(first.residual_digest == second.residual_digest, "fixed-point residual digests diverged")
    _require(not first.fixed_point, "first quiescent census unexpectedly claimed a two-census fixed point")
    _require(second.fixed_point, "second matching quiescent census did not claim fixed point")
    _require(second.predecessor_census_id == first.census_id, "fixed-point censuses are not consecutive")
    return (first.census_id, second.census_id), second.inventory_digest, second.residual_digest


def _exercise_fault(topology: PreparedTopology, boundary: FaultBoundary) -> ExecutionEvidence:
    crashes: list[CrashEvidence] = []
    if boundary is FaultBoundary.BEFORE_OUTCOME_COMMIT:
        crashes.append(
            _inject_before_outcome_commit(
                topology,
                expected_role="solo-one",
                label="before first outcome commit",
                boundary=boundary,
            )
        )
    elif boundary is FaultBoundary.AFTER_OUTCOME_COMMIT_BEFORE_CENSUS_FINALIZATION:
        crashes.append(_inject_after_outcomes_before_finalization(topology))
    elif boundary is FaultBoundary.DURING_RESUMED_BATCH:
        crashes.append(
            _inject_before_outcome_commit(
                topology,
                expected_role="solo-one",
                label="initial crash before first outcome commit",
                boundary=boundary,
            )
        )
        crashes.append(
            _inject_before_outcome_commit(
                topology,
                expected_role="solo-two",
                label="crash before outcome commit during resumed batch",
                boundary=boundary,
                require_interrupted_census=True,
            )
        )
    else:  # pragma: no cover - exhaustive StrEnum guard
        raise AssertionError(boundary)

    apply_passes = _resume_and_drain(topology)
    fixed_point_ids, inventory_digest, residual_digest = _confirm_fixed_point(topology)
    return ExecutionEvidence(
        crashes=tuple(crashes),
        apply_passes=apply_passes,
        fixed_point_census_ids=fixed_point_ids,
        fixed_point_inventory_digest=inventory_digest,
        fixed_point_residual_digest=residual_digest,
    )


def _plan_from_row(row: sqlite3.Row) -> RawReplayPlan:
    return RawReplayPlan(
        plan_id=str(row["plan_id"]),
        input_digest=str(row["input_digest"]),
        input_raw_ids=tuple(cast(list[str], json.loads(str(row["input_raw_ids_json"])))),
        logical_keys=tuple(cast(list[str], json.loads(str(row["logical_keys_json"])))),
        authority_witness=require_json_document(
            json.loads(str(row["authority_witness_json"])),
            context="raw authority witness",
        ),
        source_preconditions=require_json_document(
            json.loads(str(row["source_preconditions_json"])),
            context="raw source preconditions",
        ),
        index_preconditions=require_json_document(
            json.loads(str(row["index_preconditions_json"])),
            context="raw index preconditions",
        ),
    )


def _audit_case(
    topology: PreparedTopology,
    boundary: FaultBoundary,
    execution: ExecutionEvidence,
) -> dict[str, object]:
    """Audit conservation and exact postconditions from the durable source ledger."""
    with sqlite3.connect(topology.archive_root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        censuses = conn.execute(
            """
            SELECT *
            FROM raw_authority_censuses
            WHERE sequence_no > ?
            ORDER BY sequence_no
            """,
            (topology.preview_sequence_no,),
        ).fetchall()
        plan_rows = conn.execute(
            """
            SELECT c.sequence_no, c.census_id, c.lifecycle_status, c.mode,
                   cp.ordinal, cp.plan_id, cp.selected, cp.outcome_status,
                   cp.outcome_recorded, cp.application_receipt_json,
                   p.input_digest, p.input_raw_ids_json, p.logical_keys_json,
                   p.authority_witness_json, p.source_preconditions_json,
                   p.index_preconditions_json
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp ON cp.census_id = c.census_id
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE c.sequence_no > ?
            ORDER BY c.sequence_no, cp.ordinal
            """,
            (topology.preview_sequence_no,),
        ).fetchall()
        blocker_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0]
        )
        planned_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'planned'").fetchone()[0]
        )

    _require(bool(censuses), "fault case published no post-preview census evidence")
    _require(planned_count == 0, "fault case left a planned census after restart")
    _require(blocker_count == 0, "fault case left an unresolved stale-plan blocker")

    rows_by_census: dict[str, list[sqlite3.Row]] = {}
    for row in plan_rows:
        rows_by_census.setdefault(str(row["census_id"]), []).append(row)
    for census in censuses:
        census_id = str(census["census_id"])
        rows = rows_by_census.get(census_id, [])
        _require(len(rows) == int(census["plan_count"]), f"{census_id} plan_count is not conserved in ledger rows")
        plan_ids = [str(row["plan_id"]) for row in rows]
        _require(len(plan_ids) == len(set(plan_ids)), f"{census_id} contains a duplicate immutable plan")
        if str(census["lifecycle_status"]) in {"completed", "interrupted"}:
            _require(
                all(int(row["outcome_recorded"]) == 1 for row in rows),
                f"{census_id} finalized without exactly one recorded outcome per plan",
            )
        for row in rows:
            selected = bool(row["selected"])
            status = str(row["outcome_status"])
            if selected:
                _require(status in _SELECTED_CONSERVED_STATUSES, f"selected plan has non-conserved status {status}")
                _require(status != RawReplayPlanStatus.CARRIED_FORWARD.value, "selected plan was carried forward")
            else:
                _require(
                    status == RawReplayPlanStatus.CARRIED_FORWARD.value,
                    "unselected immutable plan was not conserved as carried-forward",
                )

        page = raw_authority.read_raw_authority_census(topology.archive_root, census_id, limit=500)
        page_census = cast(dict[str, object], page["census"])
        _require(str(page_census["census_id"]) == census_id, "production census reader resolved the wrong receipt")
        _require(
            cast(int, page_census["plan_count"]) == int(census["plan_count"]),
            "production census reader lost plan rows",
        )
        _require(page["next_query_handle"] is None, "compact census unexpectedly required pagination")

    first_fault_census_id = execution.crashes[0].census_id
    first_fault_rows = rows_by_census.get(first_fault_census_id, [])
    initial_plan_ids = set(topology.plan_ids_by_role.values())
    _require(
        {str(row["plan_id"]) for row in first_fault_rows} == initial_plan_ids,
        "fault census did not represent every previewed immutable plan exactly once",
    )
    _require(all(bool(row["selected"]) for row in first_fault_rows), "fault census did not select every component")

    terminal_rows_by_plan: dict[str, list[sqlite3.Row]] = {plan_id: [] for plan_id in initial_plan_ids}
    selected_status_path: dict[str, list[str]] = {plan_id: [] for plan_id in initial_plan_ids}
    for row in plan_rows:
        plan_id = str(row["plan_id"])
        if plan_id not in initial_plan_ids or not bool(row["selected"]):
            continue
        status = str(row["outcome_status"])
        selected_status_path[plan_id].append(status)
        if status in _TERMINAL_STATUSES:
            terminal_rows_by_plan[plan_id].append(row)
        _require(
            status != RawReplayPlanStatus.REJECTED_STALE.value, "proof reached rejected-stale instead of convergence"
        )

    terminal_outcomes: list[dict[str, object]] = []
    for role, plan_id in topology.plan_ids_by_role.items():
        terminal_rows = terminal_rows_by_plan[plan_id]
        _require(len(terminal_rows) == 1, f"{role} has {len(terminal_rows)} terminal outcomes, expected exactly one")
        terminal_row = terminal_rows[0]
        status = str(terminal_row["outcome_status"])
        expected_status = _EXPECTED_TERMINAL_STATUSES[role]
        _require(status == expected_status, f"{role} ended as {status}, expected {expected_status}")
        receipt = cast(dict[str, object], json.loads(str(terminal_row["application_receipt_json"])))
        _require(
            receipt.get("schema") == "polylogue.raw-replay-application-receipt.v2",
            f"{role} terminal outcome lacks a production application receipt",
        )
        if status == RawReplayPlanStatus.EXECUTED.value:
            plan = _plan_from_row(terminal_row)
            valid, problems = raw_authority.validate_raw_replay_application_receipt(plan, receipt)
            _require(valid, f"{role} executed receipt failed production validation: {'; '.join(problems)}")
        terminal_outcomes.append(
            {
                "role": role,
                "plan_id": plan_id,
                "status": status,
                "census_id": str(terminal_row["census_id"]),
                "selected_status_path": selected_status_path[plan_id],
            }
        )

    terminal_status_counts = Counter(str(item["status"]) for item in terminal_outcomes)
    _require(
        terminal_status_counts
        == Counter(
            {
                RawReplayPlanStatus.EXECUTED.value: 2,
                RawReplayPlanStatus.TERMINAL.value: 1,
                RawReplayPlanStatus.DEFERRED.value: 1,
            }
        ),
        f"terminal outcome partition changed: {dict(terminal_status_counts)}",
    )

    fixed_first_id, fixed_second_id = execution.fixed_point_census_ids
    _require(len(censuses) >= 2, "fault case lacks two fixed-point censuses")
    fixed_first, fixed_second = censuses[-2:]
    _require(
        (str(fixed_first["census_id"]), str(fixed_second["census_id"])) == (fixed_first_id, fixed_second_id),
        "reported fixed-point receipts are not the final durable censuses",
    )
    for census in (fixed_first, fixed_second):
        _require(str(census["mode"]) == "dry_run", "fixed-point evidence contains an apply census")
        _require(str(census["lifecycle_status"]) == "completed", "fixed-point census is not completed")
        _require(bool(census["quiescent"]), "fixed-point census is not quiescent")
        _require(int(census["plan_count"]) == 0, "fixed-point census contains immutable plans")
        _require(int(census["executable_plan_count"]) == 0, "fixed-point census contains executable plans")
        _require(int(census["residual_plan_count"]) == 0, "fixed-point census contains residual plans")
        _require(int(census["post_plan_count"]) == 0, "fixed-point postflight contains immutable plans")
    for field in ("inventory_digest", "residual_digest", "scope_json", "parser_fingerprint"):
        _require(fixed_first[field] == fixed_second[field], f"fixed-point {field} changed between dry runs")
    _require(not bool(fixed_first["fixed_point"]), "first quiescent census claimed fixed point")
    _require(bool(fixed_second["fixed_point"]), "second quiescent census did not claim fixed point")
    _require(
        str(fixed_second["predecessor_census_id"]) == fixed_first_id,
        "fixed-point receipts are not predecessor-linked",
    )

    interrupted_ids = [
        str(census["census_id"]) for census in censuses if str(census["lifecycle_status"]) == "interrupted"
    ]
    expected_interrupted = 2 if boundary is FaultBoundary.DURING_RESUMED_BATCH else 1
    _require(
        len(interrupted_ids) == expected_interrupted,
        f"{boundary.value} finalized {len(interrupted_ids)} interrupted censuses, expected {expected_interrupted}",
    )
    _require(
        set(interrupted_ids) == {crash.census_id for crash in execution.crashes},
        "interrupted census identities do not match the injected crashes",
    )

    return {
        "schema": _CASE_SCHEMA,
        "boundary": boundary.value,
        "archive_root": str(topology.archive_root),
        "topology": {
            "raw_count": topology.raw_count,
            "membership_row_count": topology.membership_row_count,
            "component_count": len(topology.component_sizes),
            "component_sizes": list(topology.component_sizes),
            "preview_census_id": topology.preview_census_id,
            "preview_sequence_no": topology.preview_sequence_no,
            "plan_ids_by_role": topology.plan_ids_by_role,
            "deferred_raw_id": topology.deferred_raw_id,
        },
        "crashes": [asdict(crash) for crash in execution.crashes],
        "apply_passes": list(execution.apply_passes),
        "interrupted_census_ids": interrupted_ids,
        "terminal_outcomes": terminal_outcomes,
        "terminal_status_counts": dict(sorted(terminal_status_counts.items())),
        "fixed_point": {
            "census_ids": list(execution.fixed_point_census_ids),
            "inventory_digest": execution.fixed_point_inventory_digest,
            "residual_digest": execution.fixed_point_residual_digest,
            "second_census_fixed_point": True,
        },
        "conservation": {
            "law": (
                "for every finalized census, plan_count equals distinct census-plan rows; "
                "each selected plan has one recorded non-carried outcome; each unselected plan "
                "has one recorded carried-forward outcome; each preview plan reaches exactly one "
                "terminal executed/deferred/terminal outcome across retries"
            ),
            "initial_plan_count": len(initial_plan_ids),
            "all_census_rows_conserved": True,
            "each_initial_plan_terminal_once": True,
            "open_blocker_count": blocker_count,
            "planned_census_count": planned_count,
        },
    }


def _run_case(case_root: Path, boundary: FaultBoundary) -> dict[str, object]:
    topology = _prepare_case(case_root)
    execution = _exercise_fault(topology, boundary)
    return _audit_case(topology, boundary, execution)


def _stable_reparse_case_summary(reparse_case: dict[str, object]) -> dict[str, object]:
    return {
        "ordering": reparse_case["ordering"],
        "application_decisions": reparse_case["application_decisions"],
        "application_count": reparse_case["application_count"],
        "head_content_hash": reparse_case["head_content_hash"],
        "session_content_hash": reparse_case["session_content_hash"],
        "repair_status": reparse_case["repair_status"],
        "repair_reason": reparse_case["repair_reason"],
        "source_raw_count_after_refusal": reparse_case["source_raw_count_after_refusal"],
    }


def _proof_identity(cases: list[dict[str, object]], reparse_case: dict[str, object]) -> str:
    stable = [
        {
            "boundary": case["boundary"],
            "terminal_status_counts": case["terminal_status_counts"],
            "fixed_point": cast(dict[str, object], case["fixed_point"]),
            "interrupted_census_count": len(cast(list[str], case["interrupted_census_ids"])),
        }
        for case in cases
    ]
    digest_input = {
        "fault_matrix": stable,
        "accepted_head_reparse": _stable_reparse_case_summary(reparse_case),
    }
    digest = hashlib.sha256(json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{_PROOF_IDENTITY_PREFIX}{digest[:24]}"


def run_raw_authority_restart_proof(
    workdir: Path,
    *,
    keep: bool = False,
) -> dict[str, object]:
    """Run the complete fault matrix against isolated production-format archives."""
    root = workdir.expanduser().resolve() / "raw-authority-restart-proof"
    if root.exists():
        shutil.rmtree(root)
    cases_root = root / "cases"
    cases_root.mkdir(parents=True)

    cases = [_run_case(cases_root / boundary.value, boundary) for boundary in FaultBoundary]
    reparse_case = _exercise_accepted_head_reparse(cases_root / "accepted-head-reparse")
    report: dict[str, object] = {
        "schema": _PROOF_SCHEMA,
        "proof_id": _proof_identity(cases, reparse_case),
        "work_root": str(root),
        "case_archives_retained": keep,
        "production_limits": {
            "raw_artifact_limit": None,
            "max_payload_bytes": repair.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
            "parser_census_component_limit": repair.RAW_MATERIALIZATION_CENSUS_COMPONENT_LIMIT,
        },
        "fault_matrix": cases,
        "accepted_head_reparse": reparse_case,
    }
    report_path = root / "raw-authority-restart-proof.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not keep:
        shutil.rmtree(cases_root)
    return report


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".cache"))
    parser.add_argument("--keep", action="store_true", help="Retain the four fresh synthetic case archives.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_raw_authority_restart_proof(args.workdir, keep=args.keep)
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    else:
        print(payload["proof_id"], file=out)
    return 0


__all__ = [
    "FaultBoundary",
    "RawAuthorityRestartProofError",
    "main",
    "run_raw_authority_restart_proof",
]


if __name__ == "__main__":
    raise SystemExit(main())
