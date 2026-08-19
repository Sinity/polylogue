"""Focused real-generation tests for the reindex canary differ.

The production dependency is ``compare_reindex_generations`` reading two
canonical ``index.db`` files.  The anti-vacuity mutation for the core test is
changing the candidate block text: a synthetic summary comparator would stay
green, while the real blocks read model must report the changed row.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance import reindex_canary as reindex_canary_module
from polylogue.maintenance.rebuild_index import (
    RebuildIndexReceipt,
    RebuildIndexRequest,
    rebuild_index_from_source_sync,
    rebuild_selection_evidence,
)
from polylogue.maintenance.reindex_canary import (
    CanaryDifferenceReview,
    CanaryDiffReport,
    CanarySelection,
    CanarySelectionError,
    DeltaExpectation,
    DifferenceClassification,
    DifferenceOperation,
    ExpectedDifference,
    RowDifference,
    UnclassifiedCanaryDiffError,
    compare_reindex_generations,
    index_delta_expectations,
    load_canary_report,
    load_canary_review_manifest,
    run_reindex_canary,
    select_canary_sessions,
    write_canary_report,
)
from polylogue.sources.revision_backfill import (
    RebuildDeadlineExceededError,
    backfill_historical_revision_evidence,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import rebuild_source_evidence_snapshot
from polylogue.storage.sqlite import lifecycle as lifecycle_module
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.lifecycle import (
    CanaryChangeOperation,
    DerivedDeltaClass,
    ExpectedCanaryChange,
    FastForwardOperation,
    FastForwardOperationKind,
    IndexDeltaDeclaration,
    TargetedReprocessScope,
    undeclared_index_delta_versions,
)
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


@pytest.fixture(autouse=True)
def _synthetic_report_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep structural report tests independent of the CLI lifecycle route.

    The CLI red twins exercise the real archive-owned provenance capture and
    reload path. These focused tests construct standalone index pairs solely
    to pin comparison and review serialization behavior.
    """

    monkeypatch.setattr(
        reindex_canary_module,
        "_capture_archive_provenance",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        reindex_canary_module,
        "_validate_archive_provenance",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reindex_canary_module,
        "_validate_authoritative_rebuild_receipt",
        lambda *args, **kwargs: None,
    )


def _receipt_path(tmp_path: Path) -> Path:
    """A path placeholder for routes whose rebuild call is replaced in-test."""
    path = tmp_path / "schema-inference-gate-receipt.json"
    path.touch()
    return path


def _write_candidate_receipt(archive_root: Path, receipt_path: Path) -> Path:
    """Bind a receipt to a fixture that already completed phase 2."""
    return write_valid_rebuild_receipt(archive_root, receipt_path)


def _prepare_candidate_ready_archive(root: Path) -> str:
    """Build one real-route archive whose source authority is fully settled."""
    initialize_active_archive_root(root)
    payload = json.dumps(
        {
            "chat_messages": [
                {"uuid": "fresh-user", "sender": "human", "text": "hello"},
                {
                    "uuid": "fresh-assistant",
                    "sender": "assistant",
                    "text": "world",
                    "attachments": [
                        {
                            "id": "fresh-attachment",
                            "name": "fresh.txt",
                            "mimeType": "text/plain",
                            "size": 16,
                            "extracted_content": "attachment bytes",
                        }
                    ],
                },
            ]
        }
    ).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_AI,
            payload=payload,
            source_path="fresh.json",
            native_id="fresh",
            acquired_at_ms=1,
        )
    backfill_historical_revision_evidence(root)
    return raw_id


def _seed_index(
    path: Path,
    *,
    sessions: tuple[str, ...] = ("alpha",),
    block_text: str = "stable transcript",
    profile_materialized_at: str = "first-run",
    profile_message_count: int = 1,
    origins: tuple[str, ...] | None = None,
) -> None:
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as connection:
        session_origins = origins or ("codex-session",) * len(sessions)
        assert len(session_origins) == len(sessions)
        for native_id, origin in zip(sessions, session_origins, strict=True):
            session_id = f"{origin}:{native_id}"
            connection.execute(
                """
                INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (native_id, origin, f"raw-{native_id}", hashlib.sha256(native_id.encode()).digest()),
            )
            connection.execute(
                """
                INSERT INTO messages(session_id, position, role, material_origin, content_hash)
                VALUES (?, 0, 'user', 'human_authored', ?)
                """,
                (session_id, hashlib.sha256((native_id + ":message").encode()).digest()),
            )
            connection.execute(
                """
                INSERT INTO blocks(message_id, session_id, position, block_type, text)
                VALUES (?, ?, 0, 'text', ?)
                """,
                (f"{session_id}:0.0", session_id, block_text),
            )
            connection.execute(
                """
                INSERT INTO session_profiles(session_id, materialized_at, message_count, tags_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, profile_materialized_at, profile_message_count, '{"b":2,"a":1}'),
            )
        connection.commit()


def _seed_action(path: Path, *, tool_input: str) -> None:
    """Create one canonical actions-view row from the real index schema."""

    session_id = "codex-session:alpha"
    message_id = f"{session_id}:0.0"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, tool_id, tool_input)
            VALUES (?, ?, 1, 'tool_use', 'tool-alpha', ?)
            """,
            (message_id, session_id, tool_input),
        )
        connection.execute(
            """
            INSERT INTO action_pairs(
                tool_use_block_id, session_id, message_id, tool_id, use_rank, tool_name
            ) VALUES (?, ?, ?, 'tool-alpha', 1, 'shell')
            """,
            (f"{message_id}:1", session_id, message_id),
        )
        connection.commit()


def _seed_session_link(path: Path, *, inheritance: str, resolved_parent: bool = False) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                resolved_dst_session_id, inheritance, observed_at_ms
            ) VALUES (?, 'codex-session', 'parent', 'resume', ?, ?, 1)
            """,
            (
                "codex-session:alpha",
                "codex-session:parent" if resolved_parent else None,
                inheritance,
            ),
        )
        connection.commit()


def _test_selection(index_path: Path) -> CanarySelection:
    return CanarySelection(
        index_path=index_path,
        sessions_per_origin=1,
        selected_session_ids=("codex-session:alpha",),
        selected_raw_ids=("raw-alpha",),
        sampled_session_ids=("codex-session:alpha",),
        pathology_session_ids=(),
        sample_session_ids=(),
        origin_counts=(),
    )


def _empty_comparison(current: Path, candidate: Path, session_ids: tuple[str, ...]) -> CanaryDiffReport:
    return CanaryDiffReport(
        current_index=current,
        candidate_index=candidate,
        session_ids=session_ids,
        compared_tables=(),
        missing_tables=(),
        missing_columns=(),
        differences=(),
    )


def _rebuild_receipt(selection: CanarySelection, comparison: CanaryDiffReport) -> dict[str, object]:
    from polylogue.maintenance.archive_verification import (
        REINDEX_CANARY_ACCEPTANCE_CHECKS,
        REINDEX_CANARY_ACCEPTANCE_PROFILE,
    )

    generation = {
        "generation_id": "gen-canary",
        "owner_id": "owner",
        "archive_root": str(comparison.candidate_index.parent),
        "index_path": str(comparison.candidate_index),
        "state": "inactive",
        "source_snapshot": "snapshot",
    }
    return {
        "receipt_schema_version": 4,
        "archive_root": str(comparison.candidate_index.parent),
        "selected_raw_count": len(selection.selected_raw_ids),
        "status": "replayed",
        "materialized": True,
        "generation": generation,
        "selection_evidence": rebuild_selection_evidence(
            selection.selected_raw_ids,
            archive_root=comparison.candidate_index.parent,
            generation_id="gen-canary",
            generation_owner_id="owner",
            candidate_index=comparison.candidate_index,
            source_snapshot="snapshot",
            selected_session_ids=selection.selected_session_ids,
        ),
        "source_evidence_after": "0" * 64,
        "canary_acceptance": {
            "profile": REINDEX_CANARY_ACCEPTANCE_PROFILE,
            "results": [
                {"name": name, "status": "ok", "summary": "fixture acceptance", "count": 0}
                for name in REINDEX_CANARY_ACCEPTANCE_CHECKS
            ],
        },
    }


def _offline_canary_rebuild(
    *,
    archive_root: Path,
    raw_ids: tuple[str, ...],
    selected_session_ids: tuple[str, ...],
    index_schema_version: int,
    schema_inference_receipt_path: Path,
) -> RebuildIndexReceipt:
    """Exercise canary post-rebuild guards with the real rebuild service.

    Daemon transport itself is covered separately against the production UDS
    endpoint. These tests mutate production rebuild output after that route.
    """
    return rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=archive_root,
            raw_ids=raw_ids,
            selected_session_ids=selected_session_ids,
            promote=False,
            canary=True,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
    )


def _offline_canary_discard(*, archive_root: Path, generation_id: str, generation_owner_id: str) -> None:
    """Exercise daemon-owned discard semantics against the fixture archive."""
    from polylogue.maintenance.rebuild_index import discard_inactive_rebuild_candidate

    discard_inactive_rebuild_candidate(archive_root, generation_id, generation_owner_id)


@pytest.fixture(autouse=True)
def _run_canary_postflight_tests_through_real_rebuild_service(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Keep post-rebuild guards local; daemon transport has its own route test."""
    if request.node.name != "test_daemon_canary_rebuild_posts_the_bound_canary_request_to_the_existing_daemon_route":
        monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", _offline_canary_rebuild)
        monkeypatch.setattr(
            "polylogue.daemon.bulk_rebuild.discard_daemon_canary_candidate",
            _offline_canary_discard,
        )


def test_equal_real_generations_ignore_only_materialization_metadata(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, profile_materialized_at="current-build")
    _seed_index(candidate, profile_materialized_at="candidate-build")

    report = compare_reindex_generations(current, candidate)

    assert report.differences == ()
    assert report.unclassified_count == 0
    assert {"sessions", "messages", "blocks", "session_profiles"}.issubset(report.compared_tables)


def test_differ_reports_real_core_and_derived_row_changes(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("alpha", "removed"))
    _seed_index(
        candidate,
        sessions=("alpha", "added"),
        block_text="changed transcript",
        profile_message_count=2,
    )

    report = compare_reindex_generations(current, candidate)

    assert report.unexpected_count > 0
    assert report.unclassified_count == 0
    operations = {(item.table, item.operation) for item in report.differences}
    assert ("blocks", DifferenceOperation.CHANGED) in operations
    assert ("blocks", DifferenceOperation.ADDED) in operations
    assert ("blocks", DifferenceOperation.REMOVED) in operations
    assert any(
        item.table == "session_profiles"
        and item.operation is DifferenceOperation.CHANGED
        and "message_count" in item.changed_columns
        for item in report.differences
    )
    assert all(item.classification is DifferenceClassification.UNEXPECTED for item in report.differences)


def test_missing_tables_and_columns_are_explicit_unexpected_differences(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    with sqlite3.connect(candidate) as connection:
        connection.execute("ALTER TABLE session_profiles DROP COLUMN tags_json")
        connection.execute("DROP TABLE blocks")
        connection.commit()

    report = compare_reindex_generations(current, candidate)

    assert "blocks" in report.missing_tables
    assert report.missing_columns == (("session_profiles", ("tags_json",)),)
    schema_differences = [item for item in report.differences if item.identity[0][0] == "__schema__"]
    assert {
        ("blocks", DifferenceOperation.REMOVED),
        ("session_profiles", DifferenceOperation.REMOVED),
    }.issubset({(item.table, item.operation) for item in schema_differences})
    assert report.unexpected_count == len(report.differences)


def test_expected_difference_is_structurally_accounted_for(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                identity=(("session_id", "codex-session:alpha"),),
                operations=(DifferenceOperation.CHANGED,),
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="the reviewed materializer change updates this aggregate",
            ),
        ),
    )

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert profile_changes
    assert all(item.classification is DifferenceClassification.EXPECTED for item in profile_changes)
    assert all("polylogue-example" in item.rationale for item in profile_changes)
    assert report.expected_count == len(profile_changes)


def test_expected_difference_cannot_hide_extra_changed_columns(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)
    with sqlite3.connect(candidate) as connection:
        connection.execute("UPDATE session_profiles SET tags_json = ?", ('{"extra":true}',))
        connection.commit()

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                identity=(("session_id", "codex-session:alpha"),),
                operations=(DifferenceOperation.CHANGED,),
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="the reviewed materializer change updates this aggregate",
            ),
        ),
    )

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert len(profile_changes) == 1
    assert profile_changes[0].changed_columns == ("tags_json", "message_count")
    assert profile_changes[0].classification is DifferenceClassification.UNEXPECTED


def test_expected_difference_for_alpha_does_not_waive_beta(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("alpha", "beta"))
    _seed_index(candidate, sessions=("alpha", "beta"), profile_message_count=2)
    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                identity=(("session_id", "codex-session:alpha"),),
                operations=(DifferenceOperation.CHANGED,),
                columns=("message_count",),
                bead_ref="ref",
                rationale="alpha only",
            ),
        ),
    )
    classifications = {
        dict(item.identity)["session_id"]: item.classification
        for item in report.differences
        if item.table == "session_profiles"
    }
    assert classifications == {
        "codex-session:alpha": DifferenceClassification.EXPECTED,
        "codex-session:beta": DifferenceClassification.UNEXPECTED,
    }


def test_expected_difference_requires_exact_operation_and_nonempty_signature() -> None:
    """A table-wide declaration cannot classify every future row difference."""

    with pytest.raises(ValueError, match="exactly one operation"):
        ExpectedDifference(
            table="session_profiles",
            identity=(("session_id", "codex-session:alpha"),),
            bead_ref="polylogue-example",
            rationale="too broad",
        )

    with pytest.raises(ValueError, match="non-empty changed-column signature"):
        ExpectedDifference(
            table="session_profiles",
            identity=(("session_id", "codex-session:alpha"),),
            operations=(DifferenceOperation.CHANGED,),
            bead_ref="polylogue-example",
            rationale="still too broad",
        )


def test_expected_difference_requires_exact_schema_asymmetry_signature(tmp_path: Path) -> None:
    """Schema deltas need the same precise operation and column signature."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    with sqlite3.connect(candidate) as connection:
        connection.execute("ALTER TABLE session_profiles DROP COLUMN tags_json")
        connection.commit()

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                identity=(("__schema__", "column"), ("name", "tags_json")),
                operations=(DifferenceOperation.REMOVED,),
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="wrong schema signature",
            ),
        ),
    )

    schema_delta = next(item for item in report.differences if item.table == "session_profiles")
    assert schema_delta.changed_columns == ("tags_json",)
    assert schema_delta.classification is DifferenceClassification.UNEXPECTED


def test_differ_compares_canonical_actions_view(tmp_path: Path) -> None:
    """Changing the blocks-backed action payload must surface through actions."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    _seed_action(current, tool_input='{"command":"current"}')
    _seed_action(candidate, tool_input='{"command":"candidate"}')

    report = compare_reindex_generations(current, candidate)

    action_delta = next(item for item in report.differences if item.table == "actions")
    assert action_delta.operation is DifferenceOperation.CHANGED
    assert action_delta.identity == (("tool_use_block_id", "codex-session:alpha:0.0:1"),)
    assert "tool_input" in action_delta.changed_columns


def test_differ_does_not_omit_session_links_from_the_canonical_relation_frame(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    _seed_session_link(current, inheritance="prefix-sharing")
    _seed_session_link(candidate, inheritance="spawned-fresh")

    report = compare_reindex_generations(current, candidate)

    link_delta = next(item for item in report.differences if item.table == "session_links")
    assert link_delta.operation is DifferenceOperation.CHANGED
    assert "inheritance" in link_delta.changed_columns
    assert "session_links" in report.compared_tables


def test_differ_excludes_child_session_link_when_only_its_destination_is_selected(tmp_path: Path) -> None:
    """A selected destination cannot pull an unselected child edge into the canary."""
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("parent", "alpha"))
    _seed_index(candidate, sessions=("parent", "alpha"))
    _seed_session_link(current, inheritance="prefix-sharing", resolved_parent=True)
    _seed_session_link(candidate, inheritance="spawned-fresh", resolved_parent=True)

    report = compare_reindex_generations(current, candidate, session_ids=("codex-session:parent",))

    assert report.differences == ()


def test_selected_sessions_bound_the_canary_to_a_real_subset(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("kept", "outside"))
    _seed_index(candidate, sessions=("kept", "outside"))
    with sqlite3.connect(candidate) as connection:
        connection.execute(
            "UPDATE blocks SET text = 'outside changed' WHERE session_id = ?",
            ("codex-session:outside",),
        )
        connection.commit()

    report = compare_reindex_generations(current, candidate, session_ids=("codex-session:kept",))

    assert report.session_ids == ("codex-session:kept",)
    assert report.differences == ()


def test_parser_fingerprint_binding_fails_closed_when_evidence_changes() -> None:
    evidence = {
        "replay_closure": {
            "raw_session_evidence": [
                {
                    "origin": "codex-session",
                    "parser_fingerprint": "parser-v1",
                    "lowering_fingerprint": "lower-v1",
                }
            ]
        }
    }
    expected = {
        ("codex-session", "parser-v1"),
    }
    reindex_canary_module._validate_parser_binding(
        evidence,
        expected_parser_fingerprints=expected,
        expected_lowering_fingerprint="lower-v1",
    )
    changed = json.loads(json.dumps(evidence))
    changed["replay_closure"]["raw_session_evidence"][0]["parser_fingerprint"] = "parser-v2"

    with pytest.raises(UnclassifiedCanaryDiffError, match="parser fingerprints"):
        reindex_canary_module._validate_parser_binding(
            changed,
            expected_parser_fingerprints=expected,
            expected_lowering_fingerprint="lower-v1",
        )


def test_rebuild_selection_evidence_binds_compared_sessions_and_production_replay_routing(tmp_path: Path) -> None:
    """The actual rebuild receipt binds both the replay and comparison denominator."""
    root = tmp_path / "archive"
    raw_id = _prepare_candidate_ready_archive(root)
    from polylogue.sources.origin_specs import materializer_fingerprint, replay_routing_fingerprint

    evidence = rebuild_selection_evidence(
        (raw_id,),
        archive_root=root,
        generation_id="generation",
        generation_owner_id="owner",
        candidate_index=root / "index.db",
        source_snapshot="snapshot",
        selected_session_ids=("claude-ai-export:fresh",),
    )

    assert evidence["selected_session_ids"] == ["claude-ai-export:fresh"]
    assert evidence["selected_session_count"] == 1
    closure = cast(dict[str, object], evidence["replay_closure"])
    raw_evidence = cast(list[dict[str, object]], closure["raw_session_evidence"])
    assert raw_evidence[0]["replay_routing_fingerprint"] == replay_routing_fingerprint()
    assert raw_evidence[0]["materializer_fingerprint"] == materializer_fingerprint()


def test_daemon_canary_rebuild_posts_the_bound_canary_request_to_the_existing_daemon_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canary transport is the real daemon UDS route, not a local writer bridge."""
    from polylogue.daemon import bulk_rebuild

    calls: dict[str, object] = {}

    class Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["init"] = (args, kwargs)

        def probe(self, **kwargs: object) -> dict[str, object]:
            calls["probe"] = kwargs
            return {"ok": True}

        def request_json(self, method: str, path: str, body: dict[str, object], **kwargs: object) -> dict[str, object]:
            calls["request"] = (method, path, body, kwargs)
            return {"status": "replayed"}

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", Client)
    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *args, **kwargs: "token")

    receipt = bulk_rebuild.run_daemon_canary_rebuild(
        archive_root=tmp_path,
        raw_ids=("raw-1",),
        selected_session_ids=("codex-session:one",),
        index_schema_version=42,
        schema_inference_receipt_path=tmp_path / "schema.json",
    )

    assert receipt == {"status": "replayed"}
    assert calls["request"] == (
        "POST",
        "/api/maintenance/rebuild-index",
        {
            "raw_ids": ["raw-1"],
            "selected_session_ids": ["codex-session:one"],
            "promote": False,
            "canary": True,
            "schema_inference_receipt_path": str(tmp_path / "schema.json"),
        },
        {},
    )
    _socket_path, client_options = cast(tuple[object, dict[str, object]], calls["init"])
    assert client_options["timeout_s"] is None


def test_daemon_canary_report_consumption_posts_to_the_writer_owned_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consumption uses the daemon route rather than local archive ownership."""
    from polylogue.daemon import bulk_rebuild

    calls: dict[str, object] = {}

    class Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["init"] = (args, kwargs)

        def request_json(self, method: str, path: str, body: dict[str, object], **kwargs: object) -> dict[str, object]:
            calls["request"] = (method, path, body, kwargs)
            return {"review_status": "reviewed"}

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", Client)
    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *args, **kwargs: "token")

    report_path = tmp_path / "report.json"
    assert bulk_rebuild.consume_daemon_canary_report(archive_root=tmp_path, report_path=report_path) == {
        "review_status": "reviewed"
    }
    assert calls["request"] == (
        "POST",
        "/api/maintenance/consume-canary-report",
        {"report_path": str(report_path.resolve())},
        {"raise_for_status": True},
    )
    _socket_path, client_options = cast(tuple[object, dict[str, object]], calls["init"])
    assert client_options["timeout_s"] is None


def test_daemon_canary_report_consumption_preserves_typed_validation_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The maintenance adapter retains the daemon's actionable 4xx detail for the CLI."""
    from polylogue.daemon import bulk_rebuild
    from polylogue.daemon_client import DaemonResponseError

    class Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def request_json(self, *args: object, **kwargs: object) -> None:
            raise DaemonResponseError(
                status=422,
                code="canary_report_invalid",
                detail="receipt is missing the canonical acceptance profile",
            )

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", Client)
    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *args, **kwargs: "token")

    with pytest.raises(UnclassifiedCanaryDiffError, match="missing the canonical acceptance profile"):
        bulk_rebuild.consume_daemon_canary_report(archive_root=tmp_path, report_path=tmp_path / "report.json")


def test_canary_cleanup_dispatches_dictionary_receipts_to_the_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Postflight failures must release a daemon receipt candidate through its owner."""
    captured: dict[str, object] = {}

    def discard(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.discard_daemon_canary_candidate", discard)

    errors = reindex_canary_module._discard_canary_candidate(
        tmp_path,
        {
            "generation": {
                "generation_id": "candidate-1",
                "owner_id": "owner-1",
            }
        },
    )

    assert errors == []
    assert captured == {
        "archive_root": tmp_path,
        "generation_id": "candidate-1",
        "generation_owner_id": "owner-1",
    }


def test_live_replay_routing_fingerprint_rejects_changed_running_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persisted routing evidence cannot approve a changed replay dispatcher."""
    monkeypatch.setattr("polylogue.sources.origin_specs.replay_routing_fingerprint", lambda: "changed-routing")

    with pytest.raises(UnclassifiedCanaryDiffError, match="running code"):
        reindex_canary_module._validate_live_replay_routing_fingerprint("recorded-routing")


def test_live_materializer_fingerprint_rejects_changed_running_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production guard rejects a materializer code mutation."""
    monkeypatch.setattr("polylogue.sources.origin_specs.materializer_fingerprint", lambda: "changed-materializer")

    with pytest.raises(UnclassifiedCanaryDiffError, match="materializer fingerprint no longer matches"):
        reindex_canary_module._validate_live_materializer_fingerprint("recorded-materializer")


def test_expected_delta_authority_resolves_outside_a_git_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Installed canaries resolve semantic authority from packaged declarations."""
    monkeypatch.chdir(tmp_path)
    difference = RowDifference(
        table="sessions",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "codex-session:sample"),),
        before={"title_ref": None},
        after={"title_ref": "message:codex-session:sample:user"},
        changed_columns=("title_ref",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    delta = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:44",
        rationale="declared targeted title reprocess",
    )
    reindex_canary_module._validate_expected_review_authorities((delta,))


def test_expected_delta_authority_rejects_unrelated_table() -> None:
    """A historical delta number cannot bless an arbitrary semantic change."""

    difference = RowDifference(
        table="blocks",
        operation=DifferenceOperation.CHANGED,
        identity=(("block_id", "block"),),
        before={"text": "before"},
        after={"text": "after"},
        changed_columns=("text",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    unrelated = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:44",
        rationale="unrelated packaged index delta",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="does not declare table blocks"):
        reindex_canary_module._validate_expected_review_authorities((unrelated,))


def test_expected_delta_authority_rejects_session_outside_declared_origin() -> None:
    """A table declaration cannot authorize rows outside its reprocess scope."""

    difference = RowDifference(
        table="sessions",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "chatgpt-export:sample"),),
        before={"title_ref": None},
        after={"title_ref": "message:chatgpt-export:sample:user"},
        changed_columns=("title_ref",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    outside_scope = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:44",
        rationale="unrelated origin",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="outside origin codex-session"):
        reindex_canary_module._validate_expected_review_authorities((outside_scope,))


def test_expected_delta_authority_rejects_undeclared_changed_column() -> None:
    """A semantic delta authorizes named values, not every column in its table."""

    difference = RowDifference(
        table="sessions",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "codex-session:sample"),),
        before={"content_hash": "before"},
        after={"content_hash": "after"},
        changed_columns=("content_hash",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    undeclared_column = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:44",
        rationale="unrelated column",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="does not declare changed columns"):
        reindex_canary_module._validate_expected_review_authorities((undeclared_column,))


def test_expected_delta_authority_rejects_nonsemantic_delta() -> None:
    """A DDL-only delta cannot authorize a changed row value."""

    difference = RowDifference(
        table="insight_materialization",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "session"), ("insight_type", "session_profile")),
        before={"materializer_version": 1},
        after={"materializer_version": 2},
        changed_columns=("materializer_version",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    constraint_only = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:33",
        rationale="constraint-only delta",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="does not declare a semantic reparse"):
        reindex_canary_module._validate_expected_review_authorities((constraint_only,))


def test_expected_delta_authority_rejects_unscoped_semantic_delta() -> None:
    """A semantic label without comparable objects cannot bless every table."""

    difference = RowDifference(
        table="session_events",
        operation=DifferenceOperation.REMOVED,
        identity=(("event_id", "event"),),
        before={"event_type": "agent_message"},
        after=None,
        changed_columns=("event_type",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    unscoped = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:42",
        rationale="unscoped writer-materialization delta",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="does not declare comparable table scope"):
        reindex_canary_module._validate_expected_review_authorities((unscoped,))


def test_unknown_expected_delta_authority_fails_closed() -> None:
    difference = RowDifference(
        table="blocks",
        operation=DifferenceOperation.CHANGED,
        identity=(("block_id", "block"),),
        before={"text": "before"},
        after={"text": "after"},
        changed_columns=("text",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    unknown = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:999999",
        rationale="not declared",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="unknown index delta"):
        reindex_canary_module._validate_expected_review_authorities((unknown,))


def test_canary_comparison_is_read_only(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    before = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size

    compare_reindex_generations(current, candidate)

    after = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size
    assert after == before


def test_revision_receipt_run_identity_is_not_a_semantic_canary_difference(tmp_path: Path) -> None:
    """Rebuild-local ids/times normalize while revision authority stays compared."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    session_id = "codex-session:alpha"
    raw_id = "raw-alpha"
    content_hash = hashlib.sha256(b"alpha").digest()
    for path, decision_id, decided_at_ms in ((current, "decision-current", 1), (candidate, "decision-candidate", 2)):
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                INSERT INTO raw_revision_applications(
                    decision_id, raw_id, session_id, logical_source_key, source_revision,
                    acquisition_generation, decision, accepted_raw_id,
                    accepted_source_revision, accepted_content_hash, detail, decided_at_ms
                ) VALUES (?, ?, ?, 'codex:alpha', '1', 0, 'selected_baseline', ?, '1', ?, 'selected', ?)
                """,
                (decision_id, raw_id, session_id, raw_id, content_hash, decided_at_ms),
            )
            connection.execute(
                """
                INSERT INTO raw_revision_heads(
                    logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                    accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                    acquisition_generation, decided_at_ms
                ) VALUES ('codex:alpha', ?, ?, '1', ?, 'semantic', 1, 0, ?)
                """,
                (session_id, raw_id, content_hash, decided_at_ms),
            )

    report = compare_reindex_generations(current, candidate)

    assert report.differences == ()


def test_revision_receipt_semantic_decision_remains_a_canary_difference(tmp_path: Path) -> None:
    """Normalizing attempt identity must not hide a changed authority decision."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    session_id = "codex-session:alpha"
    for path, decision in ((current, "selected_baseline"), (candidate, "superseded")):
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                INSERT INTO raw_revision_applications(
                    decision_id, raw_id, session_id, logical_source_key, source_revision,
                    acquisition_generation, decision, accepted_raw_id,
                    accepted_source_revision, accepted_content_hash, detail, decided_at_ms
                ) VALUES (?, 'raw-alpha', ?, 'codex:alpha', '1', 0, ?, NULL, NULL, NULL, 'decision', 1)
                """,
                (f"decision-{decision}", session_id, decision),
            )

    report = compare_reindex_generations(current, candidate)

    assert {difference.table for difference in report.differences} == {"raw_revision_applications"}


def test_selector_samples_each_origin_and_keeps_explicit_inputs(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(
        index,
        sessions=("codex-a", "codex-pathology", "chat-a", "chat-sample", "claude-a", "claude-pathology"),
        origins=(
            "codex-session",
            "codex-session",
            "chatgpt-export",
            "chatgpt-export",
            "claude-ai-export",
            "claude-ai-export",
        ),
    )

    selection = select_canary_sessions(
        index,
        sessions_per_origin=1,
        pathology_session_ids=("codex-session:codex-pathology", "claude-ai-export:claude-pathology"),
        sample_session_ids=("chatgpt-export:chat-sample",),
    )

    assert selection.origin_counts == (
        ("chatgpt-export", 2),
        ("claude-ai-export", 2),
        ("codex-session", 2),
    )
    assert selection.selected_session_ids == (
        "chatgpt-export:chat-a",
        "chatgpt-export:chat-sample",
        "claude-ai-export:claude-a",
        "claude-ai-export:claude-pathology",
        "codex-session:codex-a",
        "codex-session:codex-pathology",
    )
    assert selection.selected_raw_ids == (
        "raw-chat-a",
        "raw-chat-sample",
        "raw-claude-a",
        "raw-claude-pathology",
        "raw-codex-a",
        "raw-codex-pathology",
    )


def test_selector_refuses_unknown_or_non_replayable_explicit_sessions(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(index)
    with pytest.raises(CanarySelectionError, match="not indexed"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:missing",))
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE sessions SET raw_id = NULL")
        connection.commit()
    with pytest.raises(CanarySelectionError, match="no raw_id"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:alpha",))


def test_run_reindex_canary_automatically_includes_production_pathology_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real canary runner supplements operator IDs from the production manifest."""
    from polylogue.maintenance.pathology_zoo import pathology_zoo_session_ids

    current = tmp_path / "index.db"
    pathology_session_ids = pathology_zoo_session_ids()
    native_ids = tuple(session_id.split(":", 1)[1] for session_id in pathology_session_ids)
    origins = tuple(session_id.split(":", 1)[0] for session_id in pathology_session_ids)
    _seed_index(current, sessions=native_ids, origins=origins)
    captured: dict[str, object] = {}
    captured_receipt_path: Path | None = None

    class Receipt:
        def to_dict(self) -> dict[str, object]:
            return {"status": "replayed"}

    def fake_rebuild(**request: object) -> Receipt:
        nonlocal captured_receipt_path
        captured["raw_ids"] = tuple(cast(tuple[str, ...], request["raw_ids"]))
        captured["has_client_profile"] = "candidate_acceptance_checks" in request
        captured["promote"] = False
        captured_receipt_path = cast(Path, request["schema_inference_receipt_path"])
        return Receipt()

    def fake_compare(
        current_index: Path,
        candidate_index: Path,
        *,
        session_ids: tuple[str, ...],
        **provenance: object,
    ) -> CanaryDiffReport:
        captured["session_ids"] = session_ids
        captured.update(provenance)
        return _empty_comparison(current_index, candidate_index, session_ids)

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", fake_rebuild)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_selection_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_authoritative_rebuild_receipt", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_canary_candidate", lambda *args, **kwargs: current
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    receipt_path = _receipt_path(tmp_path)
    result = run_reindex_canary(
        tmp_path,
        input_index=current,
        schema_inference_receipt_path=receipt_path,
        sessions_per_origin=1,
        no_promote=True,
    )

    assert result.selection.pathology_session_ids == pathology_session_ids
    captured_session_ids = captured["session_ids"]
    captured_raw_ids = captured["raw_ids"]
    assert isinstance(captured_session_ids, tuple)
    assert isinstance(captured_raw_ids, tuple)
    assert set(pathology_session_ids) <= set(captured_session_ids)
    assert len(captured_raw_ids) == len(pathology_session_ids)
    assert captured["has_client_profile"] is False
    assert captured["promote"] is False
    assert captured_receipt_path == receipt_path


def test_selector_refuses_empty_automatic_selection_before_daemon_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source row with no indexed session cannot trigger full-source replay."""
    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    with sqlite3.connect(root / "index.db") as connection:
        connection.execute("DELETE FROM sessions")
        connection.commit()
    rebuild_called = False

    def unexpected_rebuild(*args: object, **kwargs: object) -> None:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("empty canary selection must not reach the rebuild engine")

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", unexpected_rebuild)

    with pytest.raises(CanarySelectionError, match="zero sessions|zero raw ids|full-source replay"):
        run_reindex_canary(
            root,
            schema_inference_receipt_path=_receipt_path(tmp_path),
            sessions_per_origin=1,
            no_promote=True,
        )
    assert not rebuild_called


def test_run_reindex_canary_rejects_missing_receipt_even_with_ambient_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    ambient_receipt = _write_candidate_receipt(root, tmp_path / "ambient-schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(ambient_receipt))

    with pytest.raises(CanarySelectionError, match="requires an explicit schema-inference receipt path"):
        run_reindex_canary(root, schema_inference_receipt_path=None, sessions_per_origin=1, no_promote=True)


@pytest.mark.parametrize("anchor_state", ["missing", "poisoned"])
def test_run_reindex_canary_cleans_candidate_after_comparison_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anchor_state: str
) -> None:
    """A post-rebuild canary failure cannot strand its inactive candidate."""
    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    receipt_path = _write_candidate_receipt(root, tmp_path / "receipt.json")
    anchor = root / ".index-active-pointer"
    expected_anchor: bytes | None = None

    def fail_compare(*args: object, **kwargs: object) -> CanaryDiffReport:
        nonlocal expected_anchor
        del kwargs
        if anchor.exists() or anchor.is_symlink():
            anchor.unlink()
        if anchor_state == "poisoned":
            candidate_path = Path(str(args[1]))
            anchor.write_text(str(candidate_path), encoding="utf-8")
            expected_anchor = anchor.read_bytes()
        raise RuntimeError("synthetic canary comparison failure")

    monkeypatch.setattr(reindex_canary_module, "compare_reindex_generations", fail_compare)

    with pytest.raises(RuntimeError, match="synthetic canary comparison failure"):
        run_reindex_canary(
            root,
            schema_inference_receipt_path=receipt_path,
            sessions_per_origin=1,
            no_promote=True,
        )

    assert not list((root / ".index-generations").glob("gen-*"))
    assert (anchor.read_bytes() if anchor.exists() else None) == expected_anchor


def test_run_reindex_canary_refuses_source_rows_without_indexed_sessions(tmp_path: Path) -> None:
    """A canary never widens an empty index selection to a full source replay."""
    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    with sqlite3.connect(root / "index.db") as connection:
        connection.execute("DELETE FROM sessions")
        connection.commit()
    receipt_path = _write_candidate_receipt(root, tmp_path / "receipt.json")

    with pytest.raises(CanarySelectionError, match="zero sessions|full-source replay"):
        run_reindex_canary(
            root,
            schema_inference_receipt_path=receipt_path,
            sessions_per_origin=1,
            no_promote=True,
        )


def test_run_reindex_canary_rejects_input_index_outside_archive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    external_index = tmp_path / "external" / "index.db"
    external_index.parent.mkdir()
    external_index.touch()
    receipt_path = _receipt_path(tmp_path)
    monkeypatch.setattr("polylogue.config.resolve_archive_root", lambda: tmp_path / "configured-live")
    selector_called = False

    def unexpected_selector(*args: object, **kwargs: object) -> None:
        nonlocal selector_called
        selector_called = True
        raise AssertionError("an outside-root input must be rejected before selection")

    monkeypatch.setattr("polylogue.maintenance.reindex_canary.select_canary_sessions", unexpected_selector)

    with pytest.raises(CanarySelectionError, match="inside or bound to the selected archive root"):
        run_reindex_canary(
            root, input_index=external_index, schema_inference_receipt_path=receipt_path, no_promote=True
        )
    assert not selector_called


def test_run_reindex_canary_accepts_split_root_active_pointer_through_real_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    external_index_root = tmp_path / "external-index-root"
    external_index_root.mkdir()
    external_index = external_index_root / "index.db"
    shutil.move(root / "index.db", external_index)
    (root / ".index-active-pointer").write_text(str(external_index), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "configured-live"))
    active_digest = hashlib.sha256(external_index.read_bytes()).hexdigest()
    evidence_before = rebuild_source_evidence_snapshot(root)
    receipt_path = _write_candidate_receipt(root, tmp_path / "schema-inference-gate-receipt.json")

    result = run_reindex_canary(
        root,
        input_index=external_index,
        schema_inference_receipt_path=receipt_path,
        sessions_per_origin=1,
        no_promote=True,
    )

    receipt = result.rebuild_receipt
    generation = receipt["generation"]
    assert isinstance(generation, dict)
    generation_id = generation["generation_id"]
    owner_id = generation["owner_id"]
    source_snapshot = generation["source_snapshot"]
    candidate_path = Path(str(generation["index_path"]))
    expected_candidate_path = external_index_root / ".index-generations" / str(generation_id) / "index.db"
    assert result.selection.index_path == external_index
    assert result.comparison.current_index.resolve() == external_index.resolve()
    assert result.comparison.candidate_index == candidate_path
    assert candidate_path == expected_candidate_path.resolve()
    assert candidate_path.is_file()
    assert generation["archive_root"] == str(root.resolve())
    assert generation["state"] == "inactive"
    assert isinstance(owner_id, str) and owner_id
    assert isinstance(source_snapshot, str) and source_snapshot
    assert source_snapshot == evidence_before == rebuild_source_evidence_snapshot(root)
    assert receipt["receipt_schema_version"] == 4
    assert receipt["source_evidence_after"] == rebuild_source_evidence_snapshot(root)
    canary_acceptance = receipt["canary_acceptance"]
    assert isinstance(canary_acceptance, dict)
    assert canary_acceptance["profile"] == "reindex-canary-v1"
    results = canary_acceptance["results"]
    assert isinstance(results, list)
    assert all(isinstance(result, dict) for result in results)
    assert [(result["name"], result["status"]) for result in cast(list[dict[str, object]], results)] == [
        ("pathology-zoo-invariants", "ok")
    ]
    assert hashlib.sha256(external_index.read_bytes()).hexdigest() == active_digest
    assert json.loads((candidate_path.parent / "generation.json").read_text(encoding="utf-8")) == generation

    transaction = receipt["transaction"]
    operation = receipt["operation"]
    assert transaction is None
    assert isinstance(operation, dict)
    operation_owner = operation["owner"]
    operation_generation = operation["generation"]
    operation_delta = operation["delta"]
    assert isinstance(operation_owner, dict)
    assert isinstance(operation_generation, dict)
    assert isinstance(operation_delta, dict)
    assert operation_owner["generation_owner_id"] == owner_id
    assert operation_generation == {"generation_id": generation_id, "state": "inactive"}
    assert operation_delta["transaction_source_snapshot"] == source_snapshot
    assert operation_delta["source_snapshot_matches"] is True


def test_real_no_promote_rebuild_preserves_remediated_source_state(tmp_path: Path) -> None:
    """Candidate replay consumes phase-2 source state without changing it."""

    root = tmp_path / "archive"
    raw_id = _prepare_candidate_ready_archive(root)
    with sqlite3.connect(root / "source.db") as connection:
        source_state_before = connection.execute(
            """
            SELECT parsed_at_ms, parse_error,
                   (SELECT COUNT(*) FROM blob_refs WHERE ref_id = ? AND ref_type = 'attachment')
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id, raw_id),
        ).fetchone()
    assert source_state_before is not None
    assert source_state_before[0] is not None
    assert source_state_before[1] is None
    assert source_state_before[2] == 1
    active_digest = hashlib.sha256((root / "index.db").read_bytes()).hexdigest()
    evidence_before = rebuild_source_evidence_snapshot(root)

    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    receipt = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, promote=False, schema_inference_receipt_path=receipt_path)
    )

    with sqlite3.connect(root / "source.db") as connection:
        source_state_after = connection.execute(
            """
            SELECT parsed_at_ms, parse_error,
                   (SELECT COUNT(*) FROM blob_refs WHERE ref_id = ? AND ref_type = 'attachment')
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id, raw_id),
        ).fetchone()
    assert source_state_after == source_state_before
    assert receipt.generation["state"] == "inactive"
    assert receipt.generation["source_snapshot"] == evidence_before == rebuild_source_evidence_snapshot(root)
    assert receipt.source_evidence_after == rebuild_source_evidence_snapshot(root)
    assert hashlib.sha256((root / "index.db").read_bytes()).hexdigest() == active_digest


def test_run_reindex_canary_rejects_external_evidence_mutation_after_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source identity mutation after replay fails before inactive readiness."""

    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    active_index = root / "index.db"
    active_digest = hashlib.sha256(active_index.read_bytes()).hexdigest()
    receipt_path = _write_candidate_receipt(root, tmp_path / "schema-inference-gate-receipt.json")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "configured-live"))

    from polylogue.maintenance import replay as rebuild_replay

    real_replay = rebuild_replay.rebuild_index_from_source

    async def mutate_source_after_replay(*args: Any, **kwargs: Any) -> dict[str, object]:
        replay = await real_replay(*args, **kwargs)
        with sqlite3.connect(root / "source.db") as connection:
            connection.execute("UPDATE raw_sessions SET source_index = source_index + 1")
        return replay

    monkeypatch.setattr(rebuild_replay, "rebuild_index_from_source", mutate_source_after_replay)
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", _offline_canary_rebuild)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        run_reindex_canary(root, schema_inference_receipt_path=receipt_path, sessions_per_origin=1, no_promote=True)

    assert hashlib.sha256(active_index.read_bytes()).hexdigest() == active_digest


def test_run_reindex_canary_rejects_active_index_rotation_after_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canary cannot compare against an index that stopped being active."""

    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    location = ArchiveLocation.resolve(root)
    current_index = location.active_index_path
    rotated_index = tmp_path / "rotated" / "index.db"
    rotated_index.parent.mkdir(parents=True)
    shutil.copy2(current_index, rotated_index)
    receipt_path = _write_candidate_receipt(root, tmp_path / "schema-inference-gate-receipt.json")

    def rebuild_then_rotate(
        *,
        archive_root: Path,
        raw_ids: tuple[str, ...],
        selected_session_ids: tuple[str, ...],
        index_schema_version: int,
        schema_inference_receipt_path: Path,
    ) -> RebuildIndexReceipt:
        result = _offline_canary_rebuild(
            archive_root=archive_root,
            raw_ids=raw_ids,
            selected_session_ids=selected_session_ids,
            index_schema_version=index_schema_version,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
        (root / ".index-active-pointer").write_text(str(rotated_index), encoding="utf-8")
        return result

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", rebuild_then_rotate)

    with pytest.raises(CanarySelectionError, match="active index changed during rebuild"):
        run_reindex_canary(root, schema_inference_receipt_path=receipt_path, sessions_per_origin=1, no_promote=True)


def test_rebuild_rejects_evidence_mutation_in_deadline_interrupted_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred resumable pass cannot preserve a mutated source proof."""

    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)

    from polylogue.maintenance import replay as rebuild_replay

    async def mutate_source_then_interrupt(*args: Any, **kwargs: Any) -> dict[str, object]:
        with sqlite3.connect(root / "source.db") as connection:
            connection.execute("UPDATE raw_sessions SET source_path = source_path || '.mutated'")
        raise RebuildDeadlineExceededError("synthetic deadline")

    monkeypatch.setattr(rebuild_replay, "rebuild_index_from_source", mutate_source_then_interrupt)
    receipt_path = _write_candidate_receipt(root, tmp_path / "schema-inference-gate-receipt.json")

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                raw_batch_size=10,
                pass_deadline_seconds=30.0,
            )
        )


def test_run_reindex_canary_does_not_require_zoo_sessions_for_ordinary_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "index.db"
    _seed_index(current)
    selection = CanarySelection(
        index_path=current,
        sessions_per_origin=1,
        selected_session_ids=("codex-session:alpha",),
        selected_raw_ids=("raw-alpha",),
        sampled_session_ids=("codex-session:alpha",),
        pathology_session_ids=(),
        sample_session_ids=(),
        origin_counts=(("codex-session", 1),),
    )
    captured: dict[str, object] = {}

    class Receipt:
        def to_dict(self) -> dict[str, object]:
            return {"status": "replayed"}

    def fake_rebuild(**request: object) -> Receipt:
        captured["raw_ids"] = tuple(cast(tuple[str, ...], request["raw_ids"]))
        captured["has_client_profile"] = "candidate_acceptance_checks" in request
        return Receipt()

    def fake_compare(
        current_index: Path,
        candidate_index: Path,
        *,
        session_ids: tuple[str, ...],
        **provenance: object,
    ) -> CanaryDiffReport:
        captured["session_ids"] = session_ids
        captured.update(provenance)
        return _empty_comparison(current_index, candidate_index, session_ids)

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", fake_rebuild)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_selection_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_canary_candidate", lambda *args, **kwargs: current
    )
    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_selection_evidence", lambda *args, **kwargs: None
    )

    result = run_reindex_canary(
        tmp_path, input_index=current, schema_inference_receipt_path=_receipt_path(tmp_path), no_promote=True
    )

    assert result.selection.pathology_session_ids == ()
    assert captured["raw_ids"] == ("raw-alpha",)
    assert captured["has_client_profile"] is False


def test_run_reindex_canary_compares_its_own_inactive_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    current = root / "index.db"
    # A real database at a real pre-v44 version: the canary derives its expected
    # signatures from the active generation's own declared schema version, so a
    # zero-byte placeholder would silently exercise the source_version==0 path.
    with sqlite3.connect(current) as connection:
        connection.execute("PRAGMA user_version = 43")
    generation_id = "gen-canary"
    candidate = tmp_path / ".index-generations" / generation_id / "index.db"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    selection = _test_selection(current)

    class Receipt:
        archive_root = str(root.resolve())
        selected_raw_count = len(selection.selected_raw_ids)
        status = "replayed"
        materialized = True
        generation = {
            "generation_id": generation_id,
            "owner_id": "owner",
            "archive_root": str(root.resolve()),
            "index_path": str(candidate),
            "state": "inactive",
            "source_snapshot": "snapshot",
        }

        def to_dict(self) -> dict[str, object]:
            return {"generation": self.generation}

    captured: dict[str, object] = {}

    def fake_rebuild(**request: object) -> Receipt:
        captured["promote"] = False
        return Receipt()

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", fake_rebuild)
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_selection_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_authoritative_rebuild_receipt", lambda *args, **kwargs: None
    )

    def fake_compare(
        current_path: Path,
        candidate_path: Path,
        *,
        session_ids: tuple[str, ...],
        **provenance: object,
    ) -> CanaryDiffReport:
        captured.update({"paths": (current_path, candidate_path), "session_ids": session_ids})
        captured.update(provenance)
        return _empty_comparison(current_path, candidate_path, session_ids)

    monkeypatch.setattr("polylogue.maintenance.reindex_canary.compare_reindex_generations", fake_compare)

    result = run_reindex_canary(
        root, input_index=current, schema_inference_receipt_path=_receipt_path(root), no_promote=True
    )

    assert result.comparison.candidate_index == candidate
    assert captured["paths"] == (current, candidate)
    assert captured["promote"] is False
    # The classifier input is derived from the active generation, not ambient
    # configuration: v43 is crossed by the packaged v44 title_ref declaration.
    assert captured["source_index_version"] == 43
    derived = cast("tuple[DeltaExpectation, ...]", captured["delta_expectations"])
    assert derived
    assert all(43 < item.version <= INDEX_SCHEMA_VERSION for item in derived)
    assert any(item.table == "sessions" and "title_ref" in item.columns for item in derived)


def test_run_reindex_canary_rejects_arbitrary_sqlite_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "index.db"
    current.touch()
    arbitrary = tmp_path / "arbitrary.db"
    arbitrary.touch()
    selection = _test_selection(current)

    class Receipt:
        archive_root = str(tmp_path.resolve())
        selected_raw_count = len(selection.selected_raw_ids)
        status = "replayed"
        materialized = True
        generation = {
            "generation_id": "gen-canary",
            "owner_id": "owner",
            "archive_root": str(tmp_path.resolve()),
            "index_path": str(arbitrary),
            "state": "inactive",
            "source_snapshot": "snapshot",
        }

        def to_dict(self) -> dict[str, object]:
            return {"generation": self.generation}

    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary.select_canary_sessions", lambda *args, **kwargs: selection
    )
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", lambda **kwargs: Receipt())
    monkeypatch.setattr(
        "polylogue.maintenance.reindex_canary._validate_selection_evidence", lambda *args, **kwargs: None
    )

    with pytest.raises(CanarySelectionError, match="outside this archive's generation root"):
        run_reindex_canary(
            tmp_path, input_index=current, schema_inference_receipt_path=_receipt_path(tmp_path), no_promote=True
        )


def test_real_pathology_canary_rejects_cyclic_candidate_before_insight_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt inactive lineage candidate is rejected without touching active data."""
    from tests.infra.pathology_zoo import build_pathology_zoo

    zoo = build_pathology_zoo(tmp_path / "zoo")
    active_index = zoo.archive_root / "index.db"
    active_digest = hashlib.sha256(active_index.read_bytes()).hexdigest()
    receipt_path = write_valid_rebuild_receipt(zoo.archive_root, tmp_path / "schema-inference-gate-receipt.json")
    monkeypatch.setattr("polylogue.maintenance.pathology_zoo.pathology_zoo_is_present", lambda *args, **kwargs: False)
    from polylogue.maintenance import rebuild_index

    real_repopulate = rebuild_index._repopulate_bulk_build_derived_state

    def corrupt_candidate_after_replay(candidate_index: Path) -> dict[str, float]:
        timings = real_repopulate(candidate_index)
        with sqlite3.connect(candidate_index) as connection:
            connection.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE session_id = ?",
                ("codex-session:zoo-cycle-b", "codex-session:zoo-cycle-a"),
            )
            connection.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE session_id = ?",
                ("codex-session:zoo-cycle-a", "codex-session:zoo-cycle-b"),
            )
            connection.commit()
        return timings

    def unexpected_insight_repair(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid lineage candidate reached session insight materialization")

    monkeypatch.setattr(rebuild_index, "_repopulate_bulk_build_derived_state", corrupt_candidate_after_replay)
    monkeypatch.setattr("polylogue.storage.repair.repair_session_insights", unexpected_insight_repair)
    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", _offline_canary_rebuild)

    # The subject is that an invalid candidate is REJECTED before session-insight
    # materialization -- unexpected_insight_repair above is what actually pins
    # that. Which guard rejects it first is incidental, and this alternation has
    # always accepted several.
    #
    # "raw frontier integrity" stopped being reachable here in 4d35f59c4: it
    # required byte-proven authority of every `full` raw, but a first-ever
    # observation is admitted FULL/ASSERTED by design with no predecessor for a
    # byte proof to be about, so every source seen exactly once counted as a
    # broken head. With that false positive gone the candidate is now caught by
    # the next real guard, current-parser logical-key drift on a frozen raw --
    # which is the grouped-JSONL pathology's own multi-session shape being
    # detected, and just as valid a rejection.
    with pytest.raises(
        RuntimeError,
        match=(
            "session-lineage-acyclic"
            "|no longer parses to one session"
            "|raw frontier integrity"
            "|re-derived different current-parser logical keys"
        ),
    ):
        run_reindex_canary(
            zoo.archive_root,
            schema_inference_receipt_path=receipt_path,
            pathology_session_ids=("codex-session:zoo-cycle-a", "codex-session:zoo-cycle-b"),
            sessions_per_origin=1,
            no_promote=True,
        )

    assert hashlib.sha256(active_index.read_bytes()).hexdigest() == active_digest


def test_real_daemon_canary_candidate_mutation_reaches_report_writer_red_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-build canonical mutation is observed and cannot bypass review."""
    from polylogue.daemon import bulk_rebuild

    root = tmp_path / "archive"
    _prepare_candidate_ready_archive(root)
    receipt_path = _write_candidate_receipt(root, tmp_path / "receipt.json")
    rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            promote=True,
            schema_inference_receipt_path=receipt_path,
        )
    )
    active_index = ArchiveLocation.resolve(root).active_index_path
    active_digest = hashlib.sha256(active_index.read_bytes()).hexdigest()
    real_rebuild = _offline_canary_rebuild

    def build_then_mutate(
        *,
        archive_root: Path,
        raw_ids: tuple[str, ...],
        selected_session_ids: tuple[str, ...],
        index_schema_version: int,
        schema_inference_receipt_path: Path,
    ) -> Any:
        receipt = real_rebuild(
            archive_root=archive_root,
            raw_ids=raw_ids,
            selected_session_ids=selected_session_ids,
            index_schema_version=index_schema_version,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
        generation = receipt.generation
        candidate = Path(str(generation["index_path"]))
        with sqlite3.connect(candidate) as connection:
            connection.execute("UPDATE blocks SET text = 'post-build candidate mutation'")
            connection.commit()
        return receipt

    monkeypatch.setattr(bulk_rebuild, "run_daemon_canary_rebuild", build_then_mutate)
    result = run_reindex_canary(
        root,
        schema_inference_receipt_path=receipt_path,
        sessions_per_origin=1,
        no_promote=True,
    )

    assert hashlib.sha256(active_index.read_bytes()).hexdigest() == active_digest
    mutation = next(item for item in result.comparison.differences if item.table == "blocks")
    assert mutation.operation is DifferenceOperation.CHANGED
    assert mutation.changed_columns == ("text",)
    report_path = tmp_path / "reports" / "mutation.json"
    with pytest.raises(UnclassifiedCanaryDiffError, match="classification is incomplete"):
        write_canary_report(
            report_path,
            selection=result.selection,
            comparison=result.comparison,
            rebuild_receipt=result.rebuild_receipt,
            reviews=(),
        )
    assert not report_path.exists()


def test_run_reindex_canary_refuses_foreign_input_index_before_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection may only read the configured archive's active generation."""

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    active_index = archive_root / "index.db"
    foreign_index = tmp_path / "foreign.db"
    _seed_index(active_index)
    _seed_index(foreign_index)

    def fail_if_rebuild_runs(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate rebuild was invoked with a foreign input index")

    monkeypatch.setattr("polylogue.daemon.bulk_rebuild.run_daemon_canary_rebuild", fail_if_rebuild_runs)

    with pytest.raises(CanarySelectionError, match="configured archive active generation"):
        run_reindex_canary(
            archive_root,
            input_index=foreign_index,
            schema_inference_receipt_path=_receipt_path(tmp_path),
            no_promote=True,
        )


def test_durable_report_persists_unreviewed_discovery_and_refuses_consumption(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)

    durable = write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=(),
        allow_unreviewed=True,
    )
    assert durable.review_status == "unreviewed"
    assert json.loads(report_path.read_text(encoding="utf-8"))["review_status"] == "unreviewed"
    with pytest.raises(UnclassifiedCanaryDiffError, match="not fully reviewed"):
        load_canary_report(report_path)


def test_durable_report_persists_explicit_review_for_every_diff(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )

    durable = write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )
    assert durable.unclassified_count == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 9
    comparison_payload = payload["comparison"]
    assert isinstance(comparison_payload, dict)
    summary = comparison_payload["summary"]
    assert isinstance(summary, dict)
    assert summary["unclassified_count"] == 0
    assert summary["unexpected_count"] == len(reviews)
    assert "rebuild_receipt" in payload
    assert all(
        isinstance(review["authority"], dict) and review["authority"]["kind"] == "successor"
        for review in payload["reviews"]
    )


def test_review_authority_kind_is_bound_to_classification() -> None:
    difference = RowDifference(
        table="blocks",
        operation=DifferenceOperation.CHANGED,
        identity=(("block_id", "block"),),
        before={"text": "before"},
        after={"text": "after"},
        changed_columns=("text",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unexpected",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="expected canary differences"):
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.EXPECTED,
            reference="successor:polylogue-next",
            rationale="wrong authority kind",
        )
    with pytest.raises(UnclassifiedCanaryDiffError, match="unexpected canary differences"):
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="delta:33",
            rationale="wrong authority kind",
        )


def test_review_manifest_rejects_reference_that_disagrees_with_authority(tmp_path: Path) -> None:
    """The CLI manifest parser must not silently rewrite an audit authority."""

    manifest = tmp_path / "reviews.json"
    manifest.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": "blocks",
                        "operation": "changed",
                        "identity": {"block_id": "block"},
                        "changed_columns": ["text"],
                        "classification": "expected",
                        "reference": "delta:33",
                        "authority": {"kind": "delta", "id": "34"},
                        "rationale": "contradictory manifest audit fields",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="reference disagrees"):
        load_canary_review_manifest(manifest)


def test_review_manifest_accepts_packaged_delta_and_nonapproving_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review loading works outside Git and keeps unresolved differences red."""

    manifest = tmp_path / "reviews.json"
    manifest.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": "sessions",
                        "operation": "changed",
                        "identity": {"session_id": "codex-session:expected"},
                        "changed_columns": ["title_ref"],
                        "classification": "expected",
                        "reference": "delta:44",
                        "authority": {"kind": "delta", "id": "44"},
                        "rationale": "the packaged title-reprocess delta declares this expected difference",
                    },
                    {
                        "table": "blocks",
                        "operation": "changed",
                        "identity": {"block_id": "successor"},
                        "changed_columns": ["text"],
                        "classification": "unexpected",
                        "reference": "successor:polylogue-ox2iz",
                        "authority": {"kind": "successor", "id": "polylogue-ox2iz"},
                        "rationale": "open successor owns the unresolved difference",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    reviews = load_canary_review_manifest(manifest)
    assert [review.classification for review in reviews] == [
        DifferenceClassification.EXPECTED,
        DifferenceClassification.UNEXPECTED,
    ]


def test_review_manifest_rejects_bead_as_semantic_authority(tmp_path: Path) -> None:
    """Planning state cannot authorize a candidate's semantic difference."""

    manifest = tmp_path / "reviews.json"
    manifest.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "table": "blocks",
                        "operation": "changed",
                        "identity": {"block_id": "unknown"},
                        "changed_columns": ["text"],
                        "classification": "expected",
                        "reference": "bead:polylogue-does-not-exist",
                        "authority": {"kind": "bead", "id": "polylogue-does-not-exist"},
                        "rationale": "fabricated expected-difference authority",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnclassifiedCanaryDiffError, match="invalid structured authority"):
        load_canary_review_manifest(manifest)


def test_partial_canary_scopes_thread_membership_by_session_not_thread_aggregate(tmp_path: Path) -> None:
    """A selected thread member must not pull un-replayed siblings into the denominator."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("selected", "unselected"))
    _seed_index(candidate, sessions=("selected",))
    for index, session_ids in (
        (current, ("codex-session:selected", "codex-session:unselected")),
        (candidate, ("codex-session:selected",)),
    ):
        with sqlite3.connect(index) as connection:
            connection.execute(
                "INSERT INTO threads(thread_id, session_ids_json, session_count) VALUES (?, ?, ?)",
                ("thread-root", json.dumps(session_ids), len(session_ids)),
            )
            connection.executemany(
                "INSERT INTO thread_sessions(thread_id, session_id, position) VALUES (?, ?, ?)",
                (("thread-root", session_id, position) for position, session_id in enumerate(session_ids)),
            )
            connection.commit()

    report = compare_reindex_generations(current, candidate, session_ids=("codex-session:selected",))

    assert all(difference.table != "thread_sessions" for difference in report.differences)
    assert "threads" not in report.compared_tables


def test_loading_canary_report_rechecks_exact_review_coverage(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["reviews"] = payload["reviews"][:-1]
    payload["comparison"]["summary"] = {
        "difference_count": 0,
        "expected_count": 0,
        "unexpected_count": 0,
        "unclassified_count": 0,
        "counts_by_table": {},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match="review coverage is incomplete"):
        load_canary_report(report_path)


def test_loading_canary_report_rejects_difference_rationale_that_contradicts_review_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    selection = select_canary_sessions(current, sessions_per_origin=1)
    comparison = compare_reindex_generations(current, candidate, session_ids=selection.selected_session_ids)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="successor:polylogue-ox2iz",
            rationale="the structured review owns this disposition",
        )
        for difference in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["comparison"]["differences"][0]["rationale"] = "bead:unrelated: forged authority text"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(reindex_canary_module, "_validate_archive_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(reindex_canary_module, "_validate_authoritative_rebuild_receipt", lambda *args, **kwargs: None)

    with pytest.raises(UnclassifiedCanaryDiffError, match="rationale disagrees"):
        load_canary_report(report_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 4, "no authoritative rebuild receipt schema"),
        ("receipt_schema_version", 1, "invalid rebuild receipt"),
    ),
)
def test_loading_canary_report_rejects_ambiguous_prior_evidence_schema(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if field == "schema_version":
        payload[field] = value
    else:
        payload["rebuild_receipt"][field] = value
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match=message):
        load_canary_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "profile",
            "acceptance profile does not match",
        ),
        (
            "results",
            "acceptance attestation does not match",
        ),
        (
            "status",
            "acceptance check pathology-zoo-invariants is not ok",
        ),
    ),
)
def test_loading_canary_report_revalidates_canonical_acceptance_attestation(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Consumption rejects a receipt that omits or changes daemon-owned acceptance evidence."""
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    receipt = _rebuild_receipt(selection, comparison)
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=receipt,
        reviews=reviews,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    persisted_receipt = payload["rebuild_receipt"]
    assert isinstance(persisted_receipt, dict)
    persisted_acceptance = persisted_receipt["canary_acceptance"]
    assert isinstance(persisted_acceptance, dict)
    if mutation == "profile":
        persisted_acceptance["profile"] = "reindex-canary-v0"
    elif mutation == "results":
        persisted_acceptance["results"] = []
    else:
        results = persisted_acceptance["results"]
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        results[0]["status"] = "warning"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match=message):
        load_canary_report(report_path)


def test_loading_canary_report_rejects_tampered_candidate_provenance(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["rebuild_receipt"]["generation"]["index_path"] = str(tmp_path / "foreign.db")
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnclassifiedCanaryDiffError, match="does not identify the compared candidate"):
        load_canary_report(report_path)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("index", "selection index"),
        ("sessions", "selection sessions"),
        ("raw-ids", "selection does not match the authoritative rebuild receipt"),
    ),
)
def test_loading_canary_report_rejects_tampered_selection_binding(tmp_path: Path, tamper: str, message: str) -> None:
    current, candidate, report_path = tmp_path / "current.db", tmp_path / "candidate.db", tmp_path / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed")
    comparison, selection = (
        compare_reindex_generations(current, candidate),
        select_canary_sessions(current, sessions_per_origin=1),
    )
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            item, classification=DifferenceClassification.UNEXPECTED, reference="polylogue-ox2iz", rationale="r"
        )
        for item in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if tamper == "index":
        payload["selection"]["index_path"] = str(tmp_path / "foreign.db")
    elif tamper == "sessions":
        payload["selection"]["selected_session_ids"] = ["codex-session:foreign"]
    else:
        payload["selection"]["selected_raw_ids"] = ["raw-foreign"]
    report_path.write_text(json.dumps(payload))
    with pytest.raises(UnclassifiedCanaryDiffError, match=message):
        load_canary_report(report_path)


def test_loading_canary_report_recomputes_tampered_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["comparison"]["summary"] = {
        "difference_count": 0,
        "expected_count": 0,
        "unexpected_count": 0,
        "unclassified_count": 0,
        "counts_by_table": {},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(reindex_canary_module, "_validate_archive_provenance", lambda *args, **kwargs: None)

    loaded = load_canary_report(report_path)

    comparison_payload = loaded["comparison"]
    assert isinstance(comparison_payload, dict)
    summary = comparison_payload["summary"]
    assert isinstance(summary, dict)
    assert summary["difference_count"] == len(reviews)
    assert summary["unexpected_count"] == len(reviews)


def test_canary_report_uses_unique_temporary_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )
    original = tempfile.NamedTemporaryFile
    names: list[str] = []

    def recording_named_temporary_file(*args: Any, **kwargs: Any) -> Any:
        stream = original(*args, **kwargs)
        names.append(stream.name)
        return stream

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", recording_named_temporary_file)
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )
    write_canary_report(
        report_path,
        selection=selection,
        comparison=comparison,
        rebuild_receipt=_rebuild_receipt(selection, comparison),
        reviews=reviews,
    )

    assert len(names) == 2
    assert len(set(names)) == 2
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def test_canary_report_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-ox2iz",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_canary_report(
            report_path,
            selection=selection,
            comparison=comparison,
            rebuild_receipt=_rebuild_receipt(selection, comparison),
            reviews=reviews,
        )

    assert not report_path.exists()
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def _semantic_declaration(
    version: int,
    *,
    table: str,
    columns: tuple[str, ...],
    operations: tuple[CanaryChangeOperation, ...] = ("changed",),
    scope: TargetedReprocessScope | None = None,
) -> IndexDeltaDeclaration:
    """Build one packaged-shape declaration for classifier tests.

    Real declarations are product data that changes as the schema advances, so
    the classifier tests state their own input rather than binding to whichever
    live delta happens to carry a signature today.
    """

    classes = (
        (DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS,)
        if scope is not None
        else (DerivedDeltaClass.SEMANTIC_REPARSE,)
    )
    return IndexDeltaDeclaration(
        version=version,
        classes=classes,
        reprocess_scope=scope,
        expected_canary_changes=(ExpectedCanaryChange(table=table, operations=operations, columns=columns),),
    )


def test_declared_semantic_delta_classifies_its_own_predicted_difference(tmp_path: Path) -> None:
    """The crossed deltas -- not a hand-written manifest -- account for the diff."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)

    expectations = index_delta_expectations(
        40,
        41,
        declarations=(_semantic_declaration(41, table="session_profiles", columns=("message_count",)),),
    )
    report = compare_reindex_generations(current, candidate, delta_expectations=expectations)

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert profile_changes
    assert all(item.classification is DifferenceClassification.EXPECTED for item in profile_changes)
    assert all("index delta 41" in item.rationale for item in profile_changes)
    assert report.unexpected_count == 0


def test_declared_delta_does_not_swallow_a_planted_undeclared_change(tmp_path: Path) -> None:
    """Anti-vacuity (Ref polylogue-tjr4z): a real semantic diff stays unexpected.

    The planted difference is a genuine read-model change on a column no crossed
    delta declares, on the *same table* a delta does declare.  A classifier that
    matched by table -- or that reported a fabricated zero -- would call this
    expected; the real comparator must surface it.
    """

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, block_text="planted divergent transcript")
    with sqlite3.connect(candidate) as connection:
        connection.execute("UPDATE session_profiles SET tags_json = ?", ('{"planted":true}',))
        connection.commit()

    expectations = index_delta_expectations(
        40,
        41,
        declarations=(_semantic_declaration(41, table="session_profiles", columns=("message_count",)),),
    )
    report = compare_reindex_generations(current, candidate, delta_expectations=expectations)

    planted = [item for item in report.differences if item.table in {"blocks", "session_profiles"}]
    assert {item.table for item in planted} == {"blocks", "session_profiles"}
    assert all(item.classification is DifferenceClassification.UNEXPECTED for item in planted)
    assert report.unexpected_count >= 2
    assert report.expected_count == 0


def test_declared_delta_cannot_absorb_a_row_that_also_changed_an_undeclared_column(tmp_path: Path) -> None:
    """A partially declared row is unexpected in whole, never partly waived."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)
    with sqlite3.connect(candidate) as connection:
        connection.execute("UPDATE session_profiles SET tags_json = ?", ('{"extra":true}',))
        connection.commit()

    expectations = index_delta_expectations(
        40,
        41,
        declarations=(_semantic_declaration(41, table="session_profiles", columns=("message_count",)),),
    )
    report = compare_reindex_generations(current, candidate, delta_expectations=expectations)

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert len(profile_changes) == 1
    assert set(profile_changes[0].changed_columns) == {"tags_json", "message_count"}
    assert profile_changes[0].classification is DifferenceClassification.UNEXPECTED


def test_declared_delta_scope_does_not_reach_another_origin(tmp_path: Path) -> None:
    """A targeted reprocess authorizes only the population it names."""

    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    sessions = ("alpha", "beta")
    origins = ("codex-session", "chatgpt-export")
    _seed_index(current, sessions=sessions, origins=origins)
    _seed_index(candidate, sessions=sessions, origins=origins, profile_message_count=2)

    expectations = index_delta_expectations(
        40,
        41,
        declarations=(
            _semantic_declaration(
                41,
                table="session_profiles",
                columns=("message_count",),
                scope=TargetedReprocessScope(origin="codex-session"),
            ),
        ),
    )
    report = compare_reindex_generations(current, candidate, delta_expectations=expectations)

    by_session = {
        str(dict(item.identity)["session_id"]): item for item in report.differences if item.table == "session_profiles"
    }
    assert by_session["codex-session:alpha"].classification is DifferenceClassification.EXPECTED
    assert by_session["chatgpt-export:beta"].classification is DifferenceClassification.UNEXPECTED


def test_shape_only_delta_contributes_no_expectations() -> None:
    """Only declarations that claim semantic work can authorize a row change."""

    shape_only = IndexDeltaDeclaration(
        version=41,
        classes=(DerivedDeltaClass.VIEW_ONLY,),
        operations=(
            FastForwardOperation(
                name="v41-view-only",
                kind=FastForwardOperationKind.REPLACE_VIEW,
                objects=(("view", "actions"),),
            ),
        ),
    )

    assert index_delta_expectations(40, 41, declarations=(shape_only,)) == ()


def test_semantic_reparse_delta_can_authorize_a_reviewed_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A semantic delta ships no fast-forward SQL, and still declares table scope.

    Sourcing comparable table scope only from ``operations`` made every
    SEMANTIC_REPARSE declaration unable to approve anything, because that class
    routes to a full rebuild and declares no SQL surface at all.
    """

    declaration = _semantic_declaration(41, table="session_profiles", columns=("message_count",))
    difference = RowDifference(
        table="session_profiles",
        operation=DifferenceOperation.CHANGED,
        identity=(("session_id", "codex-session:alpha"),),
        before={"message_count": 1},
        after={"message_count": 2},
        changed_columns=("message_count",),
        classification=DifferenceClassification.UNEXPECTED,
        rationale="unreviewed",
    )
    review = CanaryDifferenceReview.for_difference(
        difference,
        classification=DifferenceClassification.EXPECTED,
        reference="delta:41",
        rationale="the declared semantic reparse recomputes this aggregate",
    )

    monkeypatch.setattr(lifecycle_module, "INDEX_DELTA_DECLARATIONS", (declaration,))
    reindex_canary_module._validate_expected_review_authorities((review,))


def test_undeclared_crossed_versions_are_reported_not_silently_expected() -> None:
    """A crossed version with no declaration authorizes nothing, and says so."""

    declarations = (_semantic_declaration(41, table="session_profiles", columns=("message_count",)),)

    assert index_delta_expectations(40, 43, declarations=declarations) == (
        DeltaExpectation(
            version=41,
            table="session_profiles",
            operations=(DifferenceOperation.CHANGED,),
            columns=("message_count",),
        ),
    )
    assert undeclared_index_delta_versions(40, 43, declarations) == (42, 43)
