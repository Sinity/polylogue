from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import devtools.chatgpt_lifecycle_anchor_audit as audit
from devtools.chatgpt_lifecycle_anchor_audit import SCHEMA, TARGET_PREDICATE, main, run_audit
from devtools.command_catalog import COMMANDS
from polylogue.archive.session_revision_membership import MembershipRevision, _relation
from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers import chatgpt as chatgpt_parser
from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent
from polylogue.sources.revision_backfill import _parse_one as production_parse_one
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session


def _node(node_id: str, role: str, text: str, parent: str | None, children: list[str]) -> dict[str, object]:
    return {
        "id": node_id,
        "parent": parent,
        "children": children,
        "message": {
            "id": node_id,
            "author": {"role": role},
            "content": {"content_type": "text", "parts": [text]},
            "metadata": {"finished_duration_sec": 5} if role == "assistant" else {},
            "end_turn": role == "assistant",
        },
    }


def _payload(order: list[str]) -> bytes:
    nodes = {
        "u1": _node("u1", "user", "do the work", None, ["node_a"]),
        "node_a": _node("node_a", "assistant", "first draft", "u1", ["node_b"]),
        "node_b": _node("node_b", "assistant", "final draft", "node_a", []),
    }
    return json.dumps(
        {"id": "tie-break-order", "mapping": {node_id: nodes[node_id] for node_id in order}, "current_node": "node_b"},
        separators=(",", ":"),
    ).encode()


def _write_raw(source: sqlite3.Connection, *, raw_id: str, payload: bytes) -> None:
    write_source_raw_session(
        source,
        origin=Origin.CHATGPT_EXPORT,
        capture_mode=Provider.CHATGPT,
        payload=payload,
        source_path="/redacted/chatgpt-export.json",
        source_index=0,
        acquired_at_ms=1,
        raw_id=raw_id,
    )
    source.execute(
        """
        INSERT INTO raw_session_memberships(
            raw_id, logical_source_key, provider_session_id, source_revision,
            normalized_content_hash, message_count, revision_authority
        ) VALUES (?, 'chatgpt-export:tie-break-order', 'tie-break-order', ?, ?, 3, 'quarantined')
        """,
        (raw_id, raw_id, b"x" * 32),
    )


def _archive_with_ordered_exports(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = sqlite3.connect(root / "source.db")
    try:
        blob_store = BlobStore(root / "blob")
        for payload in (_payload(["u1", "node_a", "node_b"]), _payload(["u1", "node_b", "node_a"])):
            blob_store.write_from_bytes(payload)
        _write_raw(source, raw_id="raw-left", payload=_payload(["u1", "node_a", "node_b"]))
        _write_raw(source, raw_id="raw-right", payload=_payload(["u1", "node_b", "node_a"]))
        source.commit()
    finally:
        source.close()
    index = sqlite3.connect(root / "index.db")
    try:
        index.execute(
            """
            INSERT INTO raw_revision_heads(
                logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES ('chatgpt-export:tie-break-order', 'chatgpt-export:tie-break-order', 'raw-left',
                      'raw-left', ?, 'semantic', ?, 0, NULL, 0)
            """,
            (b"y" * 32, 0),
        )
        index.commit()
    finally:
        index.close()
    return root


def _historical_parse_one(
    provider: Provider,
    payload: bytes,
    source_path: str,
    *,
    payload_path: Path | None = None,
    archive_root: Path | None = None,
    fallback_id_override: str | None = None,
) -> list[ParsedSession]:
    """Route through the parser with the pre-fix mapping-position tiebreak."""
    original_extract = chatgpt_parser._extract_generation_timings

    def historical_extract(mapping: dict[str, object]) -> list[object]:
        assert isinstance(mapping, dict)
        timings = original_extract(mapping)
        timed_message_ids: list[str] = []
        for node_id, raw_node in mapping.items():
            if not isinstance(raw_node, dict):
                continue
            raw_message = raw_node.get("message")
            if not isinstance(raw_message, dict):
                continue
            raw_author = raw_message.get("author")
            if not isinstance(raw_author, dict) or raw_author.get("role") not in {"assistant", "tool"}:
                continue
            metadata = raw_message.get("metadata")
            if not isinstance(metadata, dict) or not any(
                field in metadata for field in ("reasoning_start_time", "reasoning_end_time", "finished_duration_sec")
            ):
                continue
            timed_message_ids.append(str(raw_message.get("id") or raw_node.get("id") or node_id))
        assert timed_message_ids
        return [replace(timing, message_provider_id=timed_message_ids[0]) for timing in timings]

    with patch.object(chatgpt_parser, "_extract_generation_timings", historical_extract):
        return production_parse_one(
            provider,
            payload,
            source_path,
            payload_path=payload_path,
            archive_root=archive_root,
            fallback_id_override=fallback_id_override,
        )


def test_audit_runs_the_parser_to_classifier_route_read_only_and_is_sanitized(tmp_path: Path) -> None:
    root = _archive_with_ordered_exports(tmp_path)
    source_before = (root / "source.db").read_bytes()
    index_before = (root / "index.db").read_bytes()

    receipt = run_audit(root)

    assert receipt["schema"] == SCHEMA
    assert receipt["target_predicate"] == TARGET_PREDICATE
    assert receipt["denominators"] == {
        "selected_quarantined_chatgpt_raw_count": 2,
        "selected_membership_row_count": 2,
        "membershipless_selected_raw_count": 0,
        "logical_source_key_count": 1,
        "singleton_cohort_count": 0,
        "multi_candidate_cohort_count": 1,
        "raws_in_multi_candidate_cohorts": 2,
        "parsed_and_projected_raw_count": 2,
    }
    outcomes = cast(dict[str, object], receipt["outcomes"])
    assert cast(dict[str, int], outcomes["pair_relation_counts"]) == {
        "equal": 1,
        "a_contains_b": 0,
        "b_contains_a": 0,
        "conflict": 0,
    }
    assert outcomes["target_pair_count"] == 0
    provenance = cast(dict[str, object], receipt["provenance"])
    assert (
        provenance["producer_git_revision"]
        == subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], text=True).strip()
    )
    assert isinstance(provenance["producer_working_tree_clean"], bool)
    assert isinstance(provenance["producer_working_tree_status_sha256"], str)
    blob_store = cast(dict[str, object], provenance["blob_store"])
    assert blob_store["canonical_blob_count"] == 2
    integrity = cast(dict[str, object], blob_store["integrity"])
    assert integrity["verified_blob_count"] == 2
    assert integrity["hash_mismatch_count"] == 0
    assert integrity["invalid_namespace_entry_count"] == 0
    rendered = json.dumps(receipt, sort_keys=True)
    assert "raw-left" not in rendered
    assert "raw-right" not in rendered
    assert "/redacted/chatgpt-export.json" not in rendered
    assert (root / "source.db").read_bytes() == source_before
    assert (root / "index.db").read_bytes() == index_before


def test_audit_validates_git_before_opening_candidate_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive_with_ordered_exports(tmp_path)
    events: list[str] = []
    original_connect_read_only = audit._connect_read_only
    original_blob_snapshot = audit._blob_store_snapshot

    def tracked_git_provenance() -> dict[str, object]:
        events.append("git")
        return {
            "git_revision": "test-revision",
            "working_tree_clean": True,
            "working_tree_status_sha256": "test-status",
        }

    def tracked_connect_read_only(path: Path) -> sqlite3.Connection:
        events.append(f"open:{path.name}")
        return original_connect_read_only(path)

    def tracked_blob_snapshot(blob_store: BlobStore) -> dict[str, object]:
        events.append("scan:blob")
        return original_blob_snapshot(blob_store)

    monkeypatch.setattr(audit, "_git_provenance", tracked_git_provenance)
    monkeypatch.setattr(audit, "_connect_read_only", tracked_connect_read_only)
    monkeypatch.setattr(audit, "_blob_store_snapshot", tracked_blob_snapshot)

    run_audit(root)

    assert events == ["git", "open:source.db", "open:index.db", "scan:blob"]


def test_audit_matches_historical_moved_anchor_and_current_parser_is_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive_with_ordered_exports(tmp_path)
    current_parse_one = production_parse_one
    monkeypatch.setattr(audit, "_parse_one", _historical_parse_one)

    historical = run_audit(root)
    historical_outcomes = cast(dict[str, object], historical["outcomes"])
    assert cast(dict[str, int], historical_outcomes["pair_relation_counts"])["conflict"] == 1
    assert historical_outcomes["target_pair_count"] == 1
    historical_classifier = cast(dict[str, int], historical_outcomes["classifier_cohort_counts"])
    assert historical_classifier["cohorts_with_ambiguous_raw"] == 1
    assert historical_classifier["cohorts_with_accepted_raw"] == 0

    monkeypatch.setattr(audit, "_parse_one", current_parse_one)
    current = run_audit(root)
    current_outcomes = cast(dict[str, object], current["outcomes"])
    assert cast(dict[str, int], current_outcomes["pair_relation_counts"])["conflict"] == 0
    assert current_outcomes["target_pair_count"] == 0
    current_classifier = cast(dict[str, int], current_outcomes["classifier_cohort_counts"])
    assert current_classifier["cohorts_with_accepted_raw"] == 1
    assert current_classifier["cohorts_with_equivalent_raw"] == 1
    assert current_classifier["cohorts_with_ambiguous_raw"] == 0


def test_target_normalizes_lifecycle_measurements_and_rejects_other_event_changes(tmp_path: Path) -> None:
    _archive_with_ordered_exports(tmp_path)
    payloads = [_payload(["u1", "node_a", "node_b"]), _payload(["u1", "node_b", "node_a"])]
    left = _historical_parse_one(Provider.CHATGPT, payloads[0], "export.json", fallback_id_override="tie-break-order")[
        0
    ]
    right = _historical_parse_one(Provider.CHATGPT, payloads[1], "export.json", fallback_id_override="tie-break-order")[
        0
    ]

    left_member = audit._ParsedMember(MembershipRevision("raw-left", session_revision_projection(left)), left)
    right_member = audit._ParsedMember(MembershipRevision("raw-right", session_revision_projection(right)), right)
    relation = _relation(left_member.revision.projection, right_member.revision.projection)
    assert relation == "conflict"
    assert audit._matches_target(left_member, right_member, relation)

    measured_event = next(event for event in right.session_events if event.event_type == "generation_lifecycle")
    changed_measurement = measured_event.model_copy(
        update={
            "timestamp": "2099-01-01T00:00:00Z",
            "payload": {**measured_event.payload, "finished_duration_sec": 999},
        }
    )
    measurement_changed = right.model_copy(
        update={
            "session_events": [
                changed_measurement if event is measured_event else event for event in right.session_events
            ]
        }
    )
    measurement_member = audit._ParsedMember(
        MembershipRevision("raw-right", session_revision_projection(measurement_changed)), measurement_changed
    )
    measurement_relation = _relation(left_member.revision.projection, measurement_member.revision.projection)
    assert measurement_relation == "conflict"
    assert audit._matches_target(left_member, measurement_member, measurement_relation)

    unrelated_changed = right.model_copy(
        update={
            "session_events": [
                *right.session_events,
                ParsedSessionEvent(event_type="unrelated_observation", payload={"value": "changed"}),
            ]
        }
    )
    unrelated_member = audit._ParsedMember(
        MembershipRevision("raw-right", session_revision_projection(unrelated_changed)), unrelated_changed
    )
    unrelated_relation = _relation(left_member.revision.projection, unrelated_member.revision.projection)
    assert unrelated_relation == "conflict"
    assert not audit._matches_target(left_member, unrelated_member, unrelated_relation)


def test_target_rejects_red_twin_with_equal_attachment_contents_and_different_identities(tmp_path: Path) -> None:
    _archive_with_ordered_exports(tmp_path)
    payloads = [_payload(["u1", "node_a", "node_b"]), _payload(["u1", "node_b", "node_a"])]
    left = _historical_parse_one(Provider.CHATGPT, payloads[0], "export.json", fallback_id_override="tie-break-order")[
        0
    ]
    right = _historical_parse_one(Provider.CHATGPT, payloads[1], "export.json", fallback_id_override="tie-break-order")[
        0
    ]
    left_member = audit._ParsedMember(MembershipRevision("raw-left", session_revision_projection(left)), left)
    right_projection = session_revision_projection(right)
    red_twin_projection = replace(right_projection, attachment_identities=frozenset({b"red-twin-identity"}))
    red_twin_member = audit._ParsedMember(MembershipRevision("raw-right", red_twin_projection), right)

    assert (
        left_member.revision.projection.attachment_contents == red_twin_member.revision.projection.attachment_contents
    )
    assert (
        left_member.revision.projection.attachment_identities
        != red_twin_member.revision.projection.attachment_identities
    )
    relation = _relation(left_member.revision.projection, red_twin_member.revision.projection)
    assert relation == "conflict"
    assert not audit._matches_target(left_member, red_twin_member, relation)


def test_audit_receipt_is_deterministic_and_cli_registers_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _archive_with_ordered_exports(tmp_path)
    first = run_audit(root)
    second = run_audit(root)
    assert first == second
    receipt_path = tmp_path / "receipt.json"

    assert main(["--archive-root", str(root), "--receipt", str(receipt_path)]) == 0
    assert json.loads(receipt_path.read_text()) == first
    assert json.loads(capsys.readouterr().out) == first
    command = COMMANDS["workspace chatgpt-lifecycle-anchor-audit"]
    assert command.module == "devtools.chatgpt_lifecycle_anchor_audit"


def test_blob_integrity_identity_includes_observed_content_and_receipt_stays_outside_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _archive_with_ordered_exports(tmp_path)
    blob_store = BlobStore(root / "blob")
    blob_hash, _ = blob_store.write_from_bytes(b"extra blob")
    blob_path = blob_store.blob_path(blob_hash)
    original = blob_path.read_bytes()
    healthy = run_audit(root)

    blob_path.write_bytes(b"x" * len(original))
    first_corrupt = run_audit(root)
    blob_path.write_bytes(b"y" * len(original))
    second_corrupt = run_audit(root)
    healthy_blob = cast(dict[str, object], cast(dict[str, object], healthy["provenance"])["blob_store"])
    healthy_integrity = cast(dict[str, object], healthy_blob["integrity"])
    first_blob = cast(dict[str, object], cast(dict[str, object], first_corrupt["provenance"])["blob_store"])
    second_blob = cast(dict[str, object], cast(dict[str, object], second_corrupt["provenance"])["blob_store"])
    first_integrity = cast(dict[str, object], first_blob["integrity"])
    second_integrity = cast(dict[str, object], second_blob["integrity"])
    assert first_blob["snapshot_sha256"] == second_blob["snapshot_sha256"]
    assert first_integrity["hash_mismatch_count"] == second_integrity["hash_mismatch_count"] == 1
    assert first_integrity["integrity_sha256"] != second_integrity["integrity_sha256"]
    assert first_integrity["integrity_sha256"] != healthy_integrity["integrity_sha256"]
    assert blob_hash not in json.dumps(second_corrupt, sort_keys=True)

    with pytest.raises(SystemExit):
        main(["--archive-root", str(root), "--receipt", str(root / "receipt.json")])
    capsys.readouterr()
    assert not (root / "receipt.json").exists()


def test_audit_requires_a_real_sqlite_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "source.db").touch()
    (root / "index.db").touch()
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("CREATE TABLE raw_sessions(raw_id TEXT)")
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("CREATE TABLE raw_revision_heads(logical_source_key TEXT, accepted_raw_id TEXT)")
    try:
        run_audit(root)
    except sqlite3.OperationalError as error:
        assert "origin" in str(error)
    else:  # pragma: no cover
        raise AssertionError("audit accepted an archive without the production source schema")
