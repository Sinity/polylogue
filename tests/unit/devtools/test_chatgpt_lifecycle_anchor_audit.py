from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from devtools.chatgpt_lifecycle_anchor_audit import SCHEMA, TARGET_PREDICATE, main, run_audit
from devtools.command_catalog import COMMANDS
from polylogue.core.enums import Origin, Provider
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
    rendered = json.dumps(receipt, sort_keys=True)
    assert "raw-left" not in rendered
    assert "raw-right" not in rendered
    assert "/redacted/chatgpt-export.json" not in rendered
    assert (root / "source.db").read_bytes() == source_before
    assert (root / "index.db").read_bytes() == index_before


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
