"""Coordination envelope tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.coordination.envelope import CommandResult, build_coordination_envelope


def _seed_coordination_archive(index: Path) -> None:
    conn = sqlite3.connect(index)
    try:
        conn.executescript(
            """
            PRAGMA user_version = 24;
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                native_id TEXT NOT NULL,
                origin TEXT NOT NULL,
                title TEXT,
                parent_session_id TEXT,
                root_session_id TEXT,
                branch_type TEXT,
                git_branch TEXT,
                sort_key_ms INTEGER
            );
            CREATE TABLE session_links (
                src_session_id TEXT,
                dst_native_id TEXT,
                resolved_dst_session_id TEXT,
                link_type TEXT,
                observed_at_ms INTEGER
            );
            CREATE TABLE session_runs (
                run_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                materialized_at TEXT NOT NULL,
                source_updated_at TEXT,
                native_session_id TEXT,
                native_parent_session_id TEXT,
                parent_run_ref TEXT,
                agent_ref TEXT,
                context_snapshot_ref TEXT,
                provider_origin TEXT NOT NULL DEFAULT 'codex-session',
                harness TEXT NOT NULL DEFAULT 'codex',
                role TEXT NOT NULL DEFAULT 'main',
                status TEXT NOT NULL,
                confidence TEXT NOT NULL DEFAULT 'raw',
                title TEXT,
                cwd TEXT,
                git_branch TEXT,
                lineage_refs_json TEXT NOT NULL DEFAULT '[]',
                search_text TEXT,
                evidence_refs_json TEXT NOT NULL,
                transcript_ref TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE session_observed_events (
                event_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                position INTEGER NOT NULL,
                materialized_at TEXT NOT NULL,
                source_updated_at TEXT,
                kind TEXT NOT NULL,
                summary TEXT,
                delivery_state TEXT NOT NULL DEFAULT 'observed',
                subject_ref TEXT,
                object_refs_json TEXT NOT NULL DEFAULT '[]',
                evidence_refs_json TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE session_context_snapshots (
                snapshot_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                position INTEGER NOT NULL,
                materialized_at TEXT NOT NULL,
                source_updated_at TEXT,
                boundary TEXT NOT NULL,
                inheritance_mode TEXT,
                segment_refs_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, native_id, origin, title, parent_session_id, root_session_id, branch_type, git_branch, sort_key_ms)
            VALUES
                ('codex-session:root', 'root', 'codex-session', 'Root work', NULL, 'codex-session:root', NULL, 'feature/coordination-envelope', 10),
                ('codex-session:thread-1', 'thread-1', 'codex-session', 'Coordination child', 'codex-session:root', 'codex-session:root', 'continuation', 'feature/coordination-envelope', 20)
            """
        )
        conn.execute(
            """
            INSERT INTO session_links
                (src_session_id, dst_native_id, resolved_dst_session_id, link_type, observed_at_ms)
            VALUES ('codex-session:thread-1', 'native-missing-parent', NULL, 'fork', 20)
            """
        )
        conn.execute(
            """
            INSERT INTO session_runs
                (run_ref, session_id, position, materialized_at, source_updated_at, role, status, title, search_text, evidence_refs_json)
            VALUES
                ('run:thread-1', 'codex-session:thread-1', 0, '2026-07-04T18:00:00+00:00', '2026-07-04T18:01:00+00:00', 'main', 'completed', 'main run', 'main run', '["session:thread-1"]'),
                ('run:thread-1:subagent:0:tool-2', 'codex-session:thread-1', 1, '2026-07-04T18:00:00+00:00', '2026-07-04T18:01:30+00:00', 'subagent', 'completed', 'Map the coordination surface and report caveats.', 'Map the coordination surface and report caveats.', '["message:m2"]')
            """
        )
        conn.execute(
            """
            UPDATE session_runs
            SET native_session_id = 'codex-session:child-42',
                native_parent_session_id = 'codex-session:thread-1',
                parent_run_ref = 'run:thread-1',
                agent_ref = 'agent:codex/Explore',
                context_snapshot_ref = 'context-snapshot:run:thread-1:subagent:0:tool-2:subagent_start',
                lineage_refs_json = '["run:thread-1","run:thread-1:subagent:0:tool-2"]'
            WHERE run_ref = 'run:thread-1:subagent:0:tool-2'
            """
        )
        conn.execute(
            """
            INSERT INTO session_observed_events
                (event_ref, session_id, run_ref, position, materialized_at, source_updated_at, kind, summary, evidence_refs_json, payload_json)
            VALUES
                ('event:tool', 'codex-session:thread-1', 'run:thread-1', 1, '2026-07-04T18:00:00+00:00', '2026-07-04T18:02:00+00:00', 'tool_finished', 'pytest passed', '["message:m1"]', '{"status":"passed"}'),
                ('event:subagent-started', 'codex-session:thread-1', 'run:thread-1', 2, '2026-07-04T18:00:00+00:00', '2026-07-04T18:02:30+00:00', 'subagent_started', 'Explore subagent started', '["message:m2"]', '{}'),
                ('event:subagent-finished', 'codex-session:thread-1', 'run:thread-1:subagent:0:tool-2', 3, '2026-07-04T18:00:00+00:00', '2026-07-04T18:03:00+00:00', 'subagent_finished', 'Subagent done: coordination surface mapped; caveat: web fixture only.', '["message:m3"]', '{}')
            """
        )
        conn.execute(
            """
            INSERT INTO session_context_snapshots
                (snapshot_ref, session_id, run_ref, position, materialized_at, source_updated_at, boundary, inheritance_mode, segment_refs_json, evidence_refs_json)
            VALUES
                ('context:thread-1:start', 'codex-session:thread-1', 'run:thread-1', 0, '2026-07-04T18:00:00+00:00', '2026-07-04T18:03:00+00:00', 'session_start', 'prefix-sharing', '["segment:root"]', '["message:m0"]')
            """
        )
        conn.commit()
    finally:
        conn.close()


class FakeRunner:
    def __init__(
        self,
        root: Path,
        *,
        beads_rows: list[dict[str, object]] | None,
        gates: list[dict[str, object]] | None = None,
        merge_slot: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        self.beads_rows = beads_rows
        self.gates = gates
        self.merge_slot = merge_slot

    def __call__(self, args: Sequence[str], cwd: Path | None) -> CommandResult:
        key = tuple(args)
        if key[:4] == ("git", "-C", str(self.root), "rev-parse") and key[-1] == "--show-toplevel":
            return CommandResult(key, 0, str(self.root) + "\n", "")
        if key[:4] == ("git", "-C", str(self.root), "branch"):
            return CommandResult(key, 0, "feature/coordination-envelope\n", "")
        if key[:4] == ("git", "-C", str(self.root), "rev-parse"):
            return CommandResult(key, 0, "abcdef123456\n", "")
        if key[:4] == ("git", "-C", str(self.root), "status"):
            return CommandResult(key, 0, " M polylogue/coordination/envelope.py\n", "")
        if key == ("bd", "list", "--status=in_progress", "--json"):
            return CommandResult(key, 0, json.dumps(self.beads_rows or []), "")
        if key == ("bd", "hooks", "list", "--json"):
            return CommandResult(
                key,
                0,
                json.dumps(
                    {
                        "hooks": [
                            {
                                "Name": "pre-commit",
                                "Installed": True,
                                "Version": "1.0.4",
                                "IsShim": True,
                                "Outdated": False,
                            },
                            {
                                "Name": "pre-push",
                                "Installed": True,
                                "Version": "1.0.4",
                                "IsShim": True,
                                "Outdated": False,
                            },
                        ]
                    }
                ),
                "",
            )
        if key == ("bd", "gate", "list", "--json"):
            return CommandResult(key, 0, json.dumps(self.gates), "")
        if key == ("bd", "merge-slot", "check", "--json"):
            return CommandResult(
                key,
                0,
                json.dumps(self.merge_slot or {"id": "polylogue-merge-slot", "available": False, "error": "not found"}),
                "",
            )
        if key == ("ps", "-eo", "pid,ppid,comm,args", "--no-headers"):
            return CommandResult(
                key,
                0,
                "101 1 codex codex --profile full\n202 1 polylogued polylogued run --api-port 8766\n",
                "",
            )
        return CommandResult(key, 1, "", "unexpected command")


def test_coordination_envelope_uses_beads_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(
            root,
            beads_rows=[
                {
                    "id": "polylogue-s7ae.1",
                    "title": "Coordination envelope",
                    "status": "in_progress",
                    "priority": 1,
                    "assignee": "Sinity",
                    "labels": ["area:coordination"],
                    "updated_at": "2026-07-04T18:00:00Z",
                }
            ],
        ),
    )

    assert payload.work_item.source == "beads"
    assert payload.work_item.ref == "polylogue-s7ae.1"
    assert payload.work_item.confidence == 0.95
    assert payload.repo.branch == "feature/coordination-envelope"
    assert payload.beads is not None
    assert payload.beads.hooks_all_installed is True
    assert payload.beads.hooks_outdated_count == 0
    assert [hook.name for hook in payload.beads.hooks] == ["pre-commit", "pre-push"]
    assert payload.beads.open_gate_count == 0
    assert payload.beads.merge_slot is not None
    assert payload.beads.merge_slot.id == "polylogue-merge-slot"
    assert payload.beads.merge_slot.error == "not found"
    assert payload.resource_episodes
    assert any(overlap.kind == "resource-episode" and not overlap.blocking for overlap in payload.overlaps)


def test_coordination_envelope_falls_back_to_git_without_beads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None))

    assert payload.work_item.source == "git"
    assert payload.work_item.ref == "feature/coordination-envelope"
    assert payload.work_item.confidence == 0.35
    assert payload.work_item.provenance.note == "no .beads workspace found"
    assert payload.beads is None


def test_coordination_envelope_reports_beads_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(
            root,
            beads_rows=[{"id": "polylogue-s7ae.1", "status": "in_progress"}],
            gates=[
                {
                    "id": "gate-1",
                    "title": "Await review",
                    "status": "open",
                    "metadata": {"gate_type": "human", "await_id": "review-1"},
                }
            ],
            merge_slot={"id": "polylogue-merge-slot", "available": True, "status": "free", "waiters": ["agent-a"]},
        ),
    )

    assert payload.beads is not None
    assert payload.beads.open_gate_count == 1
    assert payload.beads.gates[0].id == "gate-1"
    assert payload.beads.gates[0].gate_type == "human"
    assert payload.beads.gates[0].await_id == "review-1"
    assert payload.beads.merge_slot is not None
    assert payload.beads.merge_slot.available is True
    assert payload.beads.merge_slot.waiters == ("agent-a",)


def test_coordination_envelope_composes_archive_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    _seed_coordination_archive(index)
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)
    monkeypatch.setenv("CODEX_THREAD_ID", "thread-1")

    payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), limit=4)

    assert len(payload.session_trees) == 1
    tree = payload.session_trees[0]
    assert tree.target_session_id == "codex-session:thread-1"
    assert tree.root_session_id == "codex-session:root"
    assert len(tree.nodes) == 2
    assert len(tree.edges) == 2
    assert any(edge.parent_native_id == "native-missing-parent" for edge in tree.edges)
    assert tree.provenance.source == "archive-session-topology"
    assert len(payload.activity_episodes) == 4
    assert {"run", "tool_finished", "subagent_finished"} <= {episode.kind for episode in payload.activity_episodes}
    assert all(episode.provenance.source == "archive-run-projection" for episode in payload.activity_episodes)
    assert len(payload.subagent_exchanges) == 1
    exchange = payload.subagent_exchanges[0]
    assert exchange.run_ref == "run:thread-1:subagent:0:tool-2"
    assert exchange.dispatch_prompt == "Map the coordination surface and report caveats."
    assert exchange.returned_final_message == "Subagent done: coordination surface mapped; caveat: web fixture only."
    assert exchange.child_session_id == "codex-session:child-42"
    assert exchange.provenance.source == "archive-subagent-exchange"
    assert len(payload.proof_refs) == 1
    assert payload.proof_refs[0].status == "passed"
    assert payload.proof_refs[0].provenance.source == "archive-proof-outcome"
    assert len(payload.context_flow_refs) == 1
    assert payload.context_flow_refs[0].segment_refs == ("segment:root",)
    assert payload.context_flow_refs[0].provenance.source == "archive-context-flow"


def test_coordination_envelope_degrades_without_archive_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), limit=1)

    assert payload.archive is not None
    assert payload.archive.index_exists is True
    assert payload.session_trees == ()
    assert payload.activity_episodes == ()
    assert payload.proof_refs == ()
    assert payload.context_flow_refs == ()


def test_coordination_view_projection_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    payload = build_coordination_envelope(
        view="work-item",
        cwd=root,
        runner=FakeRunner(root, beads_rows=[{"id": "polylogue-s7ae.1", "status": "in_progress"}]),
    )

    assert payload.view == "work-item"
    assert payload.peers == ()
    assert payload.resource_episodes == ()
    assert payload.overlaps == ()
    assert payload.archive is not None
    assert payload.beads is not None
