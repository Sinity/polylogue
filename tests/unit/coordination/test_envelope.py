"""Coordination envelope tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from time import sleep

import pytest

from polylogue.coordination.envelope import CommandResult, _session_ref, build_coordination_envelope


def _seed_coordination_archive(index: Path) -> None:
    """Seed a minimal but schema-complete archive for coordination-evidence tests.

    polylogue-dab/itvd: session_runs/session_observed_events/
    session_context_snapshots are no longer materialized tables --
    run_projection_relations.py computes them on read from `sessions` and
    `blocks`. `codex-session:child-42` is its own session (branch_type=
    'subagent', parent_session_id=thread-1), not a synthesized row under the
    parent's session_id as the old materialized writer produced -- one run
    row per subagent session is the new model's guarantee.
    """
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
                message_count INTEGER NOT NULL DEFAULT 0,
                tool_use_count INTEGER NOT NULL DEFAULT 0,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                sort_key_ms INTEGER
            );
            CREATE TABLE session_links (
                src_session_id TEXT,
                dst_native_id TEXT,
                resolved_dst_session_id TEXT,
                link_type TEXT,
                observed_at_ms INTEGER
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                block_type TEXT NOT NULL,
                message_id TEXT,
                position INTEGER,
                semantic_type TEXT,
                tool_command TEXT,
                tool_id TEXT,
                tool_name TEXT,
                tool_result_exit_code INTEGER,
                tool_result_is_error INTEGER,
                search_text TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions
                (session_id, native_id, origin, title, parent_session_id, root_session_id, branch_type,
                 git_branch, message_count, tool_use_count, created_at_ms, updated_at_ms, sort_key_ms)
            VALUES
                ('codex-session:root', 'root', 'codex-session', 'Root work', NULL, 'codex-session:root', NULL,
                 'feature/coordination-envelope', 1, 0, 1780304400000, 1780304400000, 10),
                ('codex-session:thread-1', 'thread-1', 'codex-session', 'Coordination child', 'codex-session:root',
                 'codex-session:root', 'continuation', 'feature/coordination-envelope', 2, 1, 1780308000000,
                 1780308060000, 20),
                ('codex-session:child-42', 'child-42', 'codex-session', 'Map the coordination surface and report caveats.',
                 'codex-session:thread-1', 'codex-session:root', 'subagent', 'feature/coordination-envelope', 1, 0,
                 1780308120000, 1780308120000, 30)
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
            INSERT INTO blocks
                (block_id, session_id, block_type, message_id, position, tool_id, tool_name,
                 tool_result_exit_code, search_text)
            VALUES
                ('codex-session:thread-1::m1::0', 'codex-session:thread-1', 'tool_use', 'm1', 0,
                 'tool-2', 'pytest', NULL, 'run pytest'),
                ('codex-session:thread-1::m1::1', 'codex-session:thread-1', 'tool_result', 'm1', 1,
                 'tool-2', NULL, 0, 'passed')
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
        process_rows: str | None = None,
    ) -> None:
        self.root = root
        self.beads_rows = beads_rows
        self.gates = gates
        self.merge_slot = merge_slot
        self.process_rows = process_rows

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
        if key == ("ps", "ww", "-eo", "pid=,ppid=,comm=,cgroup:200=,args="):
            return CommandResult(
                key,
                0,
                self.process_rows
                or (
                    "101 1 codex 0::/user.slice/user@1000.service/app.slice/codex.scope "
                    "codex --profile full\n"
                    "202 1 polylogued 0::/user.slice/user@1000.service/app.slice/polylogued.service "
                    "polylogued run --api-port 8766\n"
                ),
                "",
            )
        return CommandResult(key, 1, "", "unexpected command")


def test_coordination_envelope_overlaps_independent_beads_probes(tmp_path: Path) -> None:
    """The production envelope waits for all Beads facts without serializing them."""

    root = tmp_path / "repo"
    (root / ".beads").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def runner(args: Sequence[str], cwd: Path | None) -> CommandResult:
        key = tuple(args)
        if key[:4] == ("git", "-C", str(root), "rev-parse"):
            return CommandResult(key, 0, str(root) + "\n", "")
        if key[:4] == ("git", "-C", str(root), "branch"):
            return CommandResult(key, 0, "feature/coordination-envelope\n", "")
        if key[:4] == ("git", "-C", str(root), "status"):
            return CommandResult(key, 0, "", "")
        if key == ("ps", "ww", "-eo", "pid=,ppid=,comm=,cgroup:200=,args="):
            return CommandResult(key, 0, "", "")
        if key in {
            ("bd", "hooks", "list", "--json"),
            ("bd", "gate", "list", "--json"),
            ("bd", "merge-slot", "check", "--json"),
        }:
            calls.append(key)
            sleep(0.05)
            if key[1] == "hooks":
                return CommandResult(key, 0, '{"hooks": []}', "")
            if key[1] == "gate":
                return CommandResult(key, 0, "[]", "")
            return CommandResult(key, 0, '{"available": true}', "")
        return CommandResult(key, 1, "", "unexpected command")

    timings: dict[str, float] = {}
    payload = build_coordination_envelope(cwd=root, runner=runner, stage_timings_ms=timings)

    assert payload.beads is not None
    assert {call[1] for call in calls} == {"hooks", "gate", "merge-slot"}
    # Three 50 ms probes must overlap on the live code path; a sequential
    # implementation records roughly 150 ms for this stage.
    assert timings["beads"] < 110


def test_coordination_envelope_bounds_live_beads_probe_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real command path receives the interactive probe deadline."""

    root = tmp_path / "repo"
    (root / ".beads").mkdir(parents=True)
    calls: list[tuple[tuple[str, ...], float]] = []

    def fake_run(
        args: Sequence[str],
        cwd: Path | None,
        *,
        timeout_seconds: float = 2.0,
    ) -> CommandResult:
        key = tuple(args)
        calls.append((key, timeout_seconds))
        if key[:4] == ("git", "-C", str(root), "rev-parse"):
            return CommandResult(key, 0, str(root) + "\n", "")
        if key[:4] == ("git", "-C", str(root), "branch"):
            return CommandResult(key, 0, "feature/coordination-envelope\n", "")
        if key[:4] == ("git", "-C", str(root), "status"):
            return CommandResult(key, 0, "", "")
        if key == ("ps", "ww", "-eo", "pid=,ppid=,comm=,cgroup:200=,args="):
            return CommandResult(key, 0, "", "")
        if key == ("bd", "hooks", "list", "--json"):
            return CommandResult(key, 0, '{"hooks": []}', "")
        if key == ("bd", "gate", "list", "--json"):
            return CommandResult(key, 0, "[]", "")
        if key == ("bd", "merge-slot", "check", "--json"):
            return CommandResult(key, 124, "", "timed out")
        return CommandResult(key, 1, "", "unexpected command")

    monkeypatch.setattr("polylogue.coordination.envelope._run_command", fake_run)
    payload = build_coordination_envelope(cwd=root)

    assert payload.beads is not None
    probes = [
        timeout
        for command, timeout in calls
        if command
        in {
            ("bd", "hooks", "list", "--json"),
            ("bd", "gate", "list", "--json"),
            ("bd", "merge-slot", "check", "--json"),
        }
    ]
    assert len(probes) == 3
    assert all(timeout == 0.35 for timeout in probes)


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
    assert payload.archive is not None
    assert set(payload.archive.hook_flow_states) == {"claude-code", "codex"}
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

    payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), limit=4, detail=True)

    assert len(payload.session_trees) == 1
    tree = payload.session_trees[0]
    assert tree.target_session_id == "codex-session:thread-1"
    assert tree.root_session_id == "codex-session:root"
    # 3 nodes: root, thread-1 (target), and the child-42 subagent session
    # (its own session sharing root_session_id, unlike the pre-dab model
    # where the subagent run was synthesized under thread-1's own session_id).
    assert len(tree.nodes) == 3
    # 2 resolved edges (thread-1->root, child-42->thread-1) + 1 unresolved
    # fork edge from session_links.
    assert len(tree.edges) == 3
    assert any(edge.parent_native_id == "native-missing-parent" for edge in tree.edges)
    assert tree.provenance.source == "archive-session-topology"
    # polylogue-dab/itvd: activity is scoped to the exact target session
    # (thread-1) only -- its own main run, session_started event, and
    # tool_finished event. The subagent's run lives under its own session
    # (child-42) and is out of exact-session scope here (see
    # subagent_exchanges below, which uses the branch-scope fallback).
    assert len(payload.activity_episodes) == 3
    assert {episode.kind for episode in payload.activity_episodes} == {"run", "session_started", "tool_finished"}
    assert all(episode.provenance.source == "archive-run-projection" for episode in payload.activity_episodes)
    assert len(payload.subagent_exchanges) == 1
    exchange = payload.subagent_exchanges[0]
    assert exchange.run_ref == "run:codex-session:child-42"
    assert exchange.dispatch_prompt == "Map the coordination surface and report caveats."
    # polylogue-dab/itvd: the source-derived CTE has no 'subagent_finished'
    # event kind (the pre-dab writer's synthesized report-text marker), so
    # this is always None now -- only the dispatch side survives.
    assert exchange.returned_final_message is None
    assert exchange.child_session_id == "codex-session:child-42"
    assert exchange.provenance.source == "archive-subagent-exchange"
    assert len(payload.proof_refs) == 1
    # 'ok'/'failed'/'unknown' derived from tool_result_exit_code, not the
    # pre-dab writer's freeform 'status' string.
    assert payload.proof_refs[0].status == "ok"
    assert payload.proof_refs[0].provenance.source == "archive-proof-outcome"
    assert len(payload.context_flow_refs) == 1
    assert payload.context_flow_refs[0].segment_refs == ("session:codex-session:thread-1",)
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


def test_coordination_envelope_signals_archive_evidence_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuine archive-evidence query failure must not look identical to
    "no evidence" (polylogue-cpf.4). Before the fix, both cases returned the
    same five empty tuples with no advisory and no log line.
    """
    root = tmp_path / "repo"
    root.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    _seed_coordination_archive(index)
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)

    def _boom(*args: object, **kwargs: object) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("polylogue.coordination.envelope._archive_tables_present", _boom)

    with caplog.at_level("WARNING"):
        payload = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), limit=4)

    # Same empty shape a reader would see for "no matching evidence" --
    # the advisory (and the log line) is what makes the two distinguishable.
    assert payload.session_trees == ()
    assert payload.activity_episodes == ()
    assert payload.subagent_exchanges == ()
    assert payload.proof_refs == ()
    assert payload.context_flow_refs == ()
    assert any("Archive-evidence lookup degraded" in advisory for advisory in payload.advisories)
    assert any("database is locked" in advisory for advisory in payload.advisories)
    assert "coordination archive-evidence query failed" in caplog.text


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


def test_process_projection_collapses_components_and_uses_real_work_scopes(
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
    monkeypatch.setattr(
        "polylogue.coordination.envelope._proc_cwd",
        lambda pid: str(root) if pid in {101, 102, 103, 300} else None,
    )
    system = "0::/system.slice"
    rows = "\n".join(
        (
            "100 1 node 0::/user.slice/user@1000.service/app.slice/codex.scope node @openai/codex code-mode-host",
            "101 100 codex 0::/user.slice/user@1000.service/app.slice/codex.scope codex --session-id session-1",
            "102 101 node 0::/user.slice/user@1000.service/app.slice/codex.scope node code-mode-host",
            "103 101 claude-mcp 0::/user.slice/user@1000.service/app.slice/codex.scope claude-mcp mcp-server",
            "104 1 claude 0::/user.slice/user@1000.service/app.slice/claude-spare.service claude --spare-daemon",
            f"201 1 systemd-timesyncd {system}/systemd-timesyncd.service /nix/store/hash/systemd-timesyncd",
            f"202 1 systemd-resolved {system}/systemd-resolved.service /nix/store/hash/systemd-resolved",
            f"203 1 systemd-udevd {system}/systemd-udevd.service /nix/store/hash/systemd-udevd",
            f"204 1 systemd-oomd {system}/systemd-oomd.service /nix/store/hash/systemd-oomd",
            f"205 1 dbus-daemon {system}/dbus.service /nix/store/hash/dbus-daemon --system",
            f"206 1 earlyoom {system}/earlyoom.service /nix/store/hash/earlyoom",
            f"207 1 below {system}/below.service /nix/store/hash/below record",
            "208 2 nvidia_uvm 0::/ [nvidia_uvm]",
            "300 1 python 0::/user.slice/user@1000.service/background.slice/"
            "sinnix-background-polylogue-index-rebuild-v30.scope python rebuild-index "
            "--workers=4",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert [(peer.kind, peer.session_ref) for peer in payload.peers] == [("codex", "session-1")]
    assert payload.peers[0].component_count >= 3
    assert {100, 102, 103} <= set(payload.peers[0].component_pids)
    assert len(payload.resource_episodes) == 1
    rebuild = payload.resource_episodes[0]
    assert rebuild.kind == "build"
    assert rebuild.unit == "sinnix-background-polylogue-index-rebuild-v30.scope"
    assert rebuild.cwd == str(root)
    assert str(archive) in rebuild.resources
    assert all(name not in rebuild.command for name in ("timesyncd", "resolved", "udevd", "oomd", "earlyoom"))

    compact = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None, process_rows=rows))
    assert compact.resource_episodes[0].resources[0] == str(archive)


def test_caller_identity_prefers_session_environment_and_resolves_owner_process(
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
    monkeypatch.setattr("polylogue.coordination.envelope.os.getpid", lambda: 303)
    monkeypatch.setenv("CODEX_THREAD_ID", "thread-stable")
    rows = "\n".join(
        (
            "100 1 node 0::/user.slice/codex.scope node /opt/@openai/codex --profile lean",
            "101 100 codex 0::/user.slice/codex.scope codex --profile lean",
            "202 101 zsh 0::/user.slice/codex.scope zsh -lc tool-host",
            "303 202 polylogue 0::/user.slice/codex.scope polylogue agents status --json",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert payload.self.identity_status == "resolved"
    assert payload.self.agent_kind == "codex"
    assert payload.self.logical_id == "codex:thread-stable"
    assert payload.self.session_ref == "thread-stable"
    assert payload.self.owner_pid == 101
    assert payload.self.invocation_pid == 303
    assert payload.self.provenance.source == "caller-environment"
    assert payload.peers == ()


def test_nested_same_provider_caller_uses_inner_owner_and_keeps_outer_peer(
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
    monkeypatch.setattr("polylogue.coordination.envelope.os.getpid", lambda: 400)
    monkeypatch.setenv("CODEX_THREAD_ID", "inner-thread")
    rows = "\n".join(
        (
            "100 1 codex 0::/user.slice/outer.scope codex --session-id outer-thread",
            "200 100 codex 0::/user.slice/inner.scope codex --session-id inner-thread",
            "300 200 sh 0::/user.slice/inner.scope sh -c tool-host",
            "400 300 polylogue 0::/user.slice/inner.scope polylogue agents status --json",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert payload.self.owner_pid == 200
    assert payload.self.logical_id == "codex:inner-thread"
    assert [(peer.pid, peer.logical_id) for peer in payload.peers] == [(100, "codex:outer-thread")]
    assert not ({200, 300, 400} & set(payload.peers[0].component_pids))


def test_canonical_session_ref_joins_peer_logical_identity(
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
    monkeypatch.setattr("polylogue.coordination.envelope.os.getpid", lambda: 400)
    monkeypatch.setenv("POLYLOGUE_SESSION_REF", "codex-session:abc")
    rows = "\n".join(
        (
            "200 1 codex 0::/user.slice/self.scope codex --profile lean",
            "300 200 sh 0::/user.slice/self.scope sh -c tool-host",
            "400 300 polylogue 0::/user.slice/self.scope polylogue agents status --json",
            "500 1 codex 0::/user.slice/peer.scope codex --session-id abc",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert payload.self.session_ref == "codex-session:abc"
    assert payload.self.logical_id == "codex:abc"
    assert payload.peers[0].logical_id == payload.self.logical_id


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("claude --resume --model sonnet", None),
        ("claude --resume session-123 --model sonnet", "session-123"),
        ("codex --session-id thread-123", "thread-123"),
        ("codex --thread-id=thread-456", "thread-456"),
    ),
)
def test_session_ref_parses_structured_option_values(command: str, expected: str | None) -> None:
    assert _session_ref(command) == expected


def test_caller_identity_falls_back_to_agent_ancestor_without_guessing_invocation(
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
    monkeypatch.setattr("polylogue.coordination.envelope.os.getpid", lambda: 303)
    for name in (
        "POLYLOGUE_SESSION_REF",
        "CODEX_THREAD_ID",
        "CODEX_SESSION_ID",
        "CLAUDE_SESSION_ID",
        "GEMINI_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    rows = "\n".join(
        (
            "101 1 claude 0::/user.slice/claude.scope claude --session-id session-from-parent",
            "202 101 sh 0::/user.slice/claude.scope sh -c tool-host",
            "303 202 polylogue 0::/user.slice/claude.scope polylogue agents status --json",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert payload.self.identity_status == "resolved"
    assert payload.self.agent_kind == "claude"
    assert payload.self.logical_id == "claude:session-from-parent"
    assert payload.self.session_ref == "session-from-parent"
    assert payload.self.owner_pid == 101
    assert payload.self.invocation_pid == 303
    assert payload.self.provenance.source == "process-tree"


def test_caller_identity_is_typed_unknown_without_session_or_agent_ancestor(
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
    monkeypatch.setattr("polylogue.coordination.envelope.os.getpid", lambda: 303)
    for name in (
        "POLYLOGUE_SESSION_REF",
        "CODEX_THREAD_ID",
        "CODEX_SESSION_ID",
        "CLAUDE_SESSION_ID",
        "GEMINI_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    rows = "\n".join(
        (
            "202 1 sh 0::/user.slice/tool.scope sh -c tool-host",
            "303 202 polylogue 0::/user.slice/tool.scope polylogue agents status --json",
        )
    )

    payload = build_coordination_envelope(
        cwd=root,
        runner=FakeRunner(root, beads_rows=None, process_rows=rows),
        detail=True,
    )

    assert payload.self.identity_status == "unknown"
    assert payload.self.agent_kind == "unknown"
    assert payload.self.logical_id is None
    assert payload.self.session_ref is None
    assert payload.self.owner_pid is None
    assert payload.self.invocation_pid == 303
    assert payload.self.provenance.source == "caller-identity"
    assert payload.self.provenance.confidence == 0.0


def test_compact_projection_is_byte_bounded_and_detail_recovers_omitted_peers(
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
    rows = "\n".join(
        f"{100 + index} 1 codex 0::/user.slice/user@1000.service/app.slice/codex-{index}.scope "
        f"codex --session-id session-{index} --config {'x' * 180}"
        for index in range(20)
    )
    runner = FakeRunner(root, beads_rows=None, process_rows=rows)

    compact = build_coordination_envelope(cwd=root, runner=runner, limit=20)
    detailed = build_coordination_envelope(cwd=root, runner=runner, limit=20, detail=True)

    compact_bytes = len(compact.to_json(exclude_none=True).encode())
    assert compact_bytes <= 8 * 1024
    assert compact.projection.serialized_bytes == compact_bytes
    assert compact.projection.omitted_counts["peers"] == 16
    assert compact.projection.detail_hint == "CLI: --detail; MCP: detail=true"
    assert len(detailed.peers) == 20
    assert detailed.projection.detail is True
    assert detailed.projection.omitted_counts == {}
    assert len(detailed.to_json(exclude_none=True).encode()) > compact_bytes


def test_compact_projection_keeps_active_archive_writers_before_other_resources(
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
    peer_rows = tuple(
        f"{100 + index} 1 codex 0::/user.slice/codex-{index}.scope "
        f"codex --session-id session-{index} --config {'x' * 180}"
        for index in range(20)
    )
    build_rows = tuple(
        f"{300 + index} 1 cargo 0::/user.slice/sinnix-build-{index}.scope cargo build {'y' * 180}" for index in range(5)
    )
    daemon_rows = (
        "901 1 polylogued 0::/user.slice/zz-polylogued-a.service polylogued run --api-port 8766",
        "902 1 polylogued 0::/user.slice/zz-polylogued-b.service polylogued run --api-port 8767",
    )
    runner = FakeRunner(root, beads_rows=None, process_rows="\n".join((*peer_rows, *build_rows, *daemon_rows)))

    compact = build_coordination_envelope(cwd=root, runner=runner, limit=20)

    compact_bytes = len(compact.to_json(exclude_none=True).encode())
    assert compact.projection.byte_budget is not None
    assert compact_bytes <= compact.projection.byte_budget
    assert [episode.pid for episode in compact.resource_episodes[:2]] == [901, 902]
    assert compact.archive is not None
    assert [episode.pid for episode in compact.archive.daemon_processes] == [901, 902]


def test_compact_projection_bounds_adversarial_beads_fields_without_erasing_control_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".beads").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    index = archive / "index.db"
    index.touch()
    monkeypatch.setattr("polylogue.coordination.envelope.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.coordination.envelope.active_index_db_path", lambda: index)
    enormous_gate_title = "gate-" + "x" * 20_000
    enormous_holder = "holder-" + "y" * 20_000
    runner = FakeRunner(
        root,
        beads_rows=[{"id": "polylogue-8k91", "status": "in_progress"}],
        gates=[{"id": "gate-1", "title": enormous_gate_title, "status": "open"}],
        merge_slot={"id": "merge-slot", "available": False, "holder": enormous_holder},
    )

    compact = build_coordination_envelope(cwd=root, runner=runner)

    compact_bytes = len(compact.to_json(exclude_none=True).encode())
    assert compact.projection.byte_budget is not None
    assert compact_bytes <= compact.projection.byte_budget
    assert compact.beads is not None
    assert compact.beads.gates
    assert compact.beads.gates[0].title is not None
    assert len(compact.beads.gates[0].title) < len(enormous_gate_title)
    assert compact.beads.merge_slot is not None
    assert compact.beads.merge_slot.holder is not None
    assert len(compact.beads.merge_slot.holder) < len(enormous_holder)
    assert compact.projection.omitted_counts["beads_gate_text_chars"] > 0
    assert compact.projection.omitted_counts["beads_merge_text_chars"] > 0
    assert compact.resource_episodes[0].kind == "daemon"
    assert compact.archive is not None
    assert compact.archive.daemon_processes


def test_handoff_projection_uses_supported_live_sources_or_empty_list(
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

    empty = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), detail=True)
    assert empty.handoff == ()
    assert "conductor-devloop" not in empty.to_json(exclude_none=True)

    scratch = root / ".agent" / "scratch"
    scratch.mkdir(parents=True)
    handoff = scratch / "2026-07-10-agent-handoff.md"
    handoff.write_text("handoff evidence", encoding="utf-8")
    populated = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), detail=True)

    assert len(populated.handoff) == 1
    assert populated.handoff[0].ref == str(handoff)
    assert populated.handoff[0].kind == "scratch-handoff"
    assert populated.handoff[0].exists is True

    with sqlite3.connect(archive / "user.db") as conn:
        conn.executescript(
            """
            CREATE TABLE assertions (
                assertion_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                body_text TEXT,
                scope_ref TEXT,
                target_ref TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            ) STRICT;
            """
        )
        conn.execute(
            """
            INSERT INTO assertions
                (assertion_id, kind, body_text, scope_ref, target_ref, status, updated_at_ms)
            VALUES (?, 'note', ?, ?, 'blackboard:handoff', 'active', 1783700000000)
            """,
            ("blackboard-note:handoff-1", f"[handoff] Ready\n\nscope_repo: {root.name}", str(root)),
        )
    with_assertion = build_coordination_envelope(cwd=root, runner=FakeRunner(root, beads_rows=None), detail=True)
    assert {item.kind for item in with_assertion.handoff} == {"scratch-handoff", "assertion-handoff"}
