"""Tests for the `polylogue ops excise` command (polylogue-27m).

Read-only cases (preview, abort, refusal-before-dispatch) keep the original
CliRunner-with-patched-paths pattern: they never reach a writer.

Cases that assert what an excision actually *does* run against a real daemon
(`_daemon_archive`). The CLI is no longer a writer -- `ops excise` lowers to
`mutation.session.excision` / `mutation.session.lifecycle-request` and
`configured_mutation_operation` refuses without a resident daemon -- so a test
that wants the effect has to supply the owner the command requires. This is
choice (b), drive the daemon operation, and it is right here because the
behaviour these tests cover did not disappear; it moved behind the daemon.

`test_excision_without_a_daemon_refuses_and_mutates_nothing` is the file's
anti-vacuity guard: it pins the typed refusal with no daemon present, so
restoring an in-process CLI writer turns it red.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.accepted_marker_inputs import persist_pending_marker_input_sync, prepare_accepted_marker_input
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from tests.infra.daemon_operations import DaemonOperationStack, cli_daemon_archive


def _refusal_text(result: object) -> str:
    """Return everything a CLI refusal exposed, however it was delivered.

    A daemon-side refusal arrives as a typed operation error rendered into the
    command output; an in-process one would surface as the exception. Reading
    both keeps the assertion about the refusal *reason* rather than about the
    delivery mechanism.
    """

    return f"{getattr(result, 'output', '')}\n{getattr(result, 'exception', '')}"


def _no_seed(_root: Path) -> None:
    """Default seed for cases that only need a bootstrapped archive."""
    return None


@contextlib.contextmanager
def _daemon_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: Callable[[Path], Any] = _no_seed,
) -> Iterator[tuple[DaemonOperationStack, Any]]:
    """Run `ops excise` against a real daemon rooted at ``tmp_path/archive``.

    ``seed`` runs after archive bootstrap and before the daemon starts, and its
    return value (typically a seeded session id) is yielded alongside the stack.
    """

    seeded: dict[str, Any] = {}

    def _seed(root: Path) -> None:
        seeded["value"] = seed(root)

    with cli_daemon_archive(tmp_path / "archive", monkeypatch, seed_archive=_seed) as stack:
        yield stack, seeded.get("value")


def _seed_session(archive_root: Path, *, native_id: str) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"

    source_conn = sqlite3.connect(source_db)
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path=f"/fake/{native_id}.jsonl",
            source_index=0,
            payload=f"payload-{native_id}".encode(),
            acquired_at_ms=1_000,
            native_id=native_id,
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)",
            (native_id, raw_id, f"Session {native_id}"),
        )
        index_conn.commit()
        session_id = index_conn.execute("SELECT session_id FROM sessions WHERE native_id = ?", (native_id,)).fetchone()[
            0
        ]
    finally:
        index_conn.close()
    return str(session_id)


def _seed_lineage_pair(archive_root: Path) -> tuple[str, str]:
    """Seed a parent session and a prefix-sharing child `session_links` row.

    Returns ``(parent_session_id, child_session_id)``.
    """
    parent_id = _seed_session(archive_root, native_id="lineage-parent")
    child_id = _seed_session(archive_root, native_id="lineage-child")

    index_conn = sqlite3.connect(archive_root / "index.db")
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (parent_id,),
        )
        branch_point = index_conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)
        ).fetchone()[0]
        index_conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type,
                resolved_dst_session_id, branch_point_message_id, inheritance,
                status, method, confidence, evidence_json, observed_at_ms, resolved_at_ms
            ) VALUES (?, 'codex-session', 'lineage-parent', 'branch', ?, ?, 'prefix-sharing',
                      NULL, NULL, 1.0, '[]', 1000, NULL)
            """,
            (child_id, parent_id, branch_point),
        )
        index_conn.commit()
    finally:
        index_conn.close()
    return parent_id, child_id


class TestExciseStandalone:
    def test_missing_session_reports_not_found(self, tmp_path: Path) -> None:
        with patch("polylogue.cli.commands.excise.archive_root", return_value=tmp_path / "archive"):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", "codex-session:nope", "--reason", "r", "--yes", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "not_found"

    def test_dry_run_reports_plan_without_mutating(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="dry-run-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r", "--dry-run", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert payload["plan"]["found"] is True
        assert payload["plan"]["index_sessions"] == 1

        # dry-run must not have mutated anything.
        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1

    def test_plain_dry_run_exposes_marker_carrier_counts_and_digest(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="dry-run-marker-carrier")
        with sqlite3.connect(archive_root / "index.db") as conn:
            raw_id = str(conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0])
        marker = prepare_accepted_marker_input(
            raw_id, [{"session_id": session_id, "candidates": [{"body": "marker secret"}]}]
        )
        with sqlite3.connect(archive_root / "source.db") as conn:
            conn.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(conn, marker, expected_incarnation_id=str(uuid.uuid4()))

        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            result = CliRunner().invoke(cli, ["ops", "excise", "--session", session_id, "--reason", "r", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "source.db marker carriers: 1 pending, 0 accepted" in result.output
        assert marker.payload_sha256 in result.output

    def test_dry_run_does_not_construct_a_mutating_audit_executor(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="dry-run-no-executor")
        with (
            patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root),
            patch("polylogue.operations.mutation_transaction.OperationExecutor.for_archive_root") as factory,
        ):
            result = CliRunner().invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r", "--dry-run", "--json"],
            )

        assert result.exit_code == 0, result.output
        factory.assert_not_called()

    def test_declined_confirmation_does_not_construct_a_mutating_audit_executor(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="declined-no-executor")
        with (
            patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root),
            patch("polylogue.operations.mutation_transaction.OperationExecutor.for_archive_root") as factory,
        ):
            result = CliRunner().invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r"],
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        factory.assert_not_called()

    def test_without_yes_aborts_in_json_mode(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="no-yes-1")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "r", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "aborted"

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1

    def test_yes_applies_excision(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """(b) daemon route: the excision effect is the daemon's, so drive it there."""
        with _daemon_archive(tmp_path, monkeypatch, lambda root: _seed_session(root, native_id="apply-1")) as (
            stack,
            session_id,
        ):
            archive_root = stack.archive_root
            result = CliRunner().invoke(
                cli,
                ["ops", "excise", "--session", str(session_id), "--reason", "secret leak", "--yes", "--json"],
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["detail"]  # receipt assertion id

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 0

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            receipt_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE assertion_id = ?", (payload["detail"],)
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert receipt_count == 1


class TestExciseMirrorPrimary:
    def test_mirror_dry_run_does_not_write_a_request(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    "codex-session:whatever",
                    "--reason",
                    "r",
                    "--mode",
                    "mirror",
                    "--dry-run",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "preview"
        assert not (archive_root / "user.db").exists()

    def test_primary_yes_creates_pending_request_without_touching_local_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) daemon route: the lifecycle request is a durable write the daemon owns."""
        with _daemon_archive(tmp_path, monkeypatch, lambda root: _seed_session(root, native_id="primary-1")) as (
            stack,
            seeded,
        ):
            archive_root = stack.archive_root
            session_id = str(seeded)
            result = CliRunner().invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    session_id,
                    "--reason",
                    "leak",
                    "--mode",
                    "primary",
                    "--yes",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assertion_id = payload["detail"]

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            row = user_conn.execute(
                "SELECT kind, target_ref FROM assertions WHERE assertion_id = ?", (assertion_id,)
            ).fetchone()
        finally:
            user_conn.close()
        assert row is not None
        assert row[0] == "excision_request"
        assert row[1] == f"session:{session_id}"
        with sqlite3.connect(archive_root / "audit.db") as audit_connection:
            assert audit_connection.execute(
                "SELECT status FROM operation_runs WHERE operation_name = 'mutate-session-lifecycle-request'"
            ).fetchone() == ("completed",)

        # Local content is untouched by mirror/primary mode.
        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            count = index_conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
        finally:
            index_conn.close()
        assert count == 1

    def test_replayed_primary_request_is_an_audited_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CLI route records the existing lifecycle assertion as idempotent.

        Anti-vacuity: unconditionally applied actuator receipts make the
        second completed audit run report one affected target instead of zero.
        """

        command_session = "primary-replay"
        command = [
            "ops",
            "excise",
            "--session",
            "",  # filled in once the daemon-seeded session id is known
            "--reason",
            "leak",
            "--mode",
            "primary",
            "--yes",
            "--json",
        ]
        with _daemon_archive(tmp_path, monkeypatch, lambda root: _seed_session(root, native_id=command_session)) as (
            stack,
            seeded,
        ):
            archive_root = stack.archive_root
            command[command.index("")] = str(seeded)
            runner = CliRunner()
            first = runner.invoke(cli, command)
            replay = runner.invoke(cli, command)

        assert first.exit_code == replay.exit_code == 0, (first.output, replay.output)
        with sqlite3.connect(archive_root / "audit.db") as connection:
            assert connection.execute("SELECT state FROM operation_targets ORDER BY rowid").fetchall() == [
                ("applied",),
                ("already_satisfied",),
            ]
            assert connection.execute(
                "SELECT affected_count FROM operation_runs ORDER BY requested_at_ms, operation_id"
            ).fetchall() == [(1,), (0,)]

    def test_primary_refuses_missing_audit_without_writing_a_lifecycle_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) daemon route: the audit precondition is enforced by the writer.

        The audit tier is removed *after* the daemon is up, because daemon
        startup prepares the operation journals -- deleting audit.db before
        boot would simply be undone and the refusal never reached.
        """
        with _daemon_archive(
            tmp_path, monkeypatch, lambda root: _seed_session(root, native_id="primary-missing-audit")
        ) as (stack, seeded):
            archive_root = stack.archive_root
            session_id = str(seeded)
            (archive_root / "audit.db").unlink()
            result = CliRunner().invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    session_id,
                    "--reason",
                    "leak",
                    "--mode",
                    "primary",
                    "--yes",
                    "--json",
                ],
            )

        assert result.exit_code != 0
        refusal = _refusal_text(result)
        # The daemon refuses while opening the audit tier, naming the tier and
        # the leaf it could not open. That is the same precondition the
        # in-process route used to report as "missing audit.db"; the reason is
        # now typed and attributed to the writer that enforced it.
        assert "daemon refused mutation.session.lifecycle-request" in refusal
        assert "audit tier" in refusal
        assert "audit.db" in refusal
        with sqlite3.connect(archive_root / "user.db") as connection:
            assert connection.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'excision_request'").fetchone() == (
                0,
            )

    def test_primary_refuses_broken_audit_continuity_without_writing_a_lifecycle_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) daemon route: continuity regression is refused by the writer."""
        with _daemon_archive(
            tmp_path, monkeypatch, lambda root: _seed_session(root, native_id="primary-broken-continuity")
        ) as (stack, seeded):
            archive_root = stack.archive_root
            session_id = str(seeded)
            with sqlite3.connect(archive_root / "audit.db") as connection:
                connection.execute("UPDATE audit_continuity_head SET head_sha256 = ? WHERE singleton = 1", ("0" * 64,))
                connection.commit()
            result = CliRunner().invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    session_id,
                    "--reason",
                    "leak",
                    "--mode",
                    "primary",
                    "--yes",
                    "--json",
                ],
            )

        assert result.exit_code != 0
        refusal = _refusal_text(result)
        # A regressed continuity head is now a typed AuditContinuityError from
        # the daemon rather than a locally raised message.
        assert "daemon refused mutation.session.lifecycle-request" in refusal
        assert "AuditContinuityError" in refusal
        with sqlite3.connect(archive_root / "user.db") as connection:
            assert connection.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'excision_request'").fetchone() == (
                0,
            )


class TestExciseLineageSafety:
    """CLI coverage for the polylogue-27m fix-round lineage-safety guard."""

    def test_dry_run_surfaces_lineage_dependents(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        parent_id, child_id = _seed_lineage_pair(archive_root)
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", parent_id, "--reason", "r", "--dry-run", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["plan"]["lineage_dependent_session_ids"] == [child_id]

    def test_without_cascade_flag_refuses_and_does_not_mutate(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        parent_id, child_id = _seed_lineage_pair(archive_root)
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["ops", "excise", "--session", parent_id, "--reason", "r", "--yes", "--json"],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "aborted"

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 2  # neither parent nor child touched

    def test_with_cascade_flag_removes_parent_and_dependents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(b) daemon route: cascade removal is a write, so it runs under the daemon."""
        with _daemon_archive(tmp_path, monkeypatch, _seed_lineage_pair) as (stack, seeded):
            archive_root = stack.archive_root
            parent_id, _child_id = seeded
            result = CliRunner().invoke(
                cli,
                [
                    "ops",
                    "excise",
                    "--session",
                    parent_id,
                    "--reason",
                    "r",
                    "--yes",
                    "--cascade-lineage",
                    "--json",
                ],
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        # affected_count is index_sessions summed across the whole cascade.
        assert payload["affected_count"] == 2

        index_conn = sqlite3.connect(archive_root / "index.db")
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0

        user_conn = sqlite3.connect(archive_root / "user.db")
        try:
            receipt_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = 'excision_record'"
            ).fetchone()[0]
        finally:
            user_conn.close()
        assert receipt_count == 2  # one durable audit receipt per removed session


class TestExciseWriteAuthority:
    """The CLI is not a writer: `ops excise --yes` needs the resident daemon."""

    def test_excision_without_a_daemon_refuses_and_mutates_nothing(self, tmp_path: Path) -> None:
        """(a) typed refusal -- this file's anti-vacuity guard.

        Anti-vacuity: if an in-process excision writer were reintroduced in the
        CLI, this command would succeed offline and delete the seeded session,
        so both assertions below flip red. It is the guard that keeps the
        daemon-backed cases above from silently blessing a restored bypass.
        """

        archive_root = tmp_path / "archive"
        session_id = _seed_session(archive_root, native_id="no-daemon")
        with patch("polylogue.cli.commands.excise.archive_root", return_value=archive_root):
            result = CliRunner().invoke(
                cli,
                ["ops", "excise", "--session", session_id, "--reason", "secret leak", "--yes"],
            )

        assert result.exit_code != 0
        assert "daemon is unavailable; it must execute mutation.session.excision" in _refusal_text(result)
        with sqlite3.connect(archive_root / "index.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone() == (1,)
