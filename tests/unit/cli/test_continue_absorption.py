"""CLI behavior for absorbed continuation workflows (#2177)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.derived.session.rebuild import rebuild_archive_session_insights
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.storage_records import SessionBuilder

NID_CONTINUE_ROOT = native_session_id_for("codex", "cli-continue-root")


@pytest.fixture(autouse=True)
def _resident_continuation(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        yield


def _seed_continuation_session(db_path: Path, slug: str = "cli-continue-root") -> None:
    (
        SessionBuilder(db_path, slug)
        .provider("codex")
        .title("CLI Continue")
        .created_at("2026-04-21T09:00:00+00:00")
        .updated_at("2026-04-21T09:30:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Add continuation JSON output.",
            timestamp="2026-04-21T09:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Implemented JSON output and ready to verify.",
            timestamp="2026-04-21T09:25:00+00:00",
        )
        .save()
    )
    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        rebuild_archive_session_insights(archive)


def test_continue_emits_interactive_resume_command(cli_workspace: dict[str, Path]) -> None:
    _seed_continuation_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        ["--id", NID_CONTINUE_ROOT, "continue"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert result.output == "codex resume ext-cli-continue-root\n"


def test_continue_missing_session_exits_with_clear_message(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["--id", "missing-session", "continue"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "session not found" in result.output.lower(), result.output
    assert "missing-session" in result.output


def test_continue_candidates_json_repeats_recent_files(cli_workspace: dict[str, Path]) -> None:
    _seed_continuation_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "continue",
            "--candidates",
            "--repo",
            "/workspace/polylogue",
            "--cwd",
            "/workspace/polylogue",
            "--recent",
            "/workspace/polylogue/polylogue/cli/click_app.py",
            "--recent",
            "/workspace/polylogue/polylogue/cli/query_verbs.py",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    result_payload = payload["result"]
    assert result_payload["returned"] >= 1
    assert result_payload["limit"] == 10
    assert result_payload["candidates"][0]["logical_session_id"] == NID_CONTINUE_ROOT
    assert "score_breakdown" in result_payload["candidates"][0]
    assert "overlap_basis" in result_payload["candidates"][0]


def test_continue_candidates_terminal_renders_overlap_basis(cli_workspace: dict[str, Path]) -> None:
    _seed_continuation_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        [
            "continue",
            "--candidates",
            "--repo",
            "/workspace/polylogue",
            "--recent",
            "/workspace/polylogue/polylogue/cli/query_verbs.py",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "[overlap exact=" in result.output
    assert " dir=" in result.output
    assert " dead-excluded=" in result.output


def test_continue_candidates_report_the_page_they_hold_not_a_total(
    cli_workspace: dict[str, Path],
) -> None:
    """The candidate payload never calls its page size a total.

    ``find_resume_candidates`` ranks every session profile and returns
    ``candidates[:limit]``; it reports no total.  The payload's ``total`` was
    therefore ``len(page)`` -- it equalled the limit whenever more sessions
    matched, and changed when the operator changed ``--limit``.

    Anti-vacuity: EXECUTED -- restoring ``"total": len(candidates)`` in
    ``_emit_continue_candidates`` makes the first assertion below red, because
    the payload once again claims ``total: 2`` for three ranked sessions.
    """

    for slug in ("cli-continue-root", "cli-continue-second", "cli-continue-third"):
        _seed_continuation_session(cli_workspace["db_path"], slug)

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "continue",
            "--candidates",
            "--limit",
            "2",
            "--repo",
            "/workspace/polylogue",
            "--cwd",
            "/workspace/polylogue",
            "--recent",
            "/workspace/polylogue/polylogue/cli/query_verbs.py",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)["result"]
    assert "total" not in payload, "the ranking reports no total, so the payload must not claim one"
    assert payload["returned"] == 2
    assert payload["limit"] == 2
    assert len(payload["candidates"]) == 2


def test_continue_json_keeps_full_successor_recipe_and_resident_ledger(cli_workspace: dict[str, Path]) -> None:
    _seed_continuation_session(cli_workspace["db_path"])
    result = CliRunner().invoke(
        cli, ["--id", NID_CONTINUE_ROOT, "continue", "--format", "json"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    image = json.loads(result.output)
    assert image["spec"]["purpose"] == "continue"
    assert image["spec"]["unit_queries"] == [
        f"{unit} where session.id:{NID_CONTINUE_ROOT}"
        for unit in ("runs", "observed-events", "context-snapshots", "actions")
    ]
    assert "Add continuation JSON output." in json.dumps(image["segments"])
    assert len([segment for segment in image["segments"] if segment["kind"] == "query_unit"]) == 4
    assert not [omission for omission in image["omitted"] if omission.get("query")]
    assert image["ledger"]
    with sqlite3.connect(cli_workspace["archive_root"] / "ops.db") as conn:
        count, observed = conn.execute(
            "SELECT COUNT(*),MIN(observed_at_ms) FROM context_injection_ledger WHERE build_ref=?",
            (image["build_ref"],),
        ).fetchone()
    assert count == len(image["ledger"])
    assert observed > 0


def test_continue_json_file_preserves_exact_image(cli_workspace: dict[str, Path]) -> None:
    _seed_continuation_session(cli_workspace["db_path"])
    target = cli_workspace["archive_root"].parent / "successor.json"
    result = CliRunner().invoke(
        cli,
        ["--id", NID_CONTINUE_ROOT, "continue", "--format", "json", "--to", "file", "--out", str(target)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text())["spec"]["seed_refs"] == [f"session:{NID_CONTINUE_ROOT}"]


def test_continue_exec_keeps_harness_execution_client_side(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    _seed_continuation_session(cli_workspace["db_path"])
    calls: list[tuple[object, object]] = []

    def run(argv: object, *, cwd: object, check: bool) -> object:
        assert not check
        calls.append((argv, cwd))
        return SimpleNamespace(returncode=-9)

    monkeypatch.setattr("polylogue.cli.query_verbs.subprocess.run", run)
    result = CliRunner().invoke(cli, ["--id", NID_CONTINUE_ROOT, "continue", "--exec"])
    assert result.exit_code == 137
    assert calls == [(["codex", "resume", "ext-cli-continue-root"], None)]


def test_continue_daemon_absence_has_no_local_facade_fallback(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # UDS path length is a physical limit; the managed fixture root is longer.
    absent_socket = Path("/realm/tmp/work") / f"plg-absent-{uuid4().hex}.sock"
    assert not absent_socket.exists()
    monkeypatch.setattr("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: absent_socket)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("continuation must not execute the local facade")

    monkeypatch.setattr("polylogue.api.Polylogue.get_session", forbidden)
    result = CliRunner().invoke(cli, ["--id", NID_CONTINUE_ROOT, "continue"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "polylogued run" in result.output, result.output
