"""Behavioral proof for the ``polylogue setting`` get/set/list command (polylogue-at44).

``setting set`` writes ``user.db``, the archive's one irreplaceable tier, so it
lowers to the declared ``mutation.user.setting.set`` operation and the daemon
is its sole writer (polylogue-gjwto / polylogue-r29bv). The write tests here
therefore run a real daemon stack. Reads use resident User-only authority
and remain available without a derived Index.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.commands.setting import setting_command
from polylogue.cli.operation_kernel import OperationUnavailableError
from tests.infra.daemon_operations import cli_daemon_archive


def _run(args: list[str]) -> dict[str, object] | list[object]:
    result = CliRunner().invoke(cli, ["--plain", "setting", *args], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    return cast("dict[str, object] | list[object]", json.loads(result.output))


def test_setting_get_reports_unset(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch) as stack:
        payload = _run(["get", "subscription_tier", "--format", "json"])
        requests: tuple[tuple[str, dict[str, object]], ...] = (
            ("user.settings.get", {"setting_key": "subscription_tier"}),
            ("user.settings.list", {}),
        )
        for operation, operands in requests:
            response = stack.client.operation(operation, operands)
            assert response is not None and response["outcome"] == "completed"
            assert response["result"]["outcome"]["state"] == "empty"
            assert set(response["schema_versions"]) == {"user"}
    assert payload == {"setting_key": "subscription_tier", "value": None}


def test_setting_set_then_get_round_trips(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon write is what the resident reads then see.

    Anti-vacuity: point ``setting set`` at a second in-process writer and this
    still passes, which is why ``test_setting_set_refuses_without_a_daemon``
    below is the load-bearing half -- together they say the write happened
    *and* that it could only have happened through the daemon.
    """
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        written = _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        assert isinstance(written, dict)
        assert written["setting_key"] == "subscription_tier"
        assert written["value"] == "max_5x"

        fetched = _run(["get", "subscription_tier", "--format", "json"])
        assert isinstance(fetched, dict)
        assert fetched["value"] == "max_5x"

        listed = _run(["list", "--format", "json"])
        assert isinstance(listed, list)
        assert listed == [written]


def test_setting_set_refuses_without_a_daemon(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No daemon means no write, not a second writer in the CLI process.

    Anti-vacuity: restore ``Polylogue.set_setting`` behind this command and the
    row lands with no daemon at all, so the assertion on the empty registry
    goes red. That is the whole defect polylogue-gjwto names.
    """
    from polylogue.cli.machine_main import run_machine_entry

    argv = ["--plain", "setting", "set", "subscription_tier", "pro", "--format", "json"]
    monkeypatch.setattr(sys, "argv", ["polylogue", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run_machine_entry(cli, argv)
    assert exit_info.value.code == 1
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["code"] == "daemon_required"
    assert refusal["details"]["operation"] == "mutation.user.setting.set"
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        assert _run(["get", "subscription_tier", "--format", "json"]) == {
            "setting_key": "subscription_tier",
            "value": None,
        }


def test_setting_set_rejects_unknown_key(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = CliRunner().invoke(cli, ["--plain", "setting", "set", "not_a_real_setting", "x"])
    assert result.exit_code != 0
    assert "unknown setting key" in result.output


def test_setting_set_rejects_invalid_tier_value(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = CliRunner().invoke(cli, ["--plain", "setting", "set", "subscription_tier", "not-a-tier"])
    assert result.exit_code != 0
    assert "subscription_tier must be one of" in result.output


@pytest.mark.parametrize("subcommand", ("get", "set", "list"))
def test_setting_subcommands_expose_standard_format_alias(subcommand: str) -> None:
    """Every settings read/write route accepts the CLI-wide ``-f`` shorthand."""
    result = CliRunner().invoke(setting_command, [subcommand, "--help"])

    assert result.exit_code == 0, result.output
    assert "-f, --format" in result.output


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_reads_refuse_unavailable_authority(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, fault: str, subcommand: str
) -> None:
    """An unavailable User tier cannot become an unset value or empty list."""
    from polylogue.cli.operation_kernel import OperationFailedError

    user = cli_workspace["archive_root"] / "user.db"
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch) as stack:
        if fault == "missing":
            user.unlink()
        else:
            user.write_bytes(b"not sqlite")
        operation = f"user.settings.{subcommand}"
        payload: dict[str, object] = {"setting_key": "subscription_tier"} if subcommand == "get" else {}
        response = stack.client.operation(operation, payload, archive_root=str(cli_workspace["archive_root"]))
        assert response is not None
        assert response["error"]["code"] == "archive_tier_unavailable"
        assert response["error"]["data"]["tier"] == "user"
        args = ["--plain", "setting", subcommand]
        if subcommand == "get":
            args.append("subscription_tier")
        result = CliRunner().invoke(cli, args)
        assert result.exit_code != 0
        assert isinstance(result.exception, OperationFailedError)
    assert user.exists() is (fault != "missing")


def test_setting_reads_remain_resident_without_index(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Derived Index absence must not withhold durable settings authority."""
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch) as stack:
        written = _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        (cli_workspace["archive_root"] / "index.db").unlink()
        response = stack.client.operation("user.settings.get", {"setting_key": "subscription_tier"})
        assert response is not None
        assert response["served_by"]["identity"] == "daemon"
        assert response["result"]["item"] == written
        assert _run(["list", "--format", "json"]) == [written]


@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_reads_refuse_without_daemon(cli_workspace: dict[str, Path], subcommand: str) -> None:
    args = ["--plain", "setting", subcommand]
    if subcommand == "get":
        args.append("subscription_tier")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, OperationUnavailableError)
    assert result.exception.operation == f"user.settings.{subcommand}"


@pytest.mark.parametrize("unrelated_tier", ["source", "audit"])
@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_reads_do_not_require_mutation_continuity(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, unrelated_tier: str, subcommand: str
) -> None:
    """Settings authority remains User-only when mutation evidence is unavailable."""
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        written = _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        (cli_workspace["archive_root"] / f"{unrelated_tier}.db").unlink()
        args = [subcommand]
        if subcommand == "get":
            args.append("subscription_tier")
        assert _run([*args, "--format", "json"]) == (written if subcommand == "get" else [written])


@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_reads_do_not_enter_pending_mutation_continuity(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, subcommand: str
) -> None:
    from contextlib import contextmanager

    from polylogue.operations.audit import AuditContinuityPendingError, AuditRepository

    @contextmanager
    def pending(self: AuditRepository) -> Iterator[None]:
        raise AuditContinuityPendingError("synthetic pending mutation continuity")
        yield

    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        written = _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        monkeypatch.setattr(AuditRepository, "settled_machine_read", pending)
        args = [subcommand]
        if subcommand == "get":
            args.append("subscription_tier")
        assert _run([*args, "--format", "json"]) == (written if subcommand == "get" else [written])


@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_native_read_cancellation_uses_its_admitted_user_handle(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, subcommand: str
) -> None:
    """Removing the User native scope loses interruption and cleanup evidence."""
    import sqlite3
    from collections.abc import Callable
    from contextlib import contextmanager

    from polylogue.archive.query.execution_control import InterruptibleSQLiteRead, QueryExecutionContext
    from polylogue.storage.sqlite.archive_tiers import user_settings_write

    contexts: list[QueryExecutionContext] = []
    control = InterruptibleSQLiteRead.control_connection

    @contextmanager
    def capture(self: InterruptibleSQLiteRead, connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        contexts.append(self._ctx)
        with control(self, connection) as controlled:
            yield controlled

    helper = "get_user_setting" if subcommand == "get" else "list_user_settings"
    read = cast(Callable[..., object], vars(user_settings_write)[helper])

    def interrupted(connection: sqlite3.Connection, *args: object) -> object:
        def cancel() -> int:
            assert contexts
            contexts[-1].cancel()
            return 1

        connection.create_function("cancel_settings_read", 0, cancel)
        connection.execute("SELECT cancel_settings_read() FROM user_settings").fetchall()
        return read(connection, *args)

    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch) as stack:
        _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        monkeypatch.setattr(InterruptibleSQLiteRead, "control_connection", capture)
        monkeypatch.setattr(user_settings_write, helper, interrupted)
        payload: dict[str, object] = {"setting_key": "subscription_tier"} if subcommand == "get" else {}
        response = stack.client.operation(f"user.settings.{subcommand}", payload)
        assert response is not None and response["outcome"] == "cancelled"
        assert response["error"]["code"] == "QueryCancelledError"
        assert len(contexts) == 1
        assert contexts[0].receipt.interrupted
        assert contexts[0].receipt.cleanup_complete


@pytest.mark.parametrize("subcommand", ["get", "list"])
def test_setting_reads_refuse_explicit_index_version_without_index_authority(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, subcommand: str
) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch) as stack:
        payload: dict[str, object] = {"setting_key": "subscription_tier"} if subcommand == "get" else {}
        response = stack.client.operation(f"user.settings.{subcommand}", payload, index_schema_version=999)
        assert response is not None
        assert response["outcome"] == "rejected"
        assert response["error"]["code"] == "schema_version_mismatch"
        assert response["result"] is None
