"""A failed read exits on its own status, not on the empty status.

polylogue-jtrtj: ``archive_query._read_failure_as_usage_error`` turned every
typed read failure that was not ``daemon_required``/``result_too_large`` into a
``click.UsageError``.  Click exits 2 for that, and 2 is ``EMPTY_EXIT_CODE`` --
so a dropped daemon connection, a read that hit its deadline, a cancelled read
and a refused request were all indistinguishable by exit status from "matched
nothing", and three of them printed Click's usage banner, framing a transport
failure as a syntax mistake.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli.operation_kernel import (
    OperationCancelledError,
    OperationFailedError,
    OperationUnavailableError,
)
from polylogue.cli.render.outcome import (
    CANCELLED_EXIT_CODE,
    EMPTY_EXIT_CODE,
    FAILED_READ_EXIT_CODE,
    read_failure_exit_code,
    read_failure_message,
)

#: One representative of each failure class the bead names, with the exit code
#: it must reach.  ``EMPTY_EXIT_CODE`` appears in none of them: that is the
#: whole point.
_FAILURE_CLASSES: tuple[tuple[str, Exception, int], ...] = (
    (
        "transport",
        OperationFailedError("daemon_transport_error", "connection reset by peer", request_id="call-7"),
        FAILED_READ_EXIT_CODE,
    ),
    (
        "deadline",
        OperationFailedError("QueryTimeoutError", "query exceeded its deadline", {"deadline_ms": 5000}),
        FAILED_READ_EXIT_CODE,
    ),
    ("cancelled", OperationCancelledError("cli.query", "interrupted"), CANCELLED_EXIT_CODE),
    (
        "invalid-request",
        OperationFailedError("invalid_request", "sample does not combine with a cursor"),
        FAILED_READ_EXIT_CODE,
    ),
    ("unavailable", OperationUnavailableError("daemon is unavailable for operation: cli.query"), FAILED_READ_EXIT_CODE),
)


@pytest.mark.parametrize(
    ("label", "exc", "expected"),
    _FAILURE_CLASSES,
    ids=[label for label, _exc, _expected in _FAILURE_CLASSES],
)
def test_each_failure_class_has_its_own_exit_status(label: str, exc: Exception, expected: int) -> None:
    """None of them is the empty status.

    Anti-vacuity: restore the ``click.UsageError`` wrap in
    ``_read_failure_as_usage_error`` and every case exits 2, so both assertions
    below go red.
    """

    code = read_failure_exit_code(exc)

    assert code == expected
    assert code != EMPTY_EXIT_CODE


def test_cancellation_is_not_an_outcome_state() -> None:
    """130 is the shell's interrupt convention, distinct from every outcome.

    Anti-vacuity: map cancellation onto ``error`` (1) and this goes red.
    """

    from polylogue.surfaces.outcome import OUTCOME_EXIT_CODES

    assert CANCELLED_EXIT_CODE == 130
    assert CANCELLED_EXIT_CODE not in OUTCOME_EXIT_CODES.values()


def test_the_message_names_the_call_id_and_a_remedy() -> None:
    """A transport failure the operator cannot correlate is half a diagnosis.

    Anti-vacuity: drop ``request_id`` from ``OperationFailedError`` (or the
    remedy table from ``render/outcome``) and this goes red.
    """

    message = read_failure_message(
        OperationFailedError("daemon_transport_error", "connection reset by peer", request_id="call-7")
    )

    assert "connection reset by peer" in message
    assert "call call-7" in message
    assert "Remedy:" in message
    assert "ops status" in message


def test_the_deadline_is_named_when_the_operation_reported_one() -> None:
    """Anti-vacuity: stop reading ``data['deadline_ms']`` and this goes red."""

    message = read_failure_message(
        OperationFailedError("QueryTimeoutError", "query exceeded its deadline", {"deadline_ms": 5000})
    )

    assert "deadline 5000 ms" in message
    assert "Remedy:" in message


@pytest.mark.parametrize(
    ("label", "exc", "expected"),
    _FAILURE_CLASSES,
    ids=[label for label, _exc, _expected in _FAILURE_CLASSES],
)
def test_the_real_cli_route_exits_without_a_usage_banner(
    label: str, exc: Exception, expected: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production ``find`` route, not just the helper.

    Anti-vacuity: re-raise the failure as ``click.UsageError`` and both the exit
    code and the absent-banner assertion go red -- Click prints ``Usage:`` and
    exits 2.
    """

    from polylogue.cli.click_app import cli

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    (tmp_path / "index.db").write_bytes(b"")

    with patch("polylogue.cli.operation_kernel.dispatch", side_effect=exc):
        result = CliRunner().invoke(cli, ["find", "anything"])

    assert result.exit_code == expected, result.output
    assert result.exit_code != EMPTY_EXIT_CODE
    assert "Usage:" not in result.output


def test_a_machine_caller_still_gets_a_parseable_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--format json`` must stay parseable when the transport fails.

    ``machine_main`` re-raises a bare non-zero ``SystemExit`` unchanged, so the
    structured document has to be emitted at the read-failure terminal itself.

    Anti-vacuity: drop the ``wants_json`` branch from ``exit_for_read_failure``
    and this goes red on ``json.loads``.
    """

    from polylogue.cli.render.outcome import exit_for_read_failure

    monkeypatch.setattr("sys.argv", ["polylogue", "--format", "json", "find", "anything"])
    with pytest.raises(SystemExit) as caught:
        exit_for_read_failure(OperationFailedError("daemon_transport_error", "connection reset by peer"))

    assert caught.value.code == FAILED_READ_EXIT_CODE
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "error"
    assert "connection reset by peer" in document["message"]


def test_the_daemon_remedy_names_a_command_that_exists() -> None:
    """``polylogue run`` is not a verb; the daemon entry point is ``polylogued``.

    polylogue-3eexy AC4 requires the daemon-absent refusal to name
    ``polylogued run``.  At the reviewed head the remedy said "start the daemon
    with `polylogue run`, or re-run without --daemon-only" -- a verb the CLI
    does not define and a flag no command declares.

    Anti-vacuity: restore either the ``polylogue run`` wording or the
    ``--daemon-only`` clause and one of the assertions below goes red.
    """

    from polylogue.cli.click_app import cli

    message = read_failure_message(OperationUnavailableError("daemon is unavailable for operation: cli.query"))

    assert "`polylogued run`" in message
    assert "--daemon-only" not in message
    assert "run" not in cli.commands


def test_a_daemon_absent_browse_is_refused_not_reported_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon that did not answer says nothing about what the archive holds.

    polylogue-3eexy AC4.  ``_missing_archive_refusal`` renders browse mode over
    an absent index as a *valid empty answer*.  Reaching that branch from a
    ``daemon_required`` refusal made ``polylogue read --all`` print
    ``outcome: empty (no_rows_in_scope)`` and exit 0 -- "no matches" for a read
    that was never executed.  The direct-read fallback is all that hides this
    today; with the fallback gone it is what the operator would see.

    Anti-vacuity: drop the ``_is_daemon_unavailable`` guard from
    ``archive_query`` and this exits 0 with an ``outcome: empty`` body instead
    of the typed refusal -- verified by reverting the guard.
    """

    from polylogue.cli.click_app import cli

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    assert not (tmp_path / "index.db").exists()

    with patch(
        "polylogue.cli.operation_kernel.dispatch",
        side_effect=OperationUnavailableError("daemon is unavailable for operation: cli.query"),
    ):
        result = CliRunner().invoke(cli, ["read", "--all"])

    assert result.exit_code == FAILED_READ_EXIT_CODE, result.output
    assert "outcome: empty" not in result.output
    assert "polylogued run" in result.output
