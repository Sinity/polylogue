"""Contracts for the human-facing ``--plain`` CLI surface.

These tests pin the user-visible behavior of plain mode: no ANSI escape
sequences, no Rich box-drawing characters, deterministic output across
consecutive invocations, and stable not-found messages on empty archives.

Companion to ``test_json_envelope_contract.py`` (machine surface, #1080)
and ``test_help_contract.py`` (help surface, #1060).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from tests.infra.contract_evidence import ContractEvidenceRecorder

pytestmark = pytest.mark.contract

# ANSI CSI sequences (color, cursor movement, formatting).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Other ANSI escapes (e.g. OSC sequences, single-char CSI introducers).
_ESC_RE = re.compile(r"\x1b")
# Unicode box-drawing characters Rich uses for tables.
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]")
# Rich uses these block characters for progress bars / shading.
_BLOCK_RE = re.compile(r"[\u2580-\u259f]")


def _assert_plain(output: str, *, context: str) -> None:
    """Assert ``output`` contains no ANSI escapes or box-drawing glyphs."""
    assert not _ANSI_RE.search(output), f"{context}: ANSI CSI sequence in {output!r}"
    assert not _ESC_RE.search(output), f"{context}: raw ESC byte in {output!r}"
    assert not _BOX_DRAWING_RE.search(output), f"{context}: box-drawing char in {output!r}"
    assert not _BLOCK_RE.search(output), f"{context}: block element char in {output!r}"


def _invoke_plain(args: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str, str]:
    """Invoke ``cli`` in plain mode and return (exit_code, stdout, stderr)."""
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()
    result = runner.invoke(cli, args, catch_exceptions=True)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.stdout, result.stderr


class TestPlainOutputIsAscii:
    """``--plain`` output must contain no ANSI escapes or box drawing."""

    @pytest.mark.parametrize(
        ("args", "contract_id"),
        [
            (["--plain", "list"], "cli.plain.list"),
            (["--plain", "stats"], "cli.plain.stats"),
            (["--plain", "config"], "cli.plain.config"),
            (
                ["--plain", "status", "--daemon-url", "http://127.0.0.1:19999"],
                "cli.plain.status",
            ),
        ],
    )
    def test_command_plain_output_has_no_ansi(
        self,
        args: list[str],
        contract_id: str,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
        record_contract_evidence: ContractEvidenceRecorder,
    ) -> None:
        """Plain output has no ANSI CSI codes or box-drawing characters."""
        exit_code, stdout, stderr = _invoke_plain(args, monkeypatch)
        _assert_plain(stdout, context=f"stdout for {' '.join(args)}")
        _assert_plain(stderr, context=f"stderr for {' '.join(args)}")
        record_contract_evidence.record(
            contract_id,
            surface="cli",
            command=("polylogue", *args),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            facts={
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stderr_bytes": len(stderr.encode("utf-8")),
                "ansi_present": False,
            },
        )


class TestPlainEmptyArchiveMessages:
    """Empty-archive plain messages are stable and human-readable."""

    def test_plain_list_empty_archive_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
        record_contract_evidence: ContractEvidenceRecorder,
    ) -> None:
        """``polylogue --plain list`` reports no conversations in human prose."""
        exit_code, stdout, _stderr = _invoke_plain(["--plain", "list"], monkeypatch)
        # Plain list on empty archive: human message, not a JSON envelope.
        assert exit_code in (0, 2), f"unexpected exit {exit_code}: {stdout!r}"
        assert not stdout.lstrip().startswith("{"), (
            f"plain list emitted JSON-shaped output instead of human text: {stdout!r}"
        )
        assert "No conversations" in stdout, f"missing 'No conversations' in {stdout!r}"
        record_contract_evidence.record(
            "cli.plain.list_empty_message",
            surface="cli",
            command=("polylogue", "--plain", "list"),
            stdout=stdout,
            exit_code=exit_code,
            facts={"has_no_conversations_phrase": True, "is_json_shaped": False},
        )

    def test_plain_stats_empty_archive_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
        record_contract_evidence: ContractEvidenceRecorder,
    ) -> None:
        """``polylogue --plain stats`` reports empty archive in human prose."""
        exit_code, stdout, _stderr = _invoke_plain(["--plain", "stats"], monkeypatch)
        assert exit_code in (0, 2), f"unexpected exit {exit_code}: {stdout!r}"
        assert not stdout.lstrip().startswith("{"), (
            f"plain stats emitted JSON-shaped output instead of human text: {stdout!r}"
        )
        record_contract_evidence.record(
            "cli.plain.stats_empty_message",
            surface="cli",
            command=("polylogue", "--plain", "stats"),
            stdout=stdout,
            exit_code=exit_code,
            facts={"is_json_shaped": False, "exit_code": exit_code},
        )


class TestPlainOutputIsDeterministic:
    """Plain output across consecutive invocations is stable.

    Deterministic means: the same command against the same (empty) archive
    produces identical output on two consecutive invocations. Conversation
    timestamps would change the output, but on an empty archive there is no
    such drift; this pins the contract.
    """

    @pytest.mark.parametrize(
        ("args", "contract_id"),
        [
            (["--plain", "list"], "cli.plain.list_deterministic"),
            (["--plain", "stats"], "cli.plain.stats_deterministic"),
            (["--plain", "config"], "cli.plain.config_deterministic"),
        ],
    )
    def test_plain_output_is_stable_across_invocations(
        self,
        args: list[str],
        contract_id: str,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
        record_contract_evidence: ContractEvidenceRecorder,
    ) -> None:
        """Two back-to-back invocations on the same empty archive match exactly."""
        first_exit, first_stdout, _ = _invoke_plain(args, monkeypatch)
        second_exit, second_stdout, _ = _invoke_plain(args, monkeypatch)
        assert first_exit == second_exit, f"exit code drift: {first_exit} vs {second_exit} for {' '.join(args)}"
        assert first_stdout == second_stdout, (
            f"stdout drift for {' '.join(args)}:\nfirst  : {first_stdout!r}\nsecond : {second_stdout!r}"
        )
        record_contract_evidence.record(
            contract_id,
            surface="cli",
            command=("polylogue", *args),
            stdout=first_stdout,
            exit_code=first_exit,
            facts={"stable": True, "byte_len": len(first_stdout.encode("utf-8"))},
        )


class TestPlainModeRespectsStderrDiscipline:
    """``--plain`` does not collapse stderr into stdout.

    Click error responses (e.g. invalid choices) must continue to land on
    stderr in plain mode. Plain mode toggles renderer formatting, not stream
    routing.
    """

    def test_invalid_flag_writes_error_to_stderr_in_plain_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
        record_contract_evidence: ContractEvidenceRecorder,
    ) -> None:
        """Plain mode + invalid choice -> non-zero exit, error stays on stderr."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "list", "--format", "xml"], catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        assert result.exit_code != 0
        # Click writes usage errors to stderr by default.
        assert "Error" in result.stderr or "Usage" in result.stderr, (
            f"expected Click error on stderr, got stderr={result.stderr!r} stdout={result.stdout!r}"
        )
        # The actual error text should not be on stdout.
        assert "Invalid value" not in result.stdout, f"error text leaked to stdout in plain mode: {result.stdout!r}"
        record_contract_evidence.record(
            "cli.plain.stderr_discipline",
            surface="cli",
            command=("polylogue", "--plain", "list", "--format", "xml"),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            facts={
                "exit_code": result.exit_code,
                "error_on_stderr": True,
                "error_not_on_stdout": True,
            },
        )
