"""Deterministic output tests for CLI commands.

Verifies:
1. `--json` output is always valid JSON (parseable by json.loads)
2. `POLYLOGUE_FORCE_PLAIN=1` produces no ANSI escape codes
3. Frozen clock produces deterministic timestamps in `doctor --json`
4. QA report `generate_qa_session` produces deterministic timestamps
5. Commands parametrized across `doctor`, `tags`, `sources` with `--json`
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.core.json import JSONDocument
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import ProofObligation
from polylogue.proof.runners import run_cli_visual_evidence
from polylogue.scenarios import polylogue_execution
from polylogue.schemas.audit.models import AuditReport
from polylogue.schemas.validation.models import ArtifactProofReport, ProviderArtifactProof
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_report import generate_qa_session
from polylogue.showcase.qa_runner import QAResult
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult
from tests.infra.json_contracts import envelope_result, extract_json_object

# ANSI escape code pattern: ESC[ ... final-byte
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\[.*?m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(output: str) -> JSONDocument:
    """Extract the first JSON object from CLI output, skipping log/banner lines."""
    return extract_json_object(output, context="CLI output")


def _result_payload(data: JSONDocument) -> JSONDocument:
    return envelope_result(data, context="CLI envelope")


def _has_ansi(text: str) -> bool:
    """Return True if text contains ANSI escape codes."""
    return bool(_ANSI_RE.search(text))


def _plain_mode_obligation(args: list[str]) -> ProofObligation:
    command_path = tuple(arg for arg in args if arg not in {"--plain", "--help"})
    subject_id = "polylogue" if not command_path else f"polylogue {' '.join(command_path)}"
    catalog = build_verification_catalog()
    for obligation in catalog.obligations:
        if obligation.claim.id == "cli.command.plain_mode" and obligation.subject.id == subject_id:
            return obligation
    raise AssertionError(f"missing plain-mode obligation for {args!r}")


# ---------------------------------------------------------------------------
# F3: Frozen clock → deterministic doctor --json timestamp
# ---------------------------------------------------------------------------


class TestFrozenClockCheckJson:
    """doctor --json timestamp is deterministic when time.time() is frozen."""

    FROZEN_EPOCH = 1700000000  # 2023-11-14T22:13:20Z

    def test_check_json_timestamp_is_deterministic(
        self: TestFrozenClockCheckJson, cli_workspace: dict[str, Path]
    ) -> None:
        """Two runs with same frozen clock produce identical timestamps."""

        def _run_check() -> JSONDocument:
            runner = CliRunner()
            with patch("time.time", return_value=float(self.FROZEN_EPOCH)):
                result = runner.invoke(
                    cli,
                    ["--plain", "doctor", "--json"],
                    catch_exceptions=False,
                )
            assert result.exit_code == 0, result.output
            return _extract_json(result.output)

        data_a = _run_check()
        data_b = _run_check()

        # Both runs use the same frozen epoch for their timestamp
        assert "result" in data_a
        assert "result" in data_b
        ts_a = _result_payload(data_a)["timestamp"]
        ts_b = _result_payload(data_b)["timestamp"]
        assert ts_a == ts_b == self.FROZEN_EPOCH

    def test_check_json_timestamp_changes_with_clock(
        self: TestFrozenClockCheckJson, cli_workspace: dict[str, Path]
    ) -> None:
        """Different frozen times produce different timestamps."""
        runner = CliRunner()

        with patch("time.time", return_value=1700000000.0):
            result_a = runner.invoke(cli, ["--plain", "doctor", "--json"], catch_exceptions=False)
        with patch("time.time", return_value=1800000000.0):
            result_b = runner.invoke(cli, ["--plain", "doctor", "--json"], catch_exceptions=False)

        ts_a = _result_payload(_extract_json(result_a.output))["timestamp"]
        ts_b = _result_payload(_extract_json(result_b.output))["timestamp"]
        assert ts_a != ts_b


# ---------------------------------------------------------------------------
# F3: Frozen clock → deterministic showcase QA session timestamps
# ---------------------------------------------------------------------------


class TestFrozenClockShowcaseReport:
    """generate_qa_session uses datetime.now(UTC); patching produces deterministic output."""

    FROZEN_DT = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def _make_qa_result() -> QAResult:
        """Build a minimal QAResult for testing."""
        exercise = Exercise(
            name="test-exercise",
            group="structural",
            tier=0,
            description="A test exercise",
            execution=polylogue_execution("--help"),
        )
        ex_result = ExerciseResult(
            exercise=exercise,
            passed=True,
            exit_code=0,
            output="Usage: polylogue ...",
            duration_ms=42.0,
        )
        showcase_result = ShowcaseResult(
            results=[ex_result],
            total_duration_ms=42.0,
        )
        return QAResult(
            audit_report=AuditReport(
                checks=[
                    OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
                ]
            ),
            proof_report=ArtifactProofReport(
                providers={
                    "chatgpt": ProviderArtifactProof(
                        provider="chatgpt",
                        total_records=1,
                        contract_backed_records=1,
                    )
                },
                total_records=1,
            ),
            showcase_result=showcase_result,
            invariant_results=[
                InvariantResult("json_valid", "test-exercise", OutcomeStatus.OK),
            ],
        )

    def test_qa_session_timestamp_is_deterministic(self: TestFrozenClockShowcaseReport) -> None:
        """Two calls with same frozen clock produce identical timestamps."""
        result = self._make_qa_result()

        with patch("polylogue.showcase.qa_report.datetime") as mock_dt:
            mock_dt.now.return_value = self.FROZEN_DT
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            session_a = generate_qa_session(result)

        with patch("polylogue.showcase.qa_report.datetime") as mock_dt:
            mock_dt.now.return_value = self.FROZEN_DT
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            session_b = generate_qa_session(result)

        assert session_a["timestamp"] == session_b["timestamp"]
        assert session_a["timestamp"] == "2024-06-15T12:00:00+00:00"

    def test_qa_session_timestamp_changes_with_clock(self: TestFrozenClockShowcaseReport) -> None:
        """Different frozen times produce different timestamps."""
        result = self._make_qa_result()

        dt_a = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_b = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        with patch("polylogue.showcase.qa_report.datetime") as mock_dt:
            mock_dt.now.return_value = dt_a
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            session_a = generate_qa_session(result)

        with patch("polylogue.showcase.qa_report.datetime") as mock_dt:
            mock_dt.now.return_value = dt_b
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            session_b = generate_qa_session(result)

        assert session_a["timestamp"] != session_b["timestamp"]


# ---------------------------------------------------------------------------
# F4: POLYLOGUE_FORCE_PLAIN=1 → no ANSI escape codes in output
# ---------------------------------------------------------------------------


class TestPlainModeNoAnsi:
    """POLYLOGUE_FORCE_PLAIN=1 must produce zero ANSI escape codes."""

    COMMANDS: list[list[str]] = [
        ["--plain", "doctor"],
        ["--plain", "--help"],
        ["--plain", "doctor", "--help"],
        ["--plain", "tags", "--help"],
        ["--plain", "run", "--help"],
    ]

    @pytest.mark.parametrize(
        "args",
        COMMANDS,
        ids=[" ".join(c) for c in COMMANDS],
    )
    def test_no_ansi_in_plain_mode(self: object, args: list[str]) -> None:
        """Output with --plain should never contain ANSI escape sequences."""
        runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
        result = runner.invoke(cli, args, catch_exceptions=True)
        # Commands may fail (e.g., no workspace) but output must be ANSI-free
        assert not _has_ansi(result.output), f"ANSI codes found in output for {args!r}:\n{result.output[:200]}"
        evidence = run_cli_visual_evidence(_plain_mode_obligation(args), args=args)
        assert evidence.status is OutcomeStatus.OK


# ---------------------------------------------------------------------------
# F4: --json always produces valid JSON (parseable by json.loads)
# ---------------------------------------------------------------------------


class TestJsonOutputValidity:
    """--json commands must emit valid, parseable JSON."""

    def test_check_json_is_valid(self: object, cli_workspace: dict[str, Path]) -> None:
        """polylogue doctor --json produces parseable JSON."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "doctor", "--json"], catch_exceptions=False)
        assert result.exit_code == 0
        parsed = _extract_json(result.output)
        assert isinstance(parsed, dict)
        assert "status" in parsed

    def test_tags_json_is_valid(self: object, cli_workspace: dict[str, Path]) -> None:
        """polylogue tags --json produces parseable JSON."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "tags", "--json"], catch_exceptions=False)
        assert result.exit_code == 0
        parsed = _extract_json(result.output)
        assert isinstance(parsed, dict)
        assert parsed["status"] == "ok"
        assert "tags" in _result_payload(parsed)


# ---------------------------------------------------------------------------
# F4: parametrized --json envelope contract across commands
# ---------------------------------------------------------------------------


class TestJsonEnvelopeParametrized:
    """All --json commands must return the {"status": "ok", "result": {...}} envelope."""

    @pytest.mark.parametrize(
        "cmd_args,result_key",
        [
            (["doctor", "--json"], None),  # result has checks, summary, timestamp
            (["tags", "--json"], "tags"),  # result has tags
        ],
        ids=["doctor", "tags"],
    )
    def test_json_envelope_shape(
        self: object,
        cli_workspace: dict[str, Path],
        cmd_args: list[str],
        result_key: str | None,
    ) -> None:
        """All --json commands wrap output in the standard envelope."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", *cmd_args], catch_exceptions=False)
        assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

        parsed = _extract_json(result.output)
        assert parsed["status"] == "ok"
        assert "result" in parsed
        result_payload = _result_payload(parsed)

        if result_key is not None:
            assert result_key in result_payload, f"Expected key {result_key!r} in result, got {list(result_payload)}"

    @pytest.mark.parametrize(
        "cmd_args",
        [
            ["doctor", "--json"],
            ["tags", "--json"],
        ],
        ids=["doctor", "tags"],
    )
    def test_json_output_no_ansi(
        self: object,
        cli_workspace: dict[str, Path],
        cmd_args: list[str],
    ) -> None:
        """--json output must not contain ANSI escape codes."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", *cmd_args], catch_exceptions=False)
        assert result.exit_code == 0
        assert not _has_ansi(result.output), f"ANSI codes in --json output for {cmd_args!r}"

    @pytest.mark.parametrize(
        "cmd_args",
        [
            ["doctor", "--json"],
            ["tags", "--json"],
        ],
        ids=["doctor", "tags"],
    )
    def test_json_output_round_trips(
        self: object,
        cli_workspace: dict[str, Path],
        cmd_args: list[str],
    ) -> None:
        """--json output can be serialized and deserialized without loss."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", *cmd_args], catch_exceptions=False)
        assert result.exit_code == 0

        parsed = _extract_json(result.output)
        # Round-trip through json.dumps/loads must be lossless
        round_tripped = json.loads(json.dumps(parsed))
        assert round_tripped == parsed


# ---------------------------------------------------------------------------
# F4: --json determinism across runs (same input → same output)
# ---------------------------------------------------------------------------


class TestJsonDeterminism:
    """Same inputs with frozen time must produce byte-identical --json output."""

    @staticmethod
    def _normalize_check_result(data: JSONDocument) -> JSONDocument:
        """Remove provenance fields that legitimately vary between runs.

        The health system reports live provenance. That provenance is still a
        runtime artifact, not a determinism issue.
        """
        result = dict(_result_payload(data) if "result" in data else data)
        raw_provenance = result.get("provenance")
        assert raw_provenance is None or isinstance(raw_provenance, dict)
        provenance = dict(raw_provenance or {})
        provenance.pop("source", None)
        result["provenance"] = provenance
        return {**data, "result": result}

    def test_check_json_deterministic_with_frozen_time(
        self: TestJsonDeterminism, cli_workspace: dict[str, Path]
    ) -> None:
        """Two doctor --json runs with same frozen time produce identical results."""
        runner = CliRunner()
        parsed: list[JSONDocument] = []

        for _ in range(2):
            with patch("time.time", return_value=1700000000.0):
                result = runner.invoke(
                    cli,
                    ["--plain", "doctor", "--json"],
                    catch_exceptions=False,
                )
            assert result.exit_code == 0
            parsed.append(self._normalize_check_result(_extract_json(result.output)))

        assert parsed[0] == parsed[1]

    def test_tags_json_deterministic(self: object, cli_workspace: dict[str, Path]) -> None:
        """Two tags --json runs produce identical output."""
        runner = CliRunner()
        outputs: list[str] = []

        for _ in range(2):
            result = runner.invoke(
                cli,
                ["--plain", "tags", "--json"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            outputs.append(result.output)

        assert outputs[0] == outputs[1]


# ---------------------------------------------------------------------------
# Merged from test_formatting.py (2026-03-15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plain", "env_value", "tty", "expected"),
    [
        (True, None, True, True),
        (False, "1", True, True),
        (False, None, False, True),
        (False, None, True, False),
    ],
)
def test_should_use_plain_contract(
    plain: bool,
    env_value: str | None,
    tty: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.cli.shared.formatting import should_use_plain

    if env_value is None:
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", env_value)

    with patch("sys.stdout.isatty", return_value=tty), patch("sys.stderr.isatty", return_value=tty):
        assert should_use_plain(plain=plain) is expected


@pytest.mark.parametrize("falsey", ["0", "false", "no"])
def test_should_use_plain_falsey_env_values_do_not_force_plain(
    falsey: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.cli.shared.formatting import should_use_plain

    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", falsey)
    with patch("sys.stdout.isatty", return_value=True), patch("sys.stderr.isatty", return_value=True):
        assert should_use_plain(plain=False) is False


def test_announce_plain_mode_writes_to_stderr() -> None:
    from polylogue.cli.shared.formatting import announce_plain_mode

    captured = StringIO()
    with patch.object(sys, "stderr", captured):
        announce_plain_mode()
    assert "Plain output active" in captured.getvalue()


@pytest.mark.parametrize(
    ("cursors", "expected_parts"),
    [
        ({}, None),
        ({"inbox": {"file_count": 10}}, ("10 files", "inbox")),
        ({"source": {"file_count": 5, "error_count": 2}}, ("2 errors",)),
        ({"source": {"file_count": 5, "error_count": 0}}, ("5 files",)),
        ({"source": {"latest_mtime": 1704067200}}, ("latest",)),
        ({"source": {"latest_file_name": "chat.json"}}, ("latest chat.json",)),
        ({"source": {"latest_path": "/tmp/export.json"}}, ("latest export.json",)),
        ({"src": "plain_string"}, ("src", "unknown")),
        ({"inbox": {"file_count": 5}, "drive": {"file_count": 3}}, ("inbox", "drive", ";")),
    ],
)
def test_format_cursors_contract(cursors: dict[str, object], expected_parts: tuple[str, ...] | None) -> None:
    from polylogue.cli.shared.formatting import format_cursors

    result = format_cursors(cursors)
    if expected_parts is None:
        assert result is None
        return
    assert result is not None
    for part in expected_parts:
        assert part in result


@pytest.mark.parametrize(
    ("counts", "expected_parts"),
    [
        ({"conversations": 10, "messages": 100}, ("10 conv", "100 msg")),
        (
            {
                "conversations": 3,
                "new_conversations": 2,
                "changed_conversations": 1,
                "messages": 30,
            },
            ("3 conv (2 new, 1 changed)", "30 msg"),
        ),
        ({"conversations": 5, "messages": 50, "rendered": 5}, ("5 rendered",)),
        ({"acquired": 4, "validated": 4, "validation_drift": 2}, ("4 acquired", "4 validated", "2 drift")),
        ({"conversations": 5, "messages": 50, "rendered": 0}, ("5 conv", "50 msg")),
        ({}, ("0 conv", "0 msg")),
    ],
)
def test_format_counts_contract(counts: dict[str, object], expected_parts: tuple[str, ...]) -> None:
    from polylogue.cli.shared.formatting import format_counts

    result = format_counts(counts)
    for part in expected_parts:
        assert part in result


@pytest.mark.parametrize(
    ("stage", "indexed", "error", "expected"),
    [
        ("parse", False, None, "Index: up-to-date"),
        ("parse", True, None, "Index: ok"),
        ("render", True, None, "Index: skipped"),
        ("index", False, "boom", "Index: error"),
        ("index", True, None, "Index: ok"),
        ("all", False, None, "Index: up-to-date"),
    ],
)
def test_format_index_status_contract(stage: str, indexed: bool, error: str | None, expected: str) -> None:
    from polylogue.cli.shared.formatting import format_index_status

    assert format_index_status(stage, indexed, error) == expected


@pytest.mark.parametrize(
    ("source_name", "provider_name", "expected"),
    [
        ("inbox", "claude-ai", "inbox/claude-ai"),
        ("chatgpt", "chatgpt", "chatgpt"),
        (None, "codex", "codex"),
    ],
)
def test_format_source_label_contract(source_name: str | None, provider_name: str, expected: str) -> None:
    from polylogue.cli.shared.formatting import format_source_label

    assert format_source_label(source_name, provider_name) == expected


def test_format_sources_summary_contract() -> None:
    from polylogue.cli.shared.formatting import format_sources_summary
    from polylogue.config import Source

    sources = [
        Source(name="inbox", path=Path("/inbox")),
        Source(name="gemini", folder="folder-id"),
    ]
    result = format_sources_summary(sources)
    assert "inbox" in result
    assert "gemini (drive)" in result


def test_format_sources_summary_marks_missing() -> None:
    from polylogue.cli.shared.formatting import format_sources_summary

    source = MagicMock()
    source.name = "broken"
    source.path = None
    source.folder = None
    assert "broken (missing)" in format_sources_summary([source])


def test_format_sources_summary_truncates_long_lists() -> None:
    from polylogue.cli.shared.formatting import format_sources_summary
    from polylogue.config import Source

    sources = [Source(name=f"source{i}", path=Path(f"/src{i}")) for i in range(12)]
    result = format_sources_summary(sources)
    assert "+4 more" in result
    assert result.count(",") == 8
