"""Systematic test that every CLI command accepts ``--json`` without crashing (#1689).

Tests that each command, when invoked with ``--json --plain``, produces:
- Exit code 0 or a clean error (no traceback)
- Valid JSON on stdout (or empty/non-JSON output for commands that don't claim JSON)
- No ANSI escape codes in stdout (pipeability)

The root-level ``--json`` flag (#1712) forces plain output and sets
``output_format=json``. This test ensures every command either produces
valid JSON or exits cleanly without a traceback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

_TRACEBACK_SENTINEL = "Traceback (most recent call last)"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _has_ansi(text: str) -> bool:
    """Return True if text contains ANSI escape sequences."""
    return bool(_ANSI_RE.search(text))


def _try_parse_json(text: str) -> object | None:
    """Try to parse text as JSON. Returns parsed value or None."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


def _looks_like_json_output(text: str) -> bool:
    """Heuristic: text that looks like it intends to be JSON.

    JSON objects always start with ``{``. JSON arrays start with ``[``
    followed by a JSON value delimiter (``{``, ``"``, digit, ``t``, ``f``,
    ``n``, ``-``). TOML section headers like ``[archive]`` start with
    ``[`` followed by a letter but are not JSON, so we exclude those.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # JSON objects always start with {
    # JSON arrays start with [ followed by {, ", digit, t(rue), f(alse), n(ull), -
    # TOML section headers like [archive] start with [a-z] — not JSON
    return stripped.startswith("{") or (
        stripped.startswith("[") and len(stripped) > 1 and stripped[1] in '{"tfn0123456789-'
    )


# ── Command list ───────────────────────────────────────────────────────
#
# Each entry: (args: list[str], allow_nonzero_exit: bool)
#
# ``allow_nonzero_exit=True`` means non-zero exit codes (e.g. exit 2 for
# "no_results" error envelopes) are acceptable as long as there is no
# traceback.  This covers both "no results on empty archive" and "help
# shown because a required argument is missing" cases.
#
# Commands intentionally excluded:
#   - delete, reset        → destructive
#   - import               → requires source config, modifies archive
#   - tutorial, auth, init → interactive or requires external service
#   - dashboard, open      → side effect (opens browser)
#   - backup               → creates backup file; needs populated archive
#   - recent               → pre-existing bug: passes sort=updated_at which
#                            is not a valid sort field (hardcoded in verb)
#   - context, context-pack → not wired for lazy subcommand dispatch

_COMMANDS: list[tuple[list[str], bool]] = [
    # ── Query verbs (root --json flows through to output_format) ──────
    (["list"], True),
    (["count"], True),
    (["stats"], True),
    (["read"], True),
    # ── Top-level commands ───────────────────────────────────────────
    (["commands"], False),
    (["doctor"], False),
    (["config"], False),  # TOML output with root --json (not JSON, but pipeable)
    (["cost", "rollup"], False),
    (["resume"], True),
    (["resume-candidates"], True),
    (["tags"], False),
    # ── Insights subcommands ─────────────────────────────────────────
    (["insights", "status"], False),
    (["insights", "audit"], False),
    (["insights", "coverage"], False),
    (["insights", "debt"], False),
    (["insights", "profiles"], False),
    (["insights", "phases"], False),
    (["insights", "threads"], False),
    (["insights", "work-events"], False),
    (["insights", "cost-rollups"], False),
    (["insights", "costs"], False),
    (["insights", "tool-usage"], False),
    (["insights", "tags"], False),
    # ── Diagnostics subcommands ──────────────────────────────────────
    (["diagnostics", "pace"], False),
    (["diagnostics", "tools"], False),
    (["--latest", "diagnostics", "turns"], True),  # needs session; fails cleanly on empty
    # ── Embed, feedback, schema, maintenance ─────────────────────────
    (["embed", "status"], False),
    (["feedback", "list"], False),
    (["schema", "list"], False),
    (["maintenance", "status"], False),
    (["maintenance", "preview"], False),
    # ── Shell integration (emits scripts, not JSON; must not crash) ──
    (["completions", "--shell", "bash"], False),
]

# Excluded from COMMANDS for documented reasons:
_EXCLUDED: list[str] = [
    "delete (destructive)",
    "reset (destructive)",
    "import (requires source config, modifies archive)",
    "tutorial (interactive)",
    "auth (interactive, requires external service)",
    "init (interactive, writes config)",
    "dashboard (side effect: opens browser)",
    "open (side effect: opens browser)",
    "backup (creates backup file)",
    "recent (pre-existing bug: hardcoded sort=updated_at is invalid)",
    "context compose (not wired for lazy subcommand dispatch from root)",
    "context-pack (not wired for lazy subcommand dispatch from root)",
    "insights timeline (requires SESSION_ID argument)",
    "insights export (requires --out argument)",
    "blackboard list (not wired for lazy subcommand dispatch from root)",
    "user-state marks (not wired for lazy subcommand dispatch from root)",
]

_COMMAND_IDS: list[str] = [" ".join(args) for args, _ in _COMMANDS]


@pytest.mark.parametrize(
    ("args", "allow_nonzero_exit"),
    _COMMANDS,
    ids=_COMMAND_IDS,
)
class TestAllCommandsAcceptJson:
    """Every CLI command must handle ``--json --plain`` without crashing.

    The root-level ``--json`` flag (#1689, #1712) forces plain output
    and sets ``output_format=json``. Every command must either produce
    valid JSON or exit cleanly without a traceback.
    """

    def test_command_with_json_produces_clean_output(
        self,
        args: list[str],
        allow_nonzero_exit: bool,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """Run command with --json --plain; assert clean, pipeable output."""
        del workspace_env  # used to isolate XDG paths
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        # Route daemon requests to an unreachable port so status-like
        # commands don't hang trying to connect.
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:1")

        full_args = ["--json", "--plain", *args]
        runner = CliRunner()
        result = runner.invoke(cli, full_args, catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception

        # ── Exit code ────────────────────────────────────────────
        if not allow_nonzero_exit:
            assert result.exit_code == 0, (
                f"`polylogue {' '.join(full_args)}` exited {result.exit_code} (expected 0)\noutput: {result.output!r}"
            )

        # ── No traceback ─────────────────────────────────────────
        assert _TRACEBACK_SENTINEL not in result.output, (
            f"`polylogue {' '.join(full_args)}` produced a traceback:\n{result.output}"
        )

        # ── No ANSI escape codes ─────────────────────────────────
        assert not _has_ansi(result.output), (
            f"`polylogue {' '.join(full_args)}` produced ANSI escape codes (not pipeable):\n{result.output!r}"
        )

        # ── Valid JSON if it looks like JSON ──────────────────────
        stripped = result.output.strip()
        if stripped:
            parsed = _try_parse_json(stripped)
            if parsed is None and _looks_like_json_output(stripped):
                # Output looks like JSON but isn't parseable —
                # this is a real failure.
                pytest.fail(
                    f"`polylogue {' '.join(full_args)}` produced output "
                    f"that looks like JSON but is not valid:\n{stripped!r}"
                )
            # Non-JSON-looking output (e.g. TOML from `config`, shell
            # scripts from `completions`) is acceptable — those commands
            # don't claim JSON support via the root --json flag.

    def test_command_with_help_accepts_json(
        self,
        args: list[str],
        allow_nonzero_exit: bool,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """Run command with --json --plain --help; assert no crash."""
        del workspace_env, allow_nonzero_exit
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:1")

        full_args = ["--json", "--plain", *args, "--help"]
        runner = CliRunner()
        result = runner.invoke(cli, full_args, catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception

        # --help always exits 0 (Click handles this)
        assert result.exit_code == 0, f"`polylogue {' '.join(full_args)}` exited {result.exit_code}:\n{result.output!r}"
        assert _TRACEBACK_SENTINEL not in result.output
        assert not _has_ansi(result.output)
