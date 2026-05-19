"""Snapshot tests for the root and key subcommand help output (#1273).

Pins the help surface that documents query-first dispatch, the
``--diagnose`` flag, and the see-also references so accidental wording
drift fails this test rather than silently degrading the operator
experience.

Companion to ``test_diagnose_and_error_discipline.py`` (behavior
contracts) and ``test_plain_cli_snapshots.py`` (data rendering).
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

syrupy = pytest.importorskip("syrupy")

from polylogue.cli.click_app import cli

# Version line is volatile (commit hash, dirty marker), redact it.
_VERSION_LINE_RE = re.compile(r"polylogue,\s+version\s+\S+")


def _redact(text: str) -> str:
    return _VERSION_LINE_RE.sub("polylogue, version <VERSION>", text)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_root_help_snapshot(runner: CliRunner, snapshot: object) -> None:
    """Root ``--help`` text is pinned; drift requires snapshot update.

    The snapshot anchors the query-first dispatch docstring, the
    ``--diagnose`` flag wording, and the "see also" pointers from #1273.
    Changes to those strings should land with a deliberate snapshot update.
    """
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert _redact(result.output) == snapshot
