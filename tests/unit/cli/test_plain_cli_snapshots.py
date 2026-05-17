"""Snapshot tests for the plain (non-interactive) CLI output surfaces.

These pin the column layout, heading shape, and ordering of the
``polylogue --plain`` text surfaces exposed to scripts and users that pipe
the CLI into other tools. The matrix covers:

- ``list``: per-conversation rows (provider, id, title, date columns)
- ``count``: single integer
- ``stats``: aggregate summary

Anti-pattern (rejected case): we intentionally do NOT snapshot ephemeral
fields. Wall-clock timestamps, generated database paths, runtime durations,
and process IDs in the output are stripped/redacted before the snapshot
comparison. A snapshot diff caused by a change in those fields is the test
asserting against the wrong thing — the fix is to extend ``_redact`` below,
not to update the baseline.

The conversations are seeded via the ``corpus_seeded_db`` fixture (real
synthetic pipeline output), so any user-visible drift in rendering of real
conversation rows triggers a snapshot diff.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

syrupy = pytest.importorskip("syrupy")

from polylogue.cli.click_app import cli

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?")
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?\s?(?:ms|µs|us|s)\b")
_HEX_HASH_RE = re.compile(r"\b[a-f0-9]{12,}\b")
# Match path-like sequences (anything with a "/" between segments).
_PATH_RE = re.compile(r"(/[A-Za-z0-9_.\-]+){2,}")


def _redact(text: str) -> str:
    """Strip ephemeral content that is not part of the rendering contract."""
    out = _PATH_RE.sub("<PATH>", text)
    out = _TIMESTAMP_RE.sub("<TIMESTAMP>", out)
    out = _DURATION_RE.sub("<DURATION>", out)
    out = _HEX_HASH_RE.sub("<HASH>", out)
    return out


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def seeded_db_env(
    corpus_seeded_db: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Seed a deterministic corpus DB and wire env vars so the CLI finds it.

    Returns the seeded DB path so individual tests can pass it to commands
    that accept ``--db`` if they need to. The default code path (no flag)
    resolves the DB from ``XDG_DATA_HOME``.
    """
    db_path = corpus_seeded_db(providers=("chatgpt",), count=2, seed=42)

    # Wire XDG to a layout that resolves polylogue.db at our seeded path.
    data_home = tmp_path / "xdg-cli-data"
    target = data_home / "polylogue" / "polylogue.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(db_path)

    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
    return db_path


def _invoke(runner: CliRunner, args: list[str]) -> str:
    result = runner.invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, f"args={args!r} output={result.output!r}"
    return _redact(result.output)


def test_plain_list_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain list`` pins per-row column layout."""
    output = _invoke(runner, ["--plain", "list"])
    assert output == snapshot


def test_plain_count_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain count`` pins the single-integer surface."""
    output = _invoke(runner, ["--plain", "count"])
    assert output == snapshot


def test_plain_stats_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain stats`` pins the stats summary shape."""
    output = _invoke(runner, ["--plain", "stats"])
    assert output == snapshot


def test_plain_list_provider_filter_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain -p chatgpt list`` pins filtered list shape."""
    output = _invoke(runner, ["--plain", "-p", "chatgpt", "list"])
    assert output == snapshot
