"""Snapshot tests for the plain (non-interactive) CLI output surfaces.

These pin the column layout, heading shape, and ordering of the
``polylogue --plain`` text surfaces exposed to scripts and users that pipe
the CLI into other tools. The matrix covers:

- ``read --all``: per-session rows (origin, id, title, date columns)
- ``analyze --count``: single integer
- ``analyze``: aggregate summary

Anti-pattern (rejected case): we intentionally do NOT snapshot ephemeral
fields. Wall-clock timestamps, generated database paths, runtime durations,
and process IDs in the output are stripped/redacted before the snapshot
comparison. A snapshot diff caused by a change in those fields is the test
asserting against the wrong thing — the fix is to extend ``_redact`` below,
not to update the baseline.

The sessions are seeded via the ``corpus_seeded_db`` fixture (real
synthetic pipeline output), so any user-visible drift in rendering of real
session rows triggers a snapshot diff.
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
_SIZE_BYTES_FIELD_RE = re.compile(r'("(?:db_)?size_bytes": )\d+')
# Match path-like sequences (anything with a "/" between segments).
_PATH_RE = re.compile(r"(/[A-Za-z0-9_.\-]+){2,}")


def _redact(text: str) -> str:
    """Strip ephemeral content that is not part of the rendering contract."""
    out = _PATH_RE.sub("<PATH>", text)
    out = _TIMESTAMP_RE.sub("<TIMESTAMP>", out)
    out = _DURATION_RE.sub("<DURATION>", out)
    out = _HEX_HASH_RE.sub("<HASH>", out)
    out = _SIZE_BYTES_FIELD_RE.sub(r"\1<SIZE_BYTES>", out)
    return out


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def seeded_db_env(
    corpus_seeded_db: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Seed a deterministic corpus DB through the archive pipeline.

    ``corpus_seeded_db`` ingests synthetic sessions into the archive
    archive and points ``POLYLOGUE_ARCHIVE_ROOT`` at the root that
    holds the seeded ``index.db`` — exactly the store the CLI query verbs
    read — so no XDG/symlink staging is required. Returns the index db path.
    """
    db_path = corpus_seeded_db(providers=("chatgpt",), count=2, seed=42)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return db_path


@pytest.fixture
def postmortem_seeded_env(
    corpus_seeded_db: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Seed a corpus DB and materialize the session-profile insights.

    The postmortem bundle reads durable ``session_profile`` insight records,
    which the bare corpus ingest does not build. This fixture rebuilds them so
    the snapshot exercises a populated bundle rather than an empty scope.
    """
    import asyncio

    from polylogue.api import Polylogue

    db_path = corpus_seeded_db(providers=("chatgpt", "claude-code"), count=2, seed=42)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    async def _rebuild() -> None:
        plg = Polylogue.open()
        try:
            await plg.rebuild_insights()
        finally:
            await plg.close()

    asyncio.run(_rebuild())
    return db_path


def _invoke(runner: CliRunner, args: list[str]) -> str:
    result = runner.invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, f"args={args!r} output={result.output!r}"
    return _redact(result.output)


def test_plain_read_all_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain read --all`` pins per-row column layout."""
    output = _invoke(runner, ["--plain", "read", "--all"])
    assert output == snapshot


def test_plain_analyze_count_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --count`` pins the single-integer surface."""
    output = _invoke(runner, ["--plain", "analyze", "--count"])
    assert output == snapshot


def test_plain_analyze_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze`` pins the stats summary shape."""
    output = _invoke(runner, ["--plain", "analyze"])
    assert output == snapshot


def test_plain_read_all_origin_filter_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain --origin chatgpt-export read --all`` pins filtered list shape."""
    output = _invoke(runner, ["--plain", "--origin", "chatgpt-export", "read", "--all"])
    assert output == snapshot


# ───────────────────────────────────────────────────────────────────────
# JSON output snapshots (#1689)
# ───────────────────────────────────────────────────────────────────────
#
# These pin the stable JSON shapes emitted by the primary query and
# introspection commands.  We use ``--format json`` on the verb itself
# rather than the root-level ``--json`` shorthand because the shorthand
# is a root-group option that forces plain mode but does not always
# propagate ``output_format`` to query-verb subcommands through the
# Click parameter chain (known gap, tracked in #1689).


def _invoke_json(runner: CliRunner, args: list[str]) -> str:
    """Invoke with --plain and redact ephemeral content from JSON output."""
    result = runner.invoke(cli, ["--plain", *args], catch_exceptions=False)
    assert result.exit_code == 0, f"args={args!r} exit={result.exit_code} output={result.output!r}"
    return _redact(result.output)


def test_json_read_all_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain read --all --format json`` pins JSON list envelope."""
    output = _invoke_json(runner, ["read", "--all", "--format", "json"])
    assert output == snapshot


def test_json_analyze_count_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --count`` pins the integer count surface.

    ``analyze --count`` does not require ``--format``; it emits a bare integer,
    which is the simplest machine-readable contract.
    """
    output = _invoke(runner, ["--plain", "analyze", "--count"])
    assert output == snapshot


def test_json_analyze_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --format json`` pins JSON stats envelope."""
    output = _invoke_json(runner, ["analyze", "--format", "json"])
    assert output == snapshot


def test_json_facets_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --facets --format json`` pins JSON facets envelope (#1842)."""
    output = _invoke_json(runner, ["analyze", "--facets", "--format", "json"])
    assert output == snapshot


def test_plain_analyze_postmortem_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --postmortem`` pins the distilled bundle text (#2380)."""
    output = _invoke(runner, ["--plain", "analyze", "--postmortem"])
    assert output == snapshot


def test_json_analyze_postmortem_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --postmortem --format json`` pins the JSON envelope (#2380)."""
    output = _invoke_json(runner, ["analyze", "--postmortem", "--format", "json"])
    assert output == snapshot


def test_markdown_analyze_postmortem_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --postmortem --format markdown`` pins the Markdown render (#2380)."""
    output = _invoke(runner, ["--plain", "analyze", "--postmortem", "--format", "markdown"])
    assert output == snapshot


def test_analyze_postmortem_json_has_stable_headline_keys(
    runner: CliRunner,
    postmortem_seeded_env: Path,
) -> None:
    """The postmortem JSON envelope carries every named headline key (#2380 AC2)."""
    import json as _json

    result = runner.invoke(
        cli,
        ["--plain", "analyze", "--postmortem", "--format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    bundle = payload["result"]["postmortem"]
    expected_keys = {
        "schema_version",
        "scope",
        "session_count",
        "wallclock_span",
        "estimated_cost",
        "repos_touched",
        "top_expensive_session",
        "subagent_branch_count",
        "longest_tool_gap",
        "wasted_loop",
        "failure_mode",
    }
    assert expected_keys <= set(bundle)
    assert set(bundle["estimated_cost"]["tokens"]) == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }
    # longest_tool_gap still degrades; pathology fields carry an explicit status
    assert bundle["longest_tool_gap"]["status"] == "unavailable"
    assert bundle["wasted_loop"]["status"] in {"detected", "clean", "unavailable"}
    assert bundle["failure_mode"]["status"] in {"detected", "clean", "unavailable"}


def test_plain_analyze_portfolio_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --portfolio`` pins the corpus report text (#2437)."""
    output = _invoke(runner, ["--plain", "analyze", "--portfolio"])
    assert output == snapshot


def test_json_analyze_portfolio_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --portfolio --format json`` pins the JSON envelope (#2437)."""
    output = _invoke_json(runner, ["analyze", "--portfolio", "--format", "json"])
    assert output == snapshot


def test_markdown_analyze_portfolio_snapshot(
    runner: CliRunner,
    postmortem_seeded_env: Path,
    snapshot: object,
) -> None:
    """``polylogue --plain analyze --portfolio --format markdown`` pins the Markdown render (#2437)."""
    output = _invoke(runner, ["--plain", "analyze", "--portfolio", "--format", "markdown"])
    assert output == snapshot


def test_analyze_portfolio_json_has_stable_headline_keys(
    runner: CliRunner,
    postmortem_seeded_env: Path,
) -> None:
    """The portfolio JSON envelope carries every named headline key (#2437)."""
    import json as _json

    result = runner.invoke(
        cli,
        ["--plain", "analyze", "--portfolio", "--format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    bundle = payload["result"]["portfolio"]
    expected_keys = {
        "schema_version",
        "scope",
        "session_count",
        "origins",
        "repos_touched",
        "estimated_cost",
        "cost_distribution",
        "wallclock_span",
        "wallclock_distribution_s",
        "pathologies",
        "context_loss",
    }
    assert expected_keys <= set(bundle)
    assert set(bundle["estimated_cost"]["tokens"]) == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }
    assert bundle["pathologies"]["status"] in {"detected", "clean", "unavailable"}
    assert bundle["context_loss"]["status"] in {"detected", "clean", "unavailable"}


def test_analyze_facets_no_idf_omits_idf(runner: CliRunner, seeded_db_env: Path) -> None:
    """`analyze --facets --no-idf` ports the old `facets --no-idf` capability (#1842)."""
    import json as _json

    payload = _json.loads(_invoke_json(runner, ["analyze", "--facets", "--no-idf", "--format", "json"]))
    assert not payload.get("idf")  # no IDF weights when --no-idf is set


def test_facets_command_json_matches_analyze_facets(
    runner: CliRunner,
    seeded_db_env: Path,
) -> None:
    """`polylogue facets` is the direct command for the shared facets envelope."""
    import json as _json

    direct = _json.loads(_invoke_json(runner, ["facets", "--no-idf", "--format", "json"]))
    via_analyze = _json.loads(_invoke_json(runner, ["analyze", "--facets", "--no-idf", "--format", "json"]))

    assert direct == via_analyze
    assert direct["family_status"]["repos"]["state"] == "deferred"
    assert "not authoredness" in direct["family_status"]["role_counts"]["canonicalization"]


def test_facets_command_can_scope_query_and_origin(
    runner: CliRunner,
    seeded_db_env: Path,
) -> None:
    """Top-level facets supports the documented scoped forms."""
    import json as _json

    payload = _json.loads(
        _invoke_json(
            runner,
            ["facets", "--query", "synthetic", "--origin", "chatgpt-export", "--no-idf", "--format", "json"],
        )
    )

    assert payload["scoped_to_query"] is True
    assert payload["scoped"]["total_sessions"] == 0
    assert payload["scoped"]["origins"] == {}

    origin_only = _json.loads(
        _invoke_json(
            runner,
            ["facets", "--origin", "chatgpt-export", "--no-idf", "--format", "json"],
        )
    )
    assert origin_only["scoped_to_query"] is True
    assert origin_only["scoped"]["total_sessions"] > 0
    assert origin_only["scoped"]["origins"] == {"chatgpt-export": origin_only["scoped"]["total_sessions"]}


def test_analyze_facets_include_deferred_materializes_expensive_families(
    runner: CliRunner,
    seeded_db_env: Path,
) -> None:
    """`--include-deferred` opts into the full facet families."""
    import json as _json

    result = runner.invoke(
        cli,
        ["--plain", "analyze", "--facets", "--include-deferred", "--format", "json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)

    assert payload["deferred_families"] == {}
    assert {
        "repos",
        "role_counts",
        "material_origins",
        "message_types",
        "action_types",
        "has_flags",
    }.issubset(set(payload["complete_families"]))
    assert payload["family_status"]["message_types"]["state"] == "complete"
    assert (
        payload["family_status"]["role_counts"]["canonicalization"]
        == "provider-reported message role; not authoredness"
    )
    assert payload["family_status"]["repos"]["canonicalization"].startswith("prefer repo_name")
    assert payload["role_counts"]
    assert payload["material_origins"]
    assert payload["message_types"] == {"message": 10}


def test_analyze_facets_default_marks_detail_families_deferred_not_authoritative(
    runner: CliRunner,
    seeded_db_env: Path,
) -> None:
    """Default facets expose deferred metadata instead of pretending expensive empty buckets are authoritative."""
    import json as _json

    payload = _json.loads(_invoke_json(runner, ["analyze", "--facets", "--format", "json"]))

    for family in ("repos", "role_counts", "material_origins", "message_types", "action_types", "has_flags"):
        assert payload["deferred_families"][family] == "deferred_by_default"
        assert payload["family_status"][family]["state"] == "deferred"
        assert payload["family_status"][family]["expensive"] is True
    assert payload["repos"] == {}
    assert payload["role_counts"] == {}
    assert payload["material_origins"] == {}
    assert "not authoredness" in payload["family_status"]["role_counts"]["canonicalization"]
    assert "omit archive/path tokens" in payload["family_status"]["repos"]["canonicalization"]


def test_json_status_snapshot(
    runner: CliRunner,
    seeded_db_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: object,
) -> None:
    """``polylogue --plain status --format json`` pins JSON status shape when daemon is unreachable."""
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:1")
    output = _invoke_json(runner, ["ops", "status", "--format", "json"])
    assert output == snapshot
