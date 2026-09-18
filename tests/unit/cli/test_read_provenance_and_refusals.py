"""Honest totals, honest provenance, and one first-run refusal.

polylogue-jfabc: the row renderer substituted ``len(rows)`` for an operation's
honest ``None`` total, and the bare landing screen named "daemon" from the
absence of ``--no-daemon`` rather than from the result.

polylogue-ry6g5: a fresh archive root produced two refusal texts and two
machine contracts depending on which verb asked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# jfabc: a null total is unknown, not len(rows)
# ---------------------------------------------------------------------------


def _envelope(payload: dict[str, object], *, limit: int = 10, offset: int = 0) -> dict[str, object]:
    from polylogue.cli.render.rows import _page_envelope

    raw = payload.get("items", [])
    assert isinstance(raw, list)
    rows: list[dict[str, object]] = [dict(cast("Mapping[str, object]", row)) for row in raw]
    return _page_envelope(payload, mode="list", rows=rows, offset=offset, limit=limit, origin=None, source="daemon")


def test_an_unknown_total_stays_unknown() -> None:
    """A vector page reports no archive-wide cardinality; the renderer says so.

    ``daemon_reads._search_payload`` sets ``total=None`` deliberately for the
    vector lane and keeps the null on the wire.

    Anti-vacuity: restore ``payload.get("total") or len(rows)`` in
    ``_page_envelope`` and this goes red with ``total == 3``.
    """

    envelope = _envelope({"items": [{"id": f"s:{n}"} for n in range(3)], "total": None, "limit": 3}, limit=3)

    assert envelope["total"] is None


def test_an_unknown_total_serializes_as_null_in_machine_formats(capsys: pytest.CaptureFixture[str]) -> None:
    """Every machine format carries the unknown through rather than a count.

    Anti-vacuity: coerce the null to ``len(rows)`` anywhere between
    ``_page_envelope`` and ``emit_rows`` and this goes red.
    """

    from polylogue.cli.render.rows import emit_rows

    rows: list[dict[str, object]] = [{"id": f"s:{n}"} for n in range(3)]
    envelope = _envelope({"items": rows, "total": None, "limit": 3}, limit=3)

    emit_rows(envelope, rows, output_format="json", text_line=lambda row: str(row["id"]), fields=None)
    assert json.loads(capsys.readouterr().out)["total"] is None


def test_a_reported_total_is_still_reported() -> None:
    """The honest-unknown handling must not erase a real total."""

    envelope = _envelope({"items": [{"id": "s:0"}], "total": 42, "limit": 1}, limit=1)

    assert envelope["total"] == 42


# ---------------------------------------------------------------------------
# jfabc: the bare screen reports the authority that answered
# ---------------------------------------------------------------------------


def test_the_bare_screen_names_the_authority_from_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read the in-process executor answered must not print "(daemon)".

    ``cli/daemon_probe.py`` documents exactly why provenance cannot be inferred
    from success: the kernel falls back to the direct reader when no socket
    answers, so the request's intent says nothing about who served it.

    Anti-vacuity: restore ``source = "direct" if daemon_disabled else "daemon"``
    and this goes red, because the stub reports ``direct`` while the invocation
    never passed ``--no-daemon``.
    """

    import click

    from polylogue.cli.click_app import _show_bare_tty_triage, cli
    from polylogue.cli.shared.types import AppEnv
    from polylogue.config import Config

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    (archive_root / "index.db").write_bytes(b"")
    config = Config(
        archive_root=archive_root,
        db_path=archive_root / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    monkeypatch.setattr("polylogue.cli.shared.helpers.load_effective_config", lambda _env: config)
    # The invocation never passes ``--no-daemon``, so the old screen printed
    # "(daemon)" from that alone; the result says ``direct``.
    with patch("polylogue.cli.session_rows.query_session_rows_with_authority") as query:
        query.return_value = ([], "direct")
        assert _show_bare_tty_triage(click.Context(cli, info_name="polylogue"), AppEnv(plain=True))

    output = capsys.readouterr().out
    assert "Archive: ready (direct)" in output
    assert "Archive: ready (daemon)" not in output


# ---------------------------------------------------------------------------
# ry6g5: one first-run refusal for find/read/select/bare
# ---------------------------------------------------------------------------

#: ``read --all`` is deliberately absent: browse mode over an archive that does
#: not exist yet has a correct empty answer, which
#: ``_emit_missing_archive_empty_read`` already gives it.  These are the verbs
#: that have no correct answer and must refuse.
_FIRST_RUN_VERBS: tuple[list[str], ...] = (
    ["find", "polylogue"],
    ["read", "polylogue"],
    ["select"],
)


@pytest.mark.parametrize("argv", _FIRST_RUN_VERBS, ids=lambda argv: argv[0])
def test_every_verb_refuses_a_fresh_root_with_one_machine_envelope(
    argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One condition, one typed refusal, one parseable machine document.

    ``find`` used to ``click.echo`` "archive index database not found at PATH"
    and ``SystemExit(1)`` straight past ``machine_errors``, so
    ``--format json find`` emitted unparseable plain text while the other verbs
    emitted the structured envelope for the very same missing tier.

    Anti-vacuity: restore ``archive_query._fail``'s echo path for the missing
    index and the ``find`` case stops parsing as JSON.
    """

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    full_argv = [*argv, "--format", "json"]
    monkeypatch.setattr("sys.argv", ["polylogue", *full_argv])
    runner = CliRunner()

    from polylogue.cli.click_app import cli

    result = runner.invoke(cli, full_argv, standalone_mode=False)
    exc = result.exception
    assert exc is not None, f"{argv} answered a fresh root instead of refusing: {result.output}"

    from polylogue.core.errors import ArchiveTierUnavailableError

    assert isinstance(exc, ArchiveTierUnavailableError), f"{argv} raised {exc!r}"
    assert exc.tier == "index"
    assert exc.reason == "database file not found"
    assert "polylogue ingest" in exc.guidance


def test_the_first_run_refusal_is_the_storage_layers_own_wording(tmp_path: Path) -> None:
    """The CLI no longer mints a second text for the missing index tier.

    Anti-vacuity: reintroduce "archive index database not found at" anywhere in
    ``polylogue/cli`` and this goes red.
    """

    cli_root = Path(__file__).resolve().parents[3] / "polylogue" / "cli"
    offenders = [
        path.relative_to(cli_root).as_posix()
        for path in cli_root.rglob("*.py")
        if "archive index database not found" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
