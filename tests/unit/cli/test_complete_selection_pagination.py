"""The complete-selection walk over a real seeded archive (polylogue-w3s0q).

``tests/unit/cli/test_select.py::test_complete_selection_walks_every_page``
proves the *loop* terminates correctly, but it feeds the loop hand-written
pages that already carry ``next_offset``.  The production ``cli.query`` list
payload did not carry that key at all, so the loop returned after one page and
``select``/``mark``/``delete --all`` acted on the first ``COMPLETE_SELECTION_PAGE``
matches while the operator asked for every match.  These tests execute the real
declared operation against a real archive, which is the only shape that can see
the missing key.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polylogue.cli.root_request import RootModeRequest
from polylogue.config import Config

SEEDED_SESSIONS = 7
PAGE = 3


@pytest.fixture
def seeded_root(tmp_path: Path) -> Path:
    """An archive holding more sessions than one walk page."""

    from tests.infra.storage_records import SessionBuilder

    index_db = tmp_path / "index.db"
    now = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    for n in range(SEEDED_SESSIONS):
        stamp = (now - timedelta(minutes=n)).isoformat()
        (
            SessionBuilder(index_db, f"walk-{n}")
            .provider("codex")
            .title(f"Walk {n}")
            .created_at(stamp)
            .updated_at(stamp)
            .add_message(f"w{n}-1", role="user", text=f"page walk question {n}")
            .add_message(f"w{n}-2", role="assistant", text=f"page walk answer {n}")
            .save()
        )
    return tmp_path


def _config(root: Path) -> Config:
    return Config(archive_root=root, db_path=root / "index.db", render_root=root / "render", sources=[])


def test_the_list_payload_carries_the_next_page_offset(seeded_root: Path) -> None:
    """The operation, not the client, decides whether another page exists.

    Anti-vacuity: drop ``next_offset`` from ``daemon_reads._list_payload`` (or
    return ``None`` unconditionally) and this goes red.
    """

    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.operation_kernel import dispatch

    result = dispatch(
        _config(seeded_root),
        lower_cli_query(RootModeRequest.from_params({}), limit=PAGE, offset=0),
        daemon_disabled=True,
    )
    payload = result.value
    assert isinstance(payload, dict)
    assert payload["total"] == SEEDED_SESSIONS
    assert payload["next_offset"] == PAGE

    last = dispatch(
        _config(seeded_root),
        lower_cli_query(RootModeRequest.from_params({}), limit=PAGE, offset=6),
        daemon_disabled=True,
    ).value
    assert isinstance(last, dict)
    assert last["next_offset"] is None


def test_complete_selection_resolves_every_seeded_session(seeded_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--all`` means every match, across page boundaries.

    Reproduces the bead's measurement: with the walk page set below the seeded
    count, the pre-fix walk returned ``PAGE`` ids for an archive holding
    ``SEEDED_SESSIONS``.

    Anti-vacuity: remove ``next_offset`` from the list payload and this returns
    ``PAGE`` ids instead of ``SEEDED_SESSIONS``.
    """

    import polylogue.cli.session_rows as session_rows

    monkeypatch.setattr(session_rows, "COMPLETE_SELECTION_PAGE", PAGE)
    ids = session_rows.query_complete_session_ids(
        _config(seeded_root), RootModeRequest.from_params({}), daemon_disabled=True
    )

    assert len(ids) == SEEDED_SESSIONS
    assert len(set(ids)) == SEEDED_SESSIONS


def test_the_mutating_verb_route_resolves_through_the_complete_walk() -> None:
    """``select``/``mark``/``delete --all`` reach the walk above, not a page.

    ``resolve_session_ids_for_verb`` is the single route the mutating verbs and
    their cardinality guard share, so the completeness proved above is the
    completeness they get.  Pinning the delegation is what makes that transfer
    an assertion rather than a claim.

    Anti-vacuity: point ``resolve_session_ids_for_verb`` at the bounded
    ``query_session_ids`` probe instead and this goes red.
    """

    import inspect

    from polylogue.cli import verb_cardinality

    source = inspect.getsource(verb_cardinality.resolve_session_ids_for_verb)
    assert "query_complete_session_ids" in source
    assert "query_session_ids(" not in source
