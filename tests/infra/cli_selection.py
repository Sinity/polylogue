"""Explicit synthetic operation verdicts for CLI selection controls."""

from collections.abc import Mapping, Sequence

from polylogue.cli.select import SelectSessionRow
from polylogue.cli.session_rows import SessionSelection
from polylogue.surfaces.outcome import decide_outcome


def selection_for_rows(rows: Sequence[SelectSessionRow], *, authority: str = "fixture") -> SessionSelection:
    return SessionSelection(tuple(rows), decide_outcome(matched=len(rows)), authority, "fixture-selected-frame")


def selection_for_ids(ids: Sequence[str], *, authority: str = "fixture") -> SessionSelection:
    return selection_for_rows(
        [SelectSessionRow(session_id=ref, origin="codex-session", title=ref, date=None) for ref in ids],
        authority=authority,
    )


def fixture_query_page(page: Mapping[str, object]) -> dict[str, object]:
    rows = page.get("items", page.get("hits", []))
    assert isinstance(rows, list)
    return {**page, "outcome": decide_outcome(matched=len(rows)).to_dict()}
