"""Temporal-layout contracts for the CLI session table (polylogue-dpm6o).

Time is the one column a session list is usually scanned by, and it is the one
column that is both *localized* (so it depends on the reading host) and
*width-sensitive* (so it competes with the title for a narrow terminal). The
suites next door cover color (``test_color_and_layout.py``), cell geometry
(``test_terminal_cells.py``) and the 80-column grid
(``test_plain_cli_snapshots.py``); none of them pins what the date field itself
must be.

Four properties, each with the mutation that turns it red:

* **Alignment** -- the date field occupies exactly ten cells for every row,
  present or missing, at every terminal width. Widening ``_display_date`` to
  carry a time, or dropping the ``{date:10s}`` pad in
  ``polylogue.cli.query_output._summary_list_line``, misaligns every column
  after it.
* **Missing time is a word** -- an absent timestamp renders ``unknown``, not
  blank padding. Returning ``""`` makes the distinction invisible to a reader
  who cannot see column boundaries, and goes red here.
* **Instant vs declared wall day** -- ``_display_date`` reads one *instant* in
  the reader's timezone, so two records written at the same moment under
  different declared offsets show the same day; ``_canonical_date`` reads the
  *declared wall day* and keeps them apart. Routing either surface through the
  other -- the tempting "just use one date formatter" cleanup -- goes red. The
  test states this without touching ``TZ``, so it holds on any host.
* **Time is text, not a color cue** -- the rendered row contains the date
  literally and no ANSI. Replacing the column with a color-coded recency badge
  goes red.

The layout-selection property (which widths keep the date column at all) is
asserted here rather than in ``test_color_and_layout.py`` because its subject
is the date, not the breakpoint table: a future narrow-terminal change that
sacrifices ``date`` before ``origin`` goes red here.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from polylogue.archive.models import SessionSummary
from polylogue.cli.query_output import (
    _canonical_date,
    _display_date,
    _LayoutBreakpoints,
    _search_hit_layout,
    _stream_date_parts,
    _summary_list_layout,
    format_summary_list,
)
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId

pytestmark = pytest.mark.contract

#: Late-evening UTC, so a record written at the same instant under a far-eastern
#: offset declares the *next* calendar day.
_INSTANT = datetime(2026, 8, 24, 23, 30, tzinfo=timezone.utc)
#: The same instant, declared at UTC+14.
_SAME_INSTANT_EAST = _INSTANT.astimezone(timezone(timedelta(hours=14)))
_DATE_CELLS = 10
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _summary(session_id: str, when: datetime | None) -> SessionSummary:
    return SessionSummary(
        id=SessionId(session_id),
        origin=Origin.CLAUDE_CODE_SESSION,
        title="temporal layout fixture",
        created_at=when,
        updated_at=when,
        message_count=3,
    )


def _rendered_rows(*, with_date: datetime | None) -> list[str]:
    """One row with a timestamp and one without, rendered by the canonical renderer."""
    ids = ["claude-code-session:dated", "claude-code-session:undated"]
    rows = [_summary(ids[0], with_date), _summary(ids[1], None)]
    rendered = format_summary_list(rows, "text", None, message_counts=dict.fromkeys(ids, 3))
    return [line for line in rendered.splitlines() if line]


#: ``_summary_list_line`` pads each field and joins with two spaces, so a field
#: is a run of text containing no double space.
_FIELD = re.compile(r"\S+(?: \S+)*")


def _fields(line: str) -> list[tuple[int, str]]:
    """``(start offset, text)`` for each rendered column of a row."""
    return [(match.start(), match.group()) for match in _FIELD.finditer(line)]


def _date_field(line: str) -> str:
    """The date column's text, unpadded."""
    return _fields(line)[1][1]


def _date_column_cells(line: str) -> int:
    """Cells the date column occupies, separator included."""
    offsets = _fields(line)
    return offsets[2][0] - offsets[1][0]


class TestDateColumnAlignment:
    """The date field is a fixed ten cells whether or not there is a date."""

    def test_present_and_missing_dates_occupy_the_same_ten_cells(self) -> None:
        dated, undated = _rendered_rows(with_date=_INSTANT)
        assert _date_column_cells(dated) == _DATE_CELLS + 2
        assert _date_column_cells(undated) == _DATE_CELLS + 2

    def test_the_column_after_the_date_starts_at_the_same_offset_in_every_row(self) -> None:
        """A row with no timestamp does not shift the origin column left."""
        dated, undated = _rendered_rows(with_date=_INSTANT)
        assert _fields(dated)[2][0] == _fields(undated)[2][0]
        assert _fields(dated)[2][1] == _fields(undated)[2][1] == "claude-code-session"

    def test_the_date_is_a_calendar_day_not_a_timestamp(self) -> None:
        """Ten cells is only stable while the format stays ``%Y-%m-%d``."""
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", _display_date(_INSTANT))
        assert len(_display_date(_INSTANT)) == _DATE_CELLS


class TestMissingTimeIsAWord:
    def test_an_absent_timestamp_renders_the_word_unknown(self) -> None:
        _dated, undated = _rendered_rows(with_date=_INSTANT)
        assert _date_field(undated) == "unknown"

    def test_the_formatter_itself_returns_empty_so_the_row_owns_the_word(self) -> None:
        """The fallback lives in the row renderer, which is where it is readable."""
        assert _display_date(None) == ""


class TestInstantVersusDeclaredWallDay:
    """``_display_date`` reads an instant; ``_canonical_date`` reads a wall day."""

    def test_the_two_fixtures_really_are_one_instant(self) -> None:
        """Anti-vacuity guard: the rest of this class is empty if they diverge."""
        assert _SAME_INSTANT_EAST == _INSTANT
        assert _SAME_INSTANT_EAST.utcoffset() != _INSTANT.utcoffset()

    def test_display_date_collapses_one_instant_to_one_reader_day(self) -> None:
        """Whatever the host timezone is, one moment is one row-date."""
        assert _display_date(_INSTANT) == _display_date(_SAME_INSTANT_EAST)

    def test_canonical_date_keeps_the_declared_wall_day(self) -> None:
        """The machine-facing date is the record's own day, not the reader's."""
        assert _canonical_date(_INSTANT) == "2026-08-24"
        assert _canonical_date(_SAME_INSTANT_EAST) == "2026-08-25"

    def test_the_two_formatters_are_not_interchangeable(self) -> None:
        """Swapping one for the other changes an observable value here."""
        assert _canonical_date(_SAME_INSTANT_EAST) != _canonical_date(_INSTANT)
        assert _display_date(_SAME_INSTANT_EAST) == _display_date(_INSTANT)

    def test_stream_rows_carry_the_human_text_and_the_machine_instant(self) -> None:
        """Collapsing the pair into one representation loses one of the two readers."""
        text, value = _stream_date_parts(_INSTANT)
        assert text is not None and value is not None
        assert datetime.fromisoformat(value) == _INSTANT
        assert datetime.fromisoformat(value).tzinfo is not None
        # The human half is localized, minute-grained text -- not an ISO instant.
        with pytest.raises(ValueError):
            datetime.fromisoformat(text)


class TestDateSurvivesNarrowTerminals:
    """The date outranks origin when columns must be dropped."""

    @pytest.mark.parametrize(
        "width",
        (200, _LayoutBreakpoints.WIDE_MIN, _LayoutBreakpoints.MID_MIN, _LayoutBreakpoints.NARROW_MIN),
    )
    def test_the_session_list_keeps_the_date_down_to_the_narrow_breakpoint(self, width: int) -> None:
        columns = _summary_list_layout(width)
        assert "date" in columns
        assert "origin" not in columns or columns.index("date") < columns.index("origin")

    def test_below_the_narrow_breakpoint_only_the_title_survives(self) -> None:
        assert _summary_list_layout(_LayoutBreakpoints.NARROW_MIN - 1) == ("title",)

    def test_a_search_hit_trades_the_date_for_the_match_snippet_only_when_narrow(self) -> None:
        """A hit's evidence outranks its date; the trade happens once, at 60 columns."""
        assert "date" in _search_hit_layout(_LayoutBreakpoints.MID_MIN)
        assert "date" not in _search_hit_layout(_LayoutBreakpoints.MID_MIN - 1)
        assert "match" in _search_hit_layout(_LayoutBreakpoints.MID_MIN - 1)


class TestTimeIsTextNotColor:
    def test_the_rendered_row_carries_the_date_literally_and_no_ansi(self) -> None:
        """A color-coded recency badge instead of a date column goes red here."""
        dated, _undated = _rendered_rows(with_date=_INSTANT)
        assert _display_date(_INSTANT) in dated
        assert not _ANSI.search(dated)
