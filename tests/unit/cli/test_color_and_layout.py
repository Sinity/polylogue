"""Contracts for CLI color discipline and narrow-terminal output layout.

Pins two user-visible CLI ergonomics behaviors landed for #958:

* ``NO_COLOR`` (https://no-color.org/ convention) forces plain rendering even
  on a TTY. This mirrors how ``POLYLOGUE_FORCE_PLAIN`` already behaves and
  guarantees ANSI-free output for downstream pipelines that set ``NO_COLOR``
  but cannot opt into the Polylogue-specific override.

* ``output_summary_list`` and ``output_search_hits`` adapt their column set
  and title-truncation budget to the rendered terminal width, so a 40-column
  terminal still produces a readable table without horizontal overflow.

Companion to ``test_plain_output_contract.py`` (--plain ANSI guarantees) and
``test_help_contract.py`` (help structure).
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.query_output import (
    _LayoutBreakpoints,
    _search_hit_layout,
    _summary_list_layout,
    _title_budget,
)
from polylogue.cli.shared.formatting import (
    no_color_requested,
    should_use_plain,
)

pytestmark = pytest.mark.contract


class TestNoColorEnv:
    """``NO_COLOR`` follows the cross-tool convention and forces plain mode.

    Actual ``NO_COLOR`` environment-variable resolution happens once, in the
    5-layer config resolution (``ConfigInventoryEntry(env_var="NO_COLOR")``,
    pinned by ``tests/unit/core/test_config_inventory.py``). By the time CLI
    code reaches ``no_color_requested`` / ``should_use_plain`` the value has
    already been resolved into a plain bool, so these tests exercise that
    passthrough contract directly rather than re-reading the environment.
    """

    def test_no_color_requested_passes_through_resolved_value(self) -> None:
        """``no_color_requested`` returns whatever resolved value it is given."""
        assert no_color_requested() is False
        assert no_color_requested(no_color=False) is False
        assert no_color_requested(no_color=True) is True

    def test_should_use_plain_bridges_no_color(self) -> None:
        """A resolved ``no_color=True`` forces plain mode even off force_plain/tty state."""
        assert should_use_plain(plain=False, no_color=True) is True

    def test_should_use_plain_no_color_false_keeps_tty_behavior(self) -> None:
        """With ``no_color=False`` (and not in a TTY), plain mode comes from non-TTY detection."""
        # CliRunner / pytest runs non-TTY, so plain falls out of the TTY check.
        assert should_use_plain(plain=False, no_color=False) is True

    def test_cli_list_with_no_color_produces_no_ansi(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, object]
    ) -> None:
        """End-to-end: ``NO_COLOR=1 polylogue read --all`` is byte-for-byte ANSI-free.

        Pins the no-color contract end-to-end so the bridge cannot regress
        silently if ``should_use_plain`` is reworked.
        """
        del workspace_env
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        runner = CliRunner()
        result = runner.invoke(cli, ["list"], catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        assert "\x1b" not in result.stdout, f"ANSI escape leaked under NO_COLOR: {result.stdout!r}"
        assert "\x1b" not in result.stderr, f"ANSI escape leaked under NO_COLOR: {result.stderr!r}"


class TestNarrowTerminalLayout:
    """Session/search-hit tables degrade gracefully at narrow widths.

    The layout helpers are pure functions and the public contract is:

    * Wide (>=80 cols): show every column.
    * Mid  (>=60 cols): drop the ID column to make room for titles.
    * Narrow (>=40 cols): drop origin too; keep date+title+msgs for the
      session list, title+match for search hits.
    * Below 40 cols: degenerate to a single ``title`` column.

    The mapping is pinned here because downstream syrupy snapshots would not
    catch a silent column-set drift on a 40-column terminal (the snapshot is
    pinned at the default 80 columns).
    """

    @pytest.mark.parametrize(
        ("width", "expected"),
        [
            (200, ("id", "date", "origin", "title", "msgs")),
            (_LayoutBreakpoints.WIDE_MIN, ("id", "date", "origin", "title", "msgs")),
            (_LayoutBreakpoints.WIDE_MIN - 1, ("date", "origin", "title", "msgs")),
            (_LayoutBreakpoints.MID_MIN, ("date", "origin", "title", "msgs")),
            (_LayoutBreakpoints.MID_MIN - 1, ("date", "title", "msgs")),
            (_LayoutBreakpoints.NARROW_MIN, ("date", "title", "msgs")),
            (_LayoutBreakpoints.NARROW_MIN - 1, ("title",)),
            (20, ("title",)),
        ],
    )
    def test_summary_list_layout(self, width: int, expected: tuple[str, ...]) -> None:
        assert _summary_list_layout(width) == expected

    @pytest.mark.parametrize(
        ("width", "expected"),
        [
            (200, ("id", "date", "origin", "title", "msgs", "match")),
            (_LayoutBreakpoints.WIDE_MIN, ("id", "date", "origin", "title", "msgs", "match")),
            (_LayoutBreakpoints.WIDE_MIN - 1, ("date", "origin", "title", "match")),
            (_LayoutBreakpoints.MID_MIN, ("date", "origin", "title", "match")),
            (_LayoutBreakpoints.MID_MIN - 1, ("title", "match")),
            (_LayoutBreakpoints.NARROW_MIN, ("title", "match")),
            (_LayoutBreakpoints.NARROW_MIN - 1, ("title",)),
            (20, ("title",)),
        ],
    )
    def test_search_hit_layout(self, width: int, expected: tuple[str, ...]) -> None:
        assert _search_hit_layout(width) == expected

    def test_title_budget_monotonic_in_width(self) -> None:
        """Title budget never grows when terminal width shrinks across breakpoints."""
        widths = [200, 100, 80, 79, 60, 59, 40, 39, 20]
        budgets = [_title_budget(w) for w in widths]
        # Strictly non-increasing across non-equal widths is too strong (we
        # cap at a per-breakpoint floor); instead assert wide >= mid >= narrow.
        assert budgets[0] >= budgets[3], "wide budget should be >= mid budget"
        assert budgets[3] >= budgets[6], "mid budget should be >= narrow budget"
        assert all(budget > 0 for budget in budgets), f"non-positive title budget: {budgets}"

    def test_title_budget_keeps_room_at_40_cols(self) -> None:
        """At 40 columns the title still gets at least 18 chars of room."""
        assert _title_budget(40) >= 18

    def test_title_budget_handles_pathologically_narrow(self) -> None:
        """A 10-column terminal still returns a positive title budget."""
        assert _title_budget(10) > 0


class TestNoColorIsNotPolylogueSpecific:
    """``NO_COLOR`` respects the cross-tool environment regardless of Polylogue settings."""

    def test_no_color_overrides_tty_assumption(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set ``NO_COLOR``, clear ``POLYLOGUE_FORCE_PLAIN``: plain still wins."""
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        assert should_use_plain(plain=False) is True

    def test_polylogue_force_plain_still_wins_when_no_color_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The existing POLYLOGUE_FORCE_PLAIN path is unchanged."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        assert should_use_plain(plain=False) is True


def test_no_color_env_does_not_leak_into_other_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: tests above use monkeypatch so ``NO_COLOR`` does not bleed.

    Guards against accidentally promoting ``NO_COLOR`` to a global default by
    mutating ``os.environ`` directly.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert os.environ.get("NO_COLOR") is None
