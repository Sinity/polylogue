"""Semantic terminal-cell laws independent of ANSI bytes and screenshots."""

from __future__ import annotations

import pytest

from tests.infra.pty_scenarios import PtyEvent, scenario_contract
from tests.infra.terminal_cells import (
    TerminalCell,
    TerminalFrame,
    assert_frame_laws,
    display_width,
    frame_law_violations,
    normalize_terminal_text,
)


@pytest.mark.parametrize("columns", (40, 80, 120, 200))
@pytest.mark.parametrize("theme", ("light", "dark"))
@pytest.mark.parametrize("color_mode", ("full", "no-color", "low-color"))
def test_normalized_frame_is_width_bounded_across_capabilities(
    columns: int,
    theme: str,
    color_mode: str,
) -> None:
    frame = normalize_terminal_text(
        "title: 東京🙂e\u0301\n\x1b[31munsafe\x1b[0m \x1b]8;;https://example.test\x07link\x1b]8;;\x07",
        columns=columns,
        rows=8,
        semantic_role="title",
        focused=True,
        non_color_label="selected",
        redirected=color_mode == "no-color",
        color_mode=color_mode,
        theme=theme,
    )
    assert frame.columns == columns
    assert_frame_laws(frame)
    assert all(cell.column + cell.display_width <= columns for cell in frame.cells)
    assert any(cell.hyperlink for cell in frame.cells)


def test_width_handles_combining_marks_and_wide_graphemes() -> None:
    assert display_width("e\u0301") == 1
    assert display_width("🙂") == 2
    assert display_width("東京") == 4


def test_bidi_isolates_are_neutralized_as_control_safety() -> None:
    frame = normalize_terminal_text("before\u2066RTL\u2069after", columns=40)
    assert_frame_laws(frame)
    assert all("\u2066" not in cell.grapheme and "\u2069" not in cell.grapheme for cell in frame.cells)


def test_focus_without_non_color_meaning_is_a_red_mutant() -> None:
    frame = TerminalFrame(
        columns=40,
        rows=2,
        cells=(TerminalCell(0, 0, "x", 1, focused=True),),
    )
    assert "focused cell has no non-color label at 0:0" in frame_law_violations(frame)


def test_clipping_is_a_red_mutant() -> None:
    frame = TerminalFrame(
        columns=4,
        rows=1,
        cells=(TerminalCell(0, 3, "界", 2),),
    )
    assert any("clipped" in violation for violation in frame_law_violations(frame))


def test_pty_schedule_covers_resize_interrupt_and_daemon_loss_controls() -> None:
    events = (
        PtyEvent(0.01, "resize", "40x120"),
        PtyEvent(0.02, "write", "status\n"),
        PtyEvent(0.03, "interrupt"),
        PtyEvent(0.04, "terminate"),
    )
    assert scenario_contract(events) == ()
    assert scenario_contract((PtyEvent(0, "resize", "0x80"),))
