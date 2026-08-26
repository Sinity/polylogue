"""Normalized terminal-cell oracle used by CLI interaction tests.

The oracle intentionally operates on semantic cells rather than ANSI bytes,
screenshots, or a particular terminal emulator's rasterization.  It can also
consume raw output from the PTY harness so archived content containing control
sequences is treated as hostile input.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from wcwidth import wcswidth

try:
    import regex as _regex
except ImportError:  # pragma: no cover - the dev dependency is present in CI
    _regex = None

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[()][0-2])")
_OSC8_RE = re.compile(r"\x1b\]8;;(?P<uri>[^\x07\x1b]*)(?:\x07|\x1b\\)(?P<body>.*?)\x1b\]8;;(?:\x07|\x1b\\)", re.DOTALL)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


@dataclass(frozen=True, slots=True)
class TerminalCell:
    """One occupied display cell or grapheme cluster."""

    row: int
    column: int
    grapheme: str
    display_width: int
    semantic_role: str = "body"
    hyperlink: str | None = None
    focused: bool = False
    non_color_label: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalFrame:
    """A normalized, width-bounded terminal frame."""

    columns: int
    rows: int
    cells: tuple[TerminalCell, ...]
    redirected: bool = False
    color_mode: str = "full"
    theme: str = "dark"

    def row_cells(self, row: int) -> tuple[TerminalCell, ...]:
        return tuple(cell for cell in self.cells if cell.row == row)


def graphemes(value: str) -> tuple[str, ...]:
    """Split text into display graphemes, including emoji ZWJ sequences."""
    if _regex is not None:
        return tuple(_regex.findall(r"\X", value))
    # Conservative fallback: combining marks attach to the preceding base.
    result: list[str] = []
    for char in value:
        if result and unicodedata.combining(char):
            result[-1] += char
        else:
            result.append(char)
    return tuple(result)


def display_width(value: str) -> int:
    """Return terminal display width, never a negative value."""
    width = wcswidth(value)
    return max(0, width)


def control_safety_violations(value: str) -> tuple[str, ...]:
    """Report raw controls that must never reach a normalized display cell."""
    return tuple(f"U+{ord(char):04X}" for char in _CONTROL_RE.findall(value))


def _safe_text(value: str) -> str:
    # OSC/CSI are terminal controls, not content.  Removing them before
    # wrapping makes the oracle safe even when a fixture contains malicious
    # archived text rather than renderer-generated escapes.
    value = _ANSI_RE.sub("", value)
    return _CONTROL_RE.sub("�", value)


def _linked_segments(value: str) -> tuple[tuple[str, str | None], ...]:
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for match in _OSC8_RE.finditer(value):
        if match.start() > cursor:
            segments.append((value[cursor : match.start()], None))
        segments.append((match.group("body"), match.group("uri") or None))
        cursor = match.end()
    if cursor < len(value):
        segments.append((value[cursor:], None))
    return tuple(segments) if segments else ((value, None),)


def normalize_terminal_text(
    value: str,
    *,
    columns: int,
    rows: int = 1000,
    semantic_role: str = "body",
    focused: bool = False,
    non_color_label: str | None = None,
    redirected: bool = False,
    color_mode: str = "full",
    theme: str = "dark",
) -> TerminalFrame:
    """Normalize text into bounded cells with deterministic wrapping/clipping."""
    if columns < 1 or rows < 1:
        raise ValueError("terminal dimensions must be positive")
    cells: list[TerminalCell] = []
    row = 0
    column = 0
    for segment, hyperlink in _linked_segments(value):
        for grapheme in graphemes(_safe_text(segment)):
            if grapheme == "\n":
                row += 1
                column = 0
                if row >= rows:
                    return TerminalFrame(columns, rows, tuple(cells), redirected, color_mode, theme)
                continue
            if grapheme == "\r":
                column = 0
                continue
            if grapheme == "\t":
                next_column = min(columns, ((column // 4) + 1) * 4)
                column = next_column
                continue
            width = display_width(grapheme)
            if width == 0:
                if cells and cells[-1].row == row:
                    previous = cells[-1]
                    cells[-1] = TerminalCell(
                        previous.row,
                        previous.column,
                        previous.grapheme + grapheme,
                        previous.display_width,
                        previous.semantic_role,
                        previous.hyperlink,
                        previous.focused,
                        previous.non_color_label,
                    )
                continue
            if width > columns:
                # A double-width grapheme cannot fit even on an otherwise
                # empty one-column terminal. Replace it with a visible safe
                # marker rather than creating an out-of-bounds cell.
                grapheme = "�"
                width = 1
            if column + width > columns:
                row += 1
                column = 0
                if row >= rows:
                    break
            cells.append(
                TerminalCell(
                    row,
                    column,
                    grapheme,
                    width,
                    semantic_role,
                    hyperlink,
                    focused,
                    non_color_label,
                )
            )
            column += width
        if row >= rows:
            break
    return TerminalFrame(columns, rows, tuple(cells), redirected, color_mode, theme)


def frame_law_violations(frame: TerminalFrame) -> tuple[str, ...]:
    """Return stable semantic/layout violations for a normalized frame."""
    violations: list[str] = []
    occupied: set[tuple[int, int]] = set()
    focus_count = 0
    for cell in frame.cells:
        if cell.display_width < 1:
            violations.append(f"zero-width cell at {cell.row}:{cell.column}")
        if cell.column < 0 or cell.column + cell.display_width > frame.columns:
            violations.append(f"clipped cell at {cell.row}:{cell.column}")
        for column in range(cell.column, cell.column + cell.display_width):
            key = (cell.row, column)
            if key in occupied:
                violations.append(f"overlap at {cell.row}:{column}")
            occupied.add(key)
        if control_safety_violations(cell.grapheme):
            violations.append(f"control in cell at {cell.row}:{cell.column}")
        if not cell.semantic_role:
            violations.append(f"missing semantic role at {cell.row}:{cell.column}")
        if cell.focused:
            focus_count += 1
            if not cell.non_color_label:
                violations.append(f"focused cell has no non-color label at {cell.row}:{cell.column}")
        if cell.hyperlink is not None and not cell.hyperlink:
            violations.append(f"empty hyperlink at {cell.row}:{cell.column}")
        if cell.hyperlink is not None and control_safety_violations(cell.hyperlink):
            violations.append(f"control in hyperlink at {cell.row}:{cell.column}")
    if focus_count > 0 and not any(cell.focused and cell.non_color_label for cell in frame.cells):
        violations.append("focus is color-only")
    return tuple(violations)


def assert_frame_laws(frame: TerminalFrame) -> None:
    """Raise a focused assertion for any terminal-cell law violation."""
    violations = frame_law_violations(frame)
    if violations:
        raise AssertionError("terminal-cell oracle violations: " + "; ".join(violations))


__all__ = [
    "TerminalCell",
    "TerminalFrame",
    "assert_frame_laws",
    "control_safety_violations",
    "display_width",
    "frame_law_violations",
    "graphemes",
    "normalize_terminal_text",
]
