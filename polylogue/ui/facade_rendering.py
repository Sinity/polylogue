"""Rendering and status helpers for the console facade."""

from __future__ import annotations

import difflib
from typing import Literal

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name
from rich.align import Align
from rich.box import Box
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from polylogue.ui.facade_console import ConsoleLike


def _print_plain_section(console: ConsoleLike, heading: str, content: str | None = None) -> None:
    console.print(heading)
    if content:
        console.print(content)


def _render_panel(
    console: ConsoleLike,
    body: RenderableType,
    *,
    border_style: str,
    box: Box,
    padding: tuple[int, int],
    title: str | None = None,
    title_align: Literal["left", "center", "right"] = "left",
) -> None:
    console.print(
        Panel(
            body,
            border_style=border_style,
            box=box,
            padding=padding,
            title=title,
            title_align=title_align,
        )
    )


def render_banner(console: ConsoleLike, *, plain: bool, banner_box: Box, title: str, subtitle: str | None) -> None:
    if plain:
        _print_plain_section(console, f"== {title} ==", subtitle)
        return

    title_text = Text(style="banner.title")
    title_text.append("◈ ", style="banner.icon")
    title_text.append(title)
    if subtitle:
        title_text.append(f"\n{subtitle}", style="banner.subtitle")
    _render_panel(
        console,
        Align.left(title_text),
        border_style="banner.border",
        box=banner_box,
        padding=(1, 3),
    )


def render_summary(console: ConsoleLike, *, plain: bool, panel_box: Box, title: str, lines: list[str]) -> None:
    text = "\n".join(lines)
    if plain:
        _print_plain_section(console, f"-- {title} --", text)
        return

    summary_text = Text()
    for line in lines:
        summary_text.append("• ", style="summary.bullet")
        parsed = Text.from_markup(line, style="summary.text")
        summary_text.append_text(parsed)
        summary_text.append("\n")
    _render_panel(
        console,
        summary_text if summary_text.plain else Text(text, style="summary.text"),
        title=f"  {title}  ",
        title_align="left",
        border_style="panel.border",
        box=panel_box,
        padding=(1, 2),
    )


def render_markdown(console: ConsoleLike, *, plain: bool, panel_box: Box, content: str) -> None:
    if plain:
        console.print(content)
        return

    md = Markdown(content, style="summary.text")
    _render_panel(
        console,
        md,
        border_style="markdown.border",
        box=panel_box,
        padding=(1, 2),
    )


def render_code(console: ConsoleLike, *, plain: bool, panel_box: Box, code: str, language: str) -> None:
    if plain:
        console.print(code)
        return

    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    _render_panel(
        console,
        syntax,
        border_style="code.border",
        box=panel_box,
        padding=(0, 1),
    )


def render_diff(
    console: ConsoleLike,
    *,
    plain: bool,
    filename: str,
    old_text: str,
    new_text: str,
) -> None:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    diff_text = "".join(diff)

    if plain:
        console.print(diff_text)
        return

    try:
        if isinstance(console, Console):
            with console.pager():
                lexer = get_lexer_by_name("diff")
                highlighted = highlight(diff_text, lexer, TerminalFormatter())
                console.print(highlighted, markup=False, highlight=False)
        else:
            console.print(diff_text)
    except (ImportError, AttributeError):
        if isinstance(console, Console):
            syntax = Syntax(diff_text, "diff", theme="ansi_dark")
            console.print(syntax)
        else:
            console.print(diff_text)


def render_status(
    console: ConsoleLike,
    *,
    plain: bool,
    icon: str,
    icon_style: str,
    message: str,
) -> None:
    if plain:
        prefix = icon if icon in {"✓", "✗"} else icon.upper()
        console.print(f"{prefix} {message}")
        return
    text = Text()
    text.append(f"{icon} ", style=icon_style)
    text.append(message, style="status.message")
    console.print(text)
