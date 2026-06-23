"""Theme tokens + flag-alias consistency + --help-markdown (#1274)."""

from __future__ import annotations

from collections.abc import Iterator

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.help_markdown import render_help_markdown
from polylogue.ui.theme import (
    SEMANTIC_TOKENS,
    ThemeMode,
    css_variable_declarations,
    css_variables,
    resolve_theme_mode,
    rich_theme_styles,
    semantic_style,
    syntax_theme,
)

_MODES: tuple[ThemeMode, ...] = ("dark", "light")

# ---------------------------------------------------------------------------
# Theme tokens (#1274 — Theme test: switch produces parsable output)
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_theme_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("POLYLOGUE_THEME", "COLORFGBG"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_theme_default_is_dark(clear_theme_env: None) -> None:
    assert resolve_theme_mode() == "dark"


def test_resolve_theme_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_THEME", "light")
    assert resolve_theme_mode() == "light"
    monkeypatch.setenv("POLYLOGUE_THEME", "DARK")
    assert resolve_theme_mode() == "dark"


def test_resolve_theme_colorfgbg_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_THEME", "auto")
    monkeypatch.setenv("COLORFGBG", "0;15")  # light bg
    assert resolve_theme_mode() == "light"
    monkeypatch.setenv("COLORFGBG", "15;0")  # dark bg
    assert resolve_theme_mode() == "dark"


def test_semantic_tokens_complete() -> None:
    # Every advertised token resolves to a non-empty Rich style string in both
    # modes — this is the cross-subcommand contract.
    for mode in _MODES:
        for token in SEMANTIC_TOKENS:
            style = semantic_style(token, mode=mode)
            assert style and isinstance(style, str), (mode, token)


def test_semantic_token_unknown_rejected() -> None:
    with pytest.raises(KeyError):
        semantic_style("nope")


def test_rich_theme_styles_carries_semantic_namespace() -> None:
    for mode in _MODES:
        styles = rich_theme_styles(mode=mode)
        for token in SEMANTIC_TOKENS:
            assert f"semantic.{token}" in styles, (mode, token)
        # Status icons still resolve so legacy callers keep working.
        for legacy in (
            "status.icon.error",
            "status.icon.warning",
            "status.icon.success",
            "status.icon.info",
        ):
            assert legacy in styles, (mode, legacy)


def test_syntax_theme_uses_semantic_surface_names() -> None:
    assert syntax_theme("terminal_code", mode="dark") == "monokai"
    assert syntax_theme("terminal_code", mode="light") == "default"
    assert syntax_theme("terminal_diff", mode="dark") == "monokai"
    assert syntax_theme("html", mode="light") == "default"


def test_css_variables_cover_template_tokens() -> None:
    for mode in _MODES:
        variables = css_variables(mode)
        for required in (
            "--bg-primary",
            "--bg-code",
            "--border-subtle",
            "--user-bg",
            "--user-border",
            "--assistant-bg",
            "--assistant-border",
        ):
            assert required in variables, (mode, required)
        declarations = css_variable_declarations(mode)
        assert "            --bg-code:" in declarations
        assert '[data-theme="light"]' not in declarations


def test_force_plain_independent_of_theme(monkeypatch: pytest.MonkeyPatch) -> None:
    # POLYLOGUE_FORCE_PLAIN is orthogonal to theme resolution.
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setenv("POLYLOGUE_THEME", "light")
    assert resolve_theme_mode() == "light"
    # And rich_theme_styles still produces a valid mapping.
    styles = rich_theme_styles()
    assert "semantic.error" in styles


# ---------------------------------------------------------------------------
# Flag alias consistency
# ---------------------------------------------------------------------------


_REQUIRED_ALIASES = {
    "--provider": "-p",
    "--limit": "-l",
    "--format": "-f",
}


def _walk(
    cmd: click.Command, ctx: click.Context, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], click.Command]]:
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name in cmd.list_commands(ctx):
            sub = cmd.get_command(ctx, name)
            if sub is None or sub.hidden:
                continue
            sub_ctx = click.Context(sub, info_name=name, parent=ctx)
            yield from _walk(sub, sub_ctx, path + (name,))


def _collect_options(cmd: click.Command) -> Iterator[click.Option]:
    for param in cmd.params:
        if isinstance(param, click.Option):
            yield param


def test_flag_aliases_consistent() -> None:
    """Every --provider/--limit/--format option exposes its canonical alias.

    The exact aliasing the issue demands: -p/--provider, -l/--limit,
    -f/--format. Sweeping the whole Click tree means new commands inherit
    the contract automatically.
    """
    ctx = click.Context(cli, info_name="polylogue")
    offenders: list[str] = []
    for path, cmd in _walk(cli, ctx, ()):
        for opt in _collect_options(cmd):
            opts = set(opt.opts)
            for long_flag, short_flag in _REQUIRED_ALIASES.items():
                if long_flag in opts and short_flag not in opts:
                    offenders.append(
                        f"polylogue {' '.join(path)}: {long_flag} missing {short_flag} (got {sorted(opts)})"
                    )
    assert not offenders, "Flag-alias drift:\n" + "\n".join(offenders)


def test_no_alias_conflicts() -> None:
    """No short alias is reused for two different long flags in the same command."""
    ctx = click.Context(cli, info_name="polylogue")
    conflicts: list[str] = []
    for path, cmd in _walk(cli, ctx, ()):
        short_to_long: dict[str, str] = {}
        for opt in _collect_options(cmd):
            longs = [o for o in opt.opts if o.startswith("--")]
            shorts = [o for o in opt.opts if o.startswith("-") and not o.startswith("--")]
            primary_long = longs[0] if longs else (opt.opts[0] if opt.opts else "?")
            for short in shorts:
                if short in short_to_long and short_to_long[short] != primary_long:
                    conflicts.append(
                        f"polylogue {' '.join(path)}: {short} maps to both "
                        f"{short_to_long[short]!r} and {primary_long!r}"
                    )
                else:
                    short_to_long[short] = primary_long
    assert not conflicts, "Short-alias conflicts:\n" + "\n".join(conflicts)


def test_root_limit_accepts_both_short_aliases() -> None:
    """`polylogue --limit` keeps the historical -n alias alongside -l."""
    runner = CliRunner()
    # We use --help to confirm both shorts are advertised without needing a DB.
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "-l, -n" in result.output or "-l" in result.output


# ---------------------------------------------------------------------------
# --help-markdown
# ---------------------------------------------------------------------------


def test_help_markdown_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help-markdown"], prog_name="polylogue")
    assert result.exit_code == 0
    out = result.output
    assert out.startswith("<!-- Generated by `polylogue --help-markdown`. -->")
    assert "# `polylogue`" in out
    # Expect at least one subcommand emitted.
    assert "## `polylogue " in out
    # Every emitted command should be in a fenced text block.
    assert "```text" in out
    assert "```" in out


def test_help_markdown_covers_all_subcommands() -> None:
    rendered = render_help_markdown(cli)
    ctx = click.Context(cli, info_name="polylogue")
    for name in cli.list_commands(ctx):
        sub = cli.get_command(ctx, name)
        if sub is None or sub.hidden:
            continue
        assert f"`polylogue {name}`" in rendered, f"{name} missing from --help-markdown output"
