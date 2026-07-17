"""Machine-error-aware CLI entry helpers."""

from __future__ import annotations

from collections.abc import Callable

import click


def extract_option(message: str) -> str | None:
    """Try to extract the option name from a Click error message."""
    if "No such option:" in message:
        return message.split("No such option:")[-1].strip().split()[0]
    # Click 8.4+ uses quotes: "No such option '--flag'."
    import re

    m = re.search(r"No such option '([^']+)'", message)
    if m:
        return m.group(1)
    return None


def actionable_hint_for_usage_error(message: str) -> str | None:
    """Return a one- or two-line actionable hint for a Click ``UsageError`` message.

    The returned hint is appended after Click's own error line so the user
    always gets a concrete next step. Returns ``None`` when no specific
    hint applies (the generic ``--help`` hint is added by the caller).
    """
    msg = message.strip()
    if "No such option:" in msg or "No such option '" in msg:
        bad = extract_option(msg)
        if bad:
            return (
                f"Hint: `{bad}` is not a recognized option. "
                "Run `polylogue --help` for the full option list, or "
                f'`polylogue "{bad}"` if you meant to search for that string.'
            )
        return "Hint: run `polylogue --help` to list available options."
    if "Missing argument" in msg or "Missing option" in msg:
        return "Hint: run the command with `--help` to see required arguments."
    if "No such command" in msg:
        # Extract the offending token if Click formatted it.
        import re

        match = re.search(r"No such command ['\"]([^'\"]+)['\"]", msg)
        bad = match.group(1) if match is not None else msg.split("No such command")[-1].strip().split()[0]
        if bad == "status":
            return (
                "Hint: `status` is an operational command. "
                "Run `polylogue ops status` for daemon/archive status, or "
                "`polylogue find status` if you meant to search for the word."
            )
        return (
            f"Hint: `{bad}` is not a registered subcommand. "
            f"Did you mean to search? Try `polylogue find {bad}` "
            "(plain unmarked roots are refused unless they are structured query expressions), "
            "or run `polylogue --help` for the full subcommand list."
        )
    if "Invalid value" in msg or "is not a valid" in msg:
        return "Hint: run the command with `--help` to see accepted values for this option."
    if "Query filters and root output flags must appear before the verb" in msg:
        return (
            "Hint: root filters (e.g. `-p claude-ai`, `--since`) and root output flags "
            "must precede the verb. Example: `polylogue --origin claude-ai find QUERY then read --all --format json`, "
            "not `polylogue read --origin claude-ai --format json`."
        )
    return None


def _show_usage_with_hint(exc: click.UsageError) -> None:
    """Show Click's own usage error, then append our actionable hint to stderr."""
    exc.show()
    hint = actionable_hint_for_usage_error(exc.format_message())
    if hint is None:
        hint = "Hint: run `polylogue --help` for usage, or `polylogue --diagnose <args>` to debug dispatch."
    click.echo(hint, err=True)


def run_machine_entry(
    cli: Callable[..., object],
    argv: list[str],
) -> None:
    """Run the CLI, emitting JSON machine errors when requested."""
    from polylogue.cli.shared.machine_errors import (
        error_invalid_arguments,
        error_runtime,
        extract_command,
        wants_json,
    )
    from polylogue.core.errors import PolylogueError

    if not wants_json(argv):
        try:
            cli(standalone_mode=False)
        except click.UsageError as exc:
            _show_usage_with_hint(exc)
            raise SystemExit(getattr(exc, "exit_code", 2)) from exc
        except click.ClickException as exc:
            exc.show()
            raise SystemExit(exc.exit_code) from exc
        except PolylogueError as exc:
            click.ClickException(str(exc)).show()
            raise SystemExit(1) from exc
        except SystemExit:
            # ``--help`` and ``--version`` reach here via Click's ctx.exit(0).
            raise
        except Exception as exc:
            click.ClickException(f"unexpected error: {type(exc).__name__}: {exc}").show()
            raise SystemExit(1) from exc
        return

    command = extract_command(argv)
    try:
        cli(standalone_mode=False)
    except click.UsageError as exc:
        option = getattr(exc, "option_name", None) or extract_option(str(exc))
        error_invalid_arguments(
            str(exc),
            command=command,
            option=option,
        ).emit(exit_code=exc.exit_code if hasattr(exc, "exit_code") else 2)
    except click.BadParameter as exc:
        param_hint = exc.param_hint
        option_hint = param_hint if param_hint is None or isinstance(param_hint, str) else ", ".join(param_hint)
        error_invalid_arguments(
            str(exc),
            command=command,
            option=option_hint,
        ).emit(exit_code=2)
    except click.ClickException as exc:
        error_runtime(
            exc.format_message(),
            command=command,
        ).emit(exit_code=exc.exit_code)
    except PolylogueError as exc:
        error_runtime(
            str(exc),
            command=command,
            exception_type=type(exc).__qualname__,
        ).emit(exit_code=1)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            error_invalid_arguments(code, command=command).emit(exit_code=1)
        elif isinstance(code, int) and code != 0:
            raise
    except Exception as exc:
        error_runtime(
            str(exc),
            command=command,
            exception_type=type(exc).__qualname__,
        ).emit(exit_code=1)


__all__ = ["extract_option", "run_machine_entry"]
