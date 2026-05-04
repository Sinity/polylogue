"""Machine-error-aware CLI entry helpers."""

from __future__ import annotations

from collections.abc import Callable

import click


def extract_option(message: str) -> str | None:
    """Try to extract the option name from a Click error message."""
    if "No such option:" in message:
        return message.split("No such option:")[-1].strip().split()[0]
    return None


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
    from polylogue.errors import PolylogueError

    if not wants_json(argv):
        try:
            cli()
        except PolylogueError as exc:
            click.ClickException(str(exc)).show()
            raise SystemExit(1) from exc
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
