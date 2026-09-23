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
        hint = 'Hint: run `polylogue --help` for usage, or `polylogue --diagnose find "migration"` to debug dispatch.'
    click.echo(hint, err=True)


def run_machine_entry(
    cli: Callable[..., object],
    argv: list[str],
) -> None:
    """Run the CLI, emitting JSON machine errors when requested."""
    from polylogue.cli.operation_kernel import OperationUnavailableError
    from polylogue.cli.shared.helper_support import DaemonRequiredError
    from polylogue.cli.shared.machine_errors import (
        error_archive_writer_ownership,
        error_daemon_required,
        error_invalid_arguments,
        error_runtime,
        extract_command,
        wants_json,
    )
    from polylogue.cli.write_authority import ArchiveWriterOwnershipError
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
        except ArchiveWriterOwnershipError as exc:
            # Before the generic branch: the single-writer boundary's refusal
            # is a decision, not a crash, and reaching the operator as
            # ``unexpected error: ArchiveWriterOwnershipError`` buried the one
            # sentence that says what to do next (polylogue-re6s3 AC4).
            click.ClickException(str(exc)).show()
            raise SystemExit(1) from exc
        except click.Abort as exc:
            # ``standalone_mode=False`` hands Click's own abort signal back to
            # us, so ``raise click.Abort()`` -- the CLI's declined-confirmation
            # and refuse-here path -- rendered as ``unexpected error: Abort:``
            # with an empty message.
            click.echo("Aborted.", err=True)
            raise SystemExit(1) from exc
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
    except DaemonRequiredError as exc:
        # Before the generic ``ClickException`` branch: the daemon-absent
        # refusal is the one case a machine caller can act on directly, and
        # folding it into ``runtime_error`` is exactly what made it
        # unactionable.
        error_daemon_required(
            exc.format_message(),
            command=command,
            operation=exc.operation,
            archive_root=exc.archive_root,
        ).emit(exit_code=exc.exit_code)
    except OperationUnavailableError as exc:
        # The kernel's own daemon-absent refusal. It is a RuntimeError rather
        # than a ClickException, so without this branch it reached the generic
        # handler and emitted ``runtime_error`` -- the precise flattening the
        # DaemonRequiredError branch above exists to prevent, arriving by the
        # other door once the CLI stopped executing operations in-process
        # (polylogue-3eexy). ``operation`` is carried as a field so the client
        # need not parse it back out of the message (polylogue-re6s3 AC4).
        error_daemon_required(
            str(exc),
            command=command,
            operation=exc.operation,
            archive_root=None,
        ).emit(exit_code=2)
    except click.ClickException as exc:
        error_runtime(
            exc.format_message(),
            command=command,
        ).emit(exit_code=exc.exit_code)
    except ArchiveWriterOwnershipError as exc:
        # Same reasoning as the terminal branch above, and the same reason the
        # ``DaemonRequiredError`` branch precedes ``ClickException``: a machine
        # caller that cannot tell "a daemon already owns this archive" from
        # "something went wrong" cannot route the write anywhere.
        error_archive_writer_ownership(
            str(exc),
            code=exc.code,
            command=command,
            archive_root=exc.archive_root,
            resident_writer=exc.resident_writer,
        ).emit(exit_code=1)
    except click.Abort as exc:
        error_runtime("aborted", command=command, exception_type=type(exc).__qualname__).emit(exit_code=1)
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
