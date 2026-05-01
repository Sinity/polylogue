"""Shared no-results feedback helpers for CLI query surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.shared.machine_errors import error_no_results

if TYPE_CHECKING:
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.spec import ConversationQuerySpec
    from polylogue.cli.shared.types import AppEnv


def emit_no_results(
    env: AppEnv,
    *,
    selection: ConversationQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
    output_format: str = "text",
    message: str | None = None,
    hint: str | None = None,
    exit_code: int | None = 2,
) -> None:
    """Render a canonical no-results message for human and machine surfaces."""
    filters = selection.describe() if selection is not None else []
    resolved_message = message or ("No conversations matched filters." if filters else "No conversations matched.")
    if output_format == "json":
        error_no_results(
            resolved_message,
            filters=filters or None,
            diagnostics=diagnostics.to_dict() if diagnostics is not None else None,
        ).emit(exit_code=exit_code or 2)

    if filters and message is None:
        env.ui.console.print("No conversations matched filters:")
        for item in filters:
            env.ui.console.print(f"  {item}")
        env.ui.console.print(hint or "Hint: try broadening your filters or use `list` to browse")
    else:
        env.ui.console.print(resolved_message)

    if diagnostics is not None and diagnostics.reasons:
        env.ui.console.print("Why this may have missed:")
        for line in diagnostics.human_reason_lines():
            env.ui.console.print(f"  - {line}" if not line.startswith("  ") else f"    {line.strip()}")

    if exit_code is not None:
        raise SystemExit(exit_code)


__all__ = ["emit_no_results"]
