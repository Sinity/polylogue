"""Shared no-results feedback helpers for CLI query surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from polylogue.archive.query.fields import describe_spec_selection_fields
from polylogue.cli.shared.machine_errors import error_no_results

if TYPE_CHECKING:
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cli.shared.types import AppEnv


def maybe_subcommand_typo_hint(query_terms: Sequence[str]) -> str | None:
    """Return a 'did you mean to run a subcommand?' hint for a single bare token.

    Query-first dispatch is surprising when a user types something like
    ``polylogue stats-by-origin`` expecting a subcommand. If ``query_terms``
    is a single token and matches a registered subcommand by Levenshtein
    distance, we surface that explicitly in the no-results output. Both the
    archive no-results paths consume this helper so the
    hint stays identical across surfaces.
    """
    terms: tuple[str, ...] = tuple(str(term) for term in query_terms)
    if len(terms) != 1:
        return None
    token = terms[0].strip()
    if not token or " " in token:
        return None

    # Lazy import to keep startup cheap.
    from polylogue.cli.click_app import cli
    from polylogue.cli.parser_diagnostics import looks_like_subcommand_typo

    registered = sorted(cli.commands.keys())
    suggestions = looks_like_subcommand_typo(token, registered)
    if not suggestions:
        return None
    return (
        "Note: query-first dispatch interpreted "
        f"`{token}` as a search query. If you meant a subcommand, try: "
        + ", ".join(f"`polylogue {s}`" for s in suggestions)
    )


def _maybe_subcommand_typo_hint(selection: SessionQuerySpec | None) -> str | None:
    """Adapter for the legacy spec-driven no-results path."""
    if selection is None:
        return None
    return maybe_subcommand_typo_hint(getattr(selection, "query_terms", None) or ())


def emit_no_results(
    env: AppEnv,
    *,
    selection: SessionQuerySpec | None = None,
    diagnostics: QueryMissDiagnostics | None = None,
    output_format: str = "text",
    message: str | None = None,
    hint: str | None = None,
    exit_code: int | None = 2,
) -> None:
    """Render a canonical no-results message for human and machine surfaces."""
    filters = describe_spec_selection_fields(selection) if selection is not None else []
    resolved_message = message or ("No sessions matched filters." if filters else "No sessions matched.")
    if output_format == "json":
        error_no_results(
            resolved_message,
            filters=filters or None,
            diagnostics=diagnostics.to_dict() if diagnostics is not None else None,
        ).emit(exit_code=exit_code or 2)

    if filters and message is None:
        env.ui.console.print("No sessions matched filters:")
        for item in filters:
            env.ui.console.print(f"  {item}")
        env.ui.console.print(hint or "Hint: try broadening your filters or use `list` to browse")
    else:
        env.ui.console.print(resolved_message)

    typo_hint = _maybe_subcommand_typo_hint(selection)
    if typo_hint is not None:
        env.ui.console.print(typo_hint)

    if diagnostics is not None and diagnostics.reasons:
        env.ui.console.print("Why this may have missed:")
        for line in diagnostics.human_reason_lines():
            env.ui.console.print(f"  - {line}" if not line.startswith("  ") else f"    {line.strip()}")

    if exit_code is not None:
        raise SystemExit(exit_code)


__all__ = ["emit_no_results", "maybe_subcommand_typo_hint"]
