"""Typed root-mode request for the query-first CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.query_contracts import coerce_query_terms

if TYPE_CHECKING:
    import click


def _is_shell_quoted_structured_query(term: str) -> bool:
    """Return whether one shell term should be compiled as DSL, not a phrase."""

    stripped = term.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered.startswith(("exists ", "exists(", "seq(", "lineage:id:")):
        return True
    if " where " in lowered:
        from polylogue.archive.query.metadata import terminal_query_sources

        prefixes = tuple(f"{source} where " for source in terminal_query_sources()) + ("sessions where ",)
        return lowered.startswith(prefixes)
    return False


def _expression_from_query_terms(query_terms: tuple[str, ...]) -> str:
    if len(query_terms) == 1 and _is_shell_quoted_structured_query(query_terms[0]):
        return query_terms[0].strip()

    parts: list[str] = []
    for term in query_terms:
        if any(char.isspace() for char in term):
            escaped = term.replace('"', '\\"')
            parts.append(f'"{escaped}"')
        else:
            parts.append(term)
    return " ".join(parts)


@dataclass(frozen=True)
class RootModeRequest:
    """Canonical root request for stats-or-query dispatch."""

    params: Mapping[str, object]
    query_terms: tuple[str, ...]

    @classmethod
    def from_context(cls, ctx: click.Context) -> RootModeRequest:
        params = dict(ctx.params)
        query_terms = coerce_query_terms(ctx.meta.get("polylogue_query_terms") or params.pop("query_term", ()))
        return cls._from_normalized_params(params, query_terms)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> RootModeRequest:
        normalized_params = dict(params)
        query_terms = coerce_query_terms(normalized_params.pop("query", ()))
        return cls._from_normalized_params(normalized_params, query_terms)

    @classmethod
    def _from_normalized_params(cls, params: dict[str, object], query_terms: tuple[str, ...]) -> RootModeRequest:
        # --lexical and --semantic are ergonomic shortcuts that desugar
        # into existing query knobs so downstream specs stay unchanged.
        import click

        lexical = bool(params.pop("lexical", False))
        semantic = bool(params.pop("semantic", False))
        has_similar = bool(params.get("similar_text"))
        # --lexical (FTS-only) and --semantic/--similar (vector-only) are
        # opposing retrieval overrides; accepting both silently ran whichever
        # branch was checked first. Reject the contradiction (#1749).
        if lexical and (semantic or has_similar):
            conflicting = "--semantic" if semantic else "--similar"
            raise click.UsageError(
                f"{conflicting} cannot be combined with --lexical (they are opposing retrieval modes)."
            )
        if semantic and not query_terms:
            # --semantic promotes the query terms into a similarity prompt;
            # with no terms it was previously a silent no-op (#1749).
            raise click.UsageError("--semantic requires query terms to use as the similarity prompt.")
        if semantic:
            params["similar_text"] = " ".join(query_terms)
            query_terms = ()
        if lexical:
            params["retrieval_lane"] = "dialogue"
        return cls(params=params, query_terms=query_terms)

    def query_params(self) -> dict[str, object]:
        params = dict(self.params)
        params["query"] = self.query_terms
        return params

    def query_spec(self) -> SessionQuerySpec:
        """Build the canonical query spec for this request.

        When query_terms are present, they are compiled as a DSL expression
        and merged with the flag-derived spec.  Field clauses such as
        ``repo:polylogue`` or ``since:7d`` are compiled to the appropriate
        spec fields; bare words and quoted phrases continue to go to FTS.
        """
        from polylogue.archive.query.expression import compile_expression_into

        # Build the base spec from CLI flags (query_terms excluded).
        base = SessionQuerySpec.from_params(self.params)

        if not self.query_terms:
            return base

        return compile_expression_into(_expression_from_query_terms(self.query_terms), base)

    @property
    def verbose(self) -> bool:
        return bool(self.params.get("verbose", False))

    @property
    def explain_query(self) -> bool:
        return bool(self.params.get("explain_query", False))

    def has_output_mode(self) -> bool:
        return any(
            self.params.get(key)
            for key in (
                "explain_query",
                "limit",
                "stream",
                "dialogue_only",
                "message_role",
            )
        )

    def has_modifiers(self) -> bool:
        return any(
            self.params.get(key)
            for key in (
                "add_tag",
                "set_meta",
            )
        )

    def with_param_updates(self, **updates: object) -> RootModeRequest:
        next_params = dict(self.params)
        next_params.update(updates)
        return replace(self, params=next_params)

    def with_query_terms(self, query_terms: Sequence[str]) -> RootModeRequest:
        return replace(self, query_terms=tuple(str(term) for term in query_terms))

    def append_query_terms(self, extra_terms: Sequence[str]) -> RootModeRequest:
        return self.with_query_terms(self.query_terms + tuple(str(term) for term in extra_terms))

    def should_show_stats(self) -> bool:
        if self.query_terms:
            return False
        query_spec = self.query_spec()
        return not query_spec.has_filters() and not self.has_output_mode() and not self.has_modifiers()


__all__ = ["RootModeRequest"]
