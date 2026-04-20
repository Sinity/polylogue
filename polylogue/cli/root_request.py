"""Typed root-mode request for the query-first CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.lib.query_spec import ConversationQuerySpec

if TYPE_CHECKING:
    import click


@dataclass(frozen=True)
class RootModeRequest:
    """Canonical root request for stats-or-query dispatch."""

    params: dict[str, object]
    query_terms: tuple[str, ...]

    @classmethod
    def from_context(cls, ctx: click.Context) -> RootModeRequest:
        query_terms = tuple(
            str(term) for term in (ctx.meta.get("polylogue_query_terms") or ctx.params.get("query_term") or ())
        )
        return cls(params=dict(ctx.params), query_terms=query_terms)

    def query_params(self) -> dict[str, object]:
        params = dict(self.params)
        params["query"] = self.query_terms
        return params

    def query_spec(self) -> ConversationQuerySpec:
        return ConversationQuerySpec.from_params(self.query_params())

    def has_output_mode(self) -> bool:
        return any(
            self.params.get(key)
            for key in (
                "limit",
                "stream",
                "dialogue_only",
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

    def should_show_stats(self) -> bool:
        query_spec = self.query_spec()
        return (
            not self.query_terms
            and not query_spec.has_filters()
            and not self.has_output_mode()
            and not self.has_modifiers()
        )


__all__ = ["RootModeRequest"]
