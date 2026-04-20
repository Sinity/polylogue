"""Typed root-mode request for the query-first CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polylogue.cli.query_contracts import coerce_query_terms
from polylogue.lib.query_spec import ConversationQuerySpec

if TYPE_CHECKING:
    import click


@dataclass(frozen=True)
class RootModeRequest:
    """Canonical root request for stats-or-query dispatch."""

    params: Mapping[str, object]
    query_terms: tuple[str, ...]

    @classmethod
    def from_context(cls, ctx: click.Context) -> RootModeRequest:
        query_terms = coerce_query_terms(ctx.meta.get("polylogue_query_terms") or ctx.params.get("query_term"))
        return cls(params=dict(ctx.params), query_terms=query_terms)

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> RootModeRequest:
        normalized_params = dict(params)
        query_terms = coerce_query_terms(normalized_params.pop("query", ()))
        return cls(params=normalized_params, query_terms=query_terms)

    def query_params(self) -> dict[str, object]:
        params = dict(self.params)
        params["query"] = self.query_terms
        return params

    def query_spec(self) -> ConversationQuerySpec:
        return ConversationQuerySpec.from_params(self.query_params())

    @property
    def verbose(self) -> bool:
        return bool(self.params.get("verbose", False))

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

    def with_param_updates(self, **updates: object) -> RootModeRequest:
        next_params = dict(self.params)
        next_params.update(updates)
        return replace(self, params=next_params)

    def with_query_terms(self, query_terms: Sequence[str]) -> RootModeRequest:
        return replace(self, query_terms=tuple(str(term) for term in query_terms))

    def append_query_terms(self, extra_terms: Sequence[str]) -> RootModeRequest:
        return self.with_query_terms(self.query_terms + tuple(str(term) for term in extra_terms))

    def should_show_stats(self) -> bool:
        query_spec = self.query_spec()
        return (
            not self.query_terms
            and not query_spec.has_filters()
            and not self.has_output_mode()
            and not self.has_modifiers()
        )


__all__ = ["RootModeRequest"]
