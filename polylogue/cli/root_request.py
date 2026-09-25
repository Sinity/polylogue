"""Typed root-mode request for the query-first CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.query_contracts import coerce_query_terms
from polylogue.operations.query_lowering import (
    QueryLoweringError,
    cli_read_request,
    desugar_retrieval_flags,
)

if TYPE_CHECKING:
    import click


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
        """Desugar retrieval flags through the one shared lowering."""
        import click

        try:
            params, query_terms = desugar_retrieval_flags(params, query_terms)
        except QueryLoweringError as exc:
            raise click.UsageError(exc.cli_message) from exc
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
        return cli_read_request(self.query_params()).selection

    @property
    def verbose(self) -> bool:
        return bool(self.params.get("verbose", False))

    @property
    def explain_query(self) -> bool:
        return bool(self.params.get("explain_query", False))

    @property
    def why(self) -> bool:
        """Whether the full zero-hit breakdown was requested (``--why``, #jnj.12)."""
        return bool(self.params.get("why", False))

    def has_output_mode(self) -> bool:
        return any(
            self.params.get(key)
            for key in (
                "explain_query",
                "limit",
                "stream",
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

    def config(self) -> object:
        """The configuration this request runs against.

        ``_config`` is the pinned configuration the root callback threaded
        through the request; without it a route answers against whatever
        archive is active, which is not necessarily the one the operator named
        with ``--db``.  A route that read ``params["_config"]`` directly got
        ``None`` for every invocation the root callback did not pin -- which
        every real ``polylogue read`` is.
        """

        from polylogue.config import Config, get_config

        pinned = self.params.get("_config")
        return pinned if isinstance(pinned, Config) else get_config()

    def should_show_stats(self) -> bool:
        if self.query_terms:
            return False
        query_spec = self.query_spec()
        return not query_spec.has_filters() and not self.has_output_mode() and not self.has_modifiers()


__all__ = ["RootModeRequest"]
