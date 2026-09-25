"""One lowering for CLI-shaped query parameters.

``--lexical``/``--semantic`` are surface ergonomics that desugar into the
ordinary query knobs (``retrieval_lane``, ``similar_text``) before any spec is
built, and bare query terms lower to one DSL expression.  Both the Click root
request and the daemon's in-process reads used to carry their own copy of that
step, so a change to one silently gave the two surfaces different selections.
This module owns it; the surfaces only choose how the refusal is presented.

Import-light on purpose: no Click, no archive readers at module import.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.archive.query.root_lowering import expression_from_query_terms

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.surfaces.read_contract import ReadRequest

__all__ = [
    "QueryLoweringError",
    "cli_query_spec",
    "cli_read_request",
    "desugar_retrieval_flags",
    "expression_from_query_terms",
    "lower_cli_query_params",
    "lower_query_params",
]


class QueryLoweringError(ValueError):
    """A retrieval-mode combination that cannot be lowered.

    Carries the option-flavoured wording separately so the CLI can raise a
    ``click.UsageError`` naming the flags the operator typed while
    protocol callers keep a transport-neutral ``ValueError``.
    """

    def __init__(self, message: str, *, cli_message: str) -> None:
        self.cli_message = cli_message
        super().__init__(message)


def coerce_terms(value: object) -> tuple[str, ...]:
    """Coerce a protocol-supplied ``query`` value into canonical terms."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError("query must be a string or sequence")


def desugar_retrieval_flags(
    params: dict[str, object], query_terms: tuple[str, ...]
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Fold ``lexical``/``semantic`` into ordinary retrieval knobs.

    ``params`` is consumed: the two flags are removed and the equivalent
    ``similar_text``/``retrieval_lane`` values written in their place.
    """

    lexical = bool(params.pop("lexical", False))
    semantic = bool(params.pop("semantic", False))
    has_similar = bool(params.get("similar_text"))
    # --lexical (FTS-only) and --semantic/--similar (vector-only) are opposing
    # retrieval overrides; accepting both silently ran whichever branch was
    # checked first.  Reject the contradiction (#1749).
    if lexical and (semantic or has_similar):
        conflicting = "--semantic" if semantic else "--similar"
        raise QueryLoweringError(
            "semantic retrieval cannot be combined with lexical retrieval",
            cli_message=f"{conflicting} cannot be combined with --lexical (they are opposing retrieval modes).",
        )
    if semantic and not query_terms:
        # --semantic promotes the query terms into a similarity prompt; with
        # no terms it was previously a silent no-op (#1749).
        raise QueryLoweringError(
            "semantic retrieval requires query terms",
            cli_message="--semantic requires query terms to use as the similarity prompt.",
        )
    if semantic:
        params["similar_text"] = " ".join(query_terms)
        query_terms = ()
    if lexical:
        params["retrieval_lane"] = "dialogue"
    return params, query_terms


def lower_query_params(params: Mapping[str, object]) -> tuple[dict[str, object], tuple[str, ...]]:
    """Split a raw parameter map into desugared params and query terms."""

    normalized = dict(params)
    terms = coerce_terms(normalized.pop("query", ()))
    normalized, terms = desugar_retrieval_flags(normalized, terms)
    if normalized.get("retrieval_lane") == "semantic":
        normalized["retrieval_lane"] = "auto"
        if not normalized.get("similar_text"):
            prompt = " ".join(term for term in terms if term).strip()
            if prompt:
                normalized["similar_text"] = prompt
        terms = ()
    return normalized, terms


def lower_cli_query_params(params: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Lower CLI conveniences to params plus one DSL expression."""

    normalized, terms = lower_query_params(params)
    return normalized, expression_from_query_terms(terms)


def cli_read_request(params: Mapping[str, object], *, preset: str = "summary") -> ReadRequest:
    """Lower CLI-shaped query intent into the shared read contract."""

    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.surfaces.read_contract import ReadRequest

    normalized, expression = lower_cli_query_params(params)
    base = SessionQuerySpec.from_params(normalized)
    selection = compile_expression_into(expression, base) if expression else base
    return ReadRequest.normalize({"selection": selection}, preset=preset)


def cli_query_spec(params: Mapping[str, object]) -> SessionQuerySpec:
    """Expose the shared read request's selection to existing query owners."""

    return cli_read_request(params).selection
