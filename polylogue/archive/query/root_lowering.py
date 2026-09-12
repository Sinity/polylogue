"""Pure root-query term lowering shared by CLI and daemon reads."""

from __future__ import annotations

import re

_FIELD_CLAUSE_START_RE = re.compile(r"^-?([a-zA-Z_][a-zA-Z0-9_.]*):")


def _looks_like_known_field_clause(piece: str) -> bool:
    # A colon alone is not DSL intent: URLs and ordinary log labels must stay
    # literal phrases unless their field is in the canonical registry.
    match = _FIELD_CLAUSE_START_RE.match(piece)
    if match is None:
        return False
    from polylogue.archive.query.metadata import EXPRESSION_FIELD_REGISTRY

    return match.group(1).lower() in EXPRESSION_FIELD_REGISTRY


def _is_shell_quoted_structured_query(term: str) -> bool:
    stripped = term.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered.startswith(("exists ", "exists(", "seq(", "lineage:id:")):
        return True
    if " where " in lowered:
        from polylogue.archive.query.metadata import terminal_query_sources

        return lowered.startswith(
            tuple(f"{source} where " for source in terminal_query_sources()) + ("sessions where ",)
        )
    return any(_looks_like_known_field_clause(piece) for piece in stripped.split())


def expression_from_query_terms(query_terms: tuple[str, ...]) -> str:
    """Turn CLI tokenization into the common DSL expression without Click."""

    if len(query_terms) == 1 and _is_shell_quoted_structured_query(query_terms[0]):
        return query_terms[0].strip()
    parts: list[str] = []
    for term in query_terms:
        if any(char.isspace() for char in term):
            parts.append('"' + term.replace('"', '\\"') + '"')
        else:
            parts.append(term)
    return " ".join(parts)


__all__ = ["expression_from_query_terms"]
