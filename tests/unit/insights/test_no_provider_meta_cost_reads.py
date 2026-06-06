"""Lint-style guard: cost ``provider_meta`` reads must live in extractors only.

#1139 completes the carry-over from #803: cost/usage values in
``provider_meta`` are extracted into the typed cost read model
(``polylogue/insights/archive_models.py``, ``CostEstimatePayload`` from
``polylogue/archive/semantic/pricing.py``). Downstream readers (archive
runtime, insights, session/threads, storage repositories, CLI/MCP surfaces)
must consume the typed model, never reach back into ``provider_meta`` for
cost or token-usage values.

This test fails if a new ``provider_meta``-cost read appears outside the
allowlisted extractor modules. The allowlist intentionally lists exact module
paths; broadening it requires an explicit decision documented in this file.

Mirrors the ``verify-layering``/``verify-test-ownership`` pattern: declared
contract enforced by a focused test rather than ad-hoc convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

POLYLOGUE_ROOT = Path(__file__).resolve().parents[3] / "polylogue"

# Keys that name a cost/usage value when read from ``provider_meta``.
COST_KEYS = frozenset(
    {
        "cost",
        "costUSD",
        "cost_usd",
        "total_cost_usd",
        "usage",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "total_tokens",
    }
)

# Modules that own provider->typed extraction; ``provider_meta`` cost reads
# here are the canonical single source of truth for those facts.
EXTRACTOR_ALLOWLIST = frozenset(
    {
        # Parsers: provider wire -> ParsedSession.provider_meta writes
        # and (for Message.cost_usd in model_runtime) the parser-only accessor.
        "polylogue/sources/parsers",
        "polylogue/sources/providers",
        # Schema-level coercion of provider_meta into typed sub-models.
        "polylogue/schemas",
        # Canonical pricing extractor: provider_meta -> CostEstimatePayload.
        "polylogue/archive/semantic/pricing.py",
        # Parser-only Message.cost_usd accessor (documented PARSER-ONLY in
        # its docstring; returns None for hydrated messages).
        "polylogue/archive/message/model_runtime.py",
    }
)


def _is_allowlisted(path: Path) -> bool:
    rel = path.relative_to(POLYLOGUE_ROOT.parent).as_posix()
    return any(rel == entry or rel.startswith(entry.rstrip("/") + "/") for entry in EXTRACTOR_ALLOWLIST)


def _provider_meta_cost_reads(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(lineno, key)`` for each ``...provider_meta.get("<COST_KEY>")``.

    Also matches ``provider_meta["<COST_KEY>"]`` subscript reads.
    """

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # Match ``X.provider_meta.get("key", ...)``.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "provider_meta"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in COST_KEYS
        ):
            hits.append((node.lineno, node.args[0].value))
        # Match ``X.provider_meta["key"]``.
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "provider_meta"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value in COST_KEYS
        ):
            hits.append((node.lineno, node.slice.value))
    return hits


def test_no_provider_meta_cost_reads_outside_extractors() -> None:
    offenders: list[str] = []
    for path in POLYLOGUE_ROOT.rglob("*.py"):
        if _is_allowlisted(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for lineno, key in _provider_meta_cost_reads(tree):
            rel = path.relative_to(POLYLOGUE_ROOT.parent).as_posix()
            offenders.append(f"{rel}:{lineno} reads provider_meta[{key!r}]")
    assert not offenders, (
        "Cost-bearing provider_meta reads must live in extractor modules. "
        "Consume the typed cost read model (polylogue.insights.archive_models / "
        "polylogue.archive.semantic.pricing) instead. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_allowlist_entries_exist() -> None:
    """Guard against bit-rot: every allowlist entry must point at a real path."""

    repo_root = POLYLOGUE_ROOT.parent
    missing = [entry for entry in EXTRACTOR_ALLOWLIST if not (repo_root / entry).exists()]
    assert not missing, f"Stale entries in EXTRACTOR_ALLOWLIST: {missing}"
