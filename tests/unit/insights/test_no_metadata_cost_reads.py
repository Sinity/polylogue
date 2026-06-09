"""Guard: cost readers must not inspect metadata escape hatches.

Current archive cost facts are typed: message token/model columns,
``session_reported_costs``, ``session_model_usage``, and materialized
``session_profiles`` cost fields. No runtime reader may recover cost or
usage from a metadata bag.
"""

from __future__ import annotations

import ast
from pathlib import Path

POLYLOGUE_ROOT = Path(__file__).resolve().parents[3] / "polylogue"

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

METADATA_ATTRS = frozenset({"provider_meta", "origin_meta", "metadata"})


def _metadata_cost_reads(tree: ast.AST) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in METADATA_ATTRS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in COST_KEYS
        ):
            hits.append((node.lineno, node.func.value.attr, node.args[0].value))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in METADATA_ATTRS
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value in COST_KEYS
        ):
            hits.append((node.lineno, node.value.attr, node.slice.value))
    return hits


def test_cost_readers_do_not_inspect_metadata_escape_hatches() -> None:
    offenders: list[str] = []
    for path in POLYLOGUE_ROOT.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for lineno, attr, key in _metadata_cost_reads(tree):
            rel = path.relative_to(POLYLOGUE_ROOT.parent).as_posix()
            offenders.append(f"{rel}:{lineno} reads {attr}[{key!r}]")
    assert not offenders, (
        "Cost-bearing facts must come from typed cost/model/token fields, "
        "not metadata escape hatches. Offenders:\n  " + "\n  ".join(offenders)
    )
