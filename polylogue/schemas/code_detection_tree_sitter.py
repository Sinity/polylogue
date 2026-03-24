"""Optional tree-sitter language detection support."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_TS_GRAMMAR_MAP: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "bash": "tree_sitter_bash",
    "html": "tree_sitter_html",
    "css": "tree_sitter_css",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
}


@lru_cache(maxsize=1)
def tree_sitter_available() -> bool:
    """Check if the tree-sitter Python bindings are importable."""
    try:
        import tree_sitter  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=32)
def get_ts_language(lang: str) -> Any | None:
    """Load a tree-sitter Language for *lang*, or ``None`` on failure."""
    if not tree_sitter_available():
        return None

    module_name = _TS_GRAMMAR_MAP.get(lang)
    if not module_name:
        return None

    try:
        import importlib

        from tree_sitter import Language

        mod = importlib.import_module(module_name)
        lang_fn = getattr(mod, "language", None)
        if lang_fn is None:
            return None
        return Language(lang_fn())
    except Exception:
        return None


def ts_error_ratio(code: str, lang: str) -> float | None:
    """Parse *code* with the tree-sitter grammar for *lang*."""
    ts_lang = get_ts_language(lang)
    if ts_lang is None:
        return None

    try:
        from tree_sitter import Parser

        parser = Parser(ts_lang)
        tree = parser.parse(code.encode("utf-8"))

        total_nodes = 0
        error_nodes = 0

        def _walk(node: Any) -> None:
            nonlocal total_nodes, error_nodes
            total_nodes += 1
            if node.type == "ERROR" or node.is_missing:
                error_nodes += 1
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)

        if total_nodes == 0:
            return 1.0
        return error_nodes / total_nodes
    except Exception:
        return None


def detect_with_tree_sitter(code: str, candidates: list[str]) -> str | None:
    """Disambiguate *candidates* using tree-sitter parse quality."""
    if not tree_sitter_available() or not candidates:
        return None

    best_lang: str | None = None
    best_ratio = 1.0
    for lang in candidates:
        ratio = ts_error_ratio(code, lang)
        if ratio is not None and ratio < best_ratio:
            best_ratio = ratio
            best_lang = lang
    if best_lang is not None and best_ratio < 0.3:
        return best_lang
    return None


__all__ = [
    "detect_with_tree_sitter",
    "get_ts_language",
    "tree_sitter_available",
    "ts_error_ratio",
]
