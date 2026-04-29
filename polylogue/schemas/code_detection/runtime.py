"""Canonical code-language detection workflow."""

from __future__ import annotations

from polylogue.schemas.code_detection.regex import normalize_declared_language, regex_scores
from polylogue.schemas.code_detection.tree_sitter import detect_with_tree_sitter


def detect_language(code: str, declared_lang: str | None = None) -> str | None:
    """Detect programming language from code content."""
    if declared_lang:
        return normalize_declared_language(declared_lang)

    scores = regex_scores(code)
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= 2:
        return ranked[0][0]

    top_score = ranked[0][1]
    candidates = [lang for lang, score in ranked if score >= top_score - 1]
    ts_result = detect_with_tree_sitter(code, candidates)
    if ts_result is not None:
        return ts_result
    return ranked[0][0]


__all__ = ["detect_language"]
