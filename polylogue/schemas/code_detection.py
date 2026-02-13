"""Code block language detection utilities.

Supports two detection backends:

1. **tree-sitter** (preferred, optional) — Attempts to parse the code with
   each candidate grammar.  The language whose parser produces the fewest
   errors (lowest ``error_ratio``) wins.  Accurate and extensible.

2. **regex** (built-in fallback) — Scores code against pattern lists per
   language.  Fast, requires no external dependencies.

When tree-sitter grammars are installed (``pip install polylogue[tree-sitter]``),
the two backends cooperate: regex narrows candidates, tree-sitter disambiguates
the top matches.  Without tree-sitter, regex-only detection is used.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from polylogue.lib.log import get_logger

LOGGER = get_logger(__name__)


# =============================================================================
# Tree-sitter backend (optional)
# =============================================================================

# Maps our canonical language names → tree-sitter grammar package names.
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
def _tree_sitter_available() -> bool:
    """Check if the tree-sitter Python bindings are importable."""
    try:
        import tree_sitter  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=32)
def _get_ts_language(lang: str) -> Any | None:
    """Load a tree-sitter Language for *lang*, or ``None`` on failure."""
    if not _tree_sitter_available():
        return None

    module_name = _TS_GRAMMAR_MAP.get(lang)
    if not module_name:
        return None

    try:
        import importlib

        from tree_sitter import Language

        mod = importlib.import_module(module_name)

        # tree-sitter-python >= 0.21 exposes language() as a callable
        lang_fn = getattr(mod, "language", None)
        if lang_fn is None:
            return None
        return Language(lang_fn())
    except Exception:
        return None


def _ts_error_ratio(code: str, lang: str) -> float | None:
    """Parse *code* with the tree-sitter grammar for *lang*.

    Returns the fraction of AST nodes that are ERROR nodes (0.0 = perfect
    parse, 1.0 = all errors).  Returns ``None`` if the grammar is not
    available.
    """
    ts_lang = _get_ts_language(lang)
    if ts_lang is None:
        return None

    try:
        from tree_sitter import Parser

        parser = Parser(ts_lang)
        tree = parser.parse(code.encode("utf-8"))

        # Walk the tree and count ERROR nodes
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


def _ts_detect(code: str, candidates: list[str]) -> str | None:
    """Disambiguate *candidates* using tree-sitter parse quality.

    Tries each candidate language grammar and returns the one with the
    lowest error ratio.  Returns ``None`` if tree-sitter is unavailable
    or all candidates fail.
    """
    if not _tree_sitter_available() or not candidates:
        return None

    best_lang: str | None = None
    best_ratio = 1.0

    for lang in candidates:
        ratio = _ts_error_ratio(code, lang)
        if ratio is not None and ratio < best_ratio:
            best_ratio = ratio
            best_lang = lang

    # Only trust tree-sitter if the best parse is reasonably clean
    if best_lang is not None and best_ratio < 0.3:
        return best_lang
    return None


# =============================================================================
# Regex backend (always available)
# =============================================================================


LANGUAGE_PATTERNS = {
    "python": [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[:\(]",
        r"\bfrom\s+[a-zA-Z]\w*\s+import",
        r"\bimport\s+[a-zA-Z]\w*",
        r"@\w+\s*\(",
        r"\bif\s+__name__\s*==\s*['\"]__main__['\"]",
        r"\bfor\s+\w+\s+in\s+",
        r"\bwhile\s+\w+\s*:",
        r"\btry\s*:",
        r"\bwith\s+\w+",
        r"lambda\s+\w+:",
        r"\[.+\s+for\s+\w+\s+in\s+",
        r"\byield\s+",
        r"\basync\s+def\s+",
    ],
    "javascript": [
        r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bvar\s+\w+\s*=",
        r"=>\s*{",
        r"console\.log\(",
        r"\bexport\s+",
        r"\bimport\s+\w+\s+from\s+['\"]",
        r"from\s+['\"]",
        r"\basync\s+function",
        r"\bclass\s+\w+.*\}\s*$",
    ],
    "typescript": [
        r":\s*\w+\s*=>",
        r":\s*(string|number|boolean|any|void|null|true|false)\s*[,);=}]",
        r"\binterface\s+\w+\s*{",
        r"\btype\s+\w+\s*=",
        r"<\w+\s*>",
        r"\benum\s+\w+",
        r"\bas\s+const",
        r"function<",
        r"\(\w+:\s*\w+\)",
    ],
    "rust": [
        r"\bfn\s+\w+\s*\(",
        r"\blet\s+mut\s+",
        r"\bimpl\s+\w+",
        r"\bpub\s+(fn|struct|enum)",
        r"#\[derive\(",
        r"\bstruct\s+\w+\s*{",
        r"\bmatch\s+\w+\s*{",
    ],
    "go": [
        r"\bfunc\s+\w+\s*\(",
        r"\bpackage\s+\w+",
        r":=",
        r"\btype\s+\w+\s+(struct|interface)",
        r'\bimport\s+"',
        r"\bdefer\s+",
        r"\bgo\s+\w+\(",
    ],
    "java": [
        r"\bpublic\s+(class|interface|enum)\s+\w+",
        r"\bprivate\s+\w+\s+\w+",
        r"System\.out\.println\(",
        r"@\w+\s*(?:\(|$)",
        r"\bpublic\s+static\s+void\s+main",
        r"\bextends\s+",
        r"\bimplements\s+",
    ],
    "c": [
        r"#include\s*<",
        r"\bint\s+main\s*\(",
        r"\bprintf\s*\(",
        r"\bmalloc\s*\(",
        r"\bint\s+\w+\s*=",
        r"\bsizeof\s*\(",
    ],
    "cpp": [
        r"\bstd::\w+",
        r"#include\s*<",
        r"\bnamespace\s+\w+",
        r"cout\s*<<",
        r"\btemplate\s*<",
        r"\busing\s+namespace",
        r"::\w+",
        r"nullptr",
        r"\bclass\s+\w+\s*\{[^}]*\};",
    ],
    "bash": [
        r"^#!/bin/(ba)?sh",
        r"\bif\s+\[\[",
        r"\bif\s+\[\s+-f",
        r"\bfunction\s+\w+\s*\(\)",
        r"\becho\s+",
        r"\$\{?\w+",
    ],
    "sql": [
        r"\bSELECT\s+.+\s+FROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bCREATE\s+(TABLE|INDEX|VIEW)\b",
        r"\bJOIN\s+\w+\s+ON\b",
        r"\bUPDATE\s+\w+\s+SET\b",
    ],
    "html": [
        r"<(!DOCTYPE|html|head|body|div|span)",
        r"</\w+>",
        r"<head\s*>",
        r"<title>",
    ],
    "css": [
        r"\.\w+\s*{",
        r"#\w+\s*{?",
        r"@media\s+",
        r":\s*(left|right|center|flex|absolute|relative)",
        r"display\s*:",
        r"\w+-\w+:",
    ],
    "json": [
        r'^\s*[{\[]',
        r'"\w+":\s*',
    ],
    "yaml": [
        r"^\w+:",
        r"^  \w+:",
        r"^-\s+\w+",
    ],
}


def _regex_scores(code: str) -> dict[str, int]:
    """Compute regex-based detection scores for all languages."""
    scores: dict[str, int] = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                score += 1
        if score > 0:
            scores[lang] = score
    return scores


def detect_language(code: str, declared_lang: str | None = None) -> str | None:
    """Detect programming language from code content.

    Uses a two-stage approach:

    1. **Regex scoring** — fast pattern matching to generate candidate
       languages with confidence scores.
    2. **Tree-sitter disambiguation** (optional) — when the top candidates
       are close in score, tree-sitter parses the code with each grammar
       and picks the one with the fewest parse errors.

    Args:
        code: Code text to analyze
        declared_lang: Language hint from source (e.g., from fence ``python``)

    Returns:
        Detected language name (lowercase) or None if unknown

    Examples:
        >>> detect_language('def hello():\\n    print("hi")')
        'python'
        >>> detect_language('function test() { console.log("hi"); }')
        'javascript'
        >>> detect_language('SELECT * FROM users', declared_lang='sql')
        'sql'
    """
    # Trust declared language if provided
    if declared_lang:
        lang_lower = declared_lang.lower()
        aliases = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "rs": "rust",
            "sh": "bash",
            "zsh": "bash",
        }
        return aliases.get(lang_lower, lang_lower)

    # Stage 1: Regex scoring
    scores = _regex_scores(code)

    if not scores:
        return None

    # Sort candidates by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # If clear winner (>= 2 points ahead of second place), return immediately
    if len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= 2:
        return ranked[0][0]

    # Stage 2: Tree-sitter disambiguation for close calls
    # Candidates = all within 1 point of the top score
    top_score = ranked[0][1]
    candidates = [lang for lang, score in ranked if score >= top_score - 1]

    ts_result = _ts_detect(code, candidates)
    if ts_result is not None:
        return ts_result

    # Fallback to regex winner
    return ranked[0][0]


def extract_code_block_from_dict(content_block: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and enrich code block with language detection (dict-based API).

    Args:
        content_block: Content block from provider_meta (type: text, thinking, etc.)

    Returns:
        Enriched code block dict with 'language' field, or None if not code

    Examples:
        >>> block = {"type": "text", "text": "```python\\ndef hello(): pass\\n```"}
        >>> result = extract_code_block_from_dict(block)
        >>> result['type']
        'code'
        >>> result['language']
        'python'
    """
    block_type = content_block.get("type")
    text = content_block.get("text", "")

    # Check for fenced code blocks
    fence_match = re.match(r"```(\w*)\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        declared_lang = fence_match.group(1) or None
        code = fence_match.group(2)
        detected_lang = detect_language(code, declared_lang)

        return {
            "type": "code",
            "language": detected_lang,
            "text": code,
            "declared_language": declared_lang,
        }

    # Check if entire text block is likely code (no fence)
    if block_type == "text" and len(text) > 20:
        detected_lang = detect_language(text)
        if detected_lang:
            return {
                "type": "code",
                "language": detected_lang,
                "text": text,
                "declared_language": None,
            }

    return None


def extract_code_block(text: str) -> str:
    """Extract code from markdown text (string-based API).

    Supports:
    - Fenced code blocks: ```language\\ncode\\n```
    - Indented code blocks (4+ spaces, first line must be indented)
    - Thinking blocks: <thinking>code</thinking>
    - Plain code if detected

    Args:
        text: Markdown or code text

    Returns:
        Extracted code string, or empty string if no code found

    Examples:
        >>> extract_code_block("```python\\ndef hello(): pass\\n```")
        'def hello(): pass'
        >>> extract_code_block("```\\ncode\\n```")
        'code'
        >>> extract_code_block("No code here")
        ''
    """
    if not text:
        return ""

    # Check for thinking blocks
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if thinking_match:
        return thinking_match.group(1).strip()

    # Check for fenced code blocks (```language\ncode\n```)
    fence_match = re.match(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Check for indented code blocks (4+ spaces on consecutive lines)
    lines = text.split("\n")
    indented_lines = []
    in_block = False
    first_line_indented = lines[0].startswith("    ") if lines else False
    blank_line_before_indent = False

    for i, line in enumerate(lines):
        if line.startswith("    "):
            if not in_block and i > 0 and not blank_line_before_indent:
                blank_line_before_indent = lines[i - 1].strip() == ""
            in_block = True
            indented_lines.append(line[4:])
        elif in_block and line.strip() == "":
            indented_lines.append("")
        elif in_block and not line.startswith("    "):
            break

    if indented_lines:
        while indented_lines and not indented_lines[-1].strip():
            indented_lines.pop()
        if indented_lines and (first_line_indented or blank_line_before_indent):
            return "\n".join(indented_lines)

    # Check if entire text is code (detected language and reasonable length)
    if len(text) >= 10 and detect_language(text):
        return text

    return ""
