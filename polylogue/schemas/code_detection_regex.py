"""Regex-backed code detection heuristics."""

from __future__ import annotations

import re

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

LANGUAGE_ALIASES = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "rs": "rust",
    "sh": "bash",
    "zsh": "bash",
}


def normalize_declared_language(declared_lang: str) -> str:
    """Normalize a declared language hint into a canonical token."""
    lang_lower = declared_lang.lower()
    return LANGUAGE_ALIASES.get(lang_lower, lang_lower)


def regex_scores(code: str) -> dict[str, int]:
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


__all__ = ["LANGUAGE_ALIASES", "LANGUAGE_PATTERNS", "normalize_declared_language", "regex_scores"]
