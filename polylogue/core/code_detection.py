"""Code block language detection utilities."""

from __future__ import annotations

import re
from typing import Any


# Common language indicators
LANGUAGE_PATTERNS = {
    "python": [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import",
        r"@\w+\s*\n",  # Decorators
    ],
    "javascript": [
        r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"=>\s*{",  # Arrow functions
        r"console\.log\(",
    ],
    "typescript": [
        r":\s*(string|number|boolean|any)\b",
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
        r"<[\w\s,]+>",  # Generics
    ],
    "rust": [
        r"\bfn\s+\w+\s*\(",
        r"\blet\s+mut\s+",
        r"\bimpl\s+\w+",
        r"\bpub\s+(fn|struct|enum)",
        r"#\[derive\(",
    ],
    "go": [
        r"\bfunc\s+\w+\s*\(",
        r"\bpackage\s+\w+",
        r":=",  # Short variable declaration
        r"\btype\s+\w+\s+(struct|interface)",
    ],
    "java": [
        r"\bpublic\s+(class|interface|enum)",
        r"\bprivate\s+\w+\s+\w+;",
        r"System\.out\.println\(",
        r"@Override",
    ],
    "c": [
        r"#include\s*<",
        r"\bint\s+main\s*\(",
        r"\bprintf\s*\(",
        r"\bmalloc\s*\(",
    ],
    "cpp": [
        r"#include\s*<",
        r"\bstd::\w+",
        r"\bnamespace\s+\w+",
        r"cout\s*<<",
    ],
    "bash": [
        r"^#!/bin/(ba)?sh",
        r"\bif\s+\[\[",
        r"\bfunction\s+\w+\s*\(\)",
        r"\becho\s+",
        r"\$\{?\w+",  # Variable expansion
    ],
    "sql": [
        r"\bSELECT\s+.+\s+FROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bCREATE\s+(TABLE|INDEX|VIEW)\b",
        r"\bJOIN\s+\w+\s+ON\b",
    ],
    "html": [
        r"<(!DOCTYPE|html|head|body|div|span)",
        r"</\w+>",
    ],
    "css": [
        r"\.\w+\s*{",
        r"#\w+\s*{",
        r"@media\s+",
        r":\s*(left|right|center|flex)",
    ],
    "json": [
        r'^\s*[{\[]',  # Starts with { or [
        r'"\w+":\s*',  # JSON key-value
    ],
    "yaml": [
        r"^\w+:",
        r"^  \w+:",  # Indented key
        r"^-\s+\w+",  # List item
    ],
}


def detect_language(code: str, declared_lang: str | None = None) -> str | None:
    """Detect programming language from code content.

    Args:
        code: Code text to analyze
        declared_lang: Language hint from source (e.g., from fence ```python)

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
        # Normalize common aliases
        aliases = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "rs": "rust",
            "sh": "bash",
            "zsh": "bash",
        }
        return aliases.get(lang_lower, lang_lower)

    # Pattern-based detection
    scores: dict[str, int] = {}

    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                score += 1
        if score > 0:
            scores[lang] = score

    if not scores:
        return None

    # Return language with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def extract_code_block(content_block: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and enrich code block with language detection.

    Args:
        content_block: Content block from provider_meta (type: text, thinking, etc.)

    Returns:
        Enriched code block dict with 'language' field, or None if not code

    Examples:
        >>> block = {"type": "text", "text": "```python\\ndef hello(): pass\\n```"}
        >>> result = extract_code_block(block)
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
