"""Code block language detection utilities."""

from __future__ import annotations

import re
from typing import Any

# Common language indicators
LANGUAGE_PATTERNS = {
    "python": [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[:\(]",  # class definition (with or without inheritance)
        r"\bfrom\s+[a-zA-Z]\w*\s+import",  # from X import Y (not "from 'module'")
        r"\bimport\s+[a-zA-Z]\w*",  # import X (not import from 'module')
        r"@\w+\s*\(",  # Decorators with parentheses (more specific)
        r"\bif\s+__name__\s*==\s*['\"]__main__['\"]",  # Main check
        r"\bfor\s+\w+\s+in\s+",  # For loop
        r"\bwhile\s+\w+\s*:",  # While loop
        r"\btry\s*:",  # Try-except (simplified)
        r"\bwith\s+\w+",  # With statement
        r"lambda\s+\w+:",  # Lambda
        r"\[.+\s+for\s+\w+\s+in\s+",  # List comprehension
        r"\byield\s+",  # Yield
        r"\basync\s+def\s+",  # Async def
    ],
    "javascript": [
        r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bvar\s+\w+\s*=",  # Var declaration
        r"=>\s*{",  # Arrow functions
        r"console\.log\(",
        r"\bexport\s+",  # Export
        r"\bimport\s+\w+\s+from\s+['\"]",  # Import from (JS-specific with quotes)
        r"from\s+['\"]",  # from 'module' (JS style, appears after import)
        r"\basync\s+function",  # Async function
        r"\bclass\s+\w+.*\}\s*$",  # Class with closing brace at end (JS, no semicolon)
    ],
    "typescript": [
        r":\s*\w+\s*=>",  # Type annotation on arrow function (high-confidence)
        r":\s*(string|number|boolean|any|void|null|true|false)\s*[,);=}]",  # Type annotation
        r"\binterface\s+\w+\s*{",
        r"\btype\s+\w+\s*=",
        r"<\w+\s*>",  # Generics
        r"\benum\s+\w+",  # Enum
        r"\bas\s+const",  # Const assertion
        r"function<",  # Generic function
        r"\(\w+:\s*\w+\)",  # Function parameter with type
    ],
    "rust": [
        r"\bfn\s+\w+\s*\(",
        r"\blet\s+mut\s+",
        r"\bimpl\s+\w+",
        r"\bpub\s+(fn|struct|enum)",
        r"#\[derive\(",
        r"\bstruct\s+\w+\s*{",  # Struct definition
        r"\bmatch\s+\w+\s*{",  # Match
    ],
    "go": [
        r"\bfunc\s+\w+\s*\(",
        r"\bpackage\s+\w+",
        r":=",  # Short variable declaration
        r"\btype\s+\w+\s+(struct|interface)",
        r'\bimport\s+"',  # Import statement
        r"\bdefer\s+",  # Defer
        r"\bgo\s+\w+\(",  # Goroutine
    ],
    "java": [
        r"\bpublic\s+(class|interface|enum)\s+\w+",  # Be more specific
        r"\bprivate\s+\w+\s+\w+",  # Private field (with or without semicolon)
        r"System\.out\.println\(",
        r"@\w+\s*(?:\(|$)",  # Annotations (@Override, @Deprecated, etc.)
        r"\bpublic\s+static\s+void\s+main",  # Main method signature
        r"\bextends\s+",  # Extends
        r"\bimplements\s+",  # Implements
    ],
    "c": [
        r"#include\s*<",
        r"\bint\s+main\s*\(",
        r"\bprintf\s*\(",
        r"\bmalloc\s*\(",
        r"\bint\s+\w+\s*=",  # Int declaration
        r"\bsizeof\s*\(",  # Sizeof
    ],
    "cpp": [
        r"\bstd::\w+",  # std:: namespace (strong indicator)
        r"#include\s*<",
        r"\bnamespace\s+\w+",
        r"cout\s*<<",
        r"\btemplate\s*<",  # Template
        r"\busing\s+namespace",  # Using namespace
        r"::\w+",  # Scope resolution operator
        r"nullptr",  # C++ nullptr keyword
        r"\bclass\s+\w+\s*\{[^}]*\};",  # Class with semicolon (C++ style)
    ],
    "bash": [
        r"^#!/bin/(ba)?sh",
        r"\bif\s+\[\[",
        r"\bif\s+\[\s+-f",  # if [ -f test
        r"\bfunction\s+\w+\s*\(\)",
        r"\becho\s+",
        r"\$\{?\w+",  # Variable expansion
    ],
    "sql": [
        r"\bSELECT\s+.+\s+FROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bCREATE\s+(TABLE|INDEX|VIEW)\b",
        r"\bJOIN\s+\w+\s+ON\b",
        r"\bUPDATE\s+\w+\s+SET\b",  # Update statement
    ],
    "html": [
        r"<(!DOCTYPE|html|head|body|div|span)",
        r"</\w+>",
        r"<head\s*>",  # Specific head tag
        r"<title>",  # Title tag
    ],
    "css": [
        r"\.\w+\s*{",
        r"#\w+\s*{?",  # ID selector (with or without braces)
        r"@media\s+",  # Media query
        r":\s*(left|right|center|flex|absolute|relative)",
        r"display\s*:",  # CSS property
        r"\w+-\w+:",  # Hyphenated CSS property
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
    # Extract any indented block (markdown convention)
    lines = text.split("\n")
    indented_lines = []
    in_block = False
    first_line_indented = lines[0].startswith("    ") if lines else False
    blank_line_before_indent = False

    for i, line in enumerate(lines):
        if line.startswith("    "):
            # Start or continue indented block
            # Check if there was a blank line before this indented block
            if not in_block and i > 0 and not blank_line_before_indent:
                blank_line_before_indent = lines[i - 1].strip() == ""
            in_block = True
            indented_lines.append(line[4:])
        elif in_block and line.strip() == "":
            # Empty line within block
            indented_lines.append("")
        elif in_block and not line.startswith("    "):
            # End of block (non-indented line after indented)
            break

    if indented_lines:
        # Remove trailing empty lines
        while indented_lines and not indented_lines[-1].strip():
            indented_lines.pop()
        # Return indented block: either pure indented code or markdown indented code block
        if indented_lines and (first_line_indented or blank_line_before_indent):
            return "\n".join(indented_lines)

    # Check if entire text is code (detected language and reasonable length)
    if len(text) >= 10 and detect_language(text):
        return text

    return ""
