from __future__ import annotations

import pytest

from polylogue.schemas.code_detection import LANGUAGE_PATTERNS, detect_language, extract_code_block


def test_language_patterns_cover_major_languages() -> None:
    expected = {
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "c",
        "cpp",
        "bash",
        "sql",
        "html",
        "css",
        "json",
        "yaml",
    }
    assert expected.issubset(LANGUAGE_PATTERNS)
    assert all(patterns for patterns in LANGUAGE_PATTERNS.values())


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("def hello():\n    print('world')", "python"),
        ("const x = () => console.log('hi')", "javascript"),
        ("interface User {\n  name: string;\n}", "typescript"),
        ("fn main() {\n    println!(\"Hello\");\n}", "rust"),
        ("func main() {\n    fmt.Println(\"hi\")\n}", "go"),
        ("public class Main {\n    public static void main(String[] args) {}\n}", "java"),
        ("#include <stdio.h>\nint main() {}", "c"),
        ("std::cout << \"Hi\" << std::endl;", "cpp"),
        ("#!/bin/bash\necho 'test'", "bash"),
        ("SELECT * FROM users WHERE id = 1;", "sql"),
        ("<!DOCTYPE html>\n<html><body></body></html>", "html"),
        (".container {\n  display: flex;\n}", "css"),
        ('{"name": "test", "value": 123}', "json"),
        ("name: test\nvalue: 123", "yaml"),
    ],
)
def test_detect_language_exact_contracts(code: str, expected: str) -> None:
    assert detect_language(code) == expected


@pytest.mark.parametrize("code", ["", "   \n\n   ", "This is plain text without code markers", "random gibberish @@##$$"])
def test_detect_language_returns_none_for_non_code(code: str) -> None:
    assert detect_language(code) is None


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("py", "python"), ("js", "javascript"), ("ts", "typescript"), ("rs", "rust"), ("sh", "bash"), ("zsh", "bash")],
)
def test_detect_language_normalizes_alias_hints(declared: str, expected: str) -> None:
    assert detect_language("", declared_lang=declared) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\ndef hello():\n    pass\n```", "def hello():\n    pass"),
        ("```\nsome code\n```", "some code"),
        ("Text before\n\n    indented code\n    more code\n\nText after", "indented code\nmore code"),
        ("<thinking>Let me analyze this</thinking>", "Let me analyze this"),
    ],
)
def test_extract_code_block_contracts(text: str, expected: str) -> None:
    result = extract_code_block(text)
    assert expected in result or result == expected


@pytest.mark.parametrize("text", ["No code blocks here", "Just plain text", ""])
def test_extract_code_block_returns_original_or_empty_for_non_code(text: str) -> None:
    result = extract_code_block(text)
    assert result in {text, ""}
