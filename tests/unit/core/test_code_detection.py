"""Direct tests for polylogue.schemas.code_detection.detection module.

Covers detect_language(), regex patterns, alias resolution,
extract_code_block_from_dict(), and extract_code_block().
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.lib.json import JSONDocument
from polylogue.schemas.code_detection.detection import (
    LANGUAGE_PATTERNS,
    _regex_scores,
    detect_language,
    extract_code_block,
    extract_code_block_from_dict,
)


class TestDetectLanguage:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ('def hello():\n    print("hi")', "python"),
            ("class Foo:\n    pass", "python"),
            ("from os import path\nimport sys", "python"),
            ('fn main() {\n    println!("hello");\n}', "rust"),
            ("pub struct Foo {\n    bar: i32,\n}", "rust"),
            ('func main() {\n    fmt.Println("hello")\n}', "go"),
            ('package main\nimport "fmt"', "go"),
            ("SELECT name FROM users WHERE id = 1", "sql"),
            ("CREATE TABLE foo (id INT, name TEXT)", "sql"),
        ],
    )
    def test_detects_language(self, code: str, expected: str) -> None:
        assert detect_language(code) == expected

    def test_returns_none_for_plain_text(self) -> None:
        assert detect_language("Hello, this is just a normal sentence.") is None

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("py", "python"),
            ("js", "javascript"),
            ("ts", "typescript"),
            ("rs", "rust"),
            ("sh", "bash"),
            ("zsh", "bash"),
        ],
    )
    def test_declared_lang_aliases(self, alias: str, canonical: str) -> None:
        assert detect_language("anything", declared_lang=alias) == canonical

    def test_declared_lang_passthrough(self) -> None:
        assert detect_language("anything", declared_lang="ruby") == "ruby"

    def test_declared_lang_takes_priority(self) -> None:
        python_code = 'def hello():\n    print("hi")'
        assert detect_language(python_code, declared_lang="ruby") == "ruby"

    @given(st.text(max_size=500))
    @settings(max_examples=100)
    def test_never_raises(self, code: str) -> None:
        result = detect_language(code)
        assert result is None or isinstance(result, str)


class TestRegexScores:
    def test_python_code_scores_positive(self) -> None:
        scores = _regex_scores("def hello():\n    pass")
        assert "python" in scores
        assert scores["python"] > 0

    def test_empty_string_scores_nothing(self) -> None:
        scores = _regex_scores("")
        assert len(scores) == 0

    def test_all_languages_have_patterns(self) -> None:
        for lang in LANGUAGE_PATTERNS:
            assert len(LANGUAGE_PATTERNS[lang]) > 0


class TestExtractCodeBlockFromDict:
    def test_fenced_with_language(self) -> None:
        block: JSONDocument = {"type": "text", "text": "```python\ndef hello(): pass\n```"}
        result = extract_code_block_from_dict(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "python"
        assert result["declared_language"] == "python"
        text = result["text"]
        assert isinstance(text, str)
        assert "def hello" in text

    def test_fenced_without_language(self) -> None:
        block: JSONDocument = {"type": "text", "text": "```\ndef hello(): pass\n```"}
        result = extract_code_block_from_dict(block)
        assert result is not None
        assert result["type"] == "code"

    def test_non_code_text(self) -> None:
        block: JSONDocument = {"type": "text", "text": "Just a short note"}
        result = extract_code_block_from_dict(block)
        assert result is None

    def test_code_without_fence(self) -> None:
        # Long enough text that looks like code
        code_text = 'def hello():\n    print("world")\n\nclass Foo:\n    pass'
        block: JSONDocument = {"type": "text", "text": code_text}
        result = extract_code_block_from_dict(block)
        assert result is not None
        assert result["language"] == "python"


class TestExtractCodeBlock:
    def test_fenced_block(self) -> None:
        text = "```python\ndef hello(): pass\n```"
        assert extract_code_block(text) == "def hello(): pass"

    def test_fenced_no_language(self) -> None:
        text = "```\nsome code\n```"
        assert extract_code_block(text) == "some code"

    def test_thinking_block(self) -> None:
        text = "<thinking>let me think about this</thinking>"
        assert extract_code_block(text) == "let me think about this"

    def test_indented_block(self) -> None:
        text = "\n    def hello():\n        pass\n"
        result = extract_code_block(text)
        assert "def hello" in result

    def test_no_code_returns_empty(self) -> None:
        assert extract_code_block("No code here") == ""

    def test_empty_string(self) -> None:
        assert extract_code_block("") == ""

    def test_code_detected_without_markers(self) -> None:
        code = "def foo():\n    return 42\n\nclass Bar:\n    pass"
        result = extract_code_block(code)
        assert "def foo" in result
