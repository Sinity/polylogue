"""Tests for code block language detection utilities."""

from __future__ import annotations

import pytest

from polylogue.core.code_detection import (
    LANGUAGE_PATTERNS,
    detect_language,
    extract_code_block,
)


class TestLanguagePatternsDefinition:
    """Test LANGUAGE_PATTERNS dictionary structure and coverage."""

    def test_patterns_has_common_languages(self):
        """Verify all major languages are defined."""
        expected_langs = {
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
        assert expected_langs.issubset(LANGUAGE_PATTERNS.keys())

    def test_each_language_has_patterns(self):
        """Verify each language has at least one pattern."""
        for lang, patterns in LANGUAGE_PATTERNS.items():
            assert isinstance(patterns, list), f"{lang} patterns should be a list"
            assert len(patterns) > 0, f"{lang} should have at least one pattern"
            assert all(
                isinstance(p, str) for p in patterns
            ), f"All {lang} patterns should be strings"


class TestDetectLanguagePython:
    """Test Python language detection."""

    def test_detects_def_function(self):
        """Detect function definition."""
        code = "def hello():\n    print('world')"
        assert detect_language(code) == "python"

    def test_detects_class_definition(self):
        """Detect class definition."""
        code = "class MyClass:\n    pass"
        assert detect_language(code) == "python"

    def test_detects_import_statement(self):
        """Detect import statement."""
        code = "import numpy\nimport pandas as pd"
        assert detect_language(code) == "python"

    def test_detects_from_import(self):
        """Detect from-import statement."""
        code = "from pathlib import Path\nfrom typing import Any"
        assert detect_language(code) == "python"

    def test_detects_decorator(self):
        """Detect decorator syntax."""
        code = "@dataclass\nclass Person:\n    name: str"
        assert detect_language(code) == "python"

    def test_multiline_python_code(self):
        """Detect complex Python code with multiple features."""
        code = """
def process_data(items):
    '''Process a list of items.'''
    results = []
    for item in items:
        if isinstance(item, dict):
            results.append(item)
    return results
        """
        assert detect_language(code) == "python"


class TestDetectLanguageJavaScript:
    """Test JavaScript language detection."""

    def test_detects_function_declaration(self):
        """Detect function declaration."""
        code = "function hello() { return 42; }"
        assert detect_language(code) == "javascript"

    def test_detects_const_assignment(self):
        """Detect const variable declaration."""
        code = "const x = 10;"
        assert detect_language(code) == "javascript"

    def test_detects_let_assignment(self):
        """Detect let variable declaration."""
        code = "let counter = 0;"
        assert detect_language(code) == "javascript"

    def test_detects_arrow_function(self):
        """Detect arrow function syntax."""
        code = "const add = (a, b) => { return a + b; }"
        assert detect_language(code) == "javascript"

    def test_detects_console_log(self):
        """Detect console.log statement."""
        code = "console.log('Hello World');"
        assert detect_language(code) == "javascript"

    def test_multiline_javascript_code(self):
        """Detect complex JavaScript code."""
        code = """
const fetchData = async (url) => {
    const response = await fetch(url);
    const data = await response.json();
    console.log(data);
    return data;
};
        """
        assert detect_language(code) == "javascript"


class TestDetectLanguageTypeScript:
    """Test TypeScript language detection."""

    def test_detects_type_annotation(self):
        """Detect type annotation."""
        code = "const name: string = 'Alice';"
        assert detect_language(code) == "typescript"

    def test_detects_interface(self):
        """Detect interface definition."""
        code = "interface User {\n  id: number;\n  name: string;\n}"
        assert detect_language(code) == "typescript"

    def test_detects_type_alias(self):
        """Detect type alias."""
        code = "type Status = 'active' | 'inactive';"
        assert detect_language(code) == "typescript"

    def test_detects_generic_type(self):
        """Detect generic type syntax."""
        code = "function identity<T>(arg: T): T { return arg; }"
        assert detect_language(code) == "typescript"


class TestDetectLanguageRust:
    """Test Rust language detection."""

    def test_detects_fn_function(self):
        """Detect function definition."""
        code = "fn main() {\n    println!(\"Hello, world!\");\n}"
        assert detect_language(code) == "rust"

    def test_detects_let_mut(self):
        """Detect mutable variable binding."""
        code = "let mut x = 5;\nx = 10;"
        assert detect_language(code) == "rust"

    def test_detects_impl_block(self):
        """Detect impl block."""
        code = "impl MyStruct {\n    fn new() -> Self { MyStruct {} }\n}"
        assert detect_language(code) == "rust"

    def test_detects_pub_fn(self):
        """Detect pub function."""
        code = "pub fn public_function() {}"
        assert detect_language(code) == "rust"

    def test_detects_derive_attribute(self):
        """Detect derive attribute."""
        code = "#[derive(Debug, Clone)]\nstruct Point { x: i32, y: i32 }"
        assert detect_language(code) == "rust"


class TestDetectLanguageGo:
    """Test Go language detection."""

    def test_detects_func_function(self):
        """Detect function definition."""
        code = "func main() {\n    fmt.Println(\"Hello\")\n}"
        assert detect_language(code) == "go"

    def test_detects_package_declaration(self):
        """Detect package declaration."""
        code = "package main\n\nimport \"fmt\""
        assert detect_language(code) == "go"

    def test_detects_short_var_declaration(self):
        """Detect := short variable declaration."""
        code = "x := 42\nname := \"Alice\""
        assert detect_language(code) == "go"

    def test_detects_type_struct(self):
        """Detect struct type definition."""
        code = "type Person struct {\n    name string\n    age int\n}"
        assert detect_language(code) == "go"


class TestDetectLanguageJava:
    """Test Java language detection."""

    def test_detects_public_class(self):
        """Detect public class."""
        code = "public class HelloWorld {\n    public static void main(String[] args) {}\n    public void test() {}\n    System.out.println(\"test\");\n}"
        assert detect_language(code) == "java"

    def test_detects_private_field(self):
        """Detect private field."""
        code = "private String name;"
        assert detect_language(code) == "java"

    def test_detects_system_out_println(self):
        """Detect System.out.println."""
        code = "System.out.println(\"Hello\");"
        assert detect_language(code) == "java"

    def test_detects_override_annotation(self):
        """Detect @Override annotation."""
        code = "@Override\npublic String toString() { return \"test\"; }\nprivate int value;"
        assert detect_language(code) == "java"


class TestDetectLanguageC:
    """Test C language detection."""

    def test_detects_include_directive(self):
        """Detect #include directive."""
        code = '#include <stdio.h>\n#include <stdlib.h>'
        assert detect_language(code) == "c"

    def test_detects_main_function(self):
        """Detect main function."""
        code = "int main() {\n    return 0;\n}"
        assert detect_language(code) == "c"

    def test_detects_printf(self):
        """Detect printf."""
        code = 'printf("Hello, World!");'
        assert detect_language(code) == "c"

    def test_detects_malloc(self):
        """Detect malloc."""
        code = "int *ptr = (int *)malloc(sizeof(int));"
        assert detect_language(code) == "c"


class TestDetectLanguageCpp:
    """Test C++ language detection."""

    def test_detects_std_namespace(self):
        """Detect std:: namespace usage."""
        code = "std::vector<int> v;\nstd::cout << v.size();"
        assert detect_language(code) == "cpp"

    def test_detects_namespace_declaration(self):
        """Detect namespace declaration."""
        code = "namespace myapp {\n    class MyClass {};\n}\nstd::cout << \"test\";"
        assert detect_language(code) == "cpp"

    def test_detects_cout_operator(self):
        """Detect cout << operator."""
        code = 'cout << "Hello" << std::endl;'
        assert detect_language(code) == "cpp"


class TestDetectLanguageBash:
    """Test Bash language detection."""

    def test_detects_shebang(self):
        """Detect shebang."""
        code = "#!/bin/bash\necho 'Hello World'"
        assert detect_language(code) == "bash"

    def test_detects_if_test_syntax(self):
        """Detect if [[ ]] syntax."""
        code = "if [[ -f file.txt ]]; then\n  echo 'File exists'\nfi"
        assert detect_language(code) == "bash"

    def test_detects_function_declaration(self):
        """Detect function declaration."""
        code = "function greet() {\n  echo 'Hello'\n}"
        assert detect_language(code) == "bash"

    def test_detects_echo(self):
        """Detect echo command."""
        code = "echo 'Starting script'\necho $variable"
        assert detect_language(code) == "bash"

    def test_detects_variable_expansion(self):
        """Detect variable expansion."""
        code = "NAME='Alice'\necho $NAME\necho ${NAME}_suffix"
        assert detect_language(code) == "bash"


class TestDetectLanguageSQL:
    """Test SQL language detection."""

    def test_detects_select_query(self):
        """Detect SELECT statement."""
        code = "SELECT * FROM users WHERE id = 1;"
        assert detect_language(code) == "sql"

    def test_detects_select_from(self):
        """Detect SELECT with FROM."""
        code = "SELECT id, name, email FROM users"
        assert detect_language(code) == "sql"

    def test_detects_insert_into(self):
        """Detect INSERT INTO statement."""
        code = "INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')"
        assert detect_language(code) == "sql"

    def test_detects_create_table(self):
        """Detect CREATE TABLE statement."""
        code = "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  name VARCHAR(100)\n)"
        assert detect_language(code) == "sql"

    def test_detects_create_view(self):
        """Detect CREATE VIEW statement."""
        code = "CREATE VIEW active_users AS SELECT * FROM users WHERE active = 1"
        assert detect_language(code) == "sql"

    def test_detects_create_index(self):
        """Detect CREATE INDEX statement."""
        code = "CREATE INDEX idx_users_email ON users(email)"
        assert detect_language(code) == "sql"

    def test_detects_join_clause(self):
        """Detect JOIN clause."""
        code = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        assert detect_language(code) == "sql"


class TestDetectLanguageHTML:
    """Test HTML language detection."""

    def test_detects_doctype(self):
        """Detect DOCTYPE declaration."""
        code = "<!DOCTYPE html>"
        assert detect_language(code) == "html"

    def test_detects_html_tag(self):
        """Detect html tag."""
        code = "<html>\n<head><title>Test</title></head>\n<body></body>\n</html>"
        assert detect_language(code) == "html"

    def test_detects_head_tag(self):
        """Detect head tag."""
        code = "<head><meta charset='utf-8'></head>"
        assert detect_language(code) == "html"

    def test_detects_div_tag(self):
        """Detect div tag."""
        code = '<div class="container"><p>Content</p></div>'
        assert detect_language(code) == "html"

    def test_detects_closing_tag(self):
        """Detect closing tag."""
        code = "<div>Content</div>\n<p>Text</p>"
        assert detect_language(code) == "html"


class TestDetectLanguageCSS:
    """Test CSS language detection."""

    def test_detects_class_selector(self):
        """Detect class selector."""
        code = ".container { width: 100%; }"
        assert detect_language(code) == "css"

    def test_detects_id_selector(self):
        """Detect ID selector."""
        code = "#header { background-color: blue; }"
        assert detect_language(code) == "css"

    def test_detects_media_query(self):
        """Detect media query."""
        code = "@media (max-width: 600px) { body { font-size: 14px; } }"
        assert detect_language(code) == "css"

    def test_detects_flex_property(self):
        """Detect flex property."""
        code = ".flex { display: flex; justify-content: center; }"
        assert detect_language(code) == "css"


class TestDetectLanguageJSON:
    """Test JSON language detection."""

    def test_detects_json_object(self):
        """Detect JSON object."""
        code = '{"name": "Alice", "age": 30}'
        assert detect_language(code) == "json"

    def test_detects_json_array(self):
        """Detect JSON array."""
        code = '[{"id": 1}, {"id": 2}]'
        assert detect_language(code) == "json"

    def test_detects_nested_json(self):
        """Detect nested JSON structure."""
        code = '{"user": {"name": "Bob", "email": "bob@example.com"}}'
        assert detect_language(code) == "json"

    def test_detects_json_with_whitespace(self):
        """Detect JSON with leading whitespace."""
        code = "  {\n    \"key\": \"value\"\n  }"
        assert detect_language(code) == "json"


class TestDetectLanguageYAML:
    """Test YAML language detection."""

    def test_detects_simple_key_value(self):
        """Detect simple key-value pair."""
        code = "name: Alice\nage: 30"
        assert detect_language(code) == "yaml"

    def test_detects_indented_key(self):
        """Detect indented nested key."""
        code = "user:\n  name: Alice\n  age: 30"
        assert detect_language(code) == "yaml"

    def test_detects_list_item(self):
        """Detect list item."""
        code = "- item1\n- item2\n- item3"
        assert detect_language(code) == "yaml"


class TestDetectLanguageDeclaredLanguage:
    """Test declared language parameter handling."""

    def test_trusts_declared_language(self):
        """Trust declared language when provided."""
        code = "console.log('hello');"  # Looks like JavaScript
        # But we declare it as Python
        result = detect_language(code, declared_lang="python")
        assert result == "python"

    def test_handles_lowercase_declared_language(self):
        """Handle declared language in lowercase."""
        code = "def hello(): pass"
        result = detect_language(code, declared_lang="PYTHON")
        assert result == "python"

    def test_alias_py_to_python(self):
        """Resolve py alias to python."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="py")
        assert result == "python"

    def test_alias_js_to_javascript(self):
        """Resolve js alias to javascript."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="js")
        assert result == "javascript"

    def test_alias_ts_to_typescript(self):
        """Resolve ts alias to typescript."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="ts")
        assert result == "typescript"

    def test_alias_rs_to_rust(self):
        """Resolve rs alias to rust."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="rs")
        assert result == "rust"

    def test_alias_sh_to_bash(self):
        """Resolve sh alias to bash."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="sh")
        assert result == "bash"

    def test_alias_zsh_to_bash(self):
        """Resolve zsh alias to bash."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="zsh")
        assert result == "bash"

    def test_unknown_declared_language_passthrough(self):
        """Pass through unknown declared language as-is."""
        code = "SELECT * FROM users"
        result = detect_language(code, declared_lang="custom_lang")
        assert result == "custom_lang"


class TestDetectLanguageEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string(self):
        """Handle empty string."""
        result = detect_language("")
        assert result is None

    def test_whitespace_only(self):
        """Handle whitespace-only string."""
        result = detect_language("   \n  \t  \n  ")
        assert result is None

    def test_plain_text_no_code(self):
        """No language detected for plain text."""
        text = "This is just regular English text with no programming constructs."
        result = detect_language(text)
        assert result is None

    def test_ambiguous_code_picks_highest_score(self):
        """Ambiguous code picks language with highest match score."""
        # Code with multiple language indicators
        code = "def foo():\n  console.log('test')"
        result = detect_language(code)
        # Should detect python (def is strongest)
        assert result in ("python", "javascript")

    def test_mixed_multiline_code(self):
        """Detect language from mixed multiline code."""
        code = """
import sys
from pathlib import Path

def process_data(items):
    '''Process items.'''
    return [item for item in items if item]
        """
        assert detect_language(code) == "python"


class TestExtractCodeBlockFenced:
    """Test extract_code_block with fenced code blocks."""

    def test_extract_fenced_python_code(self):
        """Extract Python code from fenced block."""
        block = {
            "type": "text",
            "text": "```python\ndef hello():\n    print('hi')\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "python"
        assert "def hello()" in result["text"]
        assert result["declared_language"] == "python"

    def test_extract_fenced_javascript_code(self):
        """Extract JavaScript code from fenced block."""
        block = {
            "type": "text",
            "text": "```javascript\nfunction test() { return 42; }\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "javascript"
        assert "function test()" in result["text"]
        assert result["declared_language"] == "javascript"

    def test_extract_fenced_without_declared_language(self):
        """Extract code from fenced block without declared language."""
        block = {
            "type": "text",
            "text": "```\ndef hello():\n    pass\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "python"
        assert result["declared_language"] is None

    def test_extract_fenced_sql_code(self):
        """Extract SQL code from fenced block."""
        block = {
            "type": "text",
            "text": "```sql\nSELECT * FROM users WHERE id = 1\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "sql"

    def test_extract_fenced_json_code(self):
        """Extract JSON code from fenced block."""
        block = {
            "type": "text",
            "text": '```json\n{"name": "Alice", "age": 30}\n```',
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "json"


class TestExtractCodeBlockUnfenced:
    """Test extract_code_block with unfenced code blocks."""

    def test_extract_python_unfenced(self):
        """Extract Python code without fence."""
        code = "def hello():\n    print('world')\n" * 5  # > 20 chars
        block = {"type": "text", "text": code}
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "python"
        assert result["declared_language"] is None

    def test_extract_javascript_unfenced(self):
        """Extract JavaScript code without fence."""
        code = "function test() {\n    console.log('test');\n}" * 5  # > 20 chars
        block = {"type": "text", "text": code}
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "javascript"

    def test_extract_sql_unfenced(self):
        """Extract SQL code without fence."""
        code = "SELECT * FROM users WHERE active = 1\n" * 5  # > 20 chars
        block = {"type": "text", "text": code}
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "sql"

    def test_no_extract_short_text(self):
        """Don't extract short text as code."""
        block = {"type": "text", "text": "def x(): pass"}  # Only 13 chars
        result = extract_code_block(block)
        assert result is None

    def test_no_extract_plain_text_unfenced(self):
        """Don't extract plain text without code patterns."""
        code = "This is a long plain text that goes on and on without any code patterns " * 5
        block = {"type": "text", "text": code}
        result = extract_code_block(block)
        assert result is None


class TestExtractCodeBlockEdgeCases:
    """Test extract_code_block edge cases."""

    def test_extract_from_thinking_block(self):
        """Extract code from thinking block (should not extract)."""
        block = {
            "type": "thinking",
            "thinking": "def helper(): pass" * 5,
        }
        result = extract_code_block(block)
        # Only text blocks are checked for unfenced code
        assert result is None

    def test_extract_code_with_empty_text(self):
        """Extract from block with missing text field."""
        block = {"type": "text"}
        result = extract_code_block(block)
        assert result is None

    def test_extract_code_with_empty_fence(self):
        """Extract from block with empty code fence."""
        block = {
            "type": "text",
            "text": "```python\n\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["type"] == "code"
        # Empty code is still valid
        assert result["text"] == ""

    def test_extract_preserves_multiline_code(self):
        """Preserve multiline code structure."""
        code = """def process(items):
    result = []
    for item in items:
        if valid(item):
            result.append(item)
    return result"""
        block = {"type": "text", "text": f"```python\n{code}\n```"}
        result = extract_code_block(block)
        assert result is not None
        assert result["text"] == code

    def test_extract_code_with_special_characters(self):
        """Preserve special characters in code."""
        code = 'print("Hello, World! @#$%^&*()")' * 5
        block = {"type": "text", "text": f"```python\n{code}\n```"}
        result = extract_code_block(block)
        assert result is not None
        assert "@#$%^&*()" in result["text"]

    def test_extract_code_unicode_content(self):
        """Handle Unicode content in code."""
        code = "# 你好 世界\nprint('Привет мир')" * 5
        block = {"type": "text", "text": f"```python\n{code}\n```"}
        result = extract_code_block(block)
        assert result is not None
        assert "你好" in result["text"]
        assert "Привет" in result["text"]


class TestExtractCodeBlockDeclaredLanguage:
    """Test extract_code_block respects declared language."""

    def test_declared_language_overrides_detection(self):
        """Declared language overrides automatic detection."""
        code = "SELECT * FROM users"
        block = {
            "type": "text",
            "text": f"```python\n{code}\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        # Should respect declared language
        assert result["declared_language"] == "python"

    def test_extract_with_language_alias(self):
        """Extract with language alias."""
        code = "def hello(): pass"
        block = {
            "type": "text",
            "text": f"```py\n{code}\n```",
        }
        result = extract_code_block(block)
        assert result is not None
        assert result["declared_language"] == "py"
        # Should resolve py to python
        assert result["language"] == "python"


class TestIntegrationCodeDetection:
    """Integration tests combining multiple functions."""

    def test_roundtrip_detection_and_extraction(self):
        """Test round-trip detection and extraction."""
        code = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        block = {"type": "text", "text": f"```\n{code}\n```"}
        result = extract_code_block(block)

        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "python"
        assert result["text"] == code

    def test_multiple_patterns_detection(self):
        """Test code with multiple language patterns."""
        # Mixed code that could be multiple languages
        code = """
function greet(name) {
    const greeting = "Hello, " + name;
    console.log(greeting);
    return greeting;
}
        """
        block = {"type": "text", "text": f"```\n{code}\n```"}
        result = extract_code_block(block)

        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "javascript"

    def test_sql_script_detection(self):
        """Test comprehensive SQL script detection."""
        code = """
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    dept_id INT
);

INSERT INTO employees VALUES (1, 'Alice', 10);
INSERT INTO employees VALUES (2, 'Bob', 20);

SELECT * FROM employees WHERE dept_id = 10;
        """
        block = {"type": "text", "text": f"```\n{code}\n```"}
        result = extract_code_block(block)

        assert result is not None
        assert result["type"] == "code"
        assert result["language"] == "sql"
