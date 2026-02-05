# Fuzzing and Mutation Testing

This document describes the fuzzing and mutation testing infrastructure for polylogue.

## Fuzzing Infrastructure

Polylogue includes fuzzing tests targeting security-critical functions. The fuzzers use seed corpus testing via pytest, with optional atheris integration for extended fuzzing.

### Location

```
tests/fuzz/
├── __init__.py
├── fuzz_path_sanitizer.py   # Path traversal attacks
├── fuzz_fts5_escape.py      # FTS5 injection
├── fuzz_json_importers.py   # Malformed JSON handling
├── fuzz_timestamp.py        # Timestamp parsing DoS
└── corpus/
    └── README.md
```

### Running Fuzz Tests

```bash
# Run all fuzz tests via pytest
uv run pytest tests/fuzz/ -v

# Run individual fuzzer
uv run pytest tests/fuzz/fuzz_path_sanitizer.py -v

# Run with extended iterations (if atheris is installed)
FUZZ_ITERATIONS=100000 python tests/fuzz/fuzz_path_sanitizer.py
```

### Targets

| Fuzzer | Target | Security Property |
|--------|--------|-------------------|
| `fuzz_path_sanitizer.py` | `ParsedAttachment.sanitize_path()` | Path traversal prevention |
| `fuzz_fts5_escape.py` | `escape_fts5_query()` | FTS5 injection prevention |
| `fuzz_json_importers.py` | `chatgpt.parse()`, `codex.parse()`, etc. | Crash resistance |
| `fuzz_timestamp.py` | `parse_timestamp()`, `normalize_timestamp()` | ReDoS prevention |

### Adding New Fuzz Tests

1. Create a new file `tests/fuzz/fuzz_<target>.py`
2. Define a `fuzz_<target>(data: bytes)` function
3. Add security assertions (invariants that must hold)
4. Create a pytest test class with seed corpus
5. Optionally add atheris standalone runner

Example structure:

```python
def fuzz_my_target(data: bytes) -> None:
    """Fuzz description."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        result = my_function(text)
        # Assert security invariants
        assert "dangerous_pattern" not in result
    except (ValueError, TypeError):
        pass  # Acceptable rejections


class TestMyTargetFuzz:
    CORPUS = [b"safe_input", b"malicious_input"]

    @pytest.mark.parametrize("data", CORPUS)
    def test_corpus(self, data: bytes):
        fuzz_my_target(data)
```

## Mutation Testing

Mutation testing verifies that tests actually detect bugs by introducing small changes (mutations) and checking if tests fail.

### Configuration

See `mutmut.toml` for configuration. Key settings:

- **paths_to_mutate**: Security-critical modules
- **target**: >80% mutation kill rate

### Running Mutation Tests

```bash
# Install mutmut via uvx (temporary)
uvx mutmut run

# View results
uvx mutmut results

# Show surviving mutants
uvx mutmut show <id>
```

### Interpreting Results

- **Killed**: Test suite detected the mutation (good)
- **Survived**: Mutation went undetected (need more tests)
- **Timeout**: Test took too long (often indicates infinite loop)
- **Incompetent**: Mutation caused syntax error (skip)

### Target Modules

The mutation testing focuses on security-critical modules:

1. `polylogue/importers/base.py` - Path sanitization
2. `polylogue/storage/search.py` - FTS5 query escaping
3. `polylogue/core/timestamps.py` - Timestamp parsing
4. `polylogue/core/hashing.py` - Content hashing

## CI Integration

Fuzz tests run as part of the normal pytest suite:

```yaml
- name: Run tests
  run: uv run pytest tests/ -v
```

Mutation testing is optional and runs separately due to time requirements:

```yaml
- name: Mutation testing (optional)
  run: uvx mutmut run --max-children=4
  continue-on-error: true
```

## Findings

### FTS5 Escape Edge Cases

The fuzzer identified edge cases where `escape_fts5_query()` doesn't prevent FTS5 syntax errors:

- Single quotes (`'test'`)
- Backslashes (`test\escape`)
- Semicolons (`test;`)
- Percent signs (`test%`)

These are documented in `test_fts5_escape_edge_cases` with `xfail` markers. They do not represent SQL injection vulnerabilities (parameterized queries prevent that), but may cause FTS5 search failures.

### Path Sanitizer Coverage

The path sanitizer correctly handles:
- Directory traversal (`../../../etc/passwd`)
- Null byte injection (`file.txt\x00.jpg`)
- Control characters
- Absolute path normalization
- Symlink detection
