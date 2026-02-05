# Fuzzer Corpus

This directory contains seed inputs for the polylogue fuzzers.

## Structure

- `path_traversal/` - Path traversal attack patterns
- `fts5_injection/` - FTS5 query injection patterns
- `json_malformed/` - Malformed JSON inputs
- `timestamp_edge/` - Edge case timestamp inputs

## Usage

The corpus files are automatically loaded by atheris when running fuzzers directly:

```bash
python tests/fuzz/fuzz_path_sanitizer.py corpus/path_traversal/
```

For pytest-based fuzzing, the seed corpus is embedded in the test files.

## Adding New Corpus

When a fuzzer finds a crash, save the input to the appropriate subdirectory:

```bash
mkdir -p corpus/path_traversal
echo -n "../../../etc/passwd" > corpus/path_traversal/crash_001.txt
```
