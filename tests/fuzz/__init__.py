"""Fuzzing tests for security-critical polylogue functions.

These tests use atheris (Google's Python fuzzer) to find crashes and
security issues in security-critical code paths.

Usage:
    # Run all fuzzers with default settings
    pytest tests/fuzz/ -v --tb=short

    # Run a specific fuzzer
    python tests/fuzz/fuzz_path_sanitizer.py

    # Run with extended iteration count
    FUZZ_ITERATIONS=100000 python tests/fuzz/fuzz_fts5_escape.py

Targets:
    1. Path sanitizer - path traversal attacks
    2. escape_fts5_query() - FTS injection
    3. Importer parse() functions - malformed JSON
    4. Timestamp parsing - DoS via slow regex
"""
