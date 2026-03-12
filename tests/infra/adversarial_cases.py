"""Shared deterministic adversarial corpora for security and fuzz tests."""

from __future__ import annotations

PATH_TRAVERSAL_STRINGS: tuple[str, ...] = (
    "../../../etc/passwd",
    "..\\..\\windows\\system32",
    "foo/../../../bar",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "~/.ssh/id_rsa",
    "$HOME/.bashrc",
    "file.txt\x00.jpg",
    "../\x00",
    ".hidden/../../../etc/passwd",
    "%2e%2e/",
    "..%2f",
    "..%c0%af",
    "a" * 1000 + "/../etc/passwd",
    "valid_file.txt",
    "/tmp/safe/path",
    "symlink/../../../etc/passwd",
    "..%252f",
)

PATH_TRAVERSAL_CORPUS_BYTES: tuple[bytes, ...] = (
    b"../../../etc/passwd",
    b"..\\..\\windows\\system32",
    b"foo/../../../bar",
    b"/etc/passwd",
    b"C:\\Windows\\System32",
    b"~/.ssh/id_rsa",
    b"$HOME/.bashrc",
    b"file.txt\x00.jpg",
    b"../\x00",
    b".hidden/../../../etc/passwd",
    b"%2e%2e/",
    b"..%2f",
    b"..%c0%af",
    b"a" * 10000 + b"/../etc/passwd",
    b"\x00\x01\x02\x03\x04\x05",
    b"\x7f\x80\x81\x82",
    b"valid_file.txt",
    b"/tmp/safe/path",
    b"symlink/../../../etc/passwd",
    b"..%252f",
)

SYMLINK_PATH_STRINGS: tuple[str, ...] = (
    "/tmp/link/../../../etc/passwd",
    "symlink/../secret",
    "./link/./link/../target",
)

CONTROL_CHAR_FILENAMES: tuple[str, ...] = (
    "file\x00name.txt",
    "file\nname.txt",
    "file\rname.txt",
    "file\tname.txt",
)

DOTS_ONLY_FILENAMES: tuple[str, ...] = (".", "..", "...", "....")
RESERVED_FILENAMES: tuple[str, ...] = ("CON", "PRN", "AUX", "NUL", "COM1", "LPT1")

SQL_INJECTION_PAYLOADS: tuple[str, ...] = (
    "'; DROP TABLE messages; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    "'; DELETE FROM conversations; --",
    "' UNION SELECT * FROM sqlite_master --",
    "test OR 1=1",
    'test" OR "1"="1',
    "test AND 1=1",
    "* OR MATCH",
    "NEAR(test, 5)",
    "NOT test",
    "test OR test",
    "test AND NOT test",
    "*",
    "te*st",
    "test*",
    "text:test",
    "role:user",
    '"test"',
    "'test'",
    "test; SELECT",
    "test/* comment */",
    "test\u0000",
    "test\u001f",
)

FTS5_OPERATORS: tuple[str, ...] = ("AND", "OR", "NOT", "NEAR", "MATCH")

FTS5_ESCAPE_SECURITY_CASES: tuple[tuple[str, str, bool], ...] = (
    ("test OR anything", "test OR anything", True),
    ("test AND something", "test AND something", True),
    ("test NOT anything", "test NOT anything", True),
    ("word1 NEAR word2", "word1 NEAR word2", True),
    ("test*", '"test*"', True),
    ("test?", "test?", False),
    ("*", '""', True),
    ("?", "?", False),
    ('test"something"', '"test""something"""', True),
    ("semicolon;", '"semicolon;"', True),
    ("underscore_wildcard", "underscore_wildcard", True),
)
