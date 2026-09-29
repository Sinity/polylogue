"""Storage Atlas pointers must lead to the declarations they describe (F006/F021)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("paragraph_start", "source_path", "required_text"),
    [
        (
            "- `sessions.session_id`",
            "polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py",
            ("SESSIONS_SPEC =", "session_id", "origin || ':' || native_id", "STORED UNIQUE"),
        ),
        (
            "1. Commit one generation",
            "polylogue/storage/sqlite/archive_tiers/source.py",
            ("CREATE TABLE IF NOT EXISTS gc_generation_members", "intent_committed_at_ms", "outcome = 'pending'"),
        ),
        (
            "- GC history counters",
            "polylogue/storage/sqlite/archive_tiers/source.py",
            ("CREATE TABLE IF NOT EXISTS gc_generation_members", "outcome_at_ms", "outcome = 'pending'"),
        ),
    ],
)
def test_storage_atlas_schema_pointer_matches_its_subject(
    paragraph_start: str, source_path: str, required_text: tuple[str, ...]
) -> None:
    paragraph = next(
        line for line in (_ROOT / "docs/atlas/storage.md").read_text().splitlines() if line.startswith(paragraph_start)
    )
    match = re.search(re.escape(source_path) + r":(\d+)-(\d+)", paragraph)
    assert match is not None, paragraph
    start, end = map(int, match.groups())
    source_lines = (_ROOT / source_path).read_text().splitlines()
    assert 1 <= start <= end <= len(source_lines)
    cited = "\n".join(source_lines[start - 1 : end])
    for text in required_text:
        assert text in cited, (paragraph, text, cited)


def test_storage_atlas_writer_pointer_names_the_actual_choke_point() -> None:
    paragraph = next(
        line
        for line in (_ROOT / "docs/atlas/storage.md").read_text().splitlines()
        if line.startswith("- `write_parsed_session_to_archive`")
    )
    match = re.search(r"function `([^`]+)` in `([^`]+\.py)`", paragraph)
    assert match is not None, paragraph
    symbol, path = match.groups()
    function = next(
        node
        for node in ast.parse((_ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    )
    assert [arg.arg for arg in function.args.args][:2] == ["conn", "session"]
    defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs), function.args.kw_defaults, strict=True))
    assert isinstance(defaults["manage_transaction"], ast.Constant)
    assert defaults["manage_transaction"].value is True
