"""Source fingerprints parse each file revision once and still move on an edit."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.sources import origin_specs


@pytest.fixture
def isolated_memo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    origin_specs._invalidate_source_signatures()
    yield tmp_path
    origin_specs._invalidate_source_signatures()


def _fingerprint(paths: list[Path], namespace: str) -> str:
    signatures = tuple(origin_specs._source_signature(path) for path in paths)
    return origin_specs._fingerprint_sources_compute(signatures, namespace)


def test_namespaces_sharing_a_file_parse_it_once(isolated_memo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: parsing per namespace again makes the shared file parse twice."""
    shared = isolated_memo / "shared.py"
    shared.write_text("def f():\n    return 1\n", encoding="utf-8")
    own = isolated_memo / "own.py"
    own.write_text("VALUE = 2\n", encoding="utf-8")
    parsed: list[str] = []
    real_parse = ast.parse

    def counting_parse(source: str) -> ast.Module:
        parsed.append(source)
        return real_parse(source)

    monkeypatch.setattr(ast, "parse", counting_parse)
    first = _fingerprint([shared, own], "first")
    second = _fingerprint([shared], "second")

    assert len(parsed) == 2
    assert first != second


def test_semantic_edit_moves_the_fingerprint_and_docstring_edit_does_not(isolated_memo: Path) -> None:
    source = isolated_memo / "module.py"
    source.write_text('def f():\n    """Old."""\n    return 1\n', encoding="utf-8")
    original = _fingerprint([source], "edit")

    source.write_text('def f():\n    """New wording."""\n    return 1\n', encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    assert _fingerprint([source], "edit") == original

    source.write_text('def f():\n    """New wording."""\n    return 2\n', encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    assert _fingerprint([source], "edit") != original
