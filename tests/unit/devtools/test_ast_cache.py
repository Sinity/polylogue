"""The gate AST cache parses a file once and walks exactly what ``ast.walk`` walks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from devtools import ast_cache


def test_a_path_is_parsed_once_and_walked_in_ast_walk_order(tmp_path: Path) -> None:
    """Anti-vacuity: re-parse on every call and the two trees differ in
    identity; walk in another order and the node sequence differs."""
    source = tmp_path / "module.py"
    source.write_text("def f(x):\n    return [y for y in x if y]\n", encoding="utf-8")

    tree = ast_cache.parse_path(source)

    assert ast_cache.parse_path(source) is tree
    assert [type(node) for node in ast_cache.walk_module(tree)] == [type(node) for node in ast.walk(tree)]
    assert ast_cache.walk_module(tree) is ast_cache.walk_module(tree)


def test_a_rewritten_file_is_parsed_again(tmp_path: Path) -> None:
    """Anti-vacuity: key the cache on the path alone and the census reads the
    first version of a file a test rewrote."""
    source = tmp_path / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")
    first = ast_cache.parse_path(source)
    source.write_text("y = 2\n", encoding="utf-8")

    second = ast_cache.parse_path(source)

    assert second is not first
    assert [node.id for node in ast_cache.walk_module(second) if isinstance(node, ast.Name)] == ["y"]


def test_a_parse_failure_is_raised_again_rather_than_cached(tmp_path: Path) -> None:
    """Callers that record unreadable files must see the error on every call."""
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")

    for _attempt in range(2):
        with pytest.raises(SyntaxError):
            ast_cache.parse_path(broken)


def test_a_tree_the_cache_does_not_own_is_walked_but_not_memoized() -> None:
    tree = ast.parse("x = 1\n")

    assert ast_cache.walk_module(tree) == tuple(ast.walk(tree))
    assert id(tree) not in ast_cache._NODES


def test_source_and_tree_come_from_the_same_read(tmp_path: Path) -> None:
    """A handler digest slices the text its tree was parsed from.

    Anti-vacuity: return a cached tree with a fresh read of the file (the old
    two-read shape) and, after the rewrite, the text holds ``b`` while the
    tree still holds ``a``.
    """
    source_path = tmp_path / "module.py"
    source_path.write_text("a = 1\n", encoding="utf-8")
    ast_cache.parse_source(source_path)
    source_path.write_text("b = 2\n", encoding="utf-8")

    source, tree = ast_cache.parse_source(source_path)

    names = [node.id for node in ast_cache.walk_module(tree) if isinstance(node, ast.Name)]
    assert source == "b = 2\n"
    assert names == ["b"]


def test_a_released_path_holds_no_tree(tmp_path: Path) -> None:
    """Anti-vacuity: make ``release`` a no-op and the second parse returns the
    held tree, so a package pass would keep every module it visited."""
    source = tmp_path / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")
    first = ast_cache.parse_path(source)

    ast_cache.release(source)

    assert ast_cache.parse_path(source) is not first
