"""Parse each Python source once per gate process, and walk each module once.

The layering gate runs several independent AST censuses (imports, writer
modules, durable writes, derived sweeps, SQLite degradation sites) over the
same package. Each used to read, parse and ``ast.walk`` every file itself, some
of them several times over, which made the gate the slowest step of the quick
tier. A parse is reused only for byte-identical source -- the key is the file's
content digest, not its path alone -- so a file rewritten within one process,
as tests do, is parsed again. Nothing is persisted.

Failures are not cached: a caller that handles ``SyntaxError`` or a read error
sees the same exception it saw when it parsed the file itself.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

#: The latest parse of each path, with the digest and text it came from.
_TREES: dict[Path, tuple[bytes, str, ast.Module]] = {}
#: ``ast.walk`` order of each cached tree. Each entry holds its tree, so the
#: ``id`` key cannot be reused by another object while the entry exists.
_NODES: dict[int, tuple[ast.AST, tuple[ast.AST, ...]]] = {}


def read_source(path: Path) -> str:
    """Return the UTF-8 text of *path*."""
    return path.read_text(encoding="utf-8")


def parse_source(path: Path) -> tuple[str, ast.Module]:
    """Return *path*'s text and its parse, both from one read of the file.

    A caller that slices source segments out of the tree needs the text the
    tree was parsed from; reading the file a second time could pair a new
    tree with old text.
    """
    source = read_source(path)
    digest = hashlib.blake2b(source.encode("utf-8"), digest_size=16).digest()
    key = path.resolve()
    cached = _TREES.get(key)
    if cached is not None and cached[0] == digest:
        return cached[1], cached[2]
    tree = ast.parse(source)
    if cached is not None:
        _NODES.pop(id(cached[2]), None)
    _TREES[key] = (digest, source, tree)
    _NODES[id(tree)] = (tree, ())
    return source, tree


def parse_path(path: Path) -> ast.Module:
    """Return the parsed module for *path*, reusing the parse of identical source."""
    return parse_source(path)[1]


def walk_module(tree: ast.AST) -> tuple[ast.AST, ...]:
    """Return ``tuple(ast.walk(tree))``, computed once for a tree this cache parsed."""
    memo = _NODES.get(id(tree))
    if memo is None or memo[0] is not tree:
        return tuple(ast.walk(tree))
    if not memo[1]:
        memo = (tree, tuple(ast.walk(tree)))
        _NODES[id(tree)] = memo
    return memo[1]


__all__ = ["parse_path", "parse_source", "read_source", "walk_module"]
