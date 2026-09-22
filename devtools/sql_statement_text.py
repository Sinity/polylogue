"""Reconstruct the statement texts an ``execute`` call site can pass to SQLite.

This is the shared seam under every statement-grain SQL census in ``devtools``.
A census that greps for ``UPDATE <table>`` cannot see an ``INSERT OR REPLACE``,
an ``ON CONFLICT ... DO UPDATE``, an interpolated table name, or a predicate
that one code path omits. Reconstructing the text from the AST can see all of
them, and -- the part that matters for a census rather than a linter -- it can
report *every* text a call site can take, so a fragment that is sometimes the
empty string shows up as two candidate statements instead of one.

Three properties the callers depend on:

**Multi-candidate.** :func:`statement_texts` returns a tuple, not a string. An
interpolated name bound to two different literals in two branches yields both.
A ``for table in ("a", "b")`` loop target yields one candidate per table.

**Holes are explicit.** An interpolation this module cannot resolve is rendered
as :data:`HOLE` (``{}``) rather than dropped. A caller can therefore tell
"resolved, and there is no bound parameter here" apart from "could not tell",
and report the second as its own population instead of silently counting it as
the first.

**Empty means "not a string expression".** An empty result is the statement of
a different fact from a resolved-but-holey one: the first argument was not a
string expression at all (a dispatcher request object, a prepared-statement
handle), so no SQL census should record the call site.

``devtools/durable_write_census.py`` (polylogue-6kur AC4, durable tiers) carries
its own copy of this reconstruction because it landed first; migrating it onto
this module is the follow-up that removes the duplication.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping

__all__ = [
    "HOLE",
    "SQL_EXECUTION_METHODS",
    "function_scopes",
    "statement_texts",
    "string_values",
]

#: Rendering of an interpolation this module cannot resolve.
HOLE = "{}"

#: Cursor methods whose first argument is a SQL statement.
SQL_EXECUTION_METHODS = frozenset({"execute", "executemany", "executescript"})

#: Cap on candidate expansion, so one call site interpolating several
#: multi-valued names cannot turn into a combinatorial census.
_EXPANSION_LIMIT = 16

#: Cap on the candidate set kept per name, for the same reason.
_VALUE_LIMIT = 8


def statement_texts(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Every statement text *expression* can evaluate to, with holes rendered."""
    if isinstance(expression, ast.Constant):
        return (expression.value,) if isinstance(expression.value, str) else ()
    if isinstance(expression, ast.Name):
        return values.get(expression.id, ())
    if isinstance(expression, ast.JoinedStr):
        parts: list[tuple[str, ...]] = []
        for value in expression.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append((value.value,))
            elif isinstance(value, ast.FormattedValue):
                parts.append(statement_texts(value.value, values) or (HOLE,))
            else:
                parts.append((HOLE,))
        return _cross(parts)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = statement_texts(expression.left, values) or (HOLE,)
        right = statement_texts(expression.right, values) or (HOLE,)
        if left == (HOLE,) and right == (HOLE,):
            return ()
        return _cross([left, right])
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mod):
        return tuple(_percent_holes(text) for text in statement_texts(expression.left, values))
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        # ``"?" * len(ids)`` -- the other placeholder idiom. One repetition is
        # enough: callers ask whether a bound parameter appears, not how many.
        for side in (expression.left, expression.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, str):
                return (side.value,)
        return ()
    if isinstance(expression, ast.IfExp):
        return statement_texts(expression.body, values) + statement_texts(expression.orelse, values)
    if isinstance(expression, ast.Call):
        return _call_texts(expression, values)
    return ()


def _call_texts(expression: ast.Call, values: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    func = expression.func
    if isinstance(func, ast.Attribute) and func.attr == "format":
        return tuple(_brace_holes(text) for text in statement_texts(func.value, values))
    if isinstance(func, ast.Attribute) and func.attr == "join" and expression.args:
        separators = statement_texts(func.value, values)
        if not separators:
            return ()
        joined = _sequence_texts(expression.args[0], values)
        if joined is not None:
            return tuple(separator.join(joined) for separator in separators)
        # ``", ".join("?" for _ in values)`` -- the placeholder idiom. The
        # element is a constant, so the result is that constant repeated an
        # unknown number of times. One element is enough for every caller
        # here: they ask whether a bound parameter appears, not how many.
        repeated = _repeated_element(expression.args[0])
        if repeated is not None:
            return tuple(dict.fromkeys(repeated for _ in separators))
        return ()
    if isinstance(func, ast.Attribute) and func.attr == "dedent" and expression.args:
        return statement_texts(expression.args[0], values)
    if isinstance(func, ast.Name) and func.id == "dedent" and expression.args:
        return statement_texts(expression.args[0], values)
    return ()


def _repeated_element(expression: ast.AST) -> str | None:
    """The constant string an unknown-length repetition yields.

    Both spellings of the placeholder idiom land here: ``"?" for _ in ids``
    inside a ``join``, and ``"?" * len(ids)``, whose characters ``join`` then
    walks. One element is enough for every caller.
    """
    if isinstance(expression, ast.GeneratorExp | ast.ListComp):
        element = expression.elt
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            return element.value
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        for side in (expression.left, expression.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, str):
                return side.value
    return None


def _cross(parts: list[tuple[str, ...]]) -> tuple[str, ...]:
    combined: list[str] = [""]
    for options in parts:
        candidates = options or (HOLE,)
        if len(combined) * len(candidates) > _EXPANSION_LIMIT:
            candidates = (HOLE,)
        combined = [prefix + option for prefix in combined for option in candidates]
    return tuple(dict.fromkeys(combined))


def _sequence_texts(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> list[str] | None:
    if isinstance(expression, ast.List | ast.Tuple):
        out: list[str] = []
        for element in expression.elts:
            resolved = statement_texts(element, values)
            out.append(resolved[0] if len(resolved) == 1 else HOLE)
        return out
    return None


def _brace_holes(text: str) -> str:
    return re.sub(r"\{[^{}]*\}", HOLE, text)


def _percent_holes(text: str) -> str:
    return re.sub(r"%[-#0 +]*[0-9*]*(?:\.[0-9*]+)?[hlL]?[a-zA-Z%]", HOLE, text)


def _literal_string_sequence(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Every string a literal sequence -- or a name bound to one -- can yield."""
    if isinstance(expression, ast.Name):
        return values.get(f"[]{expression.id}", ())
    if isinstance(expression, ast.List | ast.Tuple | ast.Set):
        out: list[str] = []
        for element in expression.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                out.append(element.value)
            else:
                return ()
        return tuple(out)
    return ()


def string_values(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Resolve string-valued names to the *union* of what they can hold.

    The union rather than the last assignment is the load-bearing choice. A
    corrective sweep that takes an optional scope writes its predicate as
    ``clause = ""`` on one path and ``clause = "AND id IN (?)"`` on another;
    keeping only the second hides the unscoped statement completely. Three
    passes reach a fixpoint on this tree.

    Two namespaces share the mapping: a bare name resolves to the texts it can
    hold, and ``[]<name>`` resolves to the members of a literal string sequence
    it is bound to, which is what lets a ``for table in (...)`` target expand.
    """
    values: dict[str, tuple[str, ...]] = {}

    def record(name: str, resolved: tuple[str, ...]) -> None:
        if not resolved:
            return
        merged = tuple(dict.fromkeys(values.get(name, ()) + resolved))
        values[name] = merged[:_VALUE_LIMIT]

    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                record(name, statement_texts(node.value, values))
                members = _literal_string_sequence(node.value, values)
                if members:
                    values[f"[]{name}"] = members
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                record(node.target.id, statement_texts(node.value, values))
                members = _literal_string_sequence(node.value, values)
                if members:
                    values[f"[]{node.target.id}"] = members
            elif isinstance(node, ast.For | ast.AsyncFor) and isinstance(node.target, ast.Name):
                members = _literal_string_sequence(node.iter, values)
                if members:
                    values[node.target.id] = members
    return values


def function_scopes(tree: ast.Module) -> dict[ast.AST, str]:
    """Map every node to its qualified enclosing class/function scope name."""
    mapping: dict[ast.AST, str] = {}

    def walk(node: ast.AST, stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*stack, child.name))
            else:
                mapping[child] = ".".join(stack) or "<module>"
                walk(child, stack)

    walk(tree, ())
    return mapping
