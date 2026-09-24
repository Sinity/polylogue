"""Statement-grain census of every route that can rewrite a durable archive row.

Gate classification: **blocking architectural boundary check**, enforced as part
of ``devtools gate layering``.

Why this exists
---------------

``polylogue-6kur`` AC4 reads "current writer defects have owning producer fixes
and cannot be masked by backfills". Two prior audits established the *masking*
half for two named residuals and both recorded the same limitation in their own
words: the denominator was never enumerated, so nobody may read AC4 as a proof
of absence. The blocking evidence gap was that the durable-``UPDATE`` census was
grep-literal -- it matched ``UPDATE <table>`` against a generated durable table
list and therefore could not see ``INSERT OR REPLACE``, ``ON CONFLICT ... DO
UPDATE``, interpolated table names, or a correction expressed as
delete-then-reinsert.

This module replaces that grep with a statement-grain AST census, and the
declaration it checks classifies every site. The two censuses beside it answer
different questions and neither subsumes this one:

``layering``'s writer-module census
    is *file*-grain. It proves which modules execute DML. New DML added inside
    an already-censused module is invisible to it -- and 26 of the durable
    rewrite sites here live in modules that census already lists.

``rebuild-routes``
    censuses who can *reach* a rebuild entrypoint. A direct ``UPDATE`` against a
    durable table reaches no rebuild entrypoint at all.

What counts as a durable rewrite
--------------------------------

Only statements that can change or remove a row that is *already* durable:

``update``
    ``UPDATE <durable table>``.
``delete``
    ``DELETE FROM <durable table>`` -- the delete half of a delete-then-reinsert
    correction, which the grep census could not see.
``insert_or_replace``
    ``INSERT OR REPLACE INTO`` / ``REPLACE INTO``, which silently destroys the
    conflicting row.
``upsert_do_update``
    ``INSERT ... ON CONFLICT ... DO UPDATE``, which rewrites the conflicting row.

Deliberately excluded, with the reason: a plain ``INSERT INTO`` and an ``INSERT
OR IGNORE`` cannot overwrite a stored value -- the first raises on conflict and
the second is a no-op -- so neither can mask a producer. ``INSERT OR ABORT``,
``OR FAIL`` and ``OR ROLLBACK`` are likewise refusals, not rewrites.

What this census cannot see, stated plainly
-------------------------------------------

Static analysis resolves a statement only when its text is reconstructible from
the module. Three residues are therefore reported as *their own declared
populations* rather than silently dropped, which is what turns a blind spot into
a reviewed entry:

``dynamic_table``
    the verb resolved but the table is an interpolation hole (``f"UPDATE
    {table} ..."``). The site is censused with ``table: "?"``; it must be
    declared even though its tier cannot be proven.
``caller_supplied``
    the executed statement is a parameter of the enclosing function, so the real
    SQL lives at each call site. These helpers are censused by name so that a new
    one cannot appear unreviewed, and the callers' literals are censused
    normally.

What remains genuinely invisible to *this* pass: a statement assembled through a
data structure, a registry, or a call whose return value this module does not
constant-fold. That residue is what the runtime authorizer cross-check in
``tests/unit/storage/test_durable_write_authorizer_census.py`` exists to cover --
it observes what SQLite actually compiles on an exercised production route, so
it sees dynamic SQL by construction rather than by pattern.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER

#: The declaration this census is checked against.
DECLARATION_PATH = "docs/plans/durable-write-census.yaml"

#: Tiers whose rows are durable: losing or corrupting one is not recoverable by
#: reconvergence. ``index``/``embeddings``/``ops`` are rebuildable or disposable
#: and are out of this census's subject by the tier table in ``CLAUDE.md``.
DURABLE_TIERS: frozenset[str] = frozenset({"source", "user", "audit"})

#: Placeholder table name for a statement whose table is an interpolation hole.
UNRESOLVED_TABLE = "?"

#: How a durable rewrite is allowed to be legitimate. Every token but the last
#: records a human judgement review has to accept; ``masking_backfill`` is the
#: defect AC4 names and declaring it is itself a gate failure, so the defect can
#: never be parked in the declaration as though it were a classification.
CLASSIFICATION_VOCABULARY: dict[str, str] = {
    "lifecycle_transition": (
        "a row advances through its own declared states; the new value is not a correction of a wrong old value"
    ),
    "guarded_producer_refinement": (
        "inside the producer, guarded so it can only move a NULL or a declared "
        "placeholder to a real value -- never overwrite a confident one"
    ),
    "deliberate_redaction": "operator-invoked destruction of content, not correction of a producer's output",
    "two_phase_recovery": "crash recovery for a two-phase commit against checksum-validated prepared evidence",
    "finite_actuator": (
        "a campaign-scoped actuator with a named deletion trigger, reached only through an audited operator mutation"
    ),
    "declared_retention": (
        "removal under a declared retention or garbage-collection policy; it drops rows it is allowed to drop and "
        "never rewrites a retained one"
    ),
    "row_supersession": (
        "the durable row is replaced wholesale by a re-acquisition of the same "
        "evidence, keyed by content, so no stored judgement is corrected"
    ),
    "no_effect_lock_upgrade": (
        "the exact zero-row UPDATE used only to acquire SQLite's RESERVED lock before a caller-owned read/decision"
    ),
    "rebuildable_dynamic_target": (
        "a dynamic-table helper whose currently reachable literal table targets are all canonical rebuildable tiers"
    ),
    "index_foreign_key_cleanup": (
        "a bounded foreign-key cleanup over the active index connection while bulk ingest temporarily disables FKs"
    ),
    "test_or_fixture_construct": "a controlled mutant or fixture seeding a throwaway archive, not a route over an operator archive",
    "unclassified": (
        "observed and pinned by the census, not yet adjudicated. This token is the "
        "ratchet's floor: it stops a NEW durable rewrite from appearing unnoticed "
        "without asserting that the existing site is legitimate. Replacing it with a "
        "real classification is the adjudication pass; leaving it is honest, but it is "
        "not evidence that the site is not a masking backfill"
    ),
    "masking_backfill": (
        "an after-the-fact correction of a value a producer got wrong -- the defect AC4 forbids. "
        "Declaring this token fails the gate by design; fix the producer instead"
    ),
}

#: Classifications whose presence in the declaration is itself a violation.
FORBIDDEN_CLASSIFICATIONS: frozenset[str] = frozenset({"masking_backfill"})

_SQL_EXECUTION_METHODS = frozenset({"execute", "executemany", "executescript"})

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)

# Runtime DDL needs a second population.  ``durable_table_tiers`` intentionally
# remains a map of *canonical* archive ownership: treating every ``CREATE
# TABLE`` encountered in product code as a source/user/audit table would invent
# a tier for private shards and external registries.  The narrower check below
# instead notices a persistent relation whose creator and rewrite both live in
# one runtime path, and asks the gate to reject that unowned relation.
_RUNTIME_CREATE_RE = re.compile(
    r"\bCREATE\s+(?P<temporary>TEMP(?:ORARY)?\s+)?(?:VIRTUAL\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(?P<table>[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)

#: Rewrite verbs. ``INSERT INTO`` is matched so that an ``ON CONFLICT ... DO
#: UPDATE`` tail can promote it; on its own it is dropped.
#:
#: The optional ``schema.`` qualifier is part of the table reference, not part
#: of the table name. Without it the pattern bound ``table`` to the ATTACH
#: alias -- ``UPDATE user_tier.assertions`` resolved to the table
#: ``user_tier``, which no tier declares, so three durable ``user.db``
#: rewrites in ``storage/derived/feedback`` were dropped by the very census
#: that exists to enumerate them.
_REWRITE_RE = re.compile(
    r"\b(?P<verb>UPDATE\s+OR\s+\w+|UPDATE|DELETE\s+FROM|INSERT\s+OR\s+REPLACE\s+INTO|REPLACE\s+INTO|INSERT\s+INTO)"
    r"\s+(?:[`\"\[]?(?P<schema>[A-Za-z_][A-Za-z0-9_]*)[`\"\]]?\s*\.\s*)?"
    r"[`\"\[]?(?P<table>[A-Za-z_][A-Za-z0-9_]*|\{\})",
    re.IGNORECASE,
)
_DO_UPDATE_RE = re.compile(r"ON\s+CONFLICT\b.*?\bDO\s+UPDATE\b", re.IGNORECASE | re.DOTALL)
_NO_EFFECT_LOCK_UPGRADE_RE = re.compile(
    r"^\s*UPDATE\s+assertions\s+SET\s+updated_at_ms\s*=\s+updated_at_ms\s+WHERE\s+0\s*;?\s*$",
    re.IGNORECASE,
)

#: An interpolation hole is rendered as this token so a verb can still resolve.
_HOLE = "{}"


@dataclass(frozen=True)
class WriteSite:
    """One statement-grain durable rewrite, keyed so line drift cannot churn it."""

    file: str
    function: str
    table: str
    kind: str
    line: int
    tier: str

    @property
    def key(self) -> str:
        return f"{self.file}::{self.function}::{self.table}::{self.kind}"


@dataclass(frozen=True)
class RuntimeTableCreation:
    """A relation created directly by runtime code rather than canonical DDL.

    ``disposition`` deliberately describes the creation context, not a storage
    tier.  ``temporary`` and ``scratch`` are bounded non-archive relations;
    only ``persistent`` can be a missing durable schema declaration.
    """

    file: str
    function: str
    table: str
    disposition: str
    line: int

    @property
    def key(self) -> str:
        return f"{self.file}::{self.function}::{self.table}::{self.disposition}"


@dataclass(frozen=True)
class DynamicTableTarget:
    """One statically resolved caller target of a ``table``-parameter helper."""

    helper_file: str
    helper_function: str
    table: str
    caller_file: str
    line: int

    @property
    def helper_key(self) -> str:
        return f"{self.helper_file}::{self.helper_function}"


@dataclass(frozen=True)
class RuntimeTableDeclaration:
    """An explicitly non-durable runtime relation outside canonical tier DDL."""

    key: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class HelperSite:
    """A function that executes SQL handed to it by its caller."""

    file: str
    function: str
    parameter: str
    line: int

    @property
    def key(self) -> str:
        return f"{self.file}::{self.function}::{self.parameter}"


@dataclass(frozen=True)
class CensusObservation:
    sites: tuple[WriteSite, ...]
    helpers: tuple[HelperSite, ...]
    runtime_creations: tuple[RuntimeTableCreation, ...]
    dynamic_targets: tuple[DynamicTableTarget, ...]
    index_foreign_key_cleanup_helpers: frozenset[str]


def durable_table_tiers() -> dict[str, str]:
    """Map every archive table name to its tier, from the live DDL."""
    tiers: dict[str, str] = {}
    for tier, ddl in ARCHIVE_DDL_BY_TIER.items():
        name = str(getattr(tier, "value", tier))
        for table in _CREATE_TABLE_RE.findall(ddl):
            tiers[table] = name
    return tiers


def _fragments(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Reconstruct a statement's text, rendering unresolved holes as ``{}``.

    Returns *every* text the expression can take. An interpolated table name
    bound by a ``for`` loop over a literal tuple -- the shape five derived-tier
    deletes in this tree use -- expands into one candidate per table, so the
    census proves their tier instead of recording them as unknowns. An empty
    result means "not a string expression at all", which is a different fact
    from a resolved-but-holey statement.
    """
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
                parts.append(_fragments(value.value, values) or (_HOLE,))
            else:
                parts.append((_HOLE,))
        return _cross(parts)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _fragments(expression.left, values) or (_HOLE,)
        right = _fragments(expression.right, values) or (_HOLE,)
        if left == (_HOLE,) and right == (_HOLE,):
            return ()
        return _cross([left, right])
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mod):
        return tuple(_percent_holes(text) for text in _fragments(expression.left, values))
    if isinstance(expression, ast.Call):
        func = expression.func
        # str.format / str.join / textwrap.dedent over a literal all keep the
        # verb and the table visible enough to classify.
        if isinstance(func, ast.Attribute) and func.attr == "format":
            return tuple(_brace_holes(text) for text in _fragments(func.value, values))
        if isinstance(func, ast.Attribute) and func.attr == "join" and expression.args:
            separators = _fragments(func.value, values)
            joined = _sequence_fragments(expression.args[0], values)
            if not separators or joined is None:
                return ()
            return tuple(separator.join(joined) for separator in separators)
        if isinstance(func, ast.Name) and func.id == "dedent" and expression.args:
            return _fragments(expression.args[0], values)
        if isinstance(func, ast.Attribute) and func.attr == "dedent" and expression.args:
            return _fragments(expression.args[0], values)
        return ()
    if isinstance(expression, ast.IfExp):
        return _fragments(expression.body, values) + _fragments(expression.orelse, values)
    return ()


#: Cap on candidate expansion, so a statement interpolating several multi-valued
#: names cannot turn one call site into a combinatorial census.
_EXPANSION_LIMIT = 32


def _cross(parts: list[tuple[str, ...]]) -> tuple[str, ...]:
    combined: list[str] = [""]
    for options in parts:
        if not options:
            options = (_HOLE,)
        if len(combined) * len(options) > _EXPANSION_LIMIT:
            options = (_HOLE,)
        combined = [prefix + option for prefix in combined for option in options]
    return tuple(dict.fromkeys(combined))


def _sequence_fragments(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> list[str] | None:
    if isinstance(expression, ast.List | ast.Tuple):
        out: list[str] = []
        for element in expression.elts:
            resolved = _fragments(element, values)
            out.append(resolved[0] if len(resolved) == 1 else _HOLE)
        return out
    return None


def _brace_holes(text: str) -> str:
    return re.sub(r"\{[^{}]*\}", _HOLE, text)


def _percent_holes(text: str) -> str:
    return re.sub(r"%[-#0 +]*[0-9*]*(?:\.[0-9*]+)?[hlL]?[a-zA-Z%]", _HOLE, text)


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


def _string_values(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Resolve string-valued names to a fixpoint (3 passes suffice).

    Two namespaces share the mapping: a bare name resolves to the statement
    texts it can hold, and ``[]<name>`` resolves to the members of a literal
    string sequence it is bound to, which is what lets a ``for table in (...)``
    target expand.
    """
    values: dict[str, tuple[str, ...]] = {}
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                resolved = _fragments(node.value, values)
                if resolved:
                    values[name] = resolved
                members = _literal_string_sequence(node.value, values)
                if members:
                    values[f"[]{name}"] = members
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                resolved = _fragments(node.value, values)
                if resolved:
                    values[node.target.id] = resolved
                members = _literal_string_sequence(node.value, values)
                if members:
                    values[f"[]{node.target.id}"] = members
            elif isinstance(node, ast.For | ast.AsyncFor) and isinstance(node.target, ast.Name):
                members = _literal_string_sequence(node.iter, values)
                if members:
                    values[node.target.id] = members
    return values


def _scopes(tree: ast.Module) -> dict[ast.AST, tuple[str, ast.AST | None]]:
    """Map every node to ``(qualified enclosing scope, enclosing function)``."""
    mapping: dict[ast.AST, tuple[str, ast.AST | None]] = {}

    def walk(node: ast.AST, stack: tuple[str, ...], function: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, (*stack, child.name), function)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*stack, child.name), child)
            else:
                mapping[child] = (".".join(stack) or "<module>", function)
                walk(child, stack, function)

    walk(tree, (), None)
    return mapping


def _string_parameter_names(function: ast.AST | None) -> frozenset[str]:
    """Parameters that a caller could hand a statement through.

    The annotation is required to be string-shaped. Without it
    ``OperationKernel(...).execute(request)`` -- a dispatcher, not a cursor --
    would enter the census as an unreadable SQL route, and a census that reports
    non-SQL calls is the kind of noise that gets a declaration rubber-stamped.
    An *unannotated* parameter is kept, because absence of a type is not
    evidence of a non-string.
    """
    if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
        return frozenset()
    arguments = function.args
    names: set[str] = set()
    candidates = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        candidates.append(arguments.vararg)
    for argument in candidates:
        if argument.annotation is None or _is_string_annotation(argument.annotation):
            names.add(argument.arg)
    return frozenset(names)


def _is_string_annotation(annotation: ast.AST) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in {"str", "LiteralString"}
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value.replace(" ", "").split("|")[0] in {"str", "LiteralString"}
    if isinstance(annotation, ast.Attribute):
        return annotation.attr in {"str", "LiteralString"}
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_string_annotation(annotation.left) or _is_string_annotation(annotation.right)
    return False


def _classify_statement(
    sql: str,
    table_tiers: Mapping[str, str],
    *,
    runtime_persistent_tables: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """Return ``(table, kind, tier)`` for every durable or runtime rewrite.

    ``runtime`` is not an archive tier.  It is a deliberately separate marker
    for a relation created by product code but absent from canonical tier DDL.
    It lets the caller emit a precise missing-schema finding without claiming
    the relation belongs to source, user, or audit.
    """
    found: list[tuple[str, str, str]] = []
    has_do_update = bool(_DO_UPDATE_RE.search(sql))
    for match in _REWRITE_RE.finditer(sql):
        verb = re.sub(r"\s+", " ", match.group("verb").upper())
        table = match.group("table")
        if verb.startswith("UPDATE"):
            kind = "no_effect_update" if _NO_EFFECT_LOCK_UPGRADE_RE.fullmatch(sql) else "update"
        elif verb == "DELETE FROM":
            kind = "delete"
        elif verb in {"INSERT OR REPLACE INTO", "REPLACE INTO"}:
            kind = "insert_or_replace"
        elif verb == "INSERT INTO" and has_do_update:
            kind = "upsert_do_update"
        else:
            # A plain INSERT cannot overwrite a stored value; see the module
            # docstring for why that is not an omission.
            continue
        if table == _HOLE:
            found.append((UNRESOLVED_TABLE, kind, "unresolved"))
            continue
        tier = table_tiers.get(table)
        if tier in DURABLE_TIERS:
            found.append((table, kind, str(tier)))
        elif table in runtime_persistent_tables:
            found.append((table, kind, "runtime"))
    return found


def _memory_connection_names(tree: ast.Module) -> frozenset[str]:
    """Return local connection names constructed as private in-memory SQLite."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "connect"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "sqlite3"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == ":memory:"
        ):
            names.add(node.targets[0].id)
    return frozenset(names)


def _runtime_table_creation(
    *,
    relative: str,
    function: str,
    receiver: ast.AST,
    statement: str,
    line: int,
    memory_connections: frozenset[str],
    canonical_tables: Mapping[str, str],
) -> RuntimeTableCreation | None:
    """Classify one directly executed CREATE TABLE without assigning a tier."""
    match = _RUNTIME_CREATE_RE.search(statement)
    if match is None:
        return None
    table = match.group("table")
    if table in canonical_tables:
        return None
    if match.group("temporary") is not None:
        disposition = "temporary"
    elif isinstance(receiver, ast.Name) and receiver.id in memory_connections:
        disposition = "scratch"
    else:
        disposition = "persistent"
    return RuntimeTableCreation(
        file=relative,
        function=function,
        table=table,
        disposition=disposition,
        line=line,
    )


def _is_archive_tier_runtime_module(relative: str) -> bool:
    """Whether direct runtime DDL can create one of the six archive tiers.

    Browser-capture registries and schema-observation journals intentionally
    use SQLite too, but their tables are not archive tiers.  The census is not
    a universal SQLite inventory; its bounded subject is runtime DDL in the
    archive-tier implementation where a missing canonical source/user/audit
    relation would otherwise be hidden.
    """
    return relative.startswith("polylogue/storage/sqlite/archive_tiers/")


def _function_table_parameters(
    parsed_modules: Iterable[
        tuple[Path, ast.Module, str, dict[str, tuple[str, ...]], dict[ast.AST, tuple[str, ast.AST | None]]]
    ],
    dynamic_sites: Iterable[WriteSite],
) -> dict[str, tuple[str, int | None]]:
    """Locate dynamic write helpers that explicitly accept a ``table`` argument.

    The result is keyed by the fully qualified helper identity used by
    ``WriteSite``.  A call site can then be checked against the same canonical
    DDL map as a non-dynamic statement, rather than trusting a prose claim
    that its target happens to be rebuildable.
    """
    wanted = {(site.file, site.function) for site in dynamic_sites if site.table == UNRESOLVED_TABLE}
    parameters: dict[str, tuple[str, int | None]] = {}
    for _path, tree, relative, _values, _scopes in parsed_modules:
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            matching_functions = [
                function for file, function in wanted if file == relative and function.split(".")[-1] == node.name
            ]
            if len(matching_functions) != 1:
                continue
            qualified = matching_functions[0]
            positional = [*node.args.posonlyargs, *node.args.args]
            for index, parameter in enumerate(positional):
                if parameter.arg == "table":
                    parameters[f"{relative}::{qualified}"] = (node.name, index)
                    break
            else:
                if any(parameter.arg == "table" for parameter in node.args.kwonlyargs):
                    parameters[f"{relative}::{qualified}"] = (node.name, None)
    return parameters


def _dynamic_table_targets(
    parsed_modules: Iterable[
        tuple[Path, ast.Module, str, dict[str, tuple[str, ...]], dict[ast.AST, tuple[str, ast.AST | None]]]
    ],
    dynamic_sites: Iterable[WriteSite],
) -> tuple[DynamicTableTarget, ...]:
    """Resolve literal table arguments supplied to dynamic-table helpers."""
    parameters = _function_table_parameters(parsed_modules, dynamic_sites)
    by_name: dict[str, list[tuple[str, int | None]]] = {}
    for helper_key, descriptor in parameters.items():
        by_name.setdefault(descriptor[0], []).append((helper_key, descriptor[1]))
    targets: dict[tuple[str, str, str, int], DynamicTableTarget] = {}
    for _path, tree, relative, values, _scopes in parsed_modules:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            for helper_key, position in by_name.get(name, []):
                argument: ast.AST | None = None
                for keyword in node.keywords:
                    if keyword.arg == "table":
                        argument = keyword.value
                        break
                if argument is None and position is not None and len(node.args) > position:
                    argument = node.args[position]
                if argument is None:
                    continue
                helper_file, helper_function = helper_key.split("::", 1)
                for table in _fragments(argument, values):
                    if table == _HOLE or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                        continue
                    target = DynamicTableTarget(
                        helper_file=helper_file,
                        helper_function=helper_function,
                        table=table,
                        caller_file=relative,
                        line=node.lineno,
                    )
                    targets[(helper_key, table, relative, node.lineno)] = target
    return tuple(sorted(targets.values(), key=lambda item: (item.helper_key, item.caller_file, item.line, item.table)))


def _index_foreign_key_cleanup_helpers(
    parsed_modules: Iterable[
        tuple[Path, ast.Module, str, dict[str, tuple[str, ...]], dict[ast.AST, tuple[str, ast.AST | None]]]
    ],
) -> frozenset[str]:
    """Find helpers that derive both dynamic actions from current FK metadata."""
    helpers: set[str] = set()
    for _path, tree, relative, values, _scopes in parsed_modules:
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            statements = [
                statement
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in _SQL_EXECUTION_METHODS
                and call.args
                for statement in _fragments(call.args[0], values)
            ]
            has_fk_actions = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_session_foreign_key_actions"
                for call in ast.walk(node)
            )
            if (
                has_fk_actions
                and any("DELETE FROM {}" in item for item in statements)
                and any("UPDATE {}" in item for item in statements)
            ):
                helpers.add(f"{relative}::{node.name}")
    return frozenset(helpers)


def census_package(package_root: Path, *, repo_root: Path) -> CensusObservation:
    """Census every durable rewrite statement and caller-supplied-SQL helper."""
    table_tiers = durable_table_tiers()
    sites: dict[str, WriteSite] = {}
    helpers: dict[str, HelperSite] = {}
    runtime_creations: dict[str, RuntimeTableCreation] = {}
    parsed_modules: list[
        tuple[Path, ast.Module, str, dict[str, tuple[str, ...]], dict[ast.AST, tuple[str, ast.AST | None]]]
    ] = []

    for path in sorted(package_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(repo_root).as_posix()
        values = _string_values(tree)
        scopes = _scopes(tree)
        parsed_modules.append((path, tree, relative, values, scopes))
        if not _is_archive_tier_runtime_module(relative):
            continue
        memory_connections = _memory_connection_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in _SQL_EXECUTION_METHODS:
                continue
            if not node.args:
                continue
            scope, function = scopes.get(node, ("<module>", None))
            qualified = scope if scope != "<module>" else "<module>"
            argument = node.args[0]
            statements = _fragments(argument, values)
            receiver = node.func.value
            for statement in statements:
                creation = _runtime_table_creation(
                    relative=relative,
                    function=qualified,
                    receiver=receiver,
                    statement=statement,
                    line=node.lineno,
                    memory_connections=memory_connections,
                    canonical_tables=table_tiers,
                )
                if creation is not None:
                    runtime_creations.setdefault(creation.key, creation)

    runtime_persistent_tables = frozenset(
        creation.table for creation in runtime_creations.values() if creation.disposition == "persistent"
    )
    for _path, tree, relative, values, scopes in parsed_modules:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in _SQL_EXECUTION_METHODS:
                continue
            if not node.args:
                continue
            scope, function = scopes.get(node, ("<module>", None))
            qualified = scope if scope != "<module>" else "<module>"
            argument = node.args[0]
            statements = _fragments(argument, values)
            if not statements:
                if isinstance(argument, ast.Name) and argument.id in _string_parameter_names(function):
                    helper = HelperSite(
                        file=relative,
                        function=qualified,
                        parameter=argument.id,
                        line=node.lineno,
                    )
                    helpers.setdefault(helper.key, helper)
                continue
            resolved = [
                item
                for statement in statements
                for item in _classify_statement(
                    statement,
                    table_tiers,
                    runtime_persistent_tables=runtime_persistent_tables,
                )
            ]
            for table, kind, tier in dict.fromkeys(resolved):
                site = WriteSite(
                    file=relative,
                    function=qualified,
                    table=table,
                    kind=kind,
                    line=node.lineno,
                    tier=tier,
                )
                sites.setdefault(site.key, site)

    dynamic_targets = _dynamic_table_targets(parsed_modules, sites.values())
    return CensusObservation(
        sites=tuple(sorted(sites.values(), key=lambda item: item.key)),
        helpers=tuple(sorted(helpers.values(), key=lambda item: item.key)),
        runtime_creations=tuple(sorted(runtime_creations.values(), key=lambda item: item.key)),
        dynamic_targets=dynamic_targets,
        index_foreign_key_cleanup_helpers=_index_foreign_key_cleanup_helpers(parsed_modules),
    )


def _as_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


@dataclass(frozen=True)
class CensusEntry:
    key: str
    file: str
    function: str
    table: str
    kind: str
    tier: str
    classification: str
    reason: str


@dataclass(frozen=True)
class CensusDeclaration:
    package: str
    entries: dict[str, CensusEntry]
    helpers: dict[str, str]
    runtime_tables: dict[str, RuntimeTableDeclaration]
    malformed: tuple[str, ...]


def load_declaration(path: Path) -> CensusDeclaration:
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data: Mapping[str, object] = raw if isinstance(raw, dict) else {}
    entries: dict[str, CensusEntry] = {}
    malformed: list[str] = []
    rows = data.get("writes")
    for index, item in enumerate(list(rows) if isinstance(rows, list) else []):
        if not isinstance(item, dict):
            malformed.append(f"writes[{index}]")
            continue
        file = item.get("file")
        function = item.get("function")
        table = item.get("table")
        kind = item.get("kind")
        if not all(isinstance(value, str) for value in (file, function, table, kind)):
            malformed.append(f"writes[{index}]")
            continue
        entry = CensusEntry(
            key=f"{file}::{function}::{table}::{kind}",
            file=str(file),
            function=str(function),
            table=str(table),
            kind=str(kind),
            tier=str(item.get("tier") or ""),
            classification=str(item.get("classification") or ""),
            reason=str(item.get("reason") or "").strip(),
        )
        entries[entry.key] = entry

    helpers: dict[str, str] = {}
    helper_rows = data.get("caller_supplied_sql")
    for index, item in enumerate(list(helper_rows) if isinstance(helper_rows, list) else []):
        if not isinstance(item, dict):
            malformed.append(f"caller_supplied_sql[{index}]")
            continue
        file = item.get("file")
        function = item.get("function")
        parameter = item.get("parameter")
        if not all(isinstance(value, str) for value in (file, function, parameter)):
            malformed.append(f"caller_supplied_sql[{index}]")
            continue
        helpers[f"{file}::{function}::{parameter}"] = str(item.get("reason") or "").strip()

    runtime_tables: dict[str, RuntimeTableDeclaration] = {}
    runtime_rows = data.get("runtime_tables")
    for index, item in enumerate(list(runtime_rows) if isinstance(runtime_rows, list) else []):
        if not isinstance(item, dict):
            malformed.append(f"runtime_tables[{index}]")
            continue
        file = item.get("file")
        function = item.get("function")
        table = item.get("table")
        disposition = item.get("disposition")
        reason = str(item.get("reason") or "").strip()
        if not (
            isinstance(file, str)
            and isinstance(function, str)
            and isinstance(table, str)
            and isinstance(disposition, str)
        ):
            malformed.append(f"runtime_tables[{index}]")
            continue
        key = f"{file}::{function}::{table}::persistent"
        runtime_tables[key] = RuntimeTableDeclaration(key=key, disposition=disposition, reason=reason)

    return CensusDeclaration(
        package=str(data.get("package") or "polylogue"),
        entries=entries,
        helpers=helpers,
        runtime_tables=runtime_tables,
        malformed=tuple(malformed),
    )


def collect_violations(*, repo_root: Path, declaration_path: Path | None = None) -> list[dict[str, object]]:
    """Check the observed durable-rewrite census against its declaration."""
    path = declaration_path or (repo_root / DECLARATION_PATH)
    if not path.is_file():
        return [{"rule": "durable_write_census_declaration_missing", "key": path.as_posix()}]
    declaration = load_declaration(path)
    observation = census_package(repo_root / declaration.package, repo_root=repo_root)

    violations: list[dict[str, object]] = []
    for name in declaration.malformed:
        violations.append({"rule": "durable_write_census_row_malformed", "key": name})

    runtime_creations = {creation.key: creation for creation in observation.runtime_creations}
    for key in sorted(declaration.runtime_tables.keys() - runtime_creations.keys()):
        violations.append(
            {
                "rule": "runtime_table_census_stale",
                "key": key,
                "detail": "declared non-durable runtime relation is no longer created -- drop the entry",
            }
        )
    for key, runtime_entry in declaration.runtime_tables.items():
        creation = runtime_creations.get(key)
        if creation is None:
            continue
        if (
            runtime_entry.disposition != "disposable_ops"
            or not creation.file.endswith("/ops_write.py")
            or not runtime_entry.reason
        ):
            violations.append(
                {
                    "rule": "runtime_table_disposition_invalid",
                    "key": key,
                    "file": creation.file,
                    "detail": "only an explained runtime relation in ops_write.py may be declared disposable_ops",
                }
            )

    observed = {site.key: site for site in observation.sites}
    for key in sorted(observed.keys() - declaration.entries.keys()):
        site = observed[key]
        if site.tier == "runtime":
            creation_key = f"{site.file}::{site.function}::{site.table}::persistent"
            if creation_key in declaration.runtime_tables:
                continue
            violations.append(
                {
                    "rule": "runtime_persistent_table_rewrite_undeclared",
                    "key": key,
                    "file": site.file,
                    "line": site.line,
                    "detail": (
                        f"{site.kind} on runtime-created persistent table {site.table}; add it to canonical tier DDL "
                        "or remove the runtime rewrite"
                    ),
                }
            )
            continue
        violations.append(
            {
                "rule": "durable_write_undeclared",
                "key": key,
                "file": site.file,
                "line": site.line,
                "tier": site.tier,
                "detail": (
                    f"{site.kind} on durable table {site.table}; declare it in "
                    f"{DECLARATION_PATH} with a classification and a reason"
                ),
            }
        )
    for key in sorted(declaration.entries.keys() - observed.keys()):
        violations.append(
            {
                "rule": "durable_write_census_stale",
                "key": key,
                "file": declaration.entries[key].file,
                "detail": "declared durable rewrite is no longer present -- drop the entry",
            }
        )
    for key in sorted(declaration.entries.keys() & observed.keys()):
        entry = declaration.entries[key]
        site = observed[key]
        if entry.tier != site.tier:
            violations.append(
                {
                    "rule": "durable_write_tier_drift",
                    "key": key,
                    "file": site.file,
                    "declared": entry.tier,
                    "observed": site.tier,
                }
            )
        if entry.classification not in CLASSIFICATION_VOCABULARY:
            violations.append(
                {
                    "rule": "durable_write_unknown_classification",
                    "key": key,
                    "file": site.file,
                    "declared": entry.classification,
                    "detail": f"allowed: {', '.join(sorted(CLASSIFICATION_VOCABULARY))}",
                }
            )
        elif entry.classification in FORBIDDEN_CLASSIFICATIONS:
            violations.append(
                {
                    "rule": "durable_write_masks_a_producer",
                    "key": key,
                    "file": site.file,
                    "line": site.line,
                    "detail": CLASSIFICATION_VOCABULARY[entry.classification],
                }
            )
        if not entry.reason:
            violations.append({"rule": "durable_write_reason_missing", "key": key, "file": site.file})

    for key in sorted(declaration.entries.keys() & observed.keys()):
        entry = declaration.entries[key]
        if entry.classification != "index_foreign_key_cleanup":
            continue
        helper_key = f"{entry.file}::{entry.function}"
        if helper_key not in observation.index_foreign_key_cleanup_helpers:
            violations.append(
                {
                    "rule": "index_foreign_key_cleanup_shape_invalid",
                    "key": key,
                    "file": entry.file,
                    "detail": "classification requires current FK action discovery plus dynamic UPDATE and DELETE actions",
                }
            )

    targets_by_helper: dict[str, tuple[DynamicTableTarget, ...]] = {}
    for target in observation.dynamic_targets:
        targets_by_helper[target.helper_key] = (*targets_by_helper.get(target.helper_key, ()), target)
    canonical_tiers = durable_table_tiers()
    for key in sorted(declaration.entries.keys() & observed.keys()):
        entry = declaration.entries[key]
        if entry.classification != "rebuildable_dynamic_target":
            continue
        targets = targets_by_helper.get(f"{entry.file}::{entry.function}", ())
        if not targets:
            violations.append(
                {
                    "rule": "dynamic_table_targets_missing",
                    "key": key,
                    "file": entry.file,
                    "detail": "the dynamic-table helper has no statically resolved table caller to validate",
                }
            )
            continue
        for target in targets:
            tier = canonical_tiers.get(target.table)
            if tier == "index":
                continue
            violations.append(
                {
                    "rule": "dynamic_table_target_not_index",
                    "key": key,
                    "file": target.caller_file,
                    "line": target.line,
                    "detail": (
                        f"dynamic helper target {target.table!r} resolves to {tier or 'no canonical tier'}; "
                        "only index-tier callers satisfy rebuildable_dynamic_target"
                    ),
                }
            )

    observed_helpers = {helper.key: helper for helper in observation.helpers}
    for key in sorted(observed_helpers.keys() - declaration.helpers.keys()):
        helper = observed_helpers[key]
        violations.append(
            {
                "rule": "caller_supplied_sql_undeclared",
                "key": key,
                "file": helper.file,
                "line": helper.line,
                "detail": (
                    "executes a statement handed to it by its caller, so no static census can "
                    f"see what it writes; declare it in {DECLARATION_PATH} with a reason"
                ),
            }
        )
    for key in sorted(declaration.helpers.keys() - observed_helpers.keys()):
        violations.append(
            {
                "rule": "caller_supplied_sql_census_stale",
                "key": key,
                "detail": "declared caller-supplied-SQL helper no longer exists -- drop the entry",
            }
        )
    for key in sorted(declaration.helpers.keys() & observed_helpers.keys()):
        if not declaration.helpers[key]:
            violations.append({"rule": "caller_supplied_sql_reason_missing", "key": key})

    return violations


def summarize(observation: CensusObservation) -> dict[str, object]:
    """Counts used by the gate's success line and by its JSON output."""
    by_tier: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for site in observation.sites:
        by_tier[site.tier] = by_tier.get(site.tier, 0) + 1
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1
    return {
        "durable_write_sites": len(observation.sites),
        "caller_supplied_sql_helpers": len(observation.helpers),
        "runtime_table_creations": len(observation.runtime_creations),
        "by_tier": dict(sorted(by_tier.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def iter_site_payloads(sites: Iterable[WriteSite]) -> list[dict[str, object]]:
    return [
        {
            "file": site.file,
            "function": site.function,
            "table": site.table,
            "kind": site.kind,
            "tier": site.tier,
            "line": site.line,
        }
        for site in sites
    ]
