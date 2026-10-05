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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from devtools.ast_cache import parse_path, walk_module
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER

#: The declaration this census is checked against.
DECLARATION_PATH = "docs/plans/durable-write-census.yaml"

#: Tiers whose rows are durable: losing or corrupting one is not recoverable by
#: reconvergence. ``index``/``embeddings``/``ops`` are rebuildable or disposable
#: and are out of this census's subject by the tier table in ``AGENTS.md``.
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
    "private_witness_hydration": (
        "exact original baseline images restored only into the creator-owned private Native witness "
        "before capture, never an archive mutation or refinement"
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

# This adjudication covers one reviewed baseline seed replacement, not a
# general exemption for scratch connections or methods with private names.
_PRIVATE_WITNESS_HYDRATION_SITES = frozenset(
    {
        "polylogue/storage/sqlite/reference_seal.py::PreparedIndexMutation._seed_source_controls::?::delete::1",
    }
)

_SQL_EXECUTION_METHODS = frozenset({"execute", "executemany", "executescript"})


@dataclass(frozen=True)
class SQLExecution:
    """The actual operand and connection of one evidenced execution call."""

    argument: ast.expr
    receiver: ast.expr


_NATIVE_CURSOR = "polylogue.storage.io_phase_metrics.connection_cursor"
_NATIVE_SEAL = "polylogue.storage.sqlite.reference_seal.PreparedIndexMutation"
_NATIVE_SEAL_SQL = {
    "_owned_cursor": (1, "sql", 0),
    "_source_statement_attempt": (0, "sql", None),
    "source_statement": (0, "sql", None),
    "user_statement": (0, "sql", None),
    "_selected_statement": (0, "sql", None),
    "original_rows": (1, "sql", None),
    "source_rows": (0, "sql", None),
    "user_rows": (0, "sql", None),
    "_selected_rows": (0, "sql", None),
    "before_index_input": (2, "rowid_sql", None),
}


def sql_execution_calls(tree: ast.AST, *, relative: str = "") -> dict[ast.Call, SQLExecution]:
    """Observe the finite Native SQL family without granting SQL authority.

    Imported aliases, actual seal annotations and the canonical seal's own
    methods establish helper ownership. An unrelated object with the same
    method name is not evidence. Direct SQLite calls keep their existing law.
    """
    module_nodes = walk_module(tree)
    calls = tuple(node for node in module_nodes if isinstance(node, ast.Call))
    cursor_aliases = {
        alias.asname or alias.name
        for node in module_nodes
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
        if f"{node.module}.{alias.name}" == _NATIVE_CURSOR
    }
    result: dict[ast.Call, SQLExecution] = {}
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr in _SQL_EXECUTION_METHODS:
            parameter = "sql_script" if call.func.attr == "executescript" else "sql"
            argument = next((item.value for item in call.keywords if item.arg == parameter), None)
            if argument is None and call.args:
                argument = call.args[0]
            if argument is not None:
                result[call] = SQLExecution(argument, call.func.value)
    # Ordinary SQLite calls need no imported-Native lexical ownership proof.
    # Reuse the existing cached walk rather than traverse every lexical scope
    # for modules that cannot invoke any member of this finite helper family.
    if not any(
        isinstance(call.func, ast.Name)
        and call.func.id in cursor_aliases
        or isinstance(call.func, ast.Attribute)
        and call.func.attr in {*_NATIVE_SEAL_SQL, "connection_cursor"}
        for call in calls
    ):
        return result

    comprehensions = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def definition_inputs(node: ast.AST) -> Iterator[ast.AST]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            yield from node.args.defaults
            yield from (value for value in node.args.kw_defaults if value is not None)
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            if node.args.vararg:
                arguments += (node.args.vararg,)
            if node.args.kwarg:
                arguments += (node.args.kwarg,)
            yield from (formal.annotation for formal in arguments if formal.annotation is not None)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from node.decorator_list
                if node.returns:
                    yield node.returns
        elif isinstance(node, ast.ClassDef):
            yield from node.decorator_list
            yield from node.bases
            yield from (keyword.value for keyword in node.keywords)

    def body_children(node: ast.AST) -> Iterator[ast.AST]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield from node.body
        elif isinstance(node, ast.Lambda):
            yield node.body
        elif isinstance(node, comprehensions):
            # The first iterable executes outside the comprehension scope.
            yield node.generators[0].target
            yield from node.generators[0].ifs
            yield from node.generators[1:]
            if isinstance(node, ast.DictComp):
                yield node.key
                yield node.value
            else:
                yield node.elt
        else:
            yield from ast.iter_child_nodes(node)

    def outer_comprehension_bindings(root: ast.AST) -> Iterator[ast.NamedExpr]:
        if isinstance(root, ast.NamedExpr) and isinstance(root.target, ast.Name):
            yield root
        for child in ast.iter_child_nodes(root):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                yield from outer_comprehension_bindings(child)

    def nodes(root: ast.AST) -> Iterator[ast.AST]:
        yield root
        for child in body_children(root):
            if isinstance(
                child,
                ast.FunctionDef
                | ast.AsyncFunctionDef
                | ast.ClassDef
                | ast.Lambda
                | ast.ListComp
                | ast.SetComp
                | ast.DictComp
                | ast.GeneratorExp,
            ):
                # Definition-time operands belong here, bodies to their own scope.
                yield child
                for expression in definition_inputs(child):
                    yield from nodes(expression)
                if isinstance(child, comprehensions):
                    yield from nodes(child.generators[0].iter)
                    yield from outer_comprehension_bindings(child)
            else:
                yield from nodes(child)

    def qualified(expression: ast.AST | None, imports: Mapping[str, str]) -> str | None:
        if isinstance(expression, ast.Name):
            return imports.get(expression.id)
        if isinstance(expression, ast.Attribute):
            parent = qualified(expression.value, imports)
            return f"{parent}.{expression.attr}" if parent else None
        return None

    def seal_annotation(expression: ast.AST | None, imports: Mapping[str, str]) -> bool:
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            try:
                expression = ast.parse(expression.value, mode="eval").body
            except SyntaxError:
                return False
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
            return (
                seal_annotation(expression.left, imports)
                and isinstance(expression.right, ast.Constant)
                and expression.right.value is None
            ) or (
                seal_annotation(expression.right, imports)
                and isinstance(expression.left, ast.Constant)
                and expression.left.value is None
            )
        return qualified(expression, imports) == _NATIVE_SEAL

    def operand(call: ast.Call, index: int, name: str) -> ast.expr | None:
        return next((item.value for item in call.keywords if item.arg == name), None) or (
            call.args[index] if len(call.args) > index else None
        )

    def visit(
        root: ast.AST,
        inherited: dict[str, str],
        owners: set[str],
        native_class: bool = False,
        annotation_imports: Mapping[str, str] | None = None,
    ) -> None:
        imports = dict(inherited)
        bindings = set(owners)
        scoped = tuple(nodes(root))
        # Evidence belongs to one lexical scope. Every binding must agree;
        # constructor assignments cannot undo a loop/handler/pattern rebind.
        stored: set[str] = set()
        assignment_values: dict[str, list[ast.expr | None]] = {}
        constructor_targets: set[ast.Name] = set()
        import_values: dict[str, list[str]] = {}
        for node in scoped:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignment_values.setdefault(target.id, []).append(node.value)
                        constructor_targets.add(target)
            elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and isinstance(node.target, ast.Name):
                assignment_values.setdefault(node.target.id, []).append(node.value)
                constructor_targets.add(node.target)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    import_values.setdefault(alias.asname or alias.name, []).append(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    import_values.setdefault(alias.asname or alias.name.split(".")[0], []).append(
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node is not root
                or isinstance(node, ast.ExceptHandler)
                and node.name
                or isinstance(node, ast.MatchAs | ast.MatchStar)
                and node.name
            ):
                bound_name = node.name
                if bound_name is not None:
                    stored.add(bound_name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                stored.add(node.rest)
        for node in scoped:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store | ast.Del)
                and node not in constructor_targets
            ):
                stored.add(node.id)
        rebound = stored | assignment_values.keys() | import_values.keys()
        for name in rebound:
            imports.pop(name, None)
            bindings.discard(name)
        for name, values in import_values.items():
            if name not in stored and name not in assignment_values and len(set(values)) == 1:
                imports[name] = values[0]
        if relative == "polylogue/storage/sqlite/reference_seal.py" and isinstance(root, ast.Module):
            imports["PreparedIndexMutation"] = _NATIVE_SEAL
        unknown_formals: set[str] = set()
        if isinstance(root, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            positional = (*root.args.posonlyargs, *root.args.args)
            arguments = (*positional, *root.args.kwonlyargs)
            annotated = {
                formal.arg
                for formal in arguments
                if seal_annotation(formal.annotation, inherited if annotation_imports is None else annotation_imports)
            }
            if (
                native_class
                and positional
                and isinstance(root, ast.FunctionDef | ast.AsyncFunctionDef)
                and not any(
                    isinstance(decorator, ast.Name) and decorator.id in {"staticmethod", "classmethod"}
                    for decorator in root.decorator_list
                )
            ):
                annotated.add(positional[0].arg)
            if root.args.vararg:
                arguments += (root.args.vararg,)
            if root.args.kwarg:
                arguments += (root.args.kwarg,)
            for formal in arguments:
                imports.pop(formal.arg, None)
                bindings.discard(formal.arg)
                if formal.arg in annotated:
                    if formal.arg not in rebound:
                        bindings.add(formal.arg)
                else:
                    unknown_formals.add(formal.arg)
        for name, assigned_operands in assignment_values.items():
            if (
                name not in stored
                and name not in unknown_formals
                and name not in import_values
                and assigned_operands
                and all(
                    isinstance(value, ast.Call)
                    and qualified(value.func, imports) in {_NATIVE_SEAL, _NATIVE_SEAL + ".source_only"}
                    for value in assigned_operands
                )
            ):
                bindings.add(name)
        for node in scoped:
            if not isinstance(node, ast.Call):
                continue
            argument: ast.expr | None = None
            receiver: ast.expr | None = None
            if isinstance(node.func, ast.Attribute) and node.func.attr in _SQL_EXECUTION_METHODS:
                argument = operand(node, 0, "sql_script" if node.func.attr == "executescript" else "sql")
                receiver = node.func.value
            elif qualified(node.func, imports) == _NATIVE_CURSOR:
                argument = operand(node, 1, "sql")
                receiver = operand(node, 0, "connection")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in bindings
                and node.func.attr in _NATIVE_SEAL_SQL
            ):
                index, parameter, connection_index = _NATIVE_SEAL_SQL[node.func.attr]
                argument = operand(node, index, parameter)
                receiver = (
                    operand(node, connection_index, "connection") if connection_index is not None else node.func.value
                )
            if argument is not None and receiver is not None:
                result[node] = SQLExecution(argument, receiver)

        def dispatch(child: ast.AST) -> None:
            runtime_imports = inherited if isinstance(root, ast.ClassDef) else imports
            runtime_owners = owners if isinstance(root, ast.ClassDef) else bindings
            if isinstance(child, ast.ClassDef):
                visit(
                    child,
                    runtime_imports,
                    runtime_owners,
                    relative == "polylogue/storage/sqlite/reference_seal.py"
                    and isinstance(root, ast.Module)
                    and child.name == "PreparedIndexMutation",
                )
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(
                    child,
                    runtime_imports,
                    runtime_owners,
                    native_class and isinstance(root, ast.ClassDef),
                    annotation_imports=imports,
                )
            elif isinstance(child, ast.Lambda | ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
                visit(child, runtime_imports, runtime_owners, annotation_imports=imports)
            else:
                children(child)
                return
            for expression in definition_inputs(child):
                dispatch(expression)
            if isinstance(child, comprehensions):
                dispatch(child.generators[0].iter)

        def children(node: ast.AST) -> None:
            for child in body_children(node):
                dispatch(child)

        children(root)

    visit(tree, {}, set())
    return result


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
    occurrence: int

    @property
    def key(self) -> str:
        return f"{self.file}::{self.function}::{self.table}::{self.kind}::{self.occurrence}"


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
    file: str
    function: str
    table: str
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
        body = _fragments(expression.body, values)
        orelse = _fragments(expression.orelse, values)
        if not body and not orelse:
            return ()
        # A branch the census cannot read stays a visible hole beside one it can.
        return (body or (_HOLE,)) + (orelse or (_HOLE,))
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


#: The only node shapes :func:`_string_values` binds a name from.
_BINDING_NODES = (ast.Assign, ast.AnnAssign, ast.For, ast.AsyncFor)


def _string_values(
    tree: ast.Module, *, initial: Mapping[str, tuple[str, ...]] | None = None
) -> dict[str, tuple[str, ...]]:
    """Resolve string-valued names to a fixpoint (3 passes suffice).

    Two namespaces share the mapping: a bare name resolves to the statement
    texts it can hold, and ``[]<name>`` resolves to the members of a literal
    string sequence it is bound to, which is what lets a ``for table in (...)``
    target expand.
    """
    values: dict[str, tuple[str, ...]] = dict(initial or {})
    bindings = [node for node in walk_module(tree) if isinstance(node, _BINDING_NODES)]
    for _ in range(3):
        for node in bindings:
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


def _sql_fragments(
    expression: ast.AST, values: Mapping[str, tuple[str, ...]], function: ast.AST | None
) -> tuple[str, ...]:
    """Keep caller operands distinct from another scope's literal bindings."""
    if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = function.args
        parameters = {item.arg for item in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
        if arguments.vararg is not None:
            parameters.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            parameters.add(arguments.kwarg.arg)
        # A local operand also shadows literals collected in other functions.
        # Resolve this function's own bindings only after removing those names.
        locals_ = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        shadowed = parameters | locals_
        values = {name: fragments for name, fragments in values.items() if name.removeprefix("[]") not in shadowed}
        values = _string_values(ast.Module(body=function.body, type_ignores=[]), initial=values)
    return _fragments(expression, values)


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


def _memory_connection_names(
    tree: ast.Module, scopes: Mapping[ast.AST, tuple[str, ast.AST | None]]
) -> dict[tuple[str, str], int]:
    """Prove single, unconditional private-memory bindings in their own scope."""
    stores: dict[tuple[str, str], int] = {}
    for node in walk_module(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            scope, _ = scopes.get(node, ("<module>", None))
            key = (scope, node.id)
            stores[key] = stores.get(key, 0) + 1
    bindings: dict[tuple[str, str], int] = {}
    for node in walk_module(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        scope, function = scopes.get(node, ("<module>", None))
        body = function.body if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef) else tree.body
        key = (scope, node.targets[0].id)
        if node not in body or stores.get(key) != 1:
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
            bindings[key] = node.lineno
    return bindings


def _runtime_table_creations(
    *,
    relative: str,
    function: str,
    receiver: ast.AST,
    statement: str,
    line: int,
    memory_connections: Mapping[tuple[str, str], int],
    canonical_tables: Mapping[str, str],
) -> Iterator[RuntimeTableCreation]:
    """Classify every CREATE in one execution, including executescript batches."""
    for match in _RUNTIME_CREATE_RE.finditer(statement):
        table = match.group("table")
        if table in canonical_tables:
            continue
        if match.group("temporary") is not None:
            disposition = "temporary"
        elif (
            isinstance(receiver, ast.Name)
            and (binding_line := memory_connections.get((function, receiver.id))) is not None
            and binding_line < line
        ):
            disposition = "scratch"
        else:
            disposition = "persistent"
        yield RuntimeTableCreation(file=relative, function=function, table=table, disposition=disposition, line=line)


def _is_archive_storage_module(relative: str) -> bool:
    """Whether the module belongs to the archive persistence layer."""
    return relative.startswith("polylogue/storage/")


def _writes_archive_table(
    tree: ast.Module, values: Mapping[str, tuple[str, ...]], table_tiers: Mapping[str, str]
) -> bool:
    """Whether the module executes any write against a canonical archive table.

    Such a module holds an archive connection, so a table it creates at
    runtime may live in an archive tier. A module whose statements name only
    its own relations (a parser's spill database, a browser-capture registry)
    is not an archive writer, and its private tables are not archive state.
    """
    executions = sql_execution_calls(tree)
    for node in walk_module(tree):
        if not isinstance(node, ast.Call) or (execution := executions.get(node)) is None:
            continue
        for text in _fragments(execution.argument, values):
            if any(match.group("table") in table_tiers for match in _REWRITE_RE.finditer(text)):
                return True
    return False


#: One parsed module as the post-passes read it: path, tree, repository-relative
#: path, resolved string bindings and node scopes.
ParsedModule = tuple[Path, ast.Module, str, dict[str, tuple[str, ...]], dict[ast.AST, tuple[str, ast.AST | None]]]


def _function_table_parameters(
    parsed_modules: Iterable[ParsedModule],
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
    wanted_files = {file for file, _function in wanted}
    for _path, tree, relative, _values, _scopes in parsed_modules:
        if relative not in wanted_files:
            continue
        for node in walk_module(tree):
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
    parse_modules: Callable[[Iterable[str]], Iterable[ParsedModule]],
    dynamic_sites: Iterable[WriteSite],
    called_names: Mapping[str, frozenset[str]],
) -> tuple[DynamicTableTarget, ...]:
    """Resolve literal table arguments supplied to dynamic-table helpers.

    *parse_modules* parses the named modules; *called_names* holds every name
    each module calls, so only modules that can call a helper are parsed.
    """
    dynamic_sites = tuple(dynamic_sites)
    wanted_files = {site.file for site in dynamic_sites if site.table == UNRESOLVED_TABLE}
    parameters = _function_table_parameters(parse_modules(wanted_files), dynamic_sites)
    by_name: dict[str, list[tuple[str, int | None]]] = {}
    for helper_key, descriptor in parameters.items():
        by_name.setdefault(descriptor[0], []).append((helper_key, descriptor[1]))
    targets: dict[tuple[str, str, str, int], DynamicTableTarget] = {}
    callers = [relative for relative, names in called_names.items() if not names.isdisjoint(by_name)]
    for _path, tree, relative, _values, _scopes in parse_modules(callers):
        for node in walk_module(tree):
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
                helper_file, helper_function = helper_key.split("::", 1)
                # A module-wide name cache is not proof of a caller's local value.
                # Keep missing, computed and partially unknown arguments visible.
                fragments = _fragments(argument, {}) if argument is not None else ()
                for fragment in fragments or (UNRESOLVED_TABLE,):
                    table = fragment if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", fragment) else UNRESOLVED_TABLE
                    target = DynamicTableTarget(
                        helper_file=helper_file,
                        helper_function=helper_function,
                        table=table,
                        caller_file=relative,
                        line=node.lineno,
                    )
                    targets[(helper_key, table, relative, node.lineno)] = target
    return tuple(sorted(targets.values(), key=lambda item: (item.helper_key, item.caller_file, item.line, item.table)))


def _dynamic_sql_identifiers(expression: ast.AST) -> tuple[ast.expr, ...]:
    """Recover the table interpolation, not unrelated WHERE/value holes."""
    if not isinstance(expression, ast.JoinedStr):
        return ()
    parts: list[str] = []
    holes: dict[str, ast.expr] = {}
    for part in expression.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
        elif isinstance(part, ast.FormattedValue):
            marker = f"__census_identifier_{len(holes)}__"
            holes[marker] = part.value
            parts.append(marker)
        else:
            return ()
    return tuple(holes[match["table"]] for match in _REWRITE_RE.finditer("".join(parts)) if match["table"] in holes)


def _metadata_identifier(expression: ast.expr, loop: ast.For, *, row: bool) -> bool:
    """Prove a table identifier from this loop's metadata binding.

    Only transparent quoting and single-assignment aliases are accepted.
    Rebinding the metadata name anywhere in the loop defeats the proof.
    """
    target = loop.target.elts[0] if isinstance(loop.target, ast.Tuple) else loop.target
    if not isinstance(target, ast.Name):
        return False
    stores: dict[str, list[ast.AST]] = {}
    assignments: dict[str, ast.AST] = {}
    for statement in loop.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                stores.setdefault(node.id, []).append(node)
            if (
                node in loop.body
                and isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.lineno < expression.lineno
            ):
                assignments[node.targets[0].id] = node.value
    if stores.get(target.id):
        return False

    def resolve(value: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(value, ast.Name):
            if value.id == target.id:
                return not row
            if value.id in seen or len(stores.get(value.id, ())) != 1 or value.id not in assignments:
                return False
            return resolve(assignments[value.id], seen | {value.id})
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"_quote_identifier", "str"}
            and len(value.args) == 1
            and not value.keywords
        ):
            return resolve(value.args[0], seen)
        return (
            row
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == target.id
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == 0
        )

    return resolve(expression)


def _metadata_loop_connection(loop: ast.For, function: ast.AST) -> tuple[ast.AST, bool] | None:
    iterator = loop.iter
    if (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "_session_foreign_key_actions"
        and len(iterator.args) == 1
        and isinstance(loop.target, ast.Tuple)
        and len(loop.target.elts) == 3
    ):
        return iterator.args[0], False
    if isinstance(iterator, ast.Name):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        if (
            sum(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del) and node.id == iterator.id
                for node in ast.walk(function)
            )
            != 1
        ):
            return None
        bindings = [
            node.value
            for node in function.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == iterator.id
            and node.lineno < loop.lineno
        ]
        if len(bindings) != 1:
            return None
        iterator = bindings[0]
    if isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Attribute) and iterator.func.attr == "fetchall":
        iterator = iterator.func.value
    if (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Attribute)
        and iterator.func.attr == "execute"
        and iterator.args
        and isinstance(iterator.args[0], ast.Constant)
        and isinstance(iterator.args[0].value, str)
        and re.fullmatch(
            r"\s*SELECT\s+name\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'table'\s*;?\s*",
            iterator.args[0].value,
            re.IGNORECASE,
        )
    ):
        return iterator.func.value, True
    return None


def _index_foreign_key_cleanup_helpers(
    parsed_modules: Iterable[ParsedModule],
) -> frozenset[str]:
    """Prove each dynamic write's identifier and connection from metadata."""
    helpers: set[str] = set()
    for _path, tree, relative, values, scopes in parsed_modules:
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_session_foreign_key_actions"
            for node in walk_module(tree)
        ):
            continue
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        executions = sql_execution_calls(tree, relative=relative)
        candidates: dict[str, list[bool]] = {}
        for call in walk_module(tree):
            if not isinstance(call, ast.Call) or (execution := executions.get(call)) is None:
                continue
            if not any(
                table == UNRESOLVED_TABLE
                for text in _fragments(execution.argument, values)
                for table, _kind, _tier in _classify_statement(text, table_tiers={})
            ):
                continue
            qualified, function = scopes.get(call, ("<module>", None))
            if function is None:
                continue
            identifiers = _dynamic_sql_identifiers(execution.argument)
            proven = False
            ancestor = parents.get(call)
            while ancestor is not None and ancestor is not function:
                if (
                    isinstance(ancestor, ast.For)
                    and (origin := _metadata_loop_connection(ancestor, function)) is not None
                ):
                    connection, row = origin
                    if ast.dump(connection) == ast.dump(execution.receiver) and identifiers:
                        proven = all(_metadata_identifier(identifier, ancestor, row=row) for identifier in identifiers)
                        if proven and row:
                            # The catalog route must inspect the FK metadata of
                            # this very table on this very connection too.
                            proven = any(
                                isinstance(query, ast.Call)
                                and (metadata_execution := executions.get(query)) is not None
                                and ast.dump(metadata_execution.receiver) == ast.dump(connection)
                                and isinstance(metadata_execution.argument, ast.JoinedStr)
                                and any(
                                    "PRAGMA foreign_key_list(" in text
                                    for text in _fragments(metadata_execution.argument, values)
                                )
                                and any(
                                    isinstance(part, ast.FormattedValue)
                                    and _metadata_identifier(part.value, ancestor, row=True)
                                    for part in metadata_execution.argument.values
                                )
                                for query in ast.walk(ancestor)
                            )
                    if proven:
                        break
                ancestor = parents.get(ancestor)
            candidates.setdefault(qualified, []).append(proven)
        helpers.update(f"{relative}::{qualified}" for qualified, proofs in candidates.items() if all(proofs))
    return frozenset(helpers)


def census_package(package_root: Path, *, repo_root: Path) -> CensusObservation:
    """Census every durable rewrite statement and caller-supplied-SQL helper."""
    paths = sorted(package_root.rglob("*.py"))
    census = DurableWriteCensus()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        try:
            tree = parse_path(path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        census.observe_runtime_ddl(tree, relative=relative)
    for path in paths:
        try:
            tree = parse_path(path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        census.observe(tree, path=path, relative=path.relative_to(repo_root).as_posix())
    return census.finish()


class DurableWriteCensus:
    """The census, fed parsed modules in path order, in two rounds.

    Every archive-tier runtime module goes to :meth:`observe_runtime_ddl`
    first, because classifying a statement needs every runtime-created table.
    Every module then goes to :meth:`observe`. Neither round keeps a tree:
    what :meth:`finish` needs from other modules afterwards -- the helpers
    that take a ``table`` argument and their literal call sites -- lives in a
    few modules found by name, which it parses again. A caller that already
    walks the package can therefore feed this census from the same parse as
    every other census, and no census holds the package's trees at once.
    """

    def __init__(self) -> None:
        self._table_tiers = durable_table_tiers()
        self._sites: dict[str, WriteSite] = {}
        self._occurrences: dict[tuple[str, str, str, str], int] = {}
        self._helpers: dict[str, HelperSite] = {}
        self._runtime_creations: dict[str, RuntimeTableCreation] = {}
        self._runtime_persistent_tables: frozenset[str] | None = None
        self._paths: dict[str, Path] = {}
        self._called_names: dict[str, frozenset[str]] = {}
        self._fk_cleanup_helpers: set[str] = set()

    def observe_runtime_ddl(self, tree: ast.Module, *, relative: str) -> None:
        if self._runtime_persistent_tables is not None:
            raise RuntimeError("every runtime DDL module must be observed before the first module")
        table_tiers = self._table_tiers
        runtime_creations = self._runtime_creations
        values = _string_values(tree)
        # Runtime DDL is archive state wherever an archive connection can be
        # held: the storage layer, and any other module writing an archive
        # table. A module outside both (a parser's spill database, a
        # browser-capture registry) owns only private relations.
        if not _is_archive_storage_module(relative) and not _writes_archive_table(tree, values, table_tiers):
            return
        scopes = _scopes(tree)
        executions = sql_execution_calls(tree, relative=relative)
        memory_connections = _memory_connection_names(tree, scopes)
        for node in walk_module(tree):
            if not isinstance(node, ast.Call):
                continue
            execution = executions.get(node)
            if execution is None:
                continue
            scope, function = scopes.get(node, ("<module>", None))
            qualified = scope if scope != "<module>" else "<module>"
            argument = execution.argument
            statements = _sql_fragments(argument, values, function)
            receiver = execution.receiver
            for statement in statements:
                for creation in _runtime_table_creations(
                    relative=relative,
                    function=qualified,
                    receiver=receiver,
                    statement=statement,
                    line=node.lineno,
                    memory_connections=memory_connections,
                    canonical_tables=table_tiers,
                ):
                    runtime_creations.setdefault(creation.key, creation)

    def observe(self, tree: ast.Module, *, path: Path, relative: str) -> None:
        if self._runtime_persistent_tables is None:
            self._runtime_persistent_tables = frozenset(
                creation.table for creation in self._runtime_creations.values() if creation.disposition == "persistent"
            )
        table_tiers = self._table_tiers
        sites = self._sites
        helpers = self._helpers
        values = _string_values(tree)
        # Private modules cannot acquire archive authority from another
        # creator that happens to use the same relation name. Match the
        # runtime-DDL observation boundary before classifying rewrites.
        runtime_persistent_tables = (
            self._runtime_persistent_tables
            if _is_archive_storage_module(relative) or _writes_archive_table(tree, values, table_tiers)
            else frozenset()
        )
        scopes = _scopes(tree)
        executions = sql_execution_calls(tree, relative=relative)
        self._paths[relative] = path
        self._called_names[relative] = frozenset(
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in walk_module(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
        )
        self._fk_cleanup_helpers.update(_index_foreign_key_cleanup_helpers(((path, tree, relative, values, scopes),)))
        # Occurrences are numbered in source order, so a statement's identity
        # does not move when an unrelated statement is nested differently.
        calls = sorted(
            (node for node in walk_module(tree) if isinstance(node, ast.Call)),
            key=lambda call: (call.lineno, call.col_offset),
        )
        for node in calls:
            execution = executions.get(node)
            if execution is None:
                continue
            scope, function = scopes.get(node, ("<module>", None))
            qualified = scope if scope != "<module>" else "<module>"
            argument = execution.argument
            statements = _sql_fragments(argument, values, function)
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
            # Texts are alternatives of one call (a conditional, a loop over
            # tables); statements inside one text (a script) are sequential.
            # Each write counts as often as the text naming it most often.
            counts: dict[tuple[str, str, str], int] = {}
            for statement in statements:
                per_text: dict[tuple[str, str, str], int] = {}
                for item in _classify_statement(
                    statement,
                    table_tiers,
                    runtime_persistent_tables=runtime_persistent_tables,
                ):
                    per_text[item] = per_text.get(item, 0) + 1
                for item, count in per_text.items():
                    counts[item] = max(counts.get(item, 0), count)
            resolved = [item for item, count in counts.items() for _ in range(count)]
            for table, kind, tier in resolved:
                group = (relative, qualified, table, kind)
                occurrence = self._occurrences.get(group, 0) + 1
                self._occurrences[group] = occurrence
                site = WriteSite(
                    file=relative,
                    function=qualified,
                    table=table,
                    kind=kind,
                    line=node.lineno,
                    tier=tier,
                    occurrence=occurrence,
                )
                sites[site.key] = site

    def finish(self) -> CensusObservation:
        sites = self._sites
        return CensusObservation(
            sites=tuple(sorted(sites.values(), key=lambda item: item.key)),
            helpers=tuple(sorted(self._helpers.values(), key=lambda item: item.key)),
            runtime_creations=tuple(sorted(self._runtime_creations.values(), key=lambda item: item.key)),
            dynamic_targets=_dynamic_table_targets(self._reparsed, sites.values(), self._called_names),
            index_foreign_key_cleanup_helpers=frozenset(self._fk_cleanup_helpers),
        )

    def _reparsed(self, relatives: Iterable[str]) -> Iterator[ParsedModule]:
        """Parse the named observed modules again, in path order."""
        for relative in sorted(set(relatives) & self._paths.keys(), key=lambda item: self._paths[item]):
            path = self._paths[relative]
            tree = parse_path(path)
            yield path, tree, relative, _string_values(tree), _scopes(tree)


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
    occurrences: dict[tuple[str, str, str, str], int] = {}
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
        group = (str(file), str(function), str(table), str(kind))
        occurrence = occurrences.get(group, 0) + 1
        occurrences[group] = occurrence
        entry = CensusEntry(
            key=f"{file}::{function}::{table}::{kind}::{occurrence}",
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
        runtime_tables[key] = RuntimeTableDeclaration(
            key=key,
            file=str(file),
            function=str(function),
            table=str(table),
            disposition=disposition,
            reason=reason,
        )

    return CensusDeclaration(
        package=str(data.get("package") or "polylogue"),
        entries=entries,
        helpers=helpers,
        runtime_tables=runtime_tables,
        malformed=tuple(malformed),
    )


def _private_witness_hydration_valid(path: Path) -> bool:
    """Recognize only the reviewed seed restoration, never arbitrary scratch DML."""
    tree = parse_path(path)
    owner = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PreparedIndexMutation"), None
    )
    if owner is None:
        return False
    function = next(
        (node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_seed_source_controls"),
        None,
    )
    if function is None:
        return False
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.For):
        return False
    loop = body[0]

    def same(node: ast.AST, source: str, *, expression: bool = False) -> bool:
        expected = ast.parse(source, mode="eval").body if expression else ast.parse(source).body[0]
        return ast.dump(node) == ast.dump(expected)

    if (
        not isinstance(loop.target, ast.Name)
        or loop.target.id != "table"
        or not same(loop.iter, "('raw_existence_journal_control', 'audit_continuity_control')", expression=True)
        or loop.orelse
        or len(loop.body) != 4
    ):
        return False
    retained, missing, identity, hydration = loop.body
    if not same(retained, "image = self.retain_tier_row('source', table, 1)"):
        return False
    if (
        not isinstance(missing, ast.If)
        or not same(missing.test, "image is None", expression=True)
        or missing.orelse
        or len(missing.body) != 1
        or not isinstance(missing.body[0], ast.Raise)
    ):
        return False
    if not same(identity, "image_id = self._retain_row_image(image)"):
        return False
    if (
        not isinstance(hydration, ast.With)
        or len(hydration.items) != 1
        or hydration.items[0].optional_vars is not None
        or not same(hydration.items[0].context_expr, "self._source_hydration()", expression=True)
        or len(hydration.body) != 3
    ):
        return False
    deletion, restoration, metadata = hydration.body
    expected_delete = (
        'with self._owned_cursor(self._scratch, f"DELETE FROM {quote_identifier(table)} WHERE rowid=1"):\n    pass'
    )
    expected_metadata = 'with self._owned_cursor(self._scratch, "INSERT INTO temp.polylogue_source_stage_rows(table_name,physical_rowid,input_image) VALUES (?,?,?)", (table, 1, image_id)):\n    pass'
    return (
        same(deletion, expected_delete)
        and same(restoration, "self._source_image_insert(image)")
        and same(metadata, expected_metadata)
    )


def collect_violations(
    *,
    repo_root: Path,
    declaration_path: Path | None = None,
    observation: CensusObservation | None = None,
) -> list[dict[str, object]]:
    """Check the observed durable-rewrite census against its declaration.

    *observation* is the census of the declaration's package when the caller
    has already taken it in its own pass over the package.
    """
    path = declaration_path or (repo_root / DECLARATION_PATH)
    if not path.is_file():
        return [{"rule": "durable_write_census_declaration_missing", "key": path.as_posix()}]
    declaration = load_declaration(path)
    if observation is None:
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
        valid_disposable_ops = runtime_entry.disposition == "disposable_ops" and creation.file.endswith("/ops_write.py")
        # Any censused writer module may keep a private scratch database; the
        # declaration must say why the relation never reaches an archive tier.
        valid_disposable_scratch = runtime_entry.disposition == "disposable_scratch"
        if not (valid_disposable_ops or valid_disposable_scratch) or not runtime_entry.reason:
            violations.append(
                {
                    "rule": "runtime_table_disposition_invalid",
                    "key": key,
                    "file": creation.file,
                    "detail": (
                        "only an explained runtime relation in ops_write.py may be disposable_ops, or an explicitly "
                        "declared private scratch relation may be disposable_scratch"
                    ),
                }
            )

    declared_scratch_tables = {
        (entry.file, entry.table)
        for entry in declaration.runtime_tables.values()
        if entry.disposition == "disposable_scratch"
    }
    valid_scratch_tables: set[tuple[str, str]] = set()
    for file, table in sorted(declared_scratch_tables):
        creators = [
            creation for creation in runtime_creations.values() if creation.file == file and creation.table == table
        ]
        if not creators or any(
            (creator_entry := declaration.runtime_tables.get(creation.key)) is None
            or creator_entry.disposition != "disposable_scratch"
            or not creator_entry.reason
            for creation in creators
        ):
            violations.append(
                {
                    "rule": "runtime_scratch_table_authority_incomplete",
                    "key": f"{file}::{table}",
                    "file": file,
                    "detail": "every runtime creator for this scratch table needs its own explained declaration",
                }
            )
        else:
            valid_scratch_tables.add((file, table))

    observed = {site.key: site for site in observation.sites}
    for key in sorted(observed.keys() - declaration.entries.keys()):
        site = observed[key]
        if site.tier == "runtime":
            creation_key = f"{site.file}::{site.function}::{site.table}::persistent"
            if creation_key in declaration.runtime_tables or (site.file, site.table) in valid_scratch_tables:
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
        if entry.classification == "private_witness_hydration" and (
            key not in _PRIVATE_WITNESS_HYDRATION_SITES or not _private_witness_hydration_valid(repo_root / entry.file)
        ):
            violations.append(
                {
                    "rule": "private_witness_hydration_site_invalid",
                    "key": key,
                    "file": site.file,
                    "detail": "classification requires the reviewed private scratch receiver, exact seeded tables/rowid, original retained image and hydration/restoration order",
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
            "occurrence": site.occurrence,
        }
        for site in sites
    ]
