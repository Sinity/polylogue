"""Census of derived-tier corrective sweeps and read-path substitutions.

Gate classification: **blocking architectural boundary check**, enforced as part
of ``devtools gate layering``.

Why this exists
---------------

``polylogue-6kur`` AC4 reads "current writer defects have owning producer fixes
and cannot be masked by backfills". Three successive passes closed the *masking*
half for named routes and each left the same two gaps open in its own words:

**G2** -- no exhaustive census of archive-wide derived corrective sweeps existed.
The one real sweep at head was found only because it happened to sit in the
named ``_repair_*`` family: ``daemon/lineage_startup.py`` ran
``repair_stale_prefix_branch_points`` unscoped, archive-wide, on every daemon
start, and PR #5372 deleted it. A ``_reconcile_*`` / ``_normalize_*`` /
``_converge_*`` twin would have been invisible to that search, and naming more
prefixes is the same grep with a longer list.

**G3** -- no systematic search for read-path masks was done. The one found was
found incidentally, and turned out to be not merely redundant but wrong: it
asserted ``is_continuation=True`` for a parented sidechain, so the same session
flipped True to False once convergence caught up.

The distinguishing property of both defects is **where the subject comes from**,
not what the routine is called. A corrective sweep picks its rows out of archive
state instead of out of what a write just touched. A read-path mask substitutes
a value for a stored field it found absent. Both are properties of the statement
and its surrounding expression, so both are checkable; a name is not.

What is censused
----------------

Five observation kinds, each a population in its own right. The subject tiers
are the **derived** ones (``index``, ``embeddings``); durable rewrites are the
subject of ``devtools/durable_write_census.py``, which is a different question
over a different tier set and does not subsume this one.

``unbound_rewrite``
    ``UPDATE`` / ``DELETE`` against a derived table whose row selection is a
    predicate over archive state, or has no ``WHERE`` at all. The subject set
    is whatever the archive happens to contain.
``unresolved_rewrite_scope``
    the same statement shape, but the ``WHERE`` region holds an interpolation
    hole, so the census cannot prove the scope either way. Reported as its own
    population rather than counted as bound.
``state_selected_subject``
    the rewrite itself is row-bound, but the rows come from a derived-tier
    ``SELECT`` whose own selection is unbound or unresolved. **This is the shape
    the deleted lineage sweep had**: every ``UPDATE`` it issued carried
    ``WHERE src_session_id = ?``, and the archive-wide part was the query that
    chose the ids. A census of rewrite statements alone cannot see it.
``omissible_scope``
    a derived-tier statement interpolates a fragment that one code path resolves
    to the empty string -- an optional scope predicate. This is the exact
    structural signature of "can be called archive-wide": the deleted sweep
    built ``scope_clause = ""`` when its ``session_ids`` argument was ``None``.
``read_path_substitution``
    a route substitutes a value for a stored field it found absent, either under
    an ``if`` that tests that same field or through an ``or`` fallback on it.
    ``module_writes`` records whether the enclosing module performs any DML, so
    a substitution on a pure-read route -- the mask shape -- is distinguishable
    from one inside a producer.

Validated against the two defects the campaign actually found
------------------------------------------------------------

Run against the tree at ``49ddd61d1^`` (the commit before PR #5372), this census
reports ``omissible_scope`` on
``storage/sqlite/archive_tiers/write.py::_repair_stale_prefix_branch_points_db``
and ``read_path_substitution`` on
``storage/derived/session/threads.py::_repair_profile_parent_ids`` -- the sweep
and the read-path mask, both named, neither by their names. At head both are
absent from those populations, because the fix made the scope argument required
and deleted the substitution. The check therefore discriminates the defect from
its own repaired successor, which a name-based search cannot.

What this census cannot see, stated plainly
-------------------------------------------

- **Reachability from a recurring entry point is not usable here, and this
  census does not claim it.** It was measured rather than assumed: over
  ``polylogue``'s 16,391 call-graph nodes, the import-following reachable set
  from ``daemon/cli.py``'s startup and from a single periodic service is 18,805
  names -- effectively the whole product, so "reachable from a daemon service"
  separates nothing. A call-only graph with the import edges removed severs at
  the first dynamic dispatch instead: 27 nodes from ``convergence_check``,
  which misses every stage and adapter that actually runs. Neither graph can
  carry the question, so the scope property is read off the statement instead.
- A statement assembled through a data structure, a registry, or a call whose
  return value this module does not constant-fold leaves the call site with no
  reconstructible text; it is skipped, not censused.
- A sweep expressed entirely in Python -- read a page of rows, loop, write each
  by id -- is censused only when the *read* is visible as an unbound derived
  ``SELECT`` in the same function. Split across two functions in two modules it
  is invisible here.
- A scope carried by something other than a bound parameter in the ``WHERE``
  region -- a join against a temporary table of ids, for instance -- is counted
  as unbound. That direction is conservative: it over-reports.
- ``read_path_substitution`` sees the two syntactic fallback shapes it names.
  A substitution written as an early ``return`` of a recomputed value, or one
  routed through a helper that takes the stored value as an argument, is not in
  its subject.

The declaration's floor is honest on purpose
--------------------------------------------

``unclassified`` means the census **saw** the site, not that the site is
legitimate. It is where a newly-observed sweep or substitution arrives: the
enforced property is that a new one cannot appear undeclared.

An empty floor is not a standing invariant and the gate does not enforce one: a
new site lands at the floor and stays there until someone reads the call site.
When a site fits no permitted member, it stays ``unclassified`` with a reason
saying so -- that is a finding about the route, not pending work. Stretching a
member to cover it would destroy the only signal this census produces.

Regenerate the declaration with :func:`render_declaration`.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from devtools.ast_cache import parse_path, walk_module
from devtools.sql_statement_text import (
    HOLE,
    SQL_EXECUTION_METHODS,
    function_scopes,
    statement_texts,
    string_values,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER

#: The declaration this census is checked against.
DECLARATION_PATH = "docs/plans/derived-sweep-census.yaml"

#: Tiers this census is about: rebuildable read models whose rows a corrective
#: sweep can quietly fix behind a broken producer. Durable rewrites are the
#: subject of ``devtools/durable_write_census.py``.
DERIVED_TIERS: frozenset[str] = frozenset({"index", "embeddings"})

#: Observation kinds. Every one is a population; none of them is a verdict.
OBSERVATION_KINDS: frozenset[str] = frozenset(
    {
        "unbound_rewrite",
        "unresolved_rewrite_scope",
        "state_selected_subject",
        "omissible_scope",
        "read_path_substitution",
    }
)

#: How a censused site is allowed to be legitimate. ``unclassified`` is the
#: ratchet's floor; the two forbidden tokens are the defects AC4 names, and
#: declaring either fails the gate so the defect can never be parked in the
#: declaration as though it were a classification.
CLASSIFICATION_VOCABULARY: dict[str, str] = {
    "producer_scoped_transaction": (
        "runs inside the write transaction that produced the rows, over the subject set that write touched"
    ),
    "whole_relation_derivation": (
        "replaces a derived relation wholesale from durable evidence; it re-derives rather than corrects, "
        "so a producer defect reappears on the next pass instead of being fixed away"
    ),
    "declared_retention": (
        "drops rows under a declared retention or garbage-collection policy; it removes rows it is allowed "
        "to remove and never rewrites a retained one"
    ),
    "bounded_key_publication": (
        "the derivation kernel's per-key atomic replacement: the subject is one required key, revalidated "
        "against its binding before publication"
    ),
    "operator_mutation": "reached only through an explicit, audited operator route, never from a recurring pass",
    "presentation_projection": (
        "the substituted value is not a stored producer output; the route composes a view over values it "
        "computed itself"
    ),
    "test_or_fixture_construct": "seeds or mutates a throwaway archive, not a route over an operator archive",
    "unclassified": (
        "observed and pinned by the census, not yet adjudicated. This token is the ratchet's floor: it stops "
        "a NEW sweep or substitution from appearing unnoticed without asserting that the existing site is "
        "legitimate. Replacing it with a real classification is the adjudication pass; leaving it is honest, "
        "but it is not evidence that the site is not a mask"
    ),
    "archive_wide_corrective_sweep": (
        "a pass that corrects derived rows the producer got wrong, over the whole archive rather than over "
        "what a write touched -- the defect AC4 forbids. Declaring this token fails the gate by design; fix "
        "the producer and report the residue instead"
    ),
    "read_path_mask": (
        "a read that re-derives or corrects a stored value instead of reporting what was stored, so a "
        "producer that stopped writing it stays invisible. Declaring this token fails the gate by design"
    ),
}

#: Classifications whose presence in the declaration is itself a violation.
FORBIDDEN_CLASSIFICATIONS: frozenset[str] = frozenset({"archive_wide_corrective_sweep", "read_path_mask"})

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_REWRITE_RE = re.compile(
    r"\b(?P<verb>UPDATE\s+OR\s+\w+|UPDATE|DELETE\s+FROM|INSERT\s+OR\s+REPLACE\s+INTO|REPLACE\s+INTO|INSERT\s+INTO)"
    r"\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?[`\"\[]?(?P<table>[A-Za-z_][A-Za-z0-9_]*|\{\})",
    re.IGNORECASE,
)
_DO_UPDATE_RE = re.compile(r"ON\s+CONFLICT\b.*?\bDO\s+UPDATE\b", re.IGNORECASE | re.DOTALL)
_SELECT_HEAD_RE = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_SOURCE_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_SELECTION_TERMINATOR_RE = re.compile(r"\b(?:ORDER\s+BY|GROUP\s+BY|LIMIT|RETURNING)\b", re.IGNORECASE)
_BOUND_PARAMETER_RE = re.compile(r"\?|:[A-Za-z_][A-Za-z0-9_]*")
_CALLER_KEY_BINDING_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
    r"(?:id|[A-Za-z_][A-Za-z0-9_]*_id)\s*(?:=|IN\s*\()\s*(?:\?|:[A-Za-z_][A-Za-z0-9_]*|\()",
    re.IGNORECASE,
)
_ANY_DML_RE = re.compile(
    r"\b(?:UPDATE|DELETE\s+FROM|INSERT\s+INTO|INSERT\s+OR\s+\w+\s+INTO|REPLACE\s+INTO)\b",
    re.IGNORECASE,
)

#: Rewrite kinds whose subject set is a row selection rather than the values
#: being inserted. An ``INSERT``/upsert names its subject by the key it writes.
_SELECTING_KINDS: frozenset[str] = frozenset({"update", "delete"})

#: Copy-with-changes constructors a substitution can be written through.
_SUBSTITUTION_METHODS: frozenset[str] = frozenset({"model_copy", "_replace"})


@dataclass(frozen=True)
class SweepSite:
    """One censused observation, keyed so line drift cannot churn the entry."""

    file: str
    function: str
    subject: str
    kind: str
    scope: str
    line: int

    @property
    def key(self) -> str:
        return f"{self.file}::{self.function}::{self.subject}::{self.kind}"


@dataclass(frozen=True)
class CensusObservation:
    sites: tuple[SweepSite, ...]


def derived_table_tiers() -> dict[str, frozenset[str]]:
    """Map every archive table name to all owning tiers, from the live DDL."""
    tiers: dict[str, set[str]] = {}
    for tier, ddl in ARCHIVE_DDL_BY_TIER.items():
        name = str(getattr(tier, "value", tier))
        for table in _CREATE_TABLE_RE.findall(ddl):
            tiers.setdefault(table, set()).add(name)
    return {table: frozenset(names) for table, names in tiers.items()}


#: Scope verdicts, worst first. A call site that can take several statement
#: texts is reported at its worst candidate, so an optional predicate cannot
#: hide behind the branch that supplies it.
_SCOPE_PRECEDENCE: tuple[str, ...] = ("no_where", "state_predicate", "unresolved", "bound")


def _worst_scope(scopes: Iterable[str]) -> str:
    seen = set(scopes)
    for verdict in _SCOPE_PRECEDENCE:
        if verdict in seen:
            return verdict
    return "bound"


def _selection_region(sql: str, start: int) -> str | None:
    """The statement's own row-selection text: its ``WHERE`` clause.

    Bounded at both ends on purpose. It begins at the first ``WHERE`` after the
    rewrite verb, because a correlated subquery's ``WHERE`` is not the
    statement's. It ends at the first depth-0 ``ORDER BY`` / ``GROUP BY`` /
    ``LIMIT`` / ``RETURNING`` / ``;``, because an optional ``LIMIT ?`` tail is a
    bound *bound*, not a bound subject -- the deleted lineage sweep carried
    exactly that tail while selecting its rows archive-wide, and counting it
    would have certified the sweep as scoped.

    Subqueries are kept. ``id IN (SELECT id FROM t WHERE k = ?)`` binds the
    subject just as ``id IN (?)`` does; what the defect looks like is a
    selection with no caller-supplied value anywhere in it.
    """
    region = sql[start:]
    depth = 0
    quote: str | None = None
    where_end: int | None = None
    for index, character in enumerate(region):
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in "'\"`":
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif (
            depth == 0
            and region[index : index + 5].upper() == "WHERE"
            and (index + 5 == len(region) or not (region[index + 5].isalnum() or region[index + 5] == "_"))
        ):
            where_end = index + 5
            break
    if where_end is None:
        return None
    tail = region[where_end:]
    depth = 0
    for index, character in enumerate(tail):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if character == ";":
                return tail[:index]
            if _SELECTION_TERMINATOR_RE.match(tail, index):
                return tail[:index]
    return tail


def _statement_scope(sql: str, start: int) -> str:
    """Classify where the rows a statement acts on come from.

    ``bound`` means a caller-supplied parameter appears in the statement's own
    selection, so the subject came from the caller. ``state_predicate``
    and ``no_where`` mean the subject is whatever the archive contains.
    ``unresolved`` means an interpolation hole sits in the selection and the
    census cannot tell.
    """
    predicate = _selection_region(sql, start)
    if predicate is None:
        return "no_where"
    if _BOUND_PARAMETER_RE.search(predicate) and _CALLER_KEY_BINDING_RE.search(predicate):
        return "bound"
    if HOLE in predicate:
        return "unresolved"
    return "state_predicate"


def _rewrites(sql: str, tiers: Mapping[str, frozenset[str]]) -> list[tuple[str, str, str]]:
    """Return ``(table, kind, scope)`` for every derived rewrite in *sql*."""
    found: list[tuple[str, str, str]] = []
    has_do_update = bool(_DO_UPDATE_RE.search(sql))
    for match in _REWRITE_RE.finditer(sql):
        verb = re.sub(r"\s+", " ", match.group("verb").upper())
        table = match.group("table")
        if verb.startswith("UPDATE"):
            kind = "update"
        elif verb == "DELETE FROM":
            kind = "delete"
        elif verb in {"INSERT OR REPLACE INTO", "REPLACE INTO"}:
            kind = "insert_or_replace"
        elif verb == "INSERT INTO" and has_do_update:
            kind = "upsert_do_update"
        else:
            continue
        if table == HOLE:
            found.append((HOLE, kind, _statement_scope(sql, match.start())))
            continue
        if tiers.get(table, frozenset()) & DERIVED_TIERS:
            found.append((table, kind, _statement_scope(sql, match.start())))
    return found


def _derived_reads(sql: str, tiers: Mapping[str, frozenset[str]]) -> list[tuple[str, str]]:
    """Return ``(table, scope)`` for a query whose subject is a derived table."""
    select = re.search(r"\bSELECT\b", sql, re.IGNORECASE)
    if select is None:
        return []
    projection = sql[select.end() :].lstrip()
    if projection.startswith("1") and (len(projection) == 1 or not (projection[1].isalnum() or projection[1] == "_")):
        # Diagnostic existence probes do not yield caller-selected subjects;
        # treating them as the row source for a later UPDATE is a false link.
        return []
    scope = _statement_scope(sql, select.start())
    return [
        (table, scope)
        for table in dict.fromkeys(_SOURCE_TABLE_RE.findall(sql[select.start() :]))
        if tiers.get(table, frozenset()) & DERIVED_TIERS
    ]


def _omissible_names(expression: ast.AST, values: Mapping[str, tuple[str, ...]]) -> set[str]:
    """Interpolated names one code path resolves to the empty string."""
    names: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name):
            candidates = values.get(node.value.id, ())
            if len(candidates) > 1 and any(not candidate.strip() for candidate in candidates):
                names.add(node.value.id)
    return names


def _attribute_names(node: ast.AST) -> set[str]:
    return {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}


def _substituted_fields(call: ast.Call) -> set[str]:
    """Fields a copy-with-changes call replaces."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "model_copy":
        for keyword in call.keywords:
            if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                return {
                    key.value
                    for key in keyword.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
        return set()
    if isinstance(func, ast.Attribute) and func.attr in _SUBSTITUTION_METHODS:
        return {keyword.arg for keyword in call.keywords if keyword.arg}
    if isinstance(func, ast.Name) and func.id == "replace":
        return {keyword.arg for keyword in call.keywords if keyword.arg}
    return set()


def _substitution_sites(
    tree: ast.Module,
    scopes: Mapping[ast.AST, str],
) -> list[tuple[str, str, int]]:
    """Find ``(function, field, line)`` for each substitute-if-absent shape.

    Two shapes, both of which say "the stored value was missing, so put one
    here": a copy-with-changes for field ``K`` inside a branch whose test reads
    ``.K``, and ``stored.K or <derived>`` supplied as the new value of ``K``.
    """
    found: list[tuple[str, str, int]] = []
    for node in walk_module(tree):
        if isinstance(node, ast.If | ast.IfExp):
            guard = _attribute_names(node.test)
            absent_in_true = isinstance(node.test, ast.Compare) and any(
                isinstance(op, ast.Is) and isinstance(value, ast.Constant) and value.value is None
                for op, value in zip(node.test.ops, node.test.comparators, strict=False)
            )
            absent_in_true |= isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
            body: list[ast.AST]
            if isinstance(node, ast.If):
                body = list(node.body if absent_in_true else node.orelse)
            else:
                body = [node.body if absent_in_true else node.orelse]
            for statement in body:
                for call in ast.walk(statement):
                    if not isinstance(call, ast.Call):
                        continue
                    for field in sorted(_substituted_fields(call) & guard):
                        found.append((scopes.get(call, "<module>"), field, call.lineno))
        if isinstance(node, ast.Call):
            updates: dict[str, ast.AST] = {}
            if isinstance(node.func, ast.Attribute) and node.func.attr == "model_copy":
                update = next((keyword.value for keyword in node.keywords if keyword.arg == "update"), None)
                if isinstance(update, ast.Dict):
                    updates = {
                        key.value: value
                        for key, value in zip(update.keys, update.values, strict=False)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
            elif isinstance(node.func, (ast.Attribute, ast.Name)) and (
                node.func.attr in _SUBSTITUTION_METHODS
                if isinstance(node.func, ast.Attribute)
                else node.func.id == "replace"
            ):
                updates = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            for field, value in updates.items():
                is_or_fallback = isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or)
                if is_or_fallback and field in _attribute_names(value):
                    found.append((scopes.get(node, "<module>"), field, node.lineno))
    return found


def census_package(package_root: Path, *, repo_root: Path) -> CensusObservation:
    """Census every derived-tier sweep candidate and read-path substitution."""
    census = DerivedSweepCensus()
    for path in sorted(package_root.rglob("*.py")):
        try:
            tree = parse_path(path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        census.observe(tree, relative=path.relative_to(repo_root).as_posix())
    return census.finish()


class DerivedSweepCensus:
    """The census, fed one parsed module at a time in path order.

    A caller that already walks the package parses each module once and hands
    it to every census that reads it, so no census holds the package's trees.
    Repeated SQL sites retain the broadest scope and its source location.
    Non-SQL sites and equal-scope sites retain their first observation.
    """

    def __init__(self) -> None:
        self._tiers = derived_table_tiers()
        self._sites: dict[str, SweepSite] = {}

    def finish(self) -> CensusObservation:
        return CensusObservation(sites=tuple(sorted(self._sites.values(), key=lambda item: item.key)))

    def observe(self, tree: ast.Module, *, relative: str) -> None:
        tiers = self._tiers
        sites = self._sites

        def record(site: SweepSite) -> None:
            previous = sites.get(site.key)
            if previous is None or (
                previous.scope in _SCOPE_PRECEDENCE
                and site.scope in _SCOPE_PRECEDENCE
                and _worst_scope((previous.scope, site.scope)) != previous.scope
            ):
                sites[site.key] = site

        scopes = function_scopes(tree)

        # First pass: what each call site executes. The populations below are
        # function-scoped -- an unbound read only matters in a function that
        # also rewrites -- so nothing can be decided until the whole module has
        # been seen.
        rewriting_functions: set[str] = set()
        reads_by_function: dict[str, set[tuple[str, str]]] = {}
        omissible_by_function: dict[str, set[tuple[str, int]]] = {}
        pending_rewrites: list[SweepSite] = []
        module_writes = False
        # A scope's string bindings are a function of the tree alone, so each
        # scope resolves once per module rather than once per call site in it.
        values_by_function: dict[str, dict[str, tuple[str, ...]]] = {}

        for node in walk_module(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in SQL_EXECUTION_METHODS or not node.args:
                continue
            function = scopes.get(node, "<module>")
            values = values_by_function.get(function)
            if values is None:
                values = values_by_function[function] = string_values(tree, scope=function, scopes=scopes)
            texts = statement_texts(node.args[0], values)
            if not texts:
                continue
            omissible = _omissible_names(node.args[0], values)
            # One call site, several candidate texts: collect every candidate's
            # verdict and report the worst, so an optional scope predicate is
            # judged on the branch that omits it.
            rewrite_scopes: dict[tuple[str, str], list[str]] = {}
            read_scopes: dict[str, list[str]] = {}
            for sql in texts:
                if _ANY_DML_RE.search(sql):
                    module_writes = True
                for table, kind, scope in _rewrites(sql, tiers):
                    rewrite_scopes.setdefault((table, kind), []).append(scope)
                for table, scope in _derived_reads(sql, tiers):
                    read_scopes.setdefault(table, []).append(scope)

            touches_derived = bool(rewrite_scopes or read_scopes)
            if rewrite_scopes:
                rewriting_functions.add(function)
            for (table, kind), candidates in rewrite_scopes.items():
                scope = _worst_scope(candidates)
                if scope == "bound":
                    continue
                if kind not in _SELECTING_KINDS and not any(
                    re.search(r"\bSELECT\b.*\bFROM\b", text, re.IGNORECASE | re.DOTALL) for text in texts
                ):
                    continue
                observation = "unresolved_rewrite_scope" if scope == "unresolved" else "unbound_rewrite"
                pending_rewrites.append(
                    SweepSite(
                        file=relative,
                        function=function,
                        subject=table,
                        kind=observation,
                        scope=scope,
                        line=node.lineno,
                    )
                )
            for table, candidates in read_scopes.items():
                scope = _worst_scope(candidates)
                if scope != "bound":
                    reads_by_function.setdefault(function, set()).add((table, scope))
            if touches_derived:
                for name in sorted(omissible):
                    omissible_by_function.setdefault(function, set()).add((name, node.lineno))

        for site in pending_rewrites:
            record(site)

        for function, reads in reads_by_function.items():
            if function not in rewriting_functions:
                continue
            for table, scope in sorted(reads):
                record(
                    SweepSite(
                        file=relative,
                        function=function,
                        subject=table,
                        kind="state_selected_subject",
                        scope=scope,
                        line=0,
                    )
                )

        # A conditionally-omitted predicate is only a scope question where the
        # statement can change rows. On a pure read route it is an ordinary
        # optional filter, and censusing those would bury the population.
        for function, names in omissible_by_function.items():
            if function not in rewriting_functions:
                continue
            for name, line in sorted(names):
                record(
                    SweepSite(
                        file=relative,
                        function=function,
                        subject=name,
                        kind="omissible_scope",
                        scope="omissible",
                        line=line,
                    )
                )

        # Substitution shape is independently meaningful even when the input
        # profile was loaded by a caller and this module contains no SQL.
        for function, field, line in _substitution_sites(tree, scopes):
            record(
                SweepSite(
                    file=relative,
                    function=function,
                    subject=field,
                    kind="read_path_substitution",
                    scope="writing_module" if module_writes else "read_only_module",
                    line=line,
                )
            )


@dataclass(frozen=True)
class CensusEntry:
    key: str
    file: str
    function: str
    subject: str
    kind: str
    scope: str
    classification: str
    reason: str


@dataclass(frozen=True)
class CensusDeclaration:
    package: str
    entries: dict[str, CensusEntry]
    malformed: tuple[str, ...]


def load_declaration(path: Path) -> CensusDeclaration:
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data: Mapping[str, object] = raw if isinstance(raw, dict) else {}
    entries: dict[str, CensusEntry] = {}
    malformed: list[str] = []
    rows = data.get("sites")
    for index, item in enumerate(list(rows) if isinstance(rows, list) else []):
        if not isinstance(item, dict):
            malformed.append(f"sites[{index}]")
            continue
        file = item.get("file")
        function = item.get("function")
        subject = item.get("subject")
        kind = item.get("kind")
        if not all(isinstance(value, str) for value in (file, function, subject, kind)):
            malformed.append(f"sites[{index}]")
            continue
        entry = CensusEntry(
            key=f"{file}::{function}::{subject}::{kind}",
            file=str(file),
            function=str(function),
            subject=str(subject),
            kind=str(kind),
            scope=str(item.get("scope") or ""),
            classification=str(item.get("classification") or ""),
            reason=str(item.get("reason") or "").strip(),
        )
        entries[entry.key] = entry
    return CensusDeclaration(
        package=str(data.get("package") or "polylogue"),
        entries=entries,
        malformed=tuple(malformed),
    )


def collect_violations(
    *,
    repo_root: Path,
    declaration_path: Path | None = None,
    observation: CensusObservation | None = None,
) -> list[dict[str, object]]:
    """Check the observed sweep census against its declaration.

    *observation* is the census of the declaration's package when the caller
    has already taken it in its own pass over the package.
    """
    path = declaration_path or (repo_root / DECLARATION_PATH)
    if not path.is_file():
        return [{"rule": "derived_sweep_census_declaration_missing", "key": path.as_posix()}]
    declaration = load_declaration(path)
    package_root = repo_root / declaration.package
    if declaration.package != "polylogue" or not package_root.is_dir():
        return [{"rule": "derived_sweep_census_package_invalid", "key": declaration.package}]
    if observation is None:
        observation = census_package(package_root, repo_root=repo_root)

    violations: list[dict[str, object]] = []
    for name in declaration.malformed:
        violations.append({"rule": "derived_sweep_census_row_malformed", "key": name})

    observed = {site.key: site for site in observation.sites}
    for key in sorted(observed.keys() - declaration.entries.keys()):
        site = observed[key]
        violations.append(
            {
                "rule": "derived_sweep_undeclared",
                "key": key,
                "file": site.file,
                "line": site.line,
                "scope": site.scope,
                "detail": (
                    f"{site.kind} on derived subject {site.subject}; declare it in "
                    f"{DECLARATION_PATH} with a classification and a reason"
                ),
            }
        )
    for key in sorted(declaration.entries.keys() - observed.keys()):
        violations.append(
            {
                "rule": "derived_sweep_census_stale",
                "key": key,
                "file": declaration.entries[key].file,
                "detail": "declared sweep candidate is no longer present -- drop the entry",
            }
        )
    for key in sorted(declaration.entries.keys() & observed.keys()):
        entry = declaration.entries[key]
        site = observed[key]
        if entry.kind not in OBSERVATION_KINDS:
            violations.append(
                {
                    "rule": "derived_sweep_unknown_kind",
                    "key": key,
                    "file": site.file,
                    "declared": entry.kind,
                    "detail": f"allowed: {', '.join(sorted(OBSERVATION_KINDS))}",
                }
            )
        if entry.scope != site.scope:
            violations.append(
                {
                    "rule": "derived_sweep_scope_drift",
                    "key": key,
                    "file": site.file,
                    "declared": entry.scope,
                    "observed": site.scope,
                }
            )
        if entry.classification not in CLASSIFICATION_VOCABULARY:
            violations.append(
                {
                    "rule": "derived_sweep_unknown_classification",
                    "key": key,
                    "file": site.file,
                    "declared": entry.classification,
                    "detail": f"allowed: {', '.join(sorted(CLASSIFICATION_VOCABULARY))}",
                }
            )
        elif entry.classification in FORBIDDEN_CLASSIFICATIONS:
            violations.append(
                {
                    "rule": "derived_sweep_masks_a_producer",
                    "key": key,
                    "file": site.file,
                    "line": site.line,
                    "detail": CLASSIFICATION_VOCABULARY[entry.classification],
                }
            )
        if not entry.reason:
            violations.append({"rule": "derived_sweep_reason_missing", "key": key, "file": site.file})
    return violations


def summarize(observation: CensusObservation) -> dict[str, object]:
    """Counts used by the gate's success line and by its JSON output."""
    by_kind: dict[str, int] = {}
    for site in observation.sites:
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1
    return {
        "derived_sweep_sites": len(observation.sites),
        "by_kind": dict(sorted(by_kind.items())),
    }


def iter_site_payloads(sites: Iterable[SweepSite]) -> list[dict[str, object]]:
    return [
        {
            "file": site.file,
            "function": site.function,
            "subject": site.subject,
            "kind": site.kind,
            "scope": site.scope,
            "line": site.line,
        }
        for site in sites
    ]


_DECLARATION_HEADER = """\
# Census of derived-tier corrective sweeps and read-path substitutions.
# Generated by devtools/derived_sweep_census.py::render_declaration;
# enforced by `devtools gate layering`.
#
# `unclassified` means the census SAW the site, not that the site is legitimate.
# It is the ratchet's floor: a NEW archive-wide derived sweep, or a NEW read-path
# substitution, cannot appear undeclared, and a new one arrives at that floor.
# Existing classifications record adjudication. New entries are pinned as
# unclassified until a reviewer assigns a permitted classification.
package: polylogue
sites:
"""


def render_declaration(
    observation: CensusObservation,
    *,
    existing: Mapping[str, CensusEntry] | None = None,
) -> str:
    """Render the declaration, preserving any classification already recorded."""
    known = dict(existing or {})
    pending = any(
        site.key not in known or known[site.key].classification == "unclassified" for site in observation.sites
    )
    header = _DECLARATION_HEADER
    if not pending:
        header = header.replace(
            "# Existing classifications record adjudication. New entries are pinned as\n# unclassified until a reviewer assigns a permitted classification.\n",
            "# Every entry below has been adjudicated; a reason that says no permitted member\n# fits is a finding, not pending work.\n",
        )
    lines = [header]
    for site in observation.sites:
        entry = known.get(site.key)
        classification = entry.classification if entry else "unclassified"
        reason = (
            entry.reason
            if entry and entry.reason
            else "newly visible to the census; route review is needed before classification"
        )
        lines.append(
            f"  - file: {json.dumps(site.file, ensure_ascii=False)}\n"
            f"    function: {json.dumps(site.function, ensure_ascii=False)}\n"
            f"    subject: {json.dumps(site.subject, ensure_ascii=False)}\n"
            f"    kind: {json.dumps(site.kind, ensure_ascii=False)}\n"
            f"    scope: {json.dumps(site.scope, ensure_ascii=False)}\n"
            f"    classification: {classification}\n"
            f"    reason: {json.dumps(reason, ensure_ascii=False)}\n"
        )
    return "".join(lines)
