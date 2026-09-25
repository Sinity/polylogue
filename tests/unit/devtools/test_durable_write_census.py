"""Regression contract for the durable-write census.

The fixtures are source trees rather than SQLite databases: this checker
answers a static ownership question, and every assertion runs the production
AST census rather than a test-local imitation of it.
"""

from __future__ import annotations

from pathlib import Path

from devtools import repo_root
from devtools.durable_write_census import (
    census_package,
    collect_violations,
    durable_table_tiers,
)


def _module(root: Path, body: str) -> None:
    package = root / "polylogue" / "storage" / "sqlite" / "archive_tiers"
    package.mkdir(parents=True, exist_ok=True)
    (package / "writer.py").write_text(body, encoding="utf-8")


def _declaration(root: Path, body: str = "package: polylogue\nwrites: []\n") -> Path:
    path = root / "census.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _rules(root: Path, declaration: Path) -> set[str]:
    return {str(item["rule"]) for item in collect_violations(repo_root=root, declaration_path=declaration)}


def test_runtime_persistent_creation_then_rewrite_is_not_hidden_by_missing_canonical_ddl(tmp_path: Path) -> None:
    """A runtime-created persistent relation is rejected when rewritten.

    Anti-vacuity: changing the UPDATE to an INSERT leaves the same table name
    and CREATE statement, but must remove the finding.  The check therefore
    proves a potentially destructive rewrite, not a name match.
    """
    _module(
        tmp_path,
        "def mutate(conn):\n"
        '    conn.execute("CREATE TABLE runtime_only (value TEXT)")\n'
        "    conn.execute(\"UPDATE runtime_only SET value = 'new'\")\n",
    )
    declaration = _declaration(tmp_path)
    assert _rules(tmp_path, declaration) == {"runtime_persistent_table_rewrite_undeclared"}

    _module(
        tmp_path,
        "def mutate(conn):\n"
        '    conn.execute("CREATE TABLE runtime_only (value TEXT)")\n'
        "    conn.execute(\"INSERT INTO runtime_only(value) VALUES ('new')\")\n",
    )
    assert _rules(tmp_path, declaration) == set()


def test_temporary_and_in_memory_scratch_creations_have_non_archive_dispositions(tmp_path: Path) -> None:
    """Temp and private-memory relations never acquire an invented durable tier.

    Anti-vacuity: removing either classification makes that relation enter the
    persistent runtime population and its UPDATE raises the missing-DDL rule.
    """
    _module(
        tmp_path,
        "import sqlite3\n"
        "def temporary(conn):\n"
        '    conn.execute("CREATE TEMP TABLE transient_rows (value TEXT)")\n'
        "    conn.execute(\"UPDATE transient_rows SET value = 'new'\")\n"
        "def scratch():\n"
        '    scratch_conn = sqlite3.connect(":memory:")\n'
        '    scratch_conn.execute("CREATE TABLE scratch_rows (value TEXT)")\n'
        "    scratch_conn.execute(\"UPDATE scratch_rows SET value = 'new'\")\n",
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert {(item.table, item.disposition) for item in observation.runtime_creations} == {
        ("scratch_rows", "scratch"),
        ("transient_rows", "temporary"),
    }
    assert _rules(tmp_path, _declaration(tmp_path)) == set()


def test_no_effect_lock_upgrade_requires_its_exact_constant_false_predicate(tmp_path: Path) -> None:
    """The lock upgrade is accepted only while its SQL remains rowless.

    Anti-vacuity: changing ``WHERE 0`` to ``WHERE 1`` changes the observed
    kind back to a regular durable update, so both a stale declaration and an
    undeclared rewrite are reported.
    """
    declaration = _declaration(
        tmp_path,
        "package: polylogue\n"
        "writes:\n"
        '  - file: "polylogue/storage/sqlite/archive_tiers/writer.py"\n'
        '    function: "lock"\n'
        '    table: "assertions"\n'
        '    kind: "no_effect_update"\n'
        '    tier: "user"\n'
        "    classification: no_effect_lock_upgrade\n"
        '    reason: "constant false lock upgrade"\n',
    )
    _module(
        tmp_path,
        'def lock(conn):\n    conn.execute("UPDATE assertions SET updated_at_ms = updated_at_ms WHERE 0")\n',
    )
    assert _rules(tmp_path, declaration) == set()

    _module(
        tmp_path,
        'def lock(conn):\n    conn.execute("UPDATE assertions SET updated_at_ms = updated_at_ms WHERE 1")\n',
    )
    assert _rules(tmp_path, declaration) == {"durable_write_census_stale", "durable_write_undeclared"}


def test_dynamic_helper_callers_are_checked_against_canonical_index_ddl(tmp_path: Path) -> None:
    """A dynamic helper remains accepted only for its current index targets.

    Anti-vacuity: changing the caller from ``session_profiles`` to durable
    ``assertions`` leaves the helper and its dynamic SQL untouched, but the
    caller target must fail the gate.
    """
    declaration = _declaration(
        tmp_path,
        "package: polylogue\n"
        "writes:\n"
        '  - file: "polylogue/storage/sqlite/archive_tiers/writer.py"\n'
        '    function: "replace"\n'
        '    table: "?"\n'
        '    kind: "delete"\n'
        '    tier: "unresolved"\n'
        "    classification: rebuildable_dynamic_target\n"
        '    reason: "checked caller targets"\n',
    )
    _module(
        tmp_path,
        "def replace(conn, table):\n"
        '    conn.execute(f"DELETE FROM {table}")\n'
        "def call(conn):\n"
        '    replace(conn, "session_profiles")\n',
    )
    assert _rules(tmp_path, declaration) == set()

    _module(
        tmp_path,
        "def replace(conn, table):\n"
        '    conn.execute(f"DELETE FROM {table}")\n'
        "def call(conn):\n"
        '    replace(conn, "assertions")\n',
    )
    assert _rules(tmp_path, declaration) == {"dynamic_table_target_not_index"}


def test_excision_policy_projection_is_owned_by_canonical_source_ddl() -> None:
    """The previously runtime-created policy table is source-owned at head.

    Anti-vacuity: removing its CREATE TABLE from canonical source DDL drops it
    from this map, which makes the assertion red even if a writer still names
    the relation.
    """
    assert durable_table_tiers()["excision_policy_projections"] == "source"


def test_head_declaration_matches_the_census() -> None:
    """The checked-in declaration names every current checked runtime route."""
    assert collect_violations(repo_root=repo_root()) == []
