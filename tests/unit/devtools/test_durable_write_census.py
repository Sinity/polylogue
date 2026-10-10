"""Regression contract for the durable-write census.

The fixtures are source trees rather than SQLite databases: this checker
answers a static ownership question, and every assertion runs the production
AST census rather than a test-local imitation of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


def test_private_name_collision_preserves_creator_boundary_and_archive_writes(tmp_path: Path) -> None:
    """A private name collision cannot lend another module runtime authority."""
    _module(
        tmp_path,
        "def mutate(conn):\n"
        '    conn.execute("CREATE TABLE shared_private (value TEXT)")\n'
        '    conn.execute("UPDATE shared_private SET value = 1")\n',
    )
    outside = tmp_path / "polylogue" / "operations" / "private_spool.py"
    outside.parent.mkdir(parents=True)
    private_body = (
        "def mutate(conn):\n"
        '    conn.execute("CREATE TABLE shared_private (value TEXT)")\n'
        '    conn.execute("UPDATE shared_private SET value = 1")\n'
    )
    outside.write_text(private_body, encoding="utf-8")
    declaration = _declaration(
        tmp_path,
        "package: polylogue\nruntime_tables:\n"
        '  - file: "polylogue/storage/sqlite/archive_tiers/writer.py"\n'
        '    function: "mutate"\n'
        '    table: "shared_private"\n'
        "    disposition: disposable_scratch\n"
        '    reason: "Private disk spool owned by this storage creator."\n'
        "writes: []\n",
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert {(row.file, row.table) for row in observation.runtime_creations} == {
        ("polylogue/storage/sqlite/archive_tiers/writer.py", "shared_private")
    }
    assert {(row.file, row.tier) for row in observation.sites} == {
        ("polylogue/storage/sqlite/archive_tiers/writer.py", "runtime")
    }
    assert _rules(tmp_path, declaration) == set()

    # Actual archive writes outside storage retain both archive detection and
    # their own runtime relation census, even when the relation name collides.
    outside.write_text(
        private_body + '    conn.execute("UPDATE assertions SET updated_at_ms = 1")\n',
        encoding="utf-8",
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert ("polylogue/operations/private_spool.py", "shared_private") in {
        (row.file, row.table) for row in observation.runtime_creations
    }
    assert ("polylogue/operations/private_spool.py", "assertions", "user") in {
        (row.file, row.table, row.tier) for row in observation.sites
    }
    assert "runtime_persistent_table_rewrite_undeclared" in _rules(tmp_path, declaration)


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


def test_declared_private_scratch_table_can_be_rewritten_across_methods(tmp_path: Path) -> None:
    """A scoped scratch authority covers writes through the scratch object's connection.

    Anti-vacuity: changing the disposition to the ops-only token must both
    reject the declaration and leave the cross-method rewrite visible.
    """
    package = tmp_path / "polylogue" / "storage" / "sqlite" / "archive_tiers"
    package.mkdir(parents=True, exist_ok=True)
    (package / "write.py").write_text(
        "import sqlite3\n"
        "class Scratch:\n"
        "    def __init__(self, path):\n"
        "        self.conn = sqlite3.connect(path)\n"
        '        self.conn.execute("CREATE TABLE scratch_rows (value TEXT)")\n'
        "    def replace(self, value):\n"
        '        self.conn.execute("INSERT OR REPLACE INTO scratch_rows VALUES (?)", (value,))\n',
        encoding="utf-8",
    )
    declaration = _declaration(
        tmp_path,
        "package: polylogue\n"
        "runtime_tables:\n"
        '  - file: "polylogue/storage/sqlite/archive_tiers/write.py"\n'
        '    function: "Scratch.__init__"\n'
        '    table: "scratch_rows"\n'
        "    disposition: disposable_scratch\n"
        '    reason: "A private per-operation scratch connection."\n'
        "writes: []\n",
    )
    assert _rules(tmp_path, declaration) == set()

    declaration.write_text(
        declaration.read_text(encoding="utf-8").replace("disposable_scratch", "disposable_ops"),
        encoding="utf-8",
    )
    assert _rules(tmp_path, declaration) == {
        "runtime_table_disposition_invalid",
        "runtime_persistent_table_rewrite_undeclared",
    }


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
    violations = collect_violations(repo_root=repo_root())
    assert not violations, violations


def test_production_membership_creator_is_temporary_on_its_owned_reader() -> None:
    root = repo_root()
    observation = census_package(root / "polylogue", repo_root=root)
    creations = [
        creation
        for creation in observation.runtime_creations
        if creation.file == "polylogue/storage/sqlite/archive_tiers/write.py"
        and creation.function == "_acompact_content_membership_ratio"
        and creation.table == "membership"
    ]
    assert len(creations) == 1
    assert creations[0].disposition == "temporary"
    assert creations[0].key.endswith("::temporary")
    assert not any(
        str(violation.get("key", "")).startswith(
            "polylogue/storage/sqlite/archive_tiers/write.py::_acompact_content_membership_ratio::membership"
        )
        for violation in collect_violations(repo_root=root)
    )


def test_private_witness_hydration_is_not_a_general_scratch_rewrite_exemption(tmp_path: Path) -> None:
    _module(tmp_path, "def hydrate(conn):\n    conn.execute('DELETE FROM raw_sessions')\n")
    declaration = _declaration(
        tmp_path,
        "package: polylogue\nwrites:\n"
        "  - file: polylogue/storage/sqlite/archive_tiers/writer.py\n"
        "    function: hydrate\n    table: raw_sessions\n    kind: delete\n"
        "    tier: source\n    classification: private_witness_hydration\n"
        "    reason: Claimed private scratch baseline.\n",
    )
    assert _rules(tmp_path, declaration) == {"private_witness_hydration_site_invalid"}
    assert len(census_package(tmp_path / "polylogue", repo_root=tmp_path).sites) == 1


@pytest.mark.parametrize(
    "mutation",
    ["none", "live_receiver", "rowid", "tables", "retained_tier", "missing_restore", "restore_order", "hydration"],
)
def test_private_witness_hydration_requires_the_reviewed_same_key_shape(tmp_path: Path, mutation: str) -> None:
    tree = ast.parse((repo_root() / "polylogue/storage/sqlite/reference_seal.py").read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PreparedIndexMutation")
    method = next(
        node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_seed_source_controls"
    )
    loop = next(node for node in method.body if isinstance(node, ast.For))
    phase = next(node for node in loop.body if isinstance(node, ast.With))
    if mutation == "live_receiver":
        deletion = phase.body[0]
        assert isinstance(deletion, ast.With)
        call = deletion.items[0].context_expr
        assert isinstance(call, ast.Call)
        call.args[0] = ast.parse("self._observers['source'].connection", mode="eval").body
    elif mutation == "rowid":
        deletion = phase.body[0]
        assert isinstance(deletion, ast.With)
        call = deletion.items[0].context_expr
        assert isinstance(call, ast.Call) and isinstance(call.args[1], ast.JoinedStr)
        call.args[1].values[-1] = ast.Constant(" WHERE rowid=2")
    elif mutation == "tables":
        loop.iter = ast.parse("('raw_existence_journal_control', 'raw_sessions')", mode="eval").body
    elif mutation == "retained_tier":
        retained = loop.body[0]
        assert isinstance(retained, ast.Assign) and isinstance(retained.value, ast.Call)
        retained.value.args[0] = ast.Constant("user")
    elif mutation == "missing_restore":
        phase.body[1] = ast.Pass()
    elif mutation == "restore_order":
        phase.body[0], phase.body[1] = phase.body[1], phase.body[0]
    elif mutation == "hydration":
        phase.items[0].context_expr = ast.parse("self._ordinary_write_phase()", mode="eval").body
    path = tmp_path / "polylogue/storage/sqlite/reference_seal.py"
    path.parent.mkdir(parents=True)
    ast.fix_missing_locations(method)
    path.write_text(
        "class PreparedIndexMutation:\n" + "\n".join("    " + line for line in ast.unparse(method).splitlines()) + "\n"
    )
    declaration = _declaration(
        tmp_path,
        "package: polylogue\nwrites:\n"
        "  - file: polylogue/storage/sqlite/reference_seal.py\n"
        "    function: PreparedIndexMutation._seed_source_controls\n"
        "    table: '?'\n    kind: delete\n    tier: unresolved\n"
        "    classification: private_witness_hydration\n"
        "    reason: Exact original private seed restoration.\n",
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert len(observation.sites) == 1
    assert _rules(tmp_path, declaration) == (
        set() if mutation == "none" else {"private_witness_hydration_site_invalid"}
    )


def test_same_module_statement_builder_censuses_shared_execution(tmp_path: Path) -> None:
    """A SQL builder preserves the rewrite target at its actual executor."""
    _module(
        tmp_path,
        'SQL = "INSERT INTO assertions(assertion_id) VALUES ({values}) " '
        '"ON CONFLICT(assertion_id) DO UPDATE SET assertion_id=excluded.assertion_id"\n'
        "def statement(operands):\n"
        "    if not operands:\n"
        '        raise ValueError("missing operands")\n'
        '    return SQL.format(values=", ".join(operands))\n'
        "class Writer:\n"
        "    def upsert(self, conn):\n"
        '        conn.execute(statement(("?",)))\n',
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [(site.function, site.table, site.kind, site.tier) for site in observation.sites] == [
        ("Writer.upsert", "assertions", "upsert_do_update", "user")
    ]
    assert _rules(tmp_path, _declaration(tmp_path)) == {"durable_write_undeclared"}
