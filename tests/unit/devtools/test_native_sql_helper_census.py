"""SQL ownership stays visible through the existing Native execution family."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from devtools import durable_write_census, verify_layering


@pytest.mark.parametrize("literal_first", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_caller_sql_cannot_inherit_another_functions_temporary_ddl(
    tmp_path: Path, literal_first: bool, native: bool
) -> None:
    """An unrelated scratch statement must not hide arbitrary caller SQL."""
    imports = "from polylogue.storage.io_phase_metrics import connection_cursor\n" if native else ""
    call = "connection_cursor(conn, sql)" if native else "conn.execute(sql)"
    caller = f"def execute_original(conn, sql: str):\n    {call}\n"
    scratch = (
        'def create_scratch(conn):\n    sql = "CREATE TEMP TABLE private_parts(value TEXT)"\n    conn.execute(sql)\n'
    )
    source = imports + (scratch + caller if literal_first else caller + scratch)
    path = tmp_path / "polylogue" / "storage" / "writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    observation = durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [(item.function, item.parameter) for item in observation.helpers] == [("execute_original", "sql")]
    assert observation.sites == ()


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("from polylogue.storage.io_phase_metrics import connection_cursor", "connection_cursor(conn, SQL)"),
        (
            "from polylogue.storage.io_phase_metrics import connection_cursor as retained",
            "retained(connection=conn, sql=SQL)",
        ),
        ("import polylogue.storage.io_phase_metrics as native", "native.connection_cursor(conn, sql=SQL)"),
        (
            "from polylogue.storage import io_phase_metrics as native",
            "native.connection_cursor(connection=conn, sql=SQL)",
        ),
    ],
)
@pytest.mark.parametrize("sql", ["UPDATE assertions SET status='deleted'", "SELECT status FROM assertions"])
def test_both_censuses_observe_original_cursor_sql_operand(tmp_path: Path, imports: str, call: str, sql: str) -> None:
    source = f"{imports}\n\ndef mutate(conn):\n    SQL = {sql!r}\n    with {call}:\n        pass\n"
    path = tmp_path / "polylogue" / "storage" / "writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    tree = ast.parse(source)
    writes = durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path)
    is_write = sql.startswith("UPDATE")
    assert [item.table for item in writes.sites] == (["assertions"] if is_write else [])
    assert len(verify_layering._mutation_calls(tree)) == int(is_write)
    assert verify_layering._mutation_tiers(tree) == (frozenset({"user"}) if is_write else frozenset())
    assert verify_layering._function_mutation_tiers(tree)["mutate"] == verify_layering._mutation_tiers(tree)


@pytest.mark.parametrize(
    "call",
    [
        "seal._owned_cursor(conn, SQL)",
        "seal._owned_cursor(connection=conn, sql=SQL)",
        "seal._source_statement_attempt(SQL, ())",
        "seal.source_statement(sql=SQL, parameters=(), table='assertions', writable_targets=())",
        "seal.user_statement(sql=SQL, parameters=(), table='assertions', writable_targets=())",
        "seal._selected_statement(SQL, (), table='assertions', writable_targets=())",
        "seal.original_rows('user', SQL)",
        "seal.source_rows(sql=SQL)",
        "seal.user_rows(sql=SQL)",
        "seal._selected_rows(SQL)",
        "seal.before_index_input('assertions', (), rowid_sql=SQL, parameters=())",
    ],
)
def test_both_censuses_bind_native_seal_helper_to_its_imported_owner(tmp_path: Path, call: str) -> None:
    source = f'from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Original\n\ndef mutate(seal: Original, conn):\n    SQL = "DELETE FROM assertions"\n    with {call}:\n        pass\n'
    path = tmp_path / "polylogue" / "storage" / "writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    tree = ast.parse(source)
    observed = durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [(item.function, item.table, item.kind) for item in observed.sites] == [("mutate", "assertions", "delete")]
    assert len(verify_layering._mutation_calls(tree)) == 1
    assert verify_layering._function_mutation_tiers(tree)["mutate"] == frozenset({"user"})


@pytest.mark.parametrize(
    "source",
    [
        "def mutate(other):\n    other._owned_cursor(None, 'DELETE FROM assertions')\n",
        "from unrelated import connection_cursor\ndef mutate(conn):\n    connection_cursor(conn, 'DELETE FROM assertions')\n",
        "from polylogue.storage.io_phase_metrics import connection_cursor\ndef mutate(connection_cursor, conn):\n    connection_cursor(conn, 'DELETE FROM assertions')\n",
        "from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Original\ndef mutate(seal: Original):\n    seal = object()\n    seal.source_statement('DELETE FROM assertions')\n",
        "from polylogue.storage.io_phase_metrics import connection_cursor\ndef connection_cursor(conn, sql):\n    pass\ndef mutate(conn):\n    connection_cursor(conn, 'DELETE FROM assertions')\n",
    ],
)
def test_unrelated_or_shadowed_helper_names_do_not_invent_native_writes(tmp_path: Path, source: str) -> None:
    path = tmp_path / "polylogue" / "storage" / "writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites == ()
    assert verify_layering._mutation_calls(ast.parse(source)) == ()


def test_imported_sql_opacity_uses_actual_native_helper_operand() -> None:
    source = "from polylogue.storage.io_phase_metrics import connection_cursor as retained\nfrom unrelated import SQL\ndef mutate(conn):\n    retained(connection=conn, sql=SQL)\n"
    assert verify_layering._imported_sql_execution_lines(ast.parse(source)) == [4]


def test_canonical_seal_self_is_bound_only_in_the_actual_owner_module() -> None:
    tree = ast.parse(
        "class PreparedIndexMutation:\n    def mutate(self, conn):\n        with self._owned_cursor(conn, 'DELETE FROM assertions'):\n            pass\n"
    )
    relative = "polylogue/storage/sqlite/reference_seal.py"
    assert len(verify_layering._mutation_calls(tree, relative=relative)) == 1
    assert verify_layering._function_mutation_tiers(tree, relative=relative)[
        "PreparedIndexMutation.mutate"
    ] == frozenset({"user"})
    assert verify_layering._mutation_calls(tree, relative="polylogue/storage/unrelated.py") == ()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("def inner():\n        def connection_cursor():\n            pass\n    connection_cursor(conn, SQL)", 1),
        ("seal = Original(conn)\n    for seal in others:\n        seal.source_statement(SQL)", 0),
        ("try:\n        pass\n    except RuntimeError as connection_cursor:\n        connection_cursor(conn, SQL)", 0),
        ("match value:\n        case {'owner': seal}:\n            seal.source_statement(SQL)", 0),
        ("with context as seal:\n        seal.source_statement(SQL)", 0),
        ("del seal\n    seal.source_statement(SQL)", 0),
        ("seal += other\n    seal.source_statement(SQL)", 0),
        ("match value:\n        case [*seal]:\n            seal.source_statement(SQL)", 0),
        ("match value:\n        case {'x': x, **seal}:\n            seal.source_statement(SQL)", 0),
        ("[seal.source_statement(SQL) for seal in others]", 0),
        ("[value for seal in others]\n    seal.source_statement(SQL)", 1),
        ("[value for value in connection_cursor(conn, SQL)]", 1),
        ("[value for connection_cursor in connection_cursor(conn, SQL)]", 1),
        ("[(seal := object()) for value in others]\n    seal.source_statement(SQL)", 0),
        ("fn = lambda seal: seal.source_statement(SQL)", 0),
        ("fn = lambda: seal.source_statement(SQL)", 1),
        ("def inner():\n        connection_cursor(conn, SQL)", 1),
        ("seal = Original(conn)\n    seal = Original.source_only(conn)\n    seal.source_statement(SQL)", 1),
        ("seal = Original(conn)\n    seal = object()\n    seal.source_statement(SQL)", 0),
    ],
)
def test_both_observers_respect_native_helper_lexical_bindings(tmp_path: Path, body: str, expected: int) -> None:
    source = (
        "from polylogue.storage.io_phase_metrics import connection_cursor\n"
        "from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Original\n"
        "SQL = 'DELETE FROM assertions'\n"
        "def mutate(conn, seal: Original, others, context, value, other):\n    " + body + "\n"
    )
    path = tmp_path / "polylogue/storage/writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    tree = ast.parse(source)
    assert len(durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites) == expected
    assert len(verify_layering._mutation_calls(tree)) == expected


@pytest.mark.parametrize(
    "class_body",
    [
        "from polylogue.storage.io_phase_metrics import connection_cursor\n    def mutate(self, conn):\n        connection_cursor(conn, 'DELETE FROM assertions')",
        "@staticmethod\n    def mutate(self, conn):\n        self.source_statement('DELETE FROM assertions')",
        "@classmethod\n    def mutate(cls, conn):\n        cls.source_statement('DELETE FROM assertions')",
    ],
)
def test_class_local_names_do_not_prove_native_instance_sql(tmp_path: Path, class_body: str) -> None:
    source = "class PreparedIndexMutation:\n    " + class_body + "\n"
    relative = "polylogue/storage/sqlite/reference_seal.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites == ()
    assert verify_layering._mutation_calls(ast.parse(source), relative=relative) == ()


def test_class_method_keeps_real_module_helper_import(tmp_path: Path) -> None:
    source = (
        "from polylogue.storage.io_phase_metrics import connection_cursor\n"
        "class Other:\n    connection_cursor = object()\n"
        "    def mutate(self, conn):\n        connection_cursor(conn, 'DELETE FROM assertions')\n"
    )
    path = tmp_path / "polylogue/storage/writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert len(durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites) == 1
    assert len(verify_layering._mutation_calls(ast.parse(source))) == 1


def test_nested_class_does_not_capture_outer_class_import(tmp_path: Path) -> None:
    source = (
        "class Outer:\n    from polylogue.storage.io_phase_metrics import connection_cursor\n"
        "    class Inner:\n        def mutate(self, conn):\n"
        "            connection_cursor(conn, 'DELETE FROM assertions')\n"
    )
    path = tmp_path / "polylogue/storage/writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites == ()
    assert verify_layering._mutation_calls(ast.parse(source)) == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def mutate(seal, conn):\n    seal.source_statement(SQL)\n    seal = Original(conn)", 0),
        ("def mutate(conn, *seal: Original):\n    seal.source_statement(SQL)", 0),
        ("def mutate(conn, **seal: Original):\n    seal.source_statement(SQL)", 0),
        (
            "def mutate(seal: Missing, conn):\n    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Missing\n    seal.source_statement(SQL)",
            0,
        ),
        (
            "class Other:\n    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Local\n    def mutate(self, seal: Local):\n        seal.source_statement(SQL)",
            1,
        ),
        ("def maker(conn):\n    def inner(connection_cursor=connection_cursor(conn, SQL)):\n        pass", 1),
        ("def maker(conn):\n    @connection_cursor(conn, SQL)\n    def inner(connection_cursor):\n        pass", 1),
        ("def maker(conn):\n    class Inner(connection_cursor(conn, SQL)):\n        pass", 1),
        ("def maker(conn):\n    class Inner(metaclass=connection_cursor(conn, SQL)):\n        pass", 1),
        ("def maker(conn):\n    fn = lambda connection_cursor=connection_cursor(conn, SQL): None", 1),
        (
            "class Outer:\n    from polylogue.storage.io_phase_metrics import connection_cursor as local\n    def inner(self, conn=local(None, SQL)):\n        pass",
            1,
        ),
    ],
)
def test_native_definition_inputs_are_separate_from_body_bindings(tmp_path: Path, source: str, expected: int) -> None:
    source = (
        "from polylogue.storage.io_phase_metrics import connection_cursor\n"
        "from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation as Original\n"
        "SQL = 'DELETE FROM assertions'\n" + source + "\n"
    )
    path = tmp_path / "polylogue/storage/writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert len(durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path).sites) == expected
    assert len(verify_layering._mutation_calls(ast.parse(source))) == expected


@pytest.mark.parametrize(
    "consumer", ["_mutation_calls", "_mutation_tiers", "_function_mutation_tiers", "_imported_sql_execution_lines"]
)
def test_empty_module_sql_observation_is_reused_for_all_calls(monkeypatch: pytest.MonkeyPatch, consumer: str) -> None:
    tree = ast.parse("def outer():\n    alpha(beta(gamma()))\n    def inner():\n        delta(epsilon())\n")
    original = durable_write_census.sql_execution_calls
    observed: list[ast.AST] = []

    def observe(node: ast.AST, *, relative: str = "") -> dict[ast.Call, durable_write_census.SQLExecution]:
        observed.append(node)
        assert node is tree
        return original(node, relative=relative)

    monkeypatch.setattr(durable_write_census, "sql_execution_calls", observe)
    getattr(verify_layering, consumer)(tree)
    assert observed == [tree]


def test_standalone_writer_collector_keeps_canonical_native_owner_path(tmp_path: Path) -> None:
    relative = "polylogue/storage/sqlite/reference_seal.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    source = (
        "class PreparedIndexMutation:\n    def mutate(self):\n        self.source_statement('DELETE FROM assertions')\n"
    )
    path.write_text(source)
    policy = verify_layering.WriterModulePolicy(
        marker="writer", mutation_roots=(), modules=(), twin_write_contracts={}, census_roots=("polylogue",)
    )
    assert verify_layering._census_mutation_files(tmp_path, policy) == {relative: frozenset({"user"})}
    assert verify_layering._entrypoint_tiers(
        ast.parse(source), "PreparedIndexMutation.mutate", {}, relative=relative
    ) == frozenset({"user"})


def test_entrypoint_empty_precomputed_observations_do_not_discover_again(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = ast.parse("def mutate():\n    alpha(beta())\n")

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("completed empty observation must not be recomputed")

    monkeypatch.setattr(verify_layering, "_function_definitions", unexpected)
    monkeypatch.setattr(verify_layering, "_imported_writer_modules", unexpected)
    monkeypatch.setattr(verify_layering, "_function_mutation_tiers", unexpected)
    assert (
        verify_layering._entrypoint_tiers(tree, "mutate", {}, functions={}, imported_modules={}, direct_tiers={})
        == frozenset()
    )


@pytest.mark.parametrize("mutation_first", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_local_sql_operand_cannot_borrow_another_functions_mutation(
    tmp_path: Path, mutation_first: bool, native: bool
) -> None:
    """An unresolved local reader is not another function's durable UPDATE."""
    imports = "from polylogue.storage.io_phase_metrics import connection_cursor\n" if native else ""
    call = "connection_cursor(conn, sql)" if native else "conn.execute(sql)"
    mutation_sql = "UPDATE assertions SET status='deleted'"
    mutation = f"def mutate(conn):\n    sql = {mutation_sql!r}\n    {call}\n"
    reader = f"def inspect(conn, query: bytes):\n    sql = query.decode()\n    {call}\n"
    source = imports + (mutation + reader if mutation_first else reader + mutation)
    path = tmp_path / "polylogue" / "storage" / "writer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    observation = durable_write_census.census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [(item.function, item.table) for item in observation.sites] == [("mutate", "assertions")]
    assert observation.helpers == ()
