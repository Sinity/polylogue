"""Concrete regressions for statement identity and census proof boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from devtools.durable_write_census import census_package, collect_violations

MODULE = "polylogue/writer.py"


def _source(root: Path, source: str, path: str = MODULE) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def _declared(root: Path, function: str, table: str, kind: str, *, classification: str, count: int = 1) -> Path:
    entry = {
        "file": MODULE,
        "function": function,
        "table": table,
        "kind": kind,
        "tier": "unresolved" if table == "?" else "user",
        "classification": classification,
        "reason": "Synthetic adjudicated write occurrence.",
    }
    path = root / "census.yaml"
    path.write_text(
        yaml.safe_dump({"package": "polylogue", "writes": [dict(entry) for _ in range(count)]}), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    "statements",
    [
        '    conn.execute("UPDATE assertions SET value = 1")\n    conn.execute("UPDATE assertions SET value = 2")\n',
        '    conn.executescript("UPDATE assertions SET value = 1; UPDATE assertions SET value = 2;")\n',
    ],
)
def test_each_rewrite_occurrence_requires_its_own_declaration(tmp_path: Path, statements: str) -> None:
    """04.F021: neither one function nor one script collapses separate writes."""
    source = "def mutate(conn):\n" + statements
    _source(tmp_path, source)
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [site.occurrence for site in observation.sites] == [1, 2]
    assert len({site.key for site in observation.sites}) == 2
    declaration = _declared(tmp_path, "mutate", "assertions", "update", classification="lifecycle_transition")
    violations = collect_violations(repo_root=tmp_path, declaration_path=declaration)
    assert [(item["rule"], item["key"]) for item in violations] == [
        ("durable_write_undeclared", observation.sites[1].key)
    ]
    _declared(tmp_path, "mutate", "assertions", "update", classification="lifecycle_transition", count=2)
    assert collect_violations(repo_root=tmp_path, declaration_path=declaration) == []
    _source(tmp_path, "\n\n" + source)
    shifted = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [site.key for site in shifted.sites] == [site.key for site in observation.sites]


@pytest.mark.parametrize(
    "source",
    [
        'import sqlite3\ndef scratch():\n    conn = sqlite3.connect(":memory:")\ndef mutate(conn):\n    conn.execute("CREATE TABLE hidden (value TEXT)")\n    conn.execute("UPDATE hidden SET value = 1")\n',
        'import sqlite3\ndef mutate(path):\n    conn = sqlite3.connect(":memory:")\n    conn = sqlite3.connect(path)\n    conn.execute("CREATE TABLE hidden (value TEXT)")\n    conn.execute("UPDATE hidden SET value = 1")\n',
        'import sqlite3\ndef outer():\n    conn = sqlite3.connect(":memory:")\n    def mutate(conn):\n        conn.execute("CREATE TABLE hidden (value TEXT)")\n        conn.execute("UPDATE hidden SET value = 1")\n',
    ],
)
def test_memory_binding_cannot_exempt_another_receiver(tmp_path: Path, source: str) -> None:
    """04.F022: same spelling, nested parameters and rebinding are not proof."""
    _source(tmp_path, source)
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert [(item.table, item.disposition) for item in observation.runtime_creations] == [("hidden", "persistent")]
    assert [(site.table, site.tier) for site in observation.sites] == [("hidden", "runtime")]


@pytest.mark.parametrize("argument", ["table", '"session_profiles" if known else table', "resolve_table()"])
def test_known_index_caller_does_not_hide_unknown_caller(tmp_path: Path, argument: str) -> None:
    """04.F023: preserve the unknown branch beside a valid index caller."""
    _source(
        tmp_path,
        'def replace(conn, table):\n    conn.execute(f"DELETE FROM {table}")\n'
        'def known(conn):\n    replace(conn, "session_profiles")\n'
        f"def unknown(conn, table, known):\n    replace(conn, {argument})\n",
    )
    declaration = _declared(tmp_path, "replace", "?", "delete", classification="rebuildable_dynamic_target")
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert {target.table for target in observation.dynamic_targets} == {"session_profiles", "?"}
    assert {item["rule"] for item in collect_violations(repo_root=tmp_path, declaration_path=declaration)} == {
        "dynamic_table_target_not_index"
    }


@pytest.mark.parametrize(
    "creator, creator_source",
    [
        # The finding's own shape: a storage-layer module outside archive_tiers/.
        (
            "polylogue/storage/sqlite/queries/foo.py",
            'def create(conn):\n    conn.execute("CREATE TABLE hidden (value TEXT)")\n',
        ),
        # Any module that also writes an archive table holds an archive connection.
        (
            "polylogue/other_store.py",
            'def create(conn):\n    conn.execute("CREATE TABLE hidden (value TEXT)")\n'
            '    conn.execute("INSERT INTO assertions (id) VALUES (1)")\n',
        ),
    ],
)
def test_runtime_creation_outside_archive_tiers_reaches_cross_module_writer(
    tmp_path: Path, creator: str, creator_source: str
) -> None:
    """04.F024: ownership cannot be evaded by moving a creator's module."""
    _source(tmp_path, creator_source, creator)
    _source(tmp_path, 'def mutate(conn):\n    conn.execute("UPDATE hidden SET value = 1")\n')
    declaration = tmp_path / "census.yaml"
    declaration.write_text("package: polylogue\nwrites: []\n", encoding="utf-8")
    violations = collect_violations(repo_root=tmp_path, declaration_path=declaration)
    assert [(item["rule"], item["file"]) for item in violations] == [
        ("runtime_persistent_table_rewrite_undeclared", MODULE)
    ]


def test_private_store_outside_the_archive_layer_is_not_archive_state(tmp_path: Path) -> None:
    """A parser spill database that never names an archive table owns its relations."""
    _source(
        tmp_path,
        'def spill(conn):\n    conn.execute("CREATE TABLE spill (value TEXT)")\n'
        '    conn.execute("UPDATE spill SET value = 1")\n',
        "polylogue/sources/parsers/spill.py",
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert observation.runtime_creations == ()
    assert observation.sites == ()


@pytest.mark.parametrize("first", ["CREATE TEMP TABLE transient (id TEXT);", "CREATE TABLE assertions (id TEXT);"])
def test_every_create_in_a_script_is_classified(tmp_path: Path, first: str) -> None:
    """04.F025: a canonical or temporary first table cannot hide the second."""
    _source(
        tmp_path,
        f'def mutate(conn):\n    conn.executescript("{first} CREATE TABLE hidden (value TEXT); UPDATE hidden SET value = 1;")\n',
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert ("hidden", "persistent") in {(item.table, item.disposition) for item in observation.runtime_creations}
    assert [(site.table, site.kind) for site in observation.sites] == [("hidden", "update")]


_FK_SOURCE = (
    "def cleanup(conn, arbitrary, other):\n"
    "    for table_name, column_name, action in _session_foreign_key_actions(conn):\n"
    "        table = _quote_identifier(table_name)\n"
    '        conn.execute(f"UPDATE {table} SET value = NULL")\n'
    '        conn.execute(f"DELETE FROM {table}")\n'
)


@pytest.mark.parametrize(
    "source, expected",
    [
        (_FK_SOURCE, True),
        (_FK_SOURCE.replace("_quote_identifier(table_name)", "_quote_identifier(arbitrary)"), False),
        (_FK_SOURCE.replace("conn.execute", "other.execute"), False),
        (_FK_SOURCE.replace("        table =", "        table_name = arbitrary\n        table ="), False),
        (_FK_SOURCE + '    conn.execute(f"DELETE FROM {arbitrary}")\n', False),
        (
            "def cleanup(conn, arbitrary, other):\n    _session_foreign_key_actions(conn)\n"
            '    conn.execute(f"UPDATE {arbitrary} SET value = NULL")\n'
            '    conn.execute(f"DELETE FROM {arbitrary}")\n',
            False,
        ),
    ],
)
def test_fk_classification_requires_the_actual_identifier_provenance(
    tmp_path: Path, source: str, expected: bool
) -> None:
    """04.F026: unused discovery, another connection or rebinding cannot authorize writes."""
    _source(tmp_path, source)
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert (f"{MODULE}::cleanup" in observation.index_foreign_key_cleanup_helpers) is expected
    declaration = _declared(tmp_path, "cleanup", "?", "delete", classification="index_foreign_key_cleanup")
    rules = {item["rule"] for item in collect_violations(repo_root=tmp_path, declaration_path=declaration)}
    assert ("index_foreign_key_cleanup_shape_invalid" in rules) is not expected


def test_occurrences_follow_source_order_across_nesting(tmp_path: Path) -> None:
    """04.F021: a nested statement keeps its place; walk order would renumber it.

    Anti-vacuity: number occurrences in ``ast.walk`` order and the top-level
    second statement becomes occurrence 1 ahead of the nested first one.
    """
    _source(
        tmp_path,
        "def mutate(conn, flag):\n"
        "    if flag:\n"
        '        conn.execute("UPDATE assertions SET value = 1")\n'
        '    conn.execute("UPDATE assertions SET value = 2")\n',
    )
    observation = census_package(tmp_path / "polylogue", repo_root=tmp_path)
    assert sorted((site.line, site.occurrence) for site in observation.sites) == [(3, 1), (4, 2)]
