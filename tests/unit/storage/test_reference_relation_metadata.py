"""Original relation metadata retains schema and physical row-address guards."""

from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def test_original_relation_reuse_observes_changed_schema_epoch(tmp_path: Path) -> None:
    with write_lease("test.original-relation-schema", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with connection_cursor(source, "CREATE TABLE relation_probe(key TEXT PRIMARY KEY,payload BLOB)"):
                pass
            source.commit()
            with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
                observer = seal._observers["source"]
                statements: list[str] = []
                observer.set_trace_callback(statements.append)
                for _ in range(1000):
                    assert seal._physical_rowid_alias(observer, "relation_probe", ("key", "payload")) == "rowid"
                    assert seal._known_tier_table_shape("source", "relation_probe") == (("key", "payload"), (0,))
                assert sum(sql == "PRAGMA main.table_list" for sql in statements) == 1
                with pytest.raises(ReferenceSealError):
                    seal._physical_rowid_alias(seal._scratch, "relation_probe", ("key", "payload"))
                with connection_cursor(source, "ALTER TABLE relation_probe ADD COLUMN provenance TEXT"):
                    pass
                source.commit()
                assert seal._known_tier_table_shape("source", "relation_probe") == (
                    ("key", "payload", "provenance"),
                    (0,),
                )
                assert sum(sql == "PRAGMA main.table_list" for sql in statements) == 2
                with connection_cursor(source, "DROP TABLE relation_probe"):
                    pass
                with connection_cursor(
                    source, "CREATE TABLE relation_probe(key TEXT PRIMARY KEY,payload BLOB) WITHOUT ROWID"
                ):
                    pass
                source.commit()
                with pytest.raises(ReferenceSealError):
                    seal._physical_rowid_alias(observer, "relation_probe", ("key", "payload"))
                assert sum(sql == "PRAGMA main.table_list" for sql in statements) == 3
                observer.set_trace_callback(None)


@pytest.mark.parametrize("descending", [False, True])
def test_original_shadowed_rowid_requires_actual_integer_primary_key(tmp_path: Path, descending: bool) -> None:
    with write_lease("test.original-rowid-alias", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            primary = "INTEGER PRIMARY KEY DESC" if descending else "INTEGER PRIMARY KEY"
            with connection_cursor(
                source, f"CREATE TABLE relation_probe(key {primary},rowid TEXT,_rowid_ TEXT,oid TEXT)"
            ):
                pass
            source.commit()
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
            observer = seal._observers["source"]
            for _ in range(2):
                if descending:
                    with pytest.raises(ReferenceSealError):
                        seal._physical_rowid_alias(observer, "relation_probe", ("key", "rowid", "_rowid_", "oid"))
                else:
                    assert (
                        seal._physical_rowid_alias(observer, "relation_probe", ("key", "rowid", "_rowid_", "oid"))
                        == "key"
                    )
