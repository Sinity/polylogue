"""Isolated actual SQLite allocator control for original-only rowid collisions."""

import _sqlite3
import ctypes
import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import patch

from polylogue.storage.sqlite.archive_tiers.source_write import PreparedParserSingletonWitness
from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
from polylogue.storage.sqlite.managed_connection import sqlite_connection
from polylogue.storage.sqlite.reference_seal import KnownTierCell, PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def run(root: Path) -> dict[str, object]:
    # sqlite3.h SQLITE_TESTCTRL_PRNG_SAVE/RESTORE are 5/6 in the supported
    # SQLite build. Resolve the actual extension's loaded dependency, never
    # another system SQLite. These process-global controls stay in this child.
    native = ctypes.CDLL(_sqlite3.__file__)
    native.sqlite3_libversion.restype = ctypes.c_char_p
    assert native.sqlite3_libversion().decode("ascii") == sqlite3.sqlite_version
    control = native.sqlite3_test_control
    control.argtypes = [ctypes.c_int]
    control.restype = ctypes.c_int
    saved = False
    try:
        maximum = (1 << 63) - 1
        with sqlite_connection(":memory:") as probe:
            with closing(probe.execute("CREATE TABLE allocation_probe(value INTEGER)")):
                pass
            with closing(probe.execute("INSERT INTO allocation_probe(rowid,value) VALUES(?,0)", (maximum,))):
                pass
            assert control(5) == 0
            saved = True
            with closing(probe.execute("INSERT INTO allocation_probe VALUES(1)")) as cursor:
                occupied = cursor.lastrowid
            with closing(probe.execute("DELETE FROM allocation_probe WHERE rowid=?", (occupied,))):
                pass
            assert control(6) == 0
            with closing(probe.execute("INSERT INTO allocation_probe VALUES(1)")) as cursor:
                assert cursor.lastrowid == occupied, "native allocator PRNG replay is unsupported"
        assert isinstance(occupied, int) and occupied != maximum
        with write_lease("test.native-source-allocation-collision", archive_root=root):
            bootstrap_archive_root(root)
            with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source:
                for rowid, raw_id in ((maximum, "maximum-raw"), (occupied, "occupied-original")):
                    with closing(
                        source.execute(
                            "INSERT INTO raw_sessions(rowid,raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                            "VALUES(?,?,'unknown-export','synthetic/source',?,1,1)",
                            (rowid, raw_id, b"o" * 32),
                        )
                    ):
                        pass
                source.commit()
            with PreparedIndexMutation.source_only(archive_root=root) as seal:
                with seal.original_read_snapshot(), seal.source_producer():
                    maximum_image = seal.retain_tier_row("source", "raw_sessions", maximum)
                    assert maximum_image is not None
                    seal.load_source_row(maximum_image)
                    maximum_fields = dict(zip(maximum_image.columns, maximum_image.cells, strict=True))
                    with seal.source_statement(
                        "UPDATE raw_sessions SET parsed_at_ms=7 WHERE rowid=?",
                        (maximum,),
                        table="raw_sessions",
                        writable_targets=(("raw_sessions", (maximum_fields["raw_id"],)),),
                    ):
                        pass
                    prior = seal.retain_literal_scalar("prior literal remains owned")
                    attempts = 0
                    original_attempt = seal._source_statement_attempt

                    @contextmanager
                    def replay_first_allocation(
                        sql: str,
                        parameters: tuple[object, ...],
                        *,
                        table: str,
                        columns: tuple[str, ...],
                        keys: tuple[int, ...],
                        prepared_cells: dict[str, KnownTierCell] | None,
                        allocation_parameter: int | None,
                        generated_primary_key: bool,
                        parser_singleton_witness: PreparedParserSingletonWitness | None,
                        binding_cells: tuple[KnownTierCell, ...],
                    ) -> Iterator[sqlite3.Cursor]:
                        nonlocal attempts
                        attempts += 1
                        if attempts == 1:
                            assert control(6) == 0
                        with original_attempt(
                            sql,
                            parameters,
                            table=table,
                            columns=columns,
                            keys=keys,
                            prepared_cells=prepared_cells,
                            allocation_parameter=allocation_parameter,
                            generated_primary_key=generated_primary_key,
                            parser_singleton_witness=parser_singleton_witness,
                            binding_cells=binding_cells,
                        ) as cursor:
                            yield cursor

                    raw = seal.retain_literal_scalar("allocated-new")
                    blob = seal.retain_literal_scalar(b"n" * 32)
                    raw_expression, raw_parameters = seal.source_literal_expression(raw)
                    blob_expression, blob_parameters = seal.source_literal_expression(blob)
                    with patch.object(seal, "_source_statement_attempt", replay_first_allocation):
                        with seal.source_statement(
                            "INSERT INTO raw_sessions(rowid,raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                            f"VALUES(?,{raw_expression},'unknown-export','synthetic/new',{blob_expression},1,1)",
                            (None, *raw_parameters, *blob_parameters),
                            table="raw_sessions",
                            writable_targets=(("raw_sessions", (raw,)),),
                            prepared_cells={"raw_id": raw},
                            allocation_parameter=0,
                        ) as cursor:
                            allocated = cursor.lastrowid
                    assert attempts >= 2 and allocated != occupied
                    assert seal._literal_scalar_equal(prior, "prior literal remains owned")
                    with seal.source_rows("SELECT rowid FROM raw_sessions WHERE raw_id='occupied-original'") as rows:
                        assert rows.fetchone()[0] == occupied
                permit = seal.prepare_source_mutation()
                with permit.hold_authority(), permit.mutation_connection() as source:
                    with closing(source.execute("BEGIN IMMEDIATE")):
                        pass
                    permit.apply_source_statements(source)
                    permit.allow_commit(source)
                    source.commit()
                    seal.accept_known_tier_commit(permit.committed())
                with seal.original_read_snapshot():
                    with seal.original_rows(
                        "source", "SELECT rowid FROM raw_sessions WHERE raw_id='allocated-new'"
                    ) as rows:
                        assert rows.fetchone()[0] == allocated
                    with seal.original_rows(
                        "source", "SELECT parsed_at_ms FROM raw_sessions WHERE rowid=?", (maximum,)
                    ) as rows:
                        assert rows.fetchone()[0] == 7
        return {"sqlite": sqlite3.sqlite_version, "attempts": attempts, "occupied": occupied, "allocated": allocated}
    finally:
        if saved:
            assert control(6) == 0


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]))))
