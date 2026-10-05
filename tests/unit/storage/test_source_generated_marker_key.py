"""SQLite-generated marker keys acquire only their exact retained root role."""

from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.accepted_marker_inputs import MarkerInputExcisionTarget, excise_marker_input_targets_sync
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.reference_seal import KnownTierCell, PreparedIndexMutation, ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def _statement(
    seal: PreparedIndexMutation, *, identity: str = "a" * 64
) -> tuple[str, tuple[object, ...], dict[str, KnownTierCell]]:
    values: dict[str, None | int | float | str | bytes] = {
        "identity": identity,
        "raw_id": "synthetic-raw",
        "payload": b"Neutral",
        "index_incarnation_id": None,
        "payload_sha256": "b" * 64,
    }
    cells = {name: seal.retain_literal_scalar(value) for name, value in values.items()}
    expressions: list[str] = []
    bindings: list[object] = [None]
    for cell in cells.values():
        expression, operands = seal.source_literal_expression(cell)
        expressions.append(expression)
        bindings.extend(operands)
    sql = (
        "INSERT INTO accepted_marker_inputs(sequence, identity, raw_id, payload, "
        "index_incarnation_id, payload_sha256) VALUES (?, " + ", ".join(expressions) + ") RETURNING sequence"
    )
    return sql, tuple(bindings), cells


@pytest.mark.parametrize("prior_sequence", [0, 4096])
def test_generated_marker_key_is_actual_monotonic_retained_root(tmp_path: Path, prior_sequence: int) -> None:
    with write_lease("test.generated-marker", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        if prior_sequence:
            with closing(
                open_isolated_write_connection(
                    tmp_path / "source.db", purpose="synthetic marker", archive_root=tmp_path
                )
            ) as source:
                with connection_cursor(
                    source,
                    "INSERT INTO accepted_marker_inputs VALUES (?, ?, 'synthetic-raw', X'00', NULL, ?)",
                    (prior_sequence, "c" * 64, "d" * 64),
                ):
                    pass
                counts = excise_marker_input_targets_sync(
                    source,
                    (
                        MarkerInputExcisionTarget(
                            identity="c" * 64,
                            raw_id="synthetic-raw",
                            carrier_digest="d" * 64,
                            state="accepted",
                            stream_id="synthetic-stream",
                            accepted_sequence=prior_sequence,
                        ),
                    ),
                    excised_at_ms=1,
                )
                assert counts == {"pending": 0, "accepted": 1}
                source.commit()
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
            with seal.original_read_snapshot(), seal.source_producer():
                seal.source_allocation_dependencies("accepted_marker_inputs")
                sql, bindings, cells = _statement(seal)
                with seal.source_statement(
                    sql,
                    bindings,
                    table="accepted_marker_inputs",
                    writable_targets=(),
                    prepared_cells=cells,
                    allocation_parameter=0,
                    generated_primary_key=True,
                ) as rows:
                    sequence = rows.fetchone()[0]
                    assert rows.fetchone() is None
                assert sequence == prior_sequence + 1
                with seal.source_rows("SELECT sequence,identity,payload FROM accepted_marker_inputs") as rows:
                    assert [tuple(row) for row in rows] == [(sequence, "a" * 64, b"Neutral")]
                with connection_cursor(
                    seal._scratch, "SELECT cell_id FROM temp.known_tier_statement_bindings WHERE position=0"
                ) as rows:
                    retained = KnownTierCell(seal, rows.fetchone()[0])
                assert seal._literal_scalar_equal(retained, sequence)
            seal.prepare_source_mutation()
        with closing(
            open_isolated_write_connection(
                tmp_path / "source.db", purpose="synthetic observation", archive_root=tmp_path
            )
        ) as source:
            with connection_cursor(source, "SELECT count(*) FROM accepted_marker_inputs") as rows:
                assert rows.fetchone()[0] == 0


@pytest.mark.parametrize("defect", ["table", "key", "cells", "upsert", "update"])
def test_generated_marker_role_refuses_noncanonical_statement(tmp_path: Path, defect: str) -> None:
    with write_lease("test.generated-marker-refusal", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
            with pytest.raises(ReferenceSealError):
                with seal.original_read_snapshot(), seal.source_producer():
                    seal.source_allocation_dependencies("accepted_marker_inputs")
                    sql, bindings, cells = _statement(seal)
                    table = "accepted_marker_inputs"
                    if defect == "table":
                        table = "pending_accepted_marker_inputs"
                    elif defect == "key":
                        bindings = (7, *bindings[1:])
                    elif defect == "cells":
                        cells["raw_id"] = seal.retain_literal_scalar("different")
                    elif defect == "upsert":
                        sql = sql.replace(" RETURNING", " ON CONFLICT(identity) DO NOTHING RETURNING")
                    else:
                        sql = "UPDATE accepted_marker_inputs SET sequence=7"
                    with seal.source_statement(
                        sql,
                        bindings,
                        table=table,
                        writable_targets=(),
                        prepared_cells=cells,
                        allocation_parameter=0,
                        generated_primary_key=True,
                    ):
                        pytest.fail("noncanonical generated root acquired authority")


def test_generated_marker_statement_rollback_discards_exact_root(tmp_path: Path) -> None:
    class InjectedRollbackError(Exception):
        pass

    injected = InjectedRollbackError()
    with write_lease("test.generated-marker-rollback", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
            with pytest.raises(InjectedRollbackError) as caught:
                with seal.original_read_snapshot(), seal.source_producer():
                    seal.source_allocation_dependencies("accepted_marker_inputs")
                    sql, bindings, cells = _statement(seal)
                    with seal.source_statement(
                        sql,
                        bindings,
                        table="accepted_marker_inputs",
                        writable_targets=(),
                        prepared_cells=cells,
                        allocation_parameter=0,
                        generated_primary_key=True,
                    ) as rows:
                        assert rows.fetchone()[0] == 1
                        assert rows.fetchone() is None
                        raise injected
            assert caught.value is injected
        with closing(
            open_isolated_write_connection(
                tmp_path / "source.db", purpose="synthetic rollback observation", archive_root=tmp_path
            )
        ) as source:
            with connection_cursor(source, "SELECT count(*) FROM accepted_marker_inputs") as rows:
                assert rows.fetchone()[0] == 0
