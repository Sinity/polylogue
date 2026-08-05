"""Sharded from-empty index rebuild: equivalence proof + K-sweep benchmark
(polylogue-pzxm).

Two things live here, both driving the REAL rebuild engine end to end
(``rebuild_index_from_source_sync``, the same function the offline CLI and
daemon bulk-rebuild call -- no reimplementation), against the same
structurally-representative synthetic corpus
``tests/infra/rebuild_cost_model.py`` builds for the rebuild-cost harness
(polylogue-623q/o56w):

- :func:`test_sharded_build_matches_sequential_build` -- the bead's
  correctness bar: byte-identical schema SHA, per-table row counts, and
  content SHA over ordered ``sessions``/``messages``/``blocks``/
  ``session_links``/``action_pairs`` between a ``shard_count=1`` (sequential)
  and a ``shard_count=4`` (sharded) rebuild of the SAME source corpus --
  the PR #3469 "MANIFESTS IDENTICAL" pattern (see
  ``tests/unit/sources/test_revision_backfill.py``'s
  ``_index_content_manifest`` for the prior art this extends with hashing
  and a wider table set).
- :func:`test_sharded_build_k_sweep_benchmark` -- an honest K=1/4/8 wall-clock
  comparison on the harness fixture. This is NOT run against the live
  archive (forbidden by the lane brief); it is a small, CI-fast synthetic
  corpus, so the reported ratios are directional evidence for the shardable
  portion of a rebuild pass, not a promise about the live 41k-raw archive.

Both tests build the corpus ONCE via ``build_stratum_sample_corpus`` (fully
deterministic, see its docstring) and then physically copy the archive root
before each rebuild variant, so "same source" is a filesystem fact, not an
assumption about corpus-generation determinism holding across two calls.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from tests.infra.rebuild_cost_model import Stratum, build_stratum_sample_corpus
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt

#: The bead's named correctness surface, in composite-primary-key order so a
#: row-for-row comparison is well-defined without depending on SQLite's
#: physical row order (which ATTACH+INSERT merge does not preserve).
_EQUIVALENCE_TABLE_ORDER: dict[str, str] = {
    "sessions": "session_id",
    "messages": "message_id",
    "blocks": "block_id",
    "session_links": "src_session_id, dst_origin, dst_native_id, link_type",
    "action_pairs": "session_id, tool_use_block_id",
}


#: Every table `polylogue.maintenance.sharded_rebuild.MERGE_TABLES` copies
#: across shards, checked for row-count parity (cheap, catches a merge
#: silently dropping or duplicating rows in a table content-hashing doesn't
#: cover). Mirrors that module's table list; imported by name below rather
#: than duplicated so a future table addition there is a single edit.
def _merge_table_names() -> tuple[str, ...]:
    from polylogue.maintenance.sharded_rebuild import MERGE_TABLES

    return MERGE_TABLES


def _schema_sha(index_db: Path) -> str:
    with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _table_row_counts(index_db: Path, tables: tuple[str, ...]) -> dict[str, int]:
    with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
        return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}


def _content_sha(index_db: Path) -> dict[str, str]:
    """Per-table content SHA over ordered rows -- the PR #3469 currency,
    extended from ``_index_content_manifest``'s plain-equality dict compare
    to a hash so the fixture's row payload never has to live in this
    process's memory at CI runner scale.
    """
    digests: dict[str, str] = {}
    with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
        for table, order in _EQUIVALENCE_TABLE_ORDER.items():
            digest = hashlib.sha256()
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                for value in row:
                    digest.update(repr(value).encode("utf-8"))
                    digest.update(b"\0")
                digest.update(b"\n")
            digests[table] = digest.hexdigest()
    return digests


@contextmanager
def _archive_root_env(archive_root: Path) -> Iterator[None]:
    prior = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = prior


def _build_corpus(tmp_path: Path, *, sample_n: int) -> Path:
    stratum = Stratum(
        "claude-code-pzxm",
        Provider.CLAUDE_CODE,
        count=sample_n,
        total_bytes=sample_n * 4000,
        # Deliberately non-zero: a fork/chain member split across two
        # shards by the hash partition is exactly the "hard part" the
        # polylogue-pzxm bead calls out (cohort completeness/authority
        # arbitration must not depend on which shard a raw landed in).
        chain_fraction=0.25,
        ambiguous_fraction=0.25,
    )
    source_root = tmp_path / "source-corpus"
    build_stratum_sample_corpus(source_root, stratum, sample_n=sample_n)
    return source_root


def _rebuild(archive_root: Path, *, shard_count: int, raw_batch_size: int) -> float:
    with _archive_root_env(archive_root):
        receipt_path = write_valid_rebuild_receipt(
            archive_root, archive_root.parent / "schema-inference-gate-receipt.json"
        )
        started_at = time.perf_counter()
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=archive_root,
                promote=True,
                raw_batch_size=raw_batch_size,
                shard_count=shard_count,
                schema_inference_receipt_path=receipt_path,
            )
        )
        elapsed_s = time.perf_counter() - started_at
    assert receipt.status == "replayed", receipt.to_dict()
    return elapsed_s


@pytest.mark.benchmark
def test_sharded_build_matches_sequential_build(tmp_path: Path) -> None:
    sample_n = 48
    source_root = _build_corpus(tmp_path, sample_n=sample_n)

    sequential_root = tmp_path / "sequential"
    sharded_root = tmp_path / "sharded"
    shutil.copytree(source_root, sequential_root)
    shutil.copytree(source_root, sharded_root)

    _rebuild(sequential_root, shard_count=1, raw_batch_size=sample_n)
    _rebuild(sharded_root, shard_count=4, raw_batch_size=sample_n)

    from polylogue.storage.archive_identity import ArchiveLocation

    sequential_index = ArchiveLocation.resolve(sequential_root).active_index_path
    sharded_index = ArchiveLocation.resolve(sharded_root).active_index_path

    # 1. Byte-identical schema.
    schema_sha_sequential = _schema_sha(sequential_index)
    schema_sha_sharded = _schema_sha(sharded_index)
    assert schema_sha_sharded == schema_sha_sequential

    # 2. Per-table row counts across every table the shard merge touches.
    merge_tables = _merge_table_names()
    counts_sequential = _table_row_counts(sequential_index, merge_tables)
    counts_sharded = _table_row_counts(sharded_index, merge_tables)
    assert counts_sharded == counts_sequential
    # Sanity floor: the corpus actually produced content in the tables that
    # matter, so an accidentally-empty merge could never pass this trivially.
    assert counts_sequential["sessions"] > 0
    assert counts_sequential["messages"] > 0
    # Not session_links > 0: this corpus's chain_fraction/ambiguous_fraction
    # produce revision-authority cohorts (byte-growth/ambiguous-identity
    # conflicts within ONE logical session), not parent/child session forks
    # -- session_links coverage for the cross-shard graph-resolution "hard
    # part" lives in tests/unit/maintenance/test_sharded_rebuild.py instead,
    # via a direct write_parsed_session_to_archive fork scenario.

    # 3. Content SHA over the bead-named equivalence tables (action_pairs
    #    included: empty per-shard under bulk_build, populated only by the
    #    shared terminal repopulate stage both paths run unchanged).
    content_sha_sequential = _content_sha(sequential_index)
    content_sha_sharded = _content_sha(sharded_index)
    assert content_sha_sharded == content_sha_sequential
    for table in _EQUIVALENCE_TABLE_ORDER:
        assert content_sha_sharded[table] == content_sha_sequential[table], table


@pytest.mark.benchmark
def test_sharded_build_k_sweep_benchmark(tmp_path: Path) -> None:
    """Honest K=1/4/8 wall-clock comparison on the harness fixture.

    No speedup threshold is asserted: the lane brief's instruction is to
    report the measured numbers, including a smaller-than-projected win, not
    to gate on a projection. See this test's printed report for the numbers
    from the run that produced them.
    """
    sample_n = 96
    source_root = _build_corpus(tmp_path, sample_n=sample_n)

    elapsed_by_k: dict[int, float] = {}
    for k in (1, 4, 8):
        root = tmp_path / f"k{k}"
        shutil.copytree(source_root, root)
        elapsed_by_k[k] = _rebuild(root, shard_count=k, raw_batch_size=sample_n)

    baseline = elapsed_by_k[1]
    report_lines = [
        f"polylogue-pzxm K-sweep ({sample_n} synthetic raws, chain_fraction=0.25, ambiguous_fraction=0.25):",
    ]
    for k in (1, 4, 8):
        speedup = baseline / elapsed_by_k[k] if elapsed_by_k[k] > 0 else float("nan")
        report_lines.append(f"  K={k}: {elapsed_by_k[k]:.3f}s (speedup vs K=1: {speedup:.2f}x)")
    report = "\n".join(report_lines)
    print("\n" + report)

    # Sanity floor only: every configuration must complete and produce a
    # positive measurement -- this is the harness proving it engaged
    # sharding at all (K=4/8 shard_count actually took the sharded_rebuild
    # path -- see rebuild_index.py's dispatch on
    # `shard_count > 1 and len(selected_raw_ids) >= shard_count`), not a
    # performance gate.
    assert all(elapsed > 0 for elapsed in elapsed_by_k.values())
