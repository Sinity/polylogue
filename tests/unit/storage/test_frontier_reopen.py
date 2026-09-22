"""Acknowledging a frontier blocker does not discharge the evidence behind it.

``resolve_raw_authority_blocker`` tombstones the row on the operator's
acknowledgement alone -- for a frontier witness it does not even rebuild the
plan from current evidence. ``pass_id`` is a content address over the inspected
inventory, so a later pass over *unchanged* blocking evidence derives the same
pass, plan and blocker ids, and the publishing
``INSERT ... ON CONFLICT DO NOTHING`` then found the resolved row and wrote
nothing. ``raw_authority_blocker_count`` read zero while the same missing,
quarantined or corrupt evidence still existed.

Anti-vacuity: put the plain
``f"raw-authority-blocker:{_digest(['frontier', pass_id, item.plan_id])}"``
back in ``_reconcile_frontier_obligations`` and
``test_acknowledged_obligation_reopens_...`` goes red -- the second pass
publishes nothing and readiness reports a clean archive. The remaining cases
pin the opposite direction: an unchanged pass must not mint a second open row,
and a genuinely disproved obligation must stay closed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.config import Config
from polylogue.core.json import json_document
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.raw_authority import resolve_raw_authority_blocker
from polylogue.storage.raw_reconciler import (
    RawAuthorityFrontierItem,
    RawAuthorityFrontierState,
    _reconcile_frontier_obligations,
)
from tests.infra.archive_templates import bootstrap_archive_root

_PASS_ID = "raw-authority-frontier-pass:" + "a" * 64


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


def _blocking_item() -> RawAuthorityFrontierItem:
    """One frontier item whose evidence is missing and stays missing."""
    return RawAuthorityFrontierItem(
        state=RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
        raw_id="raw-1",
        logical_source_key="codex-session:key-1",
        session_id="codex-session:s1",
        reason="accepted raw payload bytes are absent from the blob store",
        evidence_digest="d" * 64,
        input_raw_ids=("raw-1",),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
        plan_id="raw-replay:frontier-1",
    )


def _blocker_rows(root: Path) -> list[tuple[str, bool]]:
    with sqlite3.connect(root / "source.db") as conn:
        return [
            (str(blocker_id), resolved_at_ms is None)
            for blocker_id, resolved_at_ms in conn.execute(
                "SELECT blocker_id, resolved_at_ms FROM raw_authority_blockers ORDER BY created_at_ms, blocker_id"
            )
        ]


class TestFrontierObligationReopen:
    def test_acknowledged_obligation_reopens_while_evidence_blocks(self, tmp_path: Path) -> None:
        bootstrap_archive_root(tmp_path)
        item = _blocking_item()
        config = _config(tmp_path)

        first = _reconcile_frontier_obligations(config, _PASS_ID, (item,))
        original_id = first[item.plan_id]
        assert _blocker_rows(tmp_path) == [(original_id, True)]
        assert raw_materialization_readiness_snapshot(tmp_path)["raw_authority_blocker_count"] == 1

        resolve_raw_authority_blocker(tmp_path, original_id, resolution="acknowledged, will reacquire later")
        assert raw_materialization_readiness_snapshot(tmp_path)["raw_authority_blocker_count"] == 0

        # The evidence has not changed, so the next pass is byte-identical.
        second = _reconcile_frontier_obligations(config, _PASS_ID, (item,))
        successor_id = second[item.plan_id]
        assert successor_id != original_id
        assert _blocker_rows(tmp_path) == [(original_id, False), (successor_id, True)]
        assert raw_materialization_readiness_snapshot(tmp_path)["raw_authority_blocker_count"] == 1

    def test_an_unchanged_pass_does_not_multiply_open_rows(self, tmp_path: Path) -> None:
        """Opposite direction: minting a fresh obligation every pass would fail."""
        bootstrap_archive_root(tmp_path)
        item = _blocking_item()
        config = _config(tmp_path)

        first = _reconcile_frontier_obligations(config, _PASS_ID, (item,))
        second = _reconcile_frontier_obligations(config, _PASS_ID, (item,))
        third = _reconcile_frontier_obligations(config, _PASS_ID, (item,))

        assert first == second == third
        assert _blocker_rows(tmp_path) == [(first[item.plan_id], True)]

    def test_a_disproved_obligation_stays_closed(self, tmp_path: Path) -> None:
        """Opposite direction: a pass that no longer sees the item must not reopen it."""
        bootstrap_archive_root(tmp_path)
        item = _blocking_item()
        config = _config(tmp_path)

        published = _reconcile_frontier_obligations(config, _PASS_ID, (item,))
        original_id = published[item.plan_id]

        clean_pass = "raw-authority-frontier-pass:" + "b" * 64
        _reconcile_frontier_obligations(config, clean_pass, ())
        assert _blocker_rows(tmp_path) == [(original_id, False)]

        _reconcile_frontier_obligations(config, clean_pass, ())
        assert _blocker_rows(tmp_path) == [(original_id, False)]
        assert raw_materialization_readiness_snapshot(tmp_path)["raw_authority_blocker_count"] == 0
