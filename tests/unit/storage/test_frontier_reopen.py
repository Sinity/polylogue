"""Acknowledgement preserves evidence: blocking evidence reopens, disproof closes."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.stage_admission import admit_stage_write
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.frontier_inspection import (
    inspect_prepared_raw_authority_frontier,
    prepared_frontier_blocker_acknowledgement,
)
from polylogue.storage.raw_reconciler import _frontier_blocker_identity
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


def _blocker_rows(root: Path) -> list[tuple[str, bool]]:
    with closing(open_readonly_connection(root / "source.db")) as conn:
        return [
            (str(blocker_id), resolved_at_ms is None)
            for blocker_id, resolved_at_ms in conn.execute(
                "SELECT blocker_id,resolved_at_ms FROM raw_authority_blockers ORDER BY created_at_ms,blocker_id"
            )
        ]


class TestFrontierObligationReopen:
    def test_acknowledgement_chain_has_no_arbitrary_cap(self) -> None:
        resolved: dict[str, tuple[object]] = {}
        last = None
        for index in range(70):
            blocker_id = _frontier_blocker_identity(
                resolved.get, pass_id="raw-authority-frontier-pass:" + "a" * 64, plan_id="raw-replay:frontier-1"
            )
            assert blocker_id != last
            resolved[blocker_id] = (index,)
            last = blocker_id
        assert len(resolved) == 70
        successor = _frontier_blocker_identity(
            resolved.get, pass_id="raw-authority-frontier-pass:" + "a" * 64, plan_id="raw-replay:frontier-1"
        )
        assert successor not in resolved and successor.startswith("raw-authority-blocker:")

    @pytest.mark.asyncio
    @pytest.mark.timeout(0)
    @pytest.mark.parametrize("case", ["acknowledged_reopens", "unchanged_converges", "disproved_stays_closed"])
    async def test_real_prepared_obligation_lifecycle(self, tmp_path: Path, case: str) -> None:
        root = tmp_path / "archive"
        payload = json.dumps(
            [
                {
                    "id": "neutral-reopen",
                    "current_node": "m",
                    "mapping": {
                        "m": {
                            "id": "m",
                            "parent": None,
                            "children": [],
                            "message": {
                                "id": "m",
                                "author": {"role": "user"},
                                "create_time": 1,
                                "content": {"content_type": "text", "parts": ["neutral"]},
                            },
                        }
                    },
                }
            ]
        ).encode()

        def acquire() -> str:
            bootstrap_archive_root(root)
            with ArchiveStore.open_existing(root, read_only=False) as archive:
                return archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path="neutral.json",
                    canonical_source_path="neutral.json",
                    acquired_at_ms=1,
                )

        raw_id = await run_archive_fixture_write(root, acquire)
        async with prepared_live_convergence_owner(root) as owner:
            receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
            assert sum(len(receipt.written_session_ids) for receipt in receipts) == 1
            blob = BlobStore(root / "blob").blob_path(hashlib.sha256(payload).hexdigest())
            blob.write_bytes(payload + b"neutral-corruption")

            async def inspect() -> None:
                await owner.run_convergence_sync(
                    "fixture.frontier.reopen",
                    inspect_prepared_raw_authority_frontier,
                    root,
                    input_demand=owner._compute_adapter.amend_current_input_demand,
                    check_physical_dependencies=True,
                )

            await inspect()
            original_rows = _blocker_rows(root)
            assert len(original_rows) == 1 and original_rows[0][1]
            original_id = original_rows[0][0]
            assert raw_materialization_readiness_snapshot(root)["raw_authority_blocker_count"] == 1
            if case == "acknowledged_reopens":

                def acknowledge() -> None:
                    with prepared_frontier_blocker_acknowledgement(
                        root,
                        original_id,
                        resolution="acknowledged; reacquire later",
                        input_demand=owner._compute_adapter.amend_current_input_demand,
                    ) as prepared:
                        assert prepared.found
                        receipt = admit_stage_write("fixture.frontier.ack", prepared.publish)
                        assert receipt["blocker_id"] == original_id

                await owner.run_convergence_sync("fixture.frontier.ack", acknowledge)
                assert raw_materialization_readiness_snapshot(root)["raw_authority_blocker_count"] == 0
                await inspect()
                rows = _blocker_rows(root)
                assert len(rows) == 2 and (original_id, False) in rows
                assert sum(opened for _, opened in rows) == 1
                assert next(key for key, opened in rows if opened) != original_id
            elif case == "disproved_stays_closed":
                blob.write_bytes(payload)
                await inspect()
                assert _blocker_rows(root) == [(original_id, False)]
                await inspect()
                assert _blocker_rows(root) == [(original_id, False)]
                assert raw_materialization_readiness_snapshot(root)["raw_authority_blocker_count"] == 0
            else:
                await inspect()
                await inspect()
                assert _blocker_rows(root) == original_rows
                assert raw_materialization_readiness_snapshot(root)["raw_authority_blocker_count"] == 1
