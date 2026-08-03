"""polylogue-b5l.1 AC1: the raw-replay rebuild must hold ``RebuildLease`` for
its ENTIRE lifecycle, not merely a point-in-time check at entry.

The 2026-07-10 competing-daemon incident was exactly a narrow point-in-time
check (``_require_service_stopped``'s systemctl probe) that missed a
transient-unit window between the check and the write. PR #2872 proved
``RebuildLease``/``ActiveWriterLease`` mutual exclusion for the clone-forward
fast-forward path (``devtools/archive_schema_fast_forward.py``); this test
proves the SAME property for the raw-replay path
(``rebuild_index_from_source`` / ``ops reset --index && polylogued run``),
at a point deep inside the pass -- after replay has already committed rows to
the owned inactive generation, immediately before terminal FTS-parity /
readiness / promotion -- not just at the top of the function.

Anti-vacuity: the mutation that makes this test fail is narrowing
``with RebuildLease(root):`` in ``_rebuild_index_from_source_owned`` to wrap
only the initial checks (e.g. moving replay/terminal-stage work outside the
``with`` block) -- exactly the "checked once, not held" shape the 2026-07-10
incident exhibited. With the lease held for the whole pass, a concurrent
``ActiveWriterLease.acquire()`` attempted from inside
``repair_session_insights`` (a terminal stage that runs AFTER replay and
BEFORE promotion) must fail; if the lease were released early, that same
attempt would silently succeed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.index_generation import ActiveWriterLease, RebuildLeaseUnavailableError

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.core.protocols import ProgressCallback
    from polylogue.storage.repair import RepairResult
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_session(native_id: str) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": f"hello {native_id}"}],
            },
        },
    ]
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed_one_codex_session(root: Path) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("sess-lease-lifecycle"),
            source_path="lease-lifecycle-test/0.jsonl",
            acquired_at_ms=1,
        )


def test_rebuild_lease_blocks_a_concurrent_writer_deep_inside_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed_one_codex_session(root)

    probe_result: dict[str, object] = {"attempted": False, "blocked": False}

    import polylogue.storage.repair as repair_module

    real_repair_session_insights = repair_module.repair_session_insights

    def probing_repair_session_insights(
        config: Config,
        dry_run: bool = False,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_total: int | None = None,
        session_ids: tuple[str, ...] | None = None,
        archive_root_override: Path | None = None,
        owned_inactive_generation: tuple[str, str] | None = None,
    ) -> RepairResult:
        # This terminal stage runs strictly AFTER replay has already
        # committed rows into the owned inactive generation, and strictly
        # BEFORE FTS parity / readiness / promotion -- exactly the window
        # the 2026-07-10 incident's narrow point-in-time check missed.
        probe_result["attempted"] = True
        writer = ActiveWriterLease(root)
        try:
            writer.acquire()
        except RebuildLeaseUnavailableError:
            probe_result["blocked"] = True
        else:
            writer.close()
        return real_repair_session_insights(
            config,
            dry_run,
            progress_callback=progress_callback,
            progress_total=progress_total,
            session_ids=session_ids,
            archive_root_override=archive_root_override,
            owned_inactive_generation=owned_inactive_generation,
        )

    monkeypatch.setattr(repair_module, "repair_session_insights", probing_repair_session_insights)

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    assert receipt.status == "replayed"
    assert probe_result["attempted"] is True, "the probe never ran; the test setup itself is broken"
    assert probe_result["blocked"] is True, (
        "a concurrent ActiveWriterLease acquisition succeeded mid-pass -- RebuildLease was not held "
        "for the pass's entire lifecycle"
    )

    # After the pass returns, the lease must be released -- a later,
    # legitimate writer must not be blocked forever by a lease this rebuild
    # forgot to release.
    writer = ActiveWriterLease(root)
    writer.acquire()
    writer.close()
