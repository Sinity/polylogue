"""Daemon-internal automagic bulk-scale index rebuild routing.

polylogue-m6tp phase (c) / polylogue-gd6v. The daemon's trickle
raw-materialization conveyor (``_periodic_raw_materialization_convergence``,
``polylogue/daemon/cli.py``) is sized for steady-state drift; a bulk-scale
backlog (#3145's threshold) turns it into a weeks-scale grind. This module
lets the daemon itself route a bulk-scale backlog into a resumable,
transactional, blue-green generation build -- reusing the SAME engine the
offline ``polylogue ops maintenance rebuild-index`` CLI command drives
(``polylogue.maintenance.rebuild_index.rebuild_index_from_source``), never a
duplicate implementation -- and promote it once exact-ready, with zero
operator involvement (the automagic-invariants doctrine: the daemon
maintains the invariant itself).

Two properties this module adds on top of the existing rebuild engine:

* **Parallel, off-writer-hold parse** for the bulk path, by reusing the
  #3168 ``DaemonParseStage`` seam: the NEXT bounded pass's raw ids are known
  in advance (the transaction's own paged cursor,
  ``IndexGenerationStore.next_raw_page``), so they can be pre-parsed in a
  bounded thread pool before the writer-coordinated pass ever requests the
  writer hold. Degrades gracefully to the existing in-hold sequential parse
  on a GIL build or any prefetch miss -- see ``DaemonParseStage`` and
  ``RawParsePrefetchCache`` for the equivalence guarantee this rests on.
* **O(remaining-work) interruption recovery** (polylogue-fbte): the daemon
  resolves the SAME well-known operation id every tick
  (``DAEMON_BULK_REBUILD_OPERATION_ID``), so a daemon restart mid-build finds
  the persisted transaction (with its ``last_raw_id``/``processed_raw_count``
  cursor, populated by every bounded pass -- see
  ``IndexGenerationStore.checkpoint_transaction``) and resumes from there
  instead of re-walking the whole corpus. This is the property fbte
  identified as missing from the CLI's own invocation model (an operator
  who forgets ``--operation-id`` silently starts a fresh transaction); the
  daemon can never make that mistake because it never has an "operation id"
  input to forget -- there is exactly one daemon-owned bulk-rebuild
  operation per archive, always resolved the same way.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.storage.index_generation import (
    IndexGenerationStore,
    IndexRebuildTransaction,
    source_revision_snapshot,
)

if TYPE_CHECKING:
    from polylogue.daemon.parse_prefetch import DaemonParseStage
    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

logger = get_logger(__name__)

#: Fixed operation id for the daemon's own bulk-rebuild transaction. Exactly
#: one such operation is ever in flight per archive -- this module's only
#: caller is a single daemon asyncio loop -- so a well-known id lets every
#: tick resolve the same resumable transaction with an O(1) file read
#: instead of scanning every transaction under
#: ``.index-rebuild-transactions/``. This also keeps the daemon's own
#: automagic operation distinct from any operator-run
#: ``polylogue ops maintenance rebuild-index`` invocation, which always
#: mints its own random operation id and is untouched by this module.
DAEMON_BULK_REBUILD_OPERATION_ID = "daemon-bulk-rebuild"

#: Raw rows scheduled per bounded pass -- mirrors the offline CLI's own
#: default (``RebuildIndexRequest.raw_batch_size``), small enough to keep
#: the writer coordinator responsive to interleaved live-ingest/trickle
#: writer actors between passes.
DAEMON_BULK_REBUILD_BATCH_SIZE = 500

#: Transaction statuses that mean "not resumable, retire and start fresh at
#: the same well-known operation id": ``promoted`` (a prior build already
#: succeeded and is now the active index), ``stale`` (source evidence
#: changed mid-build), ``failed`` (a pass raised; automagic doctrine retries
#: rather than waiting on an operator to intervene).
_TERMINAL_NOT_RESUMABLE = frozenset({"promoted", "stale", "failed"})


def resolve_or_start_daemon_bulk_rebuild_transaction(root: Path) -> IndexRebuildTransaction:
    """Load the daemon's resumable bulk-rebuild transaction, starting one if needed.

    Read-only fast path when a resumable transaction already exists (a
    single JSON read); only touches the filesystem otherwise, and only to
    retire a terminal transaction/generation before creating a fresh one at
    the SAME well-known operation id (see ``DAEMON_BULK_REBUILD_OPERATION_ID``).
    Never touches the ACTIVE index or ``source.db`` -- a fresh generation is
    a brand-new SQLite file under ``.index-generations/``.
    """
    store = IndexGenerationStore(root)
    transaction: IndexRebuildTransaction | None
    try:
        transaction = store.load_transaction(DAEMON_BULK_REBUILD_OPERATION_ID)
    except FileNotFoundError:
        transaction = None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "bulk-rebuild: could not load persisted transaction %s; starting a fresh one: %s",
            DAEMON_BULK_REBUILD_OPERATION_ID,
            exc,
        )
        transaction = None

    if transaction is not None and transaction.status not in _TERMINAL_NOT_RESUMABLE:
        return transaction

    if transaction is not None:
        # Terminal: retire the old candidate/transaction record before
        # reusing the well-known operation id. A "promoted" generation is
        # already the active index (nothing to discard); "stale"/"failed"
        # candidates are still inactive and safe to discard.
        try:
            generation = store.load(transaction.generation_id)
        except (FileNotFoundError, OSError, ValueError):
            generation = None
        if generation is not None and generation.state == "inactive":
            store.discard_if_inactive(generation)
        store.discard_transaction(DAEMON_BULK_REBUILD_OPERATION_ID)

    return store.create_transaction(
        source_snapshot=source_revision_snapshot(root),
        operation_id=DAEMON_BULK_REBUILD_OPERATION_ID,
    )


def has_resumable_daemon_bulk_rebuild_transaction(root: Path) -> bool:
    """Whether a daemon bulk-rebuild operation is already in progress.

    Read-only: never creates or discards anything. Used to decide whether
    to keep driving an in-flight build even when the instantaneous
    raw-materialization backlog reading has dipped below the bulk-scale
    threshold -- abandoning a partially-built generation mid-flight would
    waste every page already replayed into it.
    """
    store = IndexGenerationStore(root)
    try:
        transaction = store.load_transaction(DAEMON_BULK_REBUILD_OPERATION_ID)
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return transaction.status not in _TERMINAL_NOT_RESUMABLE


async def run_daemon_bulk_rebuild_pass(
    *,
    config: Config,
    parse_stage: DaemonParseStage,
    batch_size: int = DAEMON_BULK_REBUILD_BATCH_SIZE,
    max_payload_bytes: int,
) -> RebuildIndexReceipt | None:
    """Drive one bounded daemon-owned bulk-rebuild pass.

    Returns ``None`` when the operation is already ``promoted`` (nothing to
    do this tick -- the caller's next threshold check will decide whether a
    new operation is warranted). Otherwise pre-warms the NEXT page's parse
    off the writer hold (the #3168 ``DaemonParseStage`` seam) before
    scheduling the writer-coordinated pass, so the writer hold covers mostly
    already-parsed SQLite writes rather than CPU-bound decode.

    The actual write pass reuses
    ``polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync``
    unmodified -- the SAME engine the offline CLI rebuild command drives --
    scheduled through the daemon's single write coordinator exactly like
    every other daemon writer actor (single-writer invariant: this module
    never opens a second writer connection of its own).
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync

    root = Path(config.archive_root)
    transaction = await asyncio.to_thread(resolve_or_start_daemon_bulk_rebuild_transaction, root)
    if transaction.status == "promoted":
        return None

    store = IndexGenerationStore(root)
    page = await asyncio.to_thread(store.next_raw_page, transaction, limit=batch_size)
    raw_ids = [raw_id for raw_id, _acquired_at_ms, _blob_size in page.rows]
    if raw_ids:
        warmed = await asyncio.to_thread(
            parse_stage.warm_raw_ids,
            config,
            raw_ids=raw_ids,
            max_payload_bytes=max_payload_bytes,
        )
        if warmed:
            logger.info(
                "bulk-rebuild: parse-stage prefetch warmed %d of %d raw(s) for the next pass off the writer hold",
                warmed,
                len(raw_ids),
            )

    request = RebuildIndexRequest(
        archive_root=root,
        promote=True,
        operation_id=transaction.operation_id,
        raw_batch_size=batch_size,
        prefetch_cache=parse_stage.cache,
    )
    return await daemon_write_coordinator().run_sync(
        "maintenance.bulk_rebuild",
        rebuild_index_from_source_sync,
        request,
    )


__all__ = [
    "DAEMON_BULK_REBUILD_BATCH_SIZE",
    "DAEMON_BULK_REBUILD_OPERATION_ID",
    "has_resumable_daemon_bulk_rebuild_transaction",
    "resolve_or_start_daemon_bulk_rebuild_transaction",
    "run_daemon_bulk_rebuild_pass",
]
