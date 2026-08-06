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
import json
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance.archive_verification import read_raw_failure_lifecycle
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation, assert_owns_archive_location
from polylogue.storage.index_generation import (
    IndexGenerationStore,
    IndexRebuildTransaction,
    rebuild_source_evidence_snapshot,
)

if TYPE_CHECKING:
    from polylogue.daemon.parse_prefetch import DaemonParseStage
    from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

logger = get_logger(__name__)


def _validate_rebuild_provenance_receipt(root: Path, receipt_path: Path | None) -> None:
    """Validate daemon rebuild provenance at the current ownership boundary."""
    from polylogue.maintenance.schema_inference_gate import (
        SchemaInferenceGateError,
        validate_schema_inference_receipt,
    )

    try:
        validate_schema_inference_receipt(root, receipt_path)
    except SchemaInferenceGateError as exc:
        raise RuntimeError(f"daemon bulk rebuild schema-inference preflight gate failed: {exc}") from exc


def _discard_daemon_transaction_after_provenance_failure(
    store: IndexGenerationStore, transaction: IndexRebuildTransaction
) -> list[BaseException]:
    """Attempt candidate and transaction cleanup independently.

    The caller still owns the archive location. Cleanup does not consume source
    evidence, so it must remain available even when the receipt that authorized
    candidate creation has just expired or the source snapshot has drifted.
    Each record is retired independently so one failed cleanup step cannot
    prevent the other record from being attempted. False return values are
    failures too: a daemon cleanup path must never silently strand an inactive
    candidate.
    """
    errors: list[BaseException] = []
    try:
        generation = store.load(transaction.generation_id)
        if generation.state != "inactive":
            errors.append(RuntimeError(f"candidate {generation.generation_id} is not inactive"))
        elif not store.discard_if_inactive(generation):
            errors.append(RuntimeError(f"candidate {generation.generation_id} was not discarded"))
    except BaseException as exc:
        logger.error("bulk-rebuild: candidate discard raised", exc_info=True)
        cleanup_error = RuntimeError(f"candidate {transaction.generation_id} discard failed: {exc}")
        cleanup_error.__cause__ = exc
        errors.append(cleanup_error)
    try:
        if not store.discard_transaction(transaction.operation_id):
            errors.append(RuntimeError(f"transaction {transaction.operation_id} was not discarded"))
    except BaseException as exc:
        logger.error("bulk-rebuild: transaction discard raised", exc_info=True)
        cleanup_error = RuntimeError(f"transaction {transaction.operation_id} discard failed: {exc}")
        cleanup_error.__cause__ = exc
        errors.append(cleanup_error)
    return errors


def _surface_daemon_cleanup_failures(
    primary: BaseException, cleanup_errors: list[BaseException], *, label: str
) -> None:
    """Keep the primary failure while surfacing every cleanup outcome."""
    if cleanup_errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
        primary.add_note(f"{label} cleanup also failed: {detail}")


def _raise_daemon_cleanup_failures(cleanup_errors: list[BaseException], *, label: str) -> None:
    """Raise when terminal daemon cleanup itself is the primary failure."""
    if cleanup_errors:
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
        raise RuntimeError(f"{label} cleanup failed: {detail}") from cleanup_errors[0]


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


def _preflight_raw_failure_lifecycle(root: Path) -> None:
    """Refuse daemon rebuild mutation while raw failures are unexplained."""
    snapshot = read_raw_failure_lifecycle(root / "source.db", sample_limit=10)
    if not snapshot.blocking:
        return
    if snapshot.available:
        reason = f"{snapshot.unexplained} raw failure(s) lack matching typed lifecycle evidence"
    else:
        reason = snapshot.reason or "raw failure lifecycle is unavailable"
    raise RuntimeError(f"daemon bulk-rebuild raw failure lifecycle preflight failed: {reason}")


def resolve_or_start_daemon_bulk_rebuild_transaction(
    root: Path, *, schema_inference_receipt_path: Path | None = None
) -> IndexRebuildTransaction:
    """Load the daemon's resumable bulk-rebuild transaction, starting one if needed.

    Read-only fast path when a resumable transaction already exists (a
    single JSON read); only touches the filesystem otherwise, and only to
    retire a terminal transaction/generation before creating a fresh one at
    the SAME well-known operation id (see ``DAEMON_BULK_REBUILD_OPERATION_ID``).
    Never touches the ACTIVE index or ``source.db`` -- a fresh generation is
    a brand-new SQLite file under ``.index-generations/``.

    Acquires :class:`~polylogue.storage.archive_identity.OwnedArchiveLocation`
    over ``root`` before any of that mutation happens (polylogue-ovme.2.1):
    this is the online/daemon-driven counterpart to
    ``rebuild_index_from_source``'s own offline ownership acquisition
    (polylogue-ovme.2 AC3) -- both discard a stale candidate and mint a fresh
    generation directory, so both must fail closed against a foreign/rotated
    archive location before touching disk, not just the eventual write pass.
    """
    from polylogue.maintenance.rebuild_index import require_rebuild_schema_currency

    require_rebuild_schema_currency(root)
    _validate_rebuild_provenance_receipt(root, schema_inference_receipt_path)
    # This must precede transaction resolution, because retiring a terminal
    # transaction and creating its replacement also creates generation state.
    # It must also precede the caller's page selection for a resumed
    # transaction, so no raw is selected while the source failure universe is
    # not in a known lifecycle state.
    _preflight_raw_failure_lifecycle(root)
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location)
    try:
        assert_owns_archive_location(owned, location)
        # The early check is a cheap rejection before receipt work. Repeat it
        # under archive ownership because a previous owner can migrate a
        # durable tier while this caller waits for the lock.
        require_rebuild_schema_currency(root)
        # The first validation is only a cheap early rejection. Revalidate
        # after ownership acquisition so receipt expiry, source revision, or
        # external-corpus drift cannot reach generation bookkeeping.
        _validate_rebuild_provenance_receipt(root, schema_inference_receipt_path)
        store = IndexGenerationStore(location)
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
            _validate_rebuild_provenance_receipt(root, schema_inference_receipt_path)
            return transaction

        if transaction is not None:
            # Terminal: retire the old candidate/transaction record before
            # reusing the well-known operation id. A "promoted" generation is
            # already the active index (nothing to discard); "stale"/"failed"
            # candidates are still inactive and safe to discard.
            cleanup_errors: list[BaseException] = []
            if transaction.status != "promoted":
                try:
                    generation = store.load(transaction.generation_id)
                except BaseException as exc:
                    logger.error("bulk-rebuild: terminal candidate load failed", exc_info=True)
                    cleanup_errors.append(exc)
                else:
                    if generation.state != "inactive":
                        cleanup_errors.append(RuntimeError(f"candidate {generation.generation_id} is not inactive"))
                    else:
                        try:
                            if not store.discard_if_inactive(generation):
                                cleanup_errors.append(
                                    RuntimeError(f"candidate {generation.generation_id} was not discarded")
                                )
                        except BaseException as exc:
                            logger.error("bulk-rebuild: terminal candidate discard raised", exc_info=True)
                            cleanup_error = RuntimeError(f"candidate {generation.generation_id} discard failed: {exc}")
                            cleanup_error.__cause__ = exc
                            cleanup_errors.append(cleanup_error)
            try:
                if not store.discard_transaction(DAEMON_BULK_REBUILD_OPERATION_ID):
                    cleanup_errors.append(
                        RuntimeError(f"transaction {DAEMON_BULK_REBUILD_OPERATION_ID} was not discarded")
                    )
            except BaseException as exc:
                logger.error("bulk-rebuild: terminal transaction discard failed", exc_info=True)
                cleanup_errors.append(exc)
            _raise_daemon_cleanup_failures(cleanup_errors, label="daemon bulk-rebuild terminal")

        _validate_rebuild_provenance_receipt(root, schema_inference_receipt_path)
        source_snapshot = rebuild_source_evidence_snapshot(root)
        transaction = store.create_transaction(
            source_snapshot=source_snapshot,
            operation_id=DAEMON_BULK_REBUILD_OPERATION_ID,
        )
        try:
            _validate_rebuild_provenance_receipt(root, schema_inference_receipt_path)
            if rebuild_source_evidence_snapshot(root) != source_snapshot:
                raise RuntimeError("daemon bulk rebuild source evidence changed during transaction creation")
        except BaseException as exc:
            cleanup_errors = _discard_daemon_transaction_after_provenance_failure(store, transaction)
            _surface_daemon_cleanup_failures(exc, cleanup_errors, label="daemon bulk-rebuild transaction")
            raise
        return transaction
    finally:
        owned.release()


def has_resumable_daemon_bulk_rebuild_transaction(root: Path) -> bool:
    """Whether a daemon bulk-rebuild operation is already in progress.

    Read-only: never creates or discards anything. Used to decide whether
    to keep driving an in-flight build even when the instantaneous
    raw-materialization backlog reading has dipped below the bulk-scale
    threshold -- abandoning a partially-built generation mid-flight would
    waste every page already replayed into it.
    """
    anchor = root / ".index-active-pointer"
    if not anchor.exists() and not anchor.is_symlink():
        return False
    try:
        pointer_target = Path(anchor.read_text(encoding="utf-8").strip())
        if not pointer_target.is_absolute() or pointer_target.name != "index.db":
            return False
        transaction_path = (
            pointer_target.parent / ".index-rebuild-transactions" / f"{DAEMON_BULK_REBUILD_OPERATION_ID}.json"
        )
        transaction = IndexRebuildTransaction(**json.loads(transaction_path.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
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
    from polylogue.maintenance.schema_inference_gate import resolve_schema_inference_receipt_reference

    root = Path(config.archive_root)
    receipt_path = resolve_schema_inference_receipt_reference(root)
    transaction = await asyncio.to_thread(
        resolve_or_start_daemon_bulk_rebuild_transaction,
        root,
        schema_inference_receipt_path=receipt_path,
    )
    if transaction.status == "promoted":
        return None

    location = ArchiveLocation.resolve(root)
    owned = await asyncio.to_thread(OwnedArchiveLocation.acquire, location)
    try:
        await asyncio.to_thread(assert_owns_archive_location, owned, location)
        await asyncio.to_thread(_validate_rebuild_provenance_receipt, root, receipt_path)
        store = IndexGenerationStore(location)
        await asyncio.to_thread(_validate_rebuild_provenance_receipt, root, receipt_path)
        page = await asyncio.to_thread(store.next_raw_page, transaction, limit=batch_size)
    finally:
        owned.release()
    raw_ids = [raw_id for raw_id, _blob_hash_hex, _blob_size in page.rows]
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
        schema_inference_receipt_path=receipt_path,
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
