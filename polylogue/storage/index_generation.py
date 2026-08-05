"""Owned inactive index generations and atomic active-index promotion."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

logger = logging.getLogger(__name__)

_LOCK_PID_PATTERN = re.compile(r"pid=(\d+)")
_LOCK_HOST_PATTERN = re.compile(r"host=(\S+)")

#: ``raw_session_memberships.decision`` values that mean "this raw is
#: resolved, durable history" rather than resume debt -- reused directly from
#: ``classify_membership_revisions``'s own closed vocabulary
#: (``polylogue.archive.session_revision_membership.MembershipDecision``)
#: instead of a rebuild-local classifier (polylogue-b5l.1 design note: reuse
#: the existing revision authority vocabulary, never fork a parallel one).
_SUPERSEDED_DECISIONS: tuple[str, ...] = (
    MembershipDecision.SUPERSEDED_EQUIVALENT.value,
    MembershipDecision.SUPERSEDED_PREFIX.value,
)
_SUPERSEDED_DECISION_PLACEHOLDERS = ",".join("?" for _ in _SUPERSEDED_DECISIONS)

#: Superseded generations kept after a promotion.  One is enough to roll back
#: to the previous index; each costs roughly the size of the index itself
#: (~35 GB on the reference archive), so keeping more is expensive storage,
#: not cheap insurance.
SUPERSEDED_GENERATION_RETENTION = 1
_GENERATIONS_DIRNAME = ".index-generations"


def _is_generation_member(path: Path) -> bool:
    """True when ``path`` lives inside a generation directory rather than beside one.

    The canonical index pointer must not name a path inside a generation,
    because ``generations_root`` and ``transactions_root`` are derived from the
    pointer's *parent*: a pointer at ``…/.index-generations/gen-X/index.db``
    makes them nest as ``…/gen-X/.index-generations``, which is the shape the
    self-poisoning bug produced.

    The test is deliberately narrower than "the path mentions
    ``.index-generations`` somewhere". Two cases must stay distinguishable:

    * ``…/.index-generations/gen-X/index.db`` -- a generation *member*, refused.
      The pointer is at least one directory below the generations root, which is
      exactly how a generation's own files sit.
    * ``…/.index-generations/index.db`` -- a file sitting *directly* in a
      directory that happens to carry that name, allowed. An archive root may
      legitimately be named anything, including this, and there the derived
      roots stay self-consistent. Rejecting it on a name match alone would break
      a valid symlink-farm target for no invariant's benefit.

    Uses ``absolute()`` rather than ``resolve()`` on purpose: resolving would
    follow ``index.db``'s own promotion symlink into the generation it targets,
    so the canonical pointer would classify itself as poisoned.
    """
    parts = path.absolute().parts
    try:
        depth = parts.index(_GENERATIONS_DIRNAME)
    except ValueError:
        return False
    # A direct child is `.index-generations/<name>` (one part after the root);
    # anything deeper is inside a generation.
    return len(parts) - depth > 2


@dataclass(frozen=True, slots=True)
class IndexGeneration:
    generation_id: str
    owner_id: str
    archive_root: str
    index_path: str
    state: str
    created_at_ms: int
    source_snapshot: str = ""


@dataclass(frozen=True, slots=True)
class IndexRebuildTransaction:
    """Durable cursor and candidate ownership for one source-index rebuild.

    A transaction is deliberately retained while it is paused or failed.  The
    inactive generation is useful work, not disposable scratch: resuming the
    same source snapshot continues from the next raw key without exposing a
    partial index to readers.
    """

    operation_id: str
    generation_id: str
    generation_owner_id: str
    source_snapshot: str
    status: str
    created_at_ms: int
    updated_at_ms: int
    last_blob_hash_hex: str | None = None
    last_raw_id: str | None = None
    processed_raw_count: int = 0
    processed_blob_bytes: int = 0
    pass_byte_budget: int | None = None
    pass_deadline_ms: int | None = None
    error: str | None = None
    # The transaction record is also the durable operation receipt.  Keep the
    # operating process separate from the opaque generation owner so a status
    # reader can tell a retained candidate from the process that last made
    # progress on it.
    owner_pid: int | None = None
    owner_host: str | None = None
    heartbeat_at_ms: int | None = None
    # polylogue-v6i3: set once a RESUMED pass has explicitly emptied this
    # generation's messages_fts/blocks_command_trigram (defensive idempotent
    # bookkeeping -- a fresh generation starts empty by construction and never
    # needs this, but a resumed pass makes "derived stores are empty" an
    # explicit, code-verified invariant instead of an assumption inherited
    # from generation creation). Missing on older persisted transaction JSON
    # defaults to ``False`` via the dataclass default, so an in-flight
    # transaction created before this field existed is treated as not yet
    # cleared and clears exactly once on its next resume.
    derived_stores_cleared: bool = False

    @property
    def cursor(self) -> str | None:
        if self.last_blob_hash_hex is None or self.last_raw_id is None:
            return None
        return f"source:{self.last_blob_hash_hex}:{self.last_raw_id}"


@dataclass(frozen=True, slots=True)
class RebuildRawPage:
    """One bounded content-order scheduling decision.

    Rows are ``(raw_id, blob_hash_hex, blob_size)``, ordered by
    ``(blob_hash, raw_id)`` (polylogue-hord) rather than acquisition time --
    see ``IndexGenerationStore.next_raw_page`` for why: content order makes
    byte-identical duplicates adjacent so the existing per-page dedup group
    in ``_parse_retained_raws`` (and the cross-page content cache layered on
    top of it) actually captures them, instead of only catching whatever
    duplicates happen to land close together in acquisition-time order.

    ``deferred_reason`` is scheduling evidence, not an admission decision: an
    oversized first row is still scheduled alone, and every later row remains
    reachable from the persisted keyset cursor on a later invocation.
    """

    rows: tuple[tuple[str, str, int], ...]
    has_more: bool
    deferred_reason: str | None = None


class RebuildLeaseUnavailableError(RuntimeError):
    """Another process owns the archive-wide rebuild lease."""


def _lock_holder_pid(path: Path) -> int | None:
    """Best-effort recorded pid from an existing lock file; ``None`` if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _LOCK_PID_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _lock_holder_host(path: Path) -> str | None:
    """Best-effort recorded hostname from an existing lock file; ``None`` if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _LOCK_HOST_PATTERN.search(text)
    return match.group(1) if match is not None else None


def _pid_is_alive(pid: int) -> bool:
    """Whether ``pid`` still names a live process, best-effort via ``kill(pid, 0)``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user but still running.
        return True
    return True


def _open_lock_fd(path: Path, lock_type: int, *, unavailable_message: str) -> int:
    """Open ``path`` and acquire ``lock_type`` (``LOCK_EX``/``LOCK_SH``), non-blocking.

    ``flock`` is scoped to the open file description's *inode*, so a holder
    that died without releasing it -- a crashed rebuild, or a forked worker
    that inherited the fd and outlived a since-reaped parent -- cannot be
    un-blocked by simply re-opening the same path; whatever still references
    the old inode keeps it locked. When the lock file's recorded pid is no
    longer a running process, the stale lock is reclaimed instead: a fresh
    file is written at the same path and swapped in atomically, handing out
    a brand-new, guaranteed-unlocked inode while any surviving reference to
    the old one is left orphaned (polylogue-k8kj finding: a dead-pid lock
    was blocking nothing in particular while confusing operators who read
    its stale content as an active rebuild).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
        return fd
    except BlockingIOError as exc:
        os.close(fd)
        blocking_error = exc

    holder_pid = _lock_holder_pid(path)
    if holder_pid is None or _pid_is_alive(holder_pid):
        suffix = f" (pid={holder_pid})" if holder_pid is not None else ""
        raise RebuildLeaseUnavailableError(unavailable_message + suffix) from blocking_error

    logger.warning(
        "reclaiming stale index rebuild lease %s: recorded holder pid=%d is no longer running",
        path,
        holder_pid,
    )
    temporary = path.with_name(f".{path.name}.reclaim-{uuid.uuid4().hex}")
    reclaimed_fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        fcntl.flock(reclaimed_fd, lock_type | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(reclaimed_fd)
        temporary.unlink(missing_ok=True)
        raise RebuildLeaseUnavailableError(unavailable_message) from blocking_error
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return reclaimed_fd


class RebuildLease:
    """Process-held exclusive lease for an offline index rebuild."""

    def __init__(self, archive_root: Path) -> None:
        self.path = archive_root / ".index-rebuild.lock"
        self._fd: int | None = None

    def __enter__(self) -> RebuildLease:
        fd = _open_lock_fd(
            self.path,
            fcntl.LOCK_EX,
            unavailable_message=f"index rebuild lease is already held: {self.path}",
        )
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} host={socket.gethostname()}\n".encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class ActiveWriterLease:
    """Shared process-held lease refused while an offline rebuild owns the archive."""

    def __init__(self, archive_root: Path) -> None:
        self.path = archive_root / ".index-rebuild.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        self._fd = _open_lock_fd(
            self.path,
            fcntl.LOCK_SH,
            unavailable_message=f"offline index rebuild owns archive: {self.path}",
        )

    def close(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


@dataclass(frozen=True, slots=True)
class RebuildLeaseStatus:
    """Read-only snapshot of the archive-root rebuild lease, for status surfaces.

    polylogue-b5l.1 AC5: an operator/agent must be able to see who owns the
    lease, whether the recorded holder is actually still alive, and whether
    the lock looks reclaimable, without disturbing a real holder and without
    duplicating ``RebuildLease``/``ActiveWriterLease`` as the sole exclusion
    mechanism.
    """

    held: bool
    holder_pid: int | None
    holder_host: str | None
    #: ``None`` when ``held`` is False (nothing to check liveness against) or
    #: when no pid could be parsed from the lock file at all.
    holder_alive: bool | None
    #: True when the lease is held but its recorded pid is provably dead --
    #: exactly the ``RebuildLease.__enter__`` reclaim condition
    #: (``_open_lock_fd``), surfaced here for operator visibility before a
    #: fresh acquisition would silently reclaim it.
    stale: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "held": self.held,
            "holder_pid": self.holder_pid,
            "holder_host": self.holder_host,
            "holder_alive": self.holder_alive,
            "stale": self.stale,
        }


def rebuild_lease_status(archive_root: Path) -> RebuildLeaseStatus:
    """Probe the rebuild lease without blocking or disturbing a genuine holder.

    Attempts a non-blocking exclusive ``flock``: if it succeeds, nothing
    currently holds the lease and the probe releases it immediately; if it
    fails with ``EAGAIN``/``EACCES`` (``BlockingIOError``), the lease is
    genuinely held and the lock file's recorded pid/host are reported
    best-effort for diagnosis (the file content may be stale or unreadable).
    """
    path = archive_root / ".index-rebuild.lock"
    if not path.exists():
        return RebuildLeaseStatus(held=False, holder_pid=None, holder_host=None, holder_alive=None, stale=False)
    holder_pid = _lock_holder_pid(path)
    holder_host = _lock_holder_host(path)
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            alive = _pid_is_alive(holder_pid) if holder_pid is not None else None
            return RebuildLeaseStatus(
                held=True,
                holder_pid=holder_pid,
                holder_host=holder_host,
                holder_alive=alive,
                stale=holder_pid is not None and alive is False,
            )
        fcntl.flock(fd, fcntl.LOCK_UN)
        return RebuildLeaseStatus(
            held=False, holder_pid=holder_pid, holder_host=holder_host, holder_alive=None, stale=False
        )
    finally:
        os.close(fd)


class IndexGenerationStore:
    """Create, checkpoint, and atomically promote inactive generations.

    Constructed from an already-resolved :class:`~polylogue.storage.archive_identity.ArchiveLocation`
    rather than a bare ``archive_root: Path`` (polylogue-ovme.2.1): the two
    boundaries have deliberately different jobs. ``ArchiveLocation.resolve()``
    is a pure, side-effect-free *read* of whatever pointer state already
    exists; this store additionally performs first-touch pointer
    **bootstrap** -- writing ``.index-active-pointer`` the first time an
    archive is opened, before any generation has ever been promoted --
    which ``ArchiveLocation.resolve()`` intentionally never does (a read-only
    resolver must never mutate the archive it is describing). Passing an
    ``ArchiveLocation`` in still lets this constructor reuse its pointer-read
    outcome instead of re-deriving it, while keeping the bootstrap-write
    behavior (and the ``.index-generations``-anchor sanity check below, which
    ``ArchiveLocation.resolve()`` also does not perform) here where the write
    authority belongs.
    """

    def __init__(self, location: ArchiveLocation) -> None:
        self.archive_root = location.configured_root
        self.location = location
        anchor = location.configured_root / ".index-active-pointer"
        configured_index = location.configured_tier("index").configured_path
        anchored = location.active_pointer
        if anchored is not None and not _is_generation_member(anchored):
            self.active_pointer = anchored
        else:
            # Recompute, and rewrite the anchor, when it is absent OR poisoned.
            #
            # The canonical pointer is the path ``promote()`` replaces with a
            # symlink -- i.e. the archive's own ``index.db`` -- never the
            # generation that symlink currently targets. Following the symlink
            # here wrote a ``.index-generations/gen-*/index.db`` path into the
            # anchor, which the next construction then rejected outright; the
            # store poisoned its own anchor on first use and refused every run
            # afterwards. It also made ``generations_root`` nest as
            # ``.index-generations/gen-*/.index-generations``.
            #
            # A symlink is still followed when it leaves the archive (an
            # archive root that is a symlink farm pointing at the real
            # location), because there the canonical pointer genuinely lives
            # elsewhere. Only a link that lands *inside a generation* is
            # refused -- see ``_is_generation_member``. Treating a poisoned
            # anchor as recoverable rather than fatal lets an archive already
            # carrying one heal on next open, instead of needing the file
            # repaired by hand.
            if configured_index.is_symlink():
                target = Path(os.readlink(configured_index))
                resolved = target if target.is_absolute() else configured_index.parent / target
                self.active_pointer = configured_index if _is_generation_member(resolved) else resolved
            else:
                self.active_pointer = configured_index
            temporary = anchor.with_suffix(".tmp")
            # Constructing the store must not require the archive root to have
            # been materialized first. Daemon bulk-rebuild routing is now
            # unconditional, so this runs on every convergence tick -- including
            # against a configured-but-not-yet-created root, where the eager
            # pointer write previously raised FileNotFoundError.
            anchor.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(str(self.active_pointer.absolute()), encoding="utf-8")
            os.replace(temporary, anchor)
            _fsync_directory(anchor.parent)
        self.generations_root = self.active_pointer.parent / ".index-generations"
        self.transactions_root = self.active_pointer.parent / ".index-rebuild-transactions"

    @classmethod
    def for_archive_root(cls, archive_root: Path) -> IndexGenerationStore:
        """Convenience constructor resolving ``archive_root`` into an :class:`ArchiveLocation` first."""
        return cls(ArchiveLocation.resolve(archive_root))

    def create_transaction(
        self,
        *,
        source_snapshot: str,
        operation_id: str | None = None,
        pass_byte_budget: int | None = None,
        pass_deadline_ms: int | None = None,
    ) -> IndexRebuildTransaction:
        """Create an inactive candidate and its resumable transaction record."""
        op_id = operation_id or str(uuid.uuid4())
        path = self._transaction_path(op_id)
        if path.exists():
            raise RuntimeError(f"rebuild transaction already exists: {op_id}")
        generation = self.create(source_snapshot=source_snapshot)
        now = int(time.time() * 1000)
        transaction = IndexRebuildTransaction(
            operation_id=op_id,
            generation_id=generation.generation_id,
            generation_owner_id=generation.owner_id,
            source_snapshot=source_snapshot,
            status="running",
            created_at_ms=now,
            updated_at_ms=now,
            pass_byte_budget=pass_byte_budget,
            pass_deadline_ms=pass_deadline_ms,
            owner_pid=os.getpid(),
            owner_host=socket.gethostname(),
            heartbeat_at_ms=now,
        )
        self.save_transaction(transaction)
        return transaction

    def load_transaction(self, operation_id: str) -> IndexRebuildTransaction:
        """Load a rebuild transaction; corrupt or missing state is never resumed."""
        payload = json.loads(self._transaction_path(operation_id).read_text(encoding="utf-8"))
        return IndexRebuildTransaction(**payload)

    def save_pass_receipt(self, operation_id: str, receipt: dict[str, object]) -> Path:
        """Durably persist one rebuild pass's receipt for post-hoc recovery.

        The CLI's only other copy of a pass receipt is a JSON blob written to
        stdout; if the invoking shell's pipe dies (killed shell, SIGPIPE) an
        orphaned rebuild process keeps working but the receipt is gone
        (polylogue-k8kj live incident: two page receipts lost this way in one
        night). Each pass gets its own numbered file under a
        ``<operation_id>.receipts/`` directory alongside the transaction
        record, written with the same tmp+os.replace+fsync pattern as
        ``save_transaction``.
        """
        directory = self.transactions_root / f"{operation_id}.receipts"
        directory.mkdir(parents=True, exist_ok=True)
        sequence = len(list(directory.glob("pass-*.json")))
        path = directory / f"pass-{sequence:06d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temporary, path)
        _fsync_directory(directory)
        return path

    def save_transaction(self, transaction: IndexRebuildTransaction) -> IndexRebuildTransaction:
        """Atomically checkpoint a transaction after one bounded replay pass."""
        now = int(time.time() * 1000)
        updated = IndexRebuildTransaction(
            **{
                **asdict(transaction),
                "updated_at_ms": now,
                "owner_pid": os.getpid(),
                "owner_host": socket.gethostname(),
                "heartbeat_at_ms": now,
            }
        )
        path = self._transaction_path(transaction.operation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(updated), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return updated

    def checkpoint_transaction(
        self,
        transaction: IndexRebuildTransaction,
        *,
        status: str,
        last_blob_hash_hex: str | None = None,
        last_raw_id: str | None = None,
        processed_raw_count: int | None = None,
        processed_blob_bytes: int | None = None,
        error: str | None = None,
        derived_stores_cleared: bool | None = None,
    ) -> IndexRebuildTransaction:
        """Persist one state transition without changing candidate ownership."""
        return self.save_transaction(
            IndexRebuildTransaction(
                **{
                    **asdict(transaction),
                    "status": status,
                    "last_blob_hash_hex": last_blob_hash_hex
                    if last_blob_hash_hex is not None
                    else transaction.last_blob_hash_hex,
                    "last_raw_id": last_raw_id if last_raw_id is not None else transaction.last_raw_id,
                    "processed_raw_count": processed_raw_count
                    if processed_raw_count is not None
                    else transaction.processed_raw_count,
                    "processed_blob_bytes": processed_blob_bytes
                    if processed_blob_bytes is not None
                    else transaction.processed_blob_bytes,
                    "error": error,
                    "derived_stores_cleared": derived_stores_cleared
                    if derived_stores_cleared is not None
                    else transaction.derived_stores_cleared,
                }
            )
        )

    def discard_transaction(self, operation_id: str) -> bool:
        """Remove a terminal transaction's record so its ``operation_id`` can be reused.

        Only the record itself is removed; pass receipts under
        ``<operation_id>.receipts/`` are left in place as audit history
        (mirroring ``save_pass_receipt``'s own retention). The candidate
        generation is a SEPARATE lifecycle -- callers that also want to
        reclaim a still-inactive generation must call
        ``discard_if_inactive`` themselves; a ``promoted`` generation is
        already the active index and must never be discarded here.
        """
        path = self._transaction_path(operation_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def next_raw_page(
        self,
        transaction: IndexRebuildTransaction,
        *,
        limit: int,
    ) -> RebuildRawPage:
        """Schedule one content-order page without materializing archive-wide IDs.

        Ordered by ``(blob_hash, raw_id)``, not acquisition time
        (polylogue-hord). ``blob_hash`` is a fixed-length ``NOT NULL`` 32-byte
        digest (``009_expand_origin_vocabulary.sql``), so byte-identical raws
        -- including re-acquisitions/re-exports of the same content under an
        entirely different ``acquired_at_ms`` -- sort adjacently and land in
        the same or a neighboring page, where ``_parse_retained_raws``'s
        existing per-page dedup grouping (and the cross-page content cache
        layered on it, ``RawParsePrefetchCache``) actually collapses them
        into a single parse. Acquisition-time order scattered duplicates
        across the entire multi-hour rebuild instead, so only whatever
        happened to land in the same bounded page or the cache's bounded
        budget was ever caught.

        ``(blob_hash, raw_id)`` is still a stable total order over
        ``raw_sessions``, so the keyset cursor below resumes correctly; which
        raws land on which page changes, but nothing about correctness does
        -- revision/membership authority selection
        (``session_revision_membership.classify_membership_revisions``) is a
        pure function of each logical cohort's persisted rows (content
        hashes, ``provider_updated_at``), and cohort expansion
        (``ArchiveStore.expand_raw_membership_selection``) already walks the
        full ``raw_sessions``/``raw_session_memberships`` graph regardless of
        which page triggered it -- neither depends on processing order.

        polylogue-b5l.1: a raw whose EVERY persisted
        ``raw_session_memberships`` row already carries a durable
        ``superseded_equivalent``/``superseded_prefix`` decision (the
        classification ``classify_membership_revisions`` itself writes back,
        durable in ``source.db`` independent of any index generation) is
        legitimate resolved history, not resume debt: it will never gain an
        ``index.sessions`` row of its own (only its cohort's accepted head
        does), so scheduling it wastes a full page slot re-parsing content a
        prior pass already resolved -- every single rebuild pass would
        otherwise re-touch it. A raw with no membership row at all (never
        censused) or with at least one non-superseded row (``applied``/
        ``ambiguous``/``deferred``/still pending) is left eligible -- this
        only ever narrows the schedule, it never risks dropping a genuinely
        unresolved or newly-accepted raw.
        """
        if limit <= 0:
            raise ValueError("rebuild raw page limit must be positive")
        source_db = self.archive_root / "source.db"
        not_resume_debt_clause = f"""(
            NOT EXISTS (SELECT 1 FROM raw_session_memberships m WHERE m.raw_id = raw_sessions.raw_id)
            OR EXISTS (
                SELECT 1 FROM raw_session_memberships m
                WHERE m.raw_id = raw_sessions.raw_id
                  AND (m.decision IS NULL OR m.decision NOT IN ({_SUPERSEDED_DECISION_PLACEHOLDERS}))
            )
        )"""
        if transaction.last_blob_hash_hex is None or transaction.last_raw_id is None:
            query = f"""
                SELECT raw_id, blob_hash, blob_size FROM raw_sessions
                WHERE {not_resume_debt_clause}
                ORDER BY blob_hash, raw_id LIMIT ?
            """
            params: tuple[object, ...] = (*_SUPERSEDED_DECISIONS, limit + 1)
        else:
            last_blob_hash = bytes.fromhex(transaction.last_blob_hash_hex)
            query = f"""
                SELECT raw_id, blob_hash, blob_size FROM raw_sessions
                WHERE (blob_hash > ?
                   OR (blob_hash = ? AND raw_id > ?))
                  AND {not_resume_debt_clause}
                ORDER BY blob_hash, raw_id LIMIT ?
            """
            params = (
                last_blob_hash,
                last_blob_hash,
                transaction.last_raw_id,
                *_SUPERSEDED_DECISIONS,
                limit + 1,
            )
        with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
            rows = conn.execute(query, params).fetchall()
        selected: list[tuple[str, str, int]] = []
        selected_bytes = 0
        deferred_reason: str | None = None
        budget = transaction.pass_byte_budget
        for raw_id, blob_hash, blob_size in rows:
            raw = (str(raw_id), bytes(blob_hash).hex(), int(blob_size or 0))
            if budget is not None and selected and selected_bytes + raw[2] > budget:
                deferred_reason = "byte-budget"
                break
            # A single oversized raw must never become permanently ineligible.
            selected.append(raw)
            selected_bytes += raw[2]
            if len(selected) == limit:
                break
        has_more = len(rows) > len(selected)
        if has_more and deferred_reason is None:
            deferred_reason = "raw-batch"
        return RebuildRawPage(rows=tuple(selected), has_more=has_more, deferred_reason=deferred_reason)

    def create(self, *, owner_id: str | None = None, source_snapshot: str) -> IndexGeneration:
        generation_id = f"gen-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        owner = owner_id or str(uuid.uuid4())
        root = self.generations_root / generation_id
        root.mkdir(parents=True, exist_ok=False)
        for filename in ("source.db", "user.db", "embeddings.db", "ops.db", "blob"):
            source = self.archive_root / filename
            if source.exists() or source.is_symlink():
                (root / filename).symlink_to(source.resolve(strict=False), target_is_directory=source.is_dir())
        index_path = root / "index.db"
        initialize_archive_database(index_path, ArchiveTier.INDEX)
        generation = IndexGeneration(
            generation_id=generation_id,
            owner_id=owner,
            archive_root=str(self.archive_root.resolve(strict=False)),
            index_path=str(index_path),
            state="inactive",
            created_at_ms=int(time.time() * 1000),
            source_snapshot=source_snapshot,
        )
        self._write(generation)
        return generation

    def load(self, generation_id: str) -> IndexGeneration:
        payload = json.loads(self._metadata_path(generation_id).read_text(encoding="utf-8"))
        return IndexGeneration(**payload)

    def promote(self, generation: IndexGeneration) -> IndexGeneration:
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("only the owning inactive generation can be promoted")
        target = Path(current.index_path).resolve(strict=True)
        _checkpoint_truncate(target, label="new index")
        pointer = self.active_pointer
        retired = self.generations_root / f"retired-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        retired.mkdir(parents=True, exist_ok=False)
        if pointer.exists() or pointer.is_symlink():
            _checkpoint_truncate(pointer, label="active index")
            for suffix in ("-wal", "-shm"):
                sidecar = pointer.with_name(pointer.name + suffix)
                if sidecar.exists():
                    if suffix == "-wal" and sidecar.stat().st_size != 0:
                        raise RuntimeError(f"non-empty active index sidecar blocks promotion: {sidecar}")
                    os.replace(sidecar, retired / sidecar.name)
        if pointer.exists() or pointer.is_symlink():
            os.link(pointer, retired / "index.db", follow_symlinks=False)
            _fsync_directory(retired)
        promoting = IndexGeneration(**{**asdict(current), "state": "promoting"})
        self._write(promoting)
        temporary = pointer.parent / f".index.db.promote-{uuid.uuid4().hex}"
        temporary.symlink_to(target)
        os.replace(temporary, pointer)
        _fsync_directory(pointer.parent)
        promoted = IndexGeneration(**{**asdict(current), "state": "active"})
        self._write(promoted)
        # Housekeeping only: a failure here must not undo a promotion that has
        # already swapped the pointer and written active metadata.
        try:
            self.prune_superseded_generations()
        except OSError:
            logger.warning("index generation pruning failed after promotion", exc_info=True)
        return promoted

    def prune_superseded_generations(self, *, keep: int = SUPERSEDED_GENERATION_RETENTION) -> list[str]:
        """Delete superseded generations beyond the retention window.

        A promoted generation is ~35 GB.  Before this existed nothing ever
        removed one: ``promote`` retires the *pointer* into a ``retired-*``
        marker (a hardlink of the symlink, a few KB) but left the superseded
        ``gen-*`` directory in place forever, and ``discard_if_inactive`` only
        disposes of candidates that were never promoted.  A live archive had
        accumulated nine dead generations, ~290 GB (polylogue-wmft).

        Retention is expressed in generations rather than bytes or age because
        the reason to keep one is rollback: ``keep=1`` leaves exactly the
        previous index reachable if a promotion turns out to be bad.

        Fails closed in every ambiguous case -- anything that is or might be
        the active target, anything mid-promotion, and anything whose metadata
        cannot be read is retained, never deleted.
        """
        if keep < 0:
            raise ValueError("keep must be non-negative")
        try:
            active_target = self.active_pointer.resolve(strict=True)
        except OSError:
            # No resolvable active index: refuse to delete anything, since the
            # thing that would tell us what is live is exactly what is missing.
            return []

        candidates: list[tuple[int, str, Path]] = []
        for metadata_path in sorted(self.generations_root.glob("gen-*/generation.json")):
            try:
                generation = IndexGeneration(**json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue  # unreadable metadata: retain
            # ONLY previously-promoted generations are superseded history.
            # `state == "inactive"` means never promoted -- which is exactly
            # what an in-flight or paused resumable rebuild candidate looks
            # like (see `create_transaction`). Treating those as prunable let
            # an unrelated promotion delete a rebuild in progress, and let a
            # newer inactive candidate consume the single retained slot so the
            # real rollback target went instead. Never-promoted candidates
            # belong to `discard_if_inactive`, which their owner drives.
            if generation.state != "active":
                continue
            try:
                if Path(generation.index_path).resolve(strict=True) == active_target:
                    continue
            except OSError:
                # index.db already gone; the directory is still reclaimable.
                pass
            candidates.append((generation.created_at_ms, generation.generation_id, metadata_path.parent))

        # generation_id breaks ties: two generations can share a millisecond,
        # and a stable sort would otherwise fall back to glob order, making
        # "newest" non-deterministic and the retained slot arbitrary.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        removed: list[str] = []
        for _created_at_ms, generation_id, directory in candidates[keep:]:
            shutil.rmtree(directory)
            removed.append(generation_id)

        # The retired-* markers only point at superseded generations, so they
        # follow the same retention -- otherwise they accumulate as dangling
        # links to directories this method just removed.
        markers = sorted(
            (path for path in self.generations_root.glob("retired-*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        pruned_markers = 0
        for marker in markers[keep:]:
            shutil.rmtree(marker)
            pruned_markers += 1

        if removed or pruned_markers:
            # Markers count toward the fsync gate too: an archive's first
            # marker is pruned a promotion before any gen-* becomes prunable,
            # so gating on `removed` alone skipped the durability barrier
            # exactly when only markers had gone.
            _fsync_directory(self.generations_root)
        if removed:
            logger.info("pruned %d superseded index generation(s): %s", len(removed), ", ".join(removed))
        return removed

    def recover_promotion(self, generation_id: str) -> IndexGeneration:
        """Reconcile a crash after the pointer swap but before active metadata."""
        generation = self.load(generation_id)
        if generation.state != "promoting":
            return generation
        pointer = self.active_pointer
        state = "inactive"
        if pointer.exists() or pointer.is_symlink():
            state = (
                "active"
                if pointer.resolve(strict=True) == Path(generation.index_path).resolve(strict=True)
                else "inactive"
            )
        recovered = IndexGeneration(**{**asdict(generation), "state": state})
        self._write(recovered)
        return recovered

    def discard_if_inactive(self, generation: IndexGeneration) -> bool:
        """Remove a terminal failed candidate without risking an active target."""
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            return False
        shutil.rmtree(self._metadata_path(generation.generation_id).parent)
        _fsync_directory(self.generations_root)
        return True

    def _metadata_path(self, generation_id: str) -> Path:
        return self.generations_root / generation_id / "generation.json"

    def _transaction_path(self, operation_id: str) -> Path:
        return self.transactions_root / f"{operation_id}.json"

    def _write(self, generation: IndexGeneration) -> None:
        path = self._metadata_path(generation.generation_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(generation), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        _fsync_directory(path.parent)


def source_revision_snapshot(archive_root: Path) -> str:
    """Hash the full mutable raw-session state after a rebuild replay."""
    import hashlib

    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id"):
            for value in row:
                encoded = value.hex() if isinstance(value, bytes) else str(value)
                digest.update(encoded.encode())
                digest.update(b"\0")
            digest.update(b"\n")
    return digest.hexdigest()


def rebuild_source_evidence_snapshot(archive_root: Path) -> str:
    """Hash the immutable source evidence a rebuild is allowed to replay.

    Parse, validation, and revision-governance state are rebuild outputs or
    post-acquisition interpretation. They can legitimately change while the
    rebuild runs, so they must never invalidate its before/after source proof.
    The selected columns capture durable raw identity, membership, and bytes in
    deterministic raw-id order.
    """
    import hashlib

    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            """
            SELECT raw_id, origin, capture_mode, native_id, source_path,
                   source_index, blob_hash, blob_size, acquired_at_ms,
                   file_mtime_ms
            FROM raw_sessions
            ORDER BY raw_id
            """
        )
        for row in rows:
            for value in row:
                if value is None:
                    encoded = b"n"
                elif isinstance(value, bytes):
                    encoded = b"b" + value
                elif isinstance(value, str):
                    encoded = b"s" + value.encode()
                else:
                    encoded = b"i" + str(value).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return digest.hexdigest()


def _checkpoint_truncate(path: Path, *, label: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0:
        raise RuntimeError(f"{label} WAL checkpoint failed: {checkpoint!r}")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "ActiveWriterLease",
    "IndexGeneration",
    "IndexRebuildTransaction",
    "IndexGenerationStore",
    "RebuildLeaseStatus",
    "RebuildRawPage",
    "RebuildLease",
    "RebuildLeaseUnavailableError",
    "rebuild_lease_status",
    "rebuild_source_evidence_snapshot",
    "source_revision_snapshot",
]
