"""Materialize hook events out of retained carrier bytes.

One acquired hook-event carrier is one key. The carrier's bytes are already
durable -- the ordinary file route acquired them once, as an artifact, exactly
like any other append-only source -- so this domain never reads the
filesystem's live spool, never publishes a blob, and never acknowledges
anything. It decodes the retained bytes and writes the rows they imply.

That is the whole point of the carrier redesign (polylogue-k3ahm). The retired
route persisted one event at capture time, one commit and four fsyncs each,
and a 690,297-event backlog projected to roughly 13 hours of serial writer
time (polylogue-xa33k). Here the durable cost is paid once per carrier
revision by acquisition, and materialization is one transaction per carrier.

The rows this publishes are a derivation output: they are recomputable from
the carrier bytes at any time, and republishing them is idempotent because
every identity -- the event id, the carrier coordinate -- is content-derived
rather than ordinal.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import WARNING, emit
from polylogue.sources.hooks import (
    CarrierLine,
    HookSpoolRecordError,
    carrier_hook_events,
    hook_carrier_dir,
    hook_spool_sources,
    read_hook_carrier,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_write import (
    CarrierHookEvent,
    hook_carrier_coordinate,
)

HOOK_EVENTS_DOMAIN = "hook_events"

#: The artifact kind acquisition stamps on a carrier. The key space is exactly
#: the raws carrying it, so a carrier that was never classified as one is not
#: silently materialized as hook evidence.
HOOK_EVENT_CARRIER_ARTIFACT_KIND = "hook_event_carrier"


class HookCarrierTopologyError(ValueError):
    """An acquired carrier does not sit under any declared hook root."""


@dataclass(frozen=True, slots=True)
class HookEventsScope:
    """Restrict a pass to exactly these carrier raw ids."""

    raw_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HookCarrierIdentity:
    """Where one acquired carrier sits in the declared hook topology."""

    source_id: str
    relative_path: str
    role: str


@dataclass(frozen=True, slots=True)
class HookEventsReplacement:
    """Every event one carrier revision materializes into."""

    key: str
    input_binding: str
    payload: tuple[CarrierHookEvent, ...]
    identity: HookCarrierIdentity
    blob_hash: str
    source_path: str
    acquired_at_ms: int
    empty: bool = False


def carrier_identity(
    source_path: str,
    *,
    specs: Sequence[tuple[str, Path, str]] | None = None,
) -> HookCarrierIdentity:
    """Resolve a carrier's durable coordinate from the declared topology.

    The coordinate is root-relative and provider-qualified
    (``<provider>/<day>/<pid>.ndjson``), so it stays stable if the archive
    root moves and it cannot collide across declared roots. A carrier under no
    declared root is refused rather than attributed to a guessed source: the
    source id is what excision and blob disposition reach hook evidence by.
    """

    declared = (
        specs if specs is not None else tuple((spec.source_id, spec.root, spec.role) for spec in hook_spool_sources())
    )
    candidate = Path(source_path).resolve()
    for source_id, root, role in declared:
        carriers = hook_carrier_dir(root).resolve()
        try:
            relative = candidate.relative_to(carriers)
        except ValueError:
            continue
        return HookCarrierIdentity(source_id, relative.as_posix(), role)
    raise HookCarrierTopologyError(f"hook carrier sits under no declared hook root: {source_path}")


class HookEventsDerivation:
    """The ``hook_events`` domain: one acquired carrier in, its events out."""

    domain = HOOK_EVENTS_DOMAIN
    prerequisites: tuple[str, ...] = ()
    #: Bump when the materialized rows change shape. The identity of an event
    #: and of its carrier coordinate are both content-derived, so a bump
    #: re-publishes over the same keys rather than duplicating them.
    recipe_version = "hook-events-carrier-v1"

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root
        self._decoded: dict[str, tuple[tuple[CarrierLine, ...], int]] = {}

    # -- reading ----------------------------------------------------------

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        source = self.archive_root / "source.db"
        conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _current(self, frame: object) -> bool:
        archive_root = getattr(frame, "archive_root", None)
        recipe = getattr(frame, "recipe_version", None)
        return (
            archive_root is not None
            and Path(str(archive_root)).resolve() == self.archive_root.resolve()
            and recipe is not None
            and recipe(self.domain) == self.recipe_version
        )

    def _carrier_lines(self, blob_hash: str) -> tuple[tuple[CarrierLine, ...], int]:
        """Decode one carrier's retained bytes, memoized on its blob identity.

        A carrier revision's bytes are immutable once acquired, so the decode
        is a pure function of the blob hash. Memoizing it is what keeps
        inspection cheap: the kernel inspects a key before computing it and
        again to certify publication, and a bare re-decode would triple the
        parse cost of every pass over an already-materialized carrier.
        """

        cached = self._decoded.get(blob_hash)
        if cached is not None:
            return cached
        payload = BlobStore(self.archive_root / "blob").read_all(blob_hash)
        lines, refusals = read_hook_carrier(payload)
        if refusals:
            emit(
                "source.hook_carrier.line_refused",
                level=WARNING,
                outcome="degraded",
                blob_hash=blob_hash,
                errors=len(refusals),
                error_detail="; ".join(f"@{refusal.byte_offset}: {refusal.reason}" for refusal in refusals),
            )
        decoded = (lines, len(refusals))
        self._decoded[blob_hash] = decoded
        return decoded

    def _descriptor(self, conn: sqlite3.Connection, key: str) -> tuple[str, str, int] | None:
        row = conn.execute(
            "SELECT source_path, blob_hash, acquired_at_ms FROM raw_sessions WHERE raw_id = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        blob_hash = row["blob_hash"]
        return (
            str(row["source_path"]),
            blob_hash.hex() if isinstance(blob_hash, (bytes, bytearray)) else str(blob_hash),
            int(row["acquired_at_ms"] or 0),
        )

    # -- kernel surface ---------------------------------------------------

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        if limit < 1:
            return (), None
        if not (self.archive_root / "source.db").exists():
            return (), None
        scope = getattr(frame, "scope", None)
        if isinstance(scope, HookEventsScope) and scope.raw_ids:
            return tuple(scope.raw_ids), None
        with self._read() as conn:
            if (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_artifacts'").fetchone()
                is None
            ):
                return (), None
            rows = conn.execute(
                "SELECT DISTINCT raw_id FROM raw_artifacts WHERE artifact_kind = ? AND raw_id > ? "
                "ORDER BY raw_id LIMIT ?",
                (HOOK_EVENT_CARRIER_ARTIFACT_KIND, cursor or "", limit),
            ).fetchall()
        keys = tuple(str(row["raw_id"]) for row in rows)
        return keys, (keys[-1] if len(keys) == limit else None)

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], None]:
        # Materialized hook events are durable evidence of retained bytes.
        # An event with no carrier is excised or deleted deliberately, never
        # swept by a derivation guessing at an absent owner.
        return (), None

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        return ()

    def quiet(self, frame: object, key: str) -> bool:
        return False

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        if not keys:
            return {}
        if not self._current(frame):
            return dict.fromkeys(keys, "stale")
        with self._read() as conn:
            return {key: self._inspect(conn, key) for key in keys}

    def _inspect(self, conn: sqlite3.Connection, key: str) -> str:
        descriptor = self._descriptor(conn, key)
        if descriptor is None:
            return "stale"
        source_path, blob_hash, _acquired_at_ms = descriptor
        try:
            identity = carrier_identity(source_path)
        except HookCarrierTopologyError:
            return "stale"
        try:
            lines, _refused = self._carrier_lines(blob_hash)
        except (OSError, ValueError):
            return "stale"
        if not lines:
            # A carrier whose every line is malformed has nothing to
            # materialize. Reporting it stale forever would pin the traversal
            # on a key no publication can ever resolve.
            return "valid"
        recorded = {
            str(row["relative_path"])
            for row in conn.execute(
                "SELECT relative_path FROM hook_event_carriers WHERE source_id = ? AND relative_path LIKE ?",
                (identity.source_id, f"{identity.relative_path}#%"),
            )
        }
        expected = {hook_carrier_coordinate(identity.relative_path, line.byte_offset) for line in lines}
        return "valid" if expected <= recorded else "stale"

    def _binding(self, conn: sqlite3.Connection, identity: HookCarrierIdentity, blob_hash: str) -> str:
        recorded = sorted(
            str(row["relative_path"])
            for row in conn.execute(
                "SELECT relative_path FROM hook_event_carriers WHERE source_id = ? AND relative_path LIKE ?",
                (identity.source_id, f"{identity.relative_path}#%"),
            )
        )
        return hashlib.sha256(
            repr((blob_hash, identity.source_id, identity.relative_path, recorded)).encode()
        ).hexdigest()

    def compute(self, frame: object, key: str) -> HookEventsReplacement:
        with self._read() as conn:
            descriptor = self._descriptor(conn, key)
            if descriptor is None:
                raise ValueError(f"hook carrier raw is not acquired: {key}")
            source_path, blob_hash, acquired_at_ms = descriptor
            identity = carrier_identity(source_path)
            binding = self._binding(conn, identity, blob_hash)
        if not BlobStore(self.archive_root / "blob").verify(blob_hash):
            raise ValueError(f"retained hook carrier does not match its identity: {key}")
        lines, _refused = self._carrier_lines(blob_hash)
        try:
            events = carrier_hook_events(lines, source_path=source_path)
        except HookSpoolRecordError as exc:
            raise ValueError(f"hook carrier holds an unmaterializable line: {exc}") from exc
        return HookEventsReplacement(
            key=key,
            input_binding=binding,
            payload=events,
            identity=identity,
            blob_hash=blob_hash,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            empty=not events,
        )

    def publish(self, frame: object, replacement: HookEventsReplacement) -> bool:
        from polylogue.sources.live.archive_open import _open_archive_for_live_write
        from polylogue.storage.index_generation import ActiveWriterLease

        lease = ActiveWriterLease(self.archive_root)
        lease.acquire()
        try:
            if not self._current(frame):
                return False
            with self._read() as conn:
                if self._binding(conn, replacement.identity, replacement.blob_hash) != replacement.input_binding:
                    return False
            if replacement.empty:
                return True
            store = _open_archive_for_live_write(self.archive_root)
            with store as archive:
                archive.write_hook_events_from_carrier(
                    carrier_source_id=replacement.identity.source_id,
                    carrier_relative_path=replacement.identity.relative_path,
                    carrier_role=replacement.identity.role,
                    carrier_blob_hash=bytes.fromhex(replacement.blob_hash),
                    carrier_source_path=replacement.source_path,
                    events=replacement.payload,
                    acquired_at_ms=replacement.acquired_at_ms,
                )
                archive.commit()
            return True
        finally:
            lease.close()


__all__ = [
    "HOOK_EVENTS_DOMAIN",
    "HOOK_EVENT_CARRIER_ARTIFACT_KIND",
    "HookCarrierIdentity",
    "HookCarrierTopologyError",
    "HookEventsDerivation",
    "HookEventsReplacement",
    "HookEventsScope",
    "carrier_identity",
]
