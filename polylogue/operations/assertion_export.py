"""Runtime-owned immutable assertion export images, sorted once on their User snapshot."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from polylogue.operations.mutation_transaction import AuthorizationMismatchError, MutationPrincipal

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(slots=True)
class _ExportImage:
    directory: tempfile.TemporaryDirectory[str]
    principal: tuple[str, str]
    scope: tuple[tuple[str, ...] | None, tuple[str, ...] | None, int | None]
    epoch: str
    count: int
    references: int = 0
    retired_epoch: str | None = None

    @property
    def path(self) -> Path:
        return Path(self.directory.name) / "rows.db"


def assertion_export_epoch(archive: ArchiveStore) -> str:
    """Read assertion revision on the same attached User snapshot as the export."""
    from polylogue.storage.archive_identity import TierFileIdentity

    archive.require_attached_user_tier()
    epoch = archive._conn.execute("SELECT epoch FROM user_tier.query_unit_frame_state WHERE singleton = 1").fetchone()
    if epoch is None:
        raise ValueError("original User assertion revision is unavailable")
    identity = TierFileIdentity.resolve("user", archive.user_db_path)
    return f"assertions:{identity.stable_id}:{int(epoch[0])}"


class AssertionExportImages:
    """Share immutable bytes, with independently releasable client references.

    Equivalent starts share one disk relation. Lost releases retain scalar
    handles, not another complete image. Observing a newer assertion revision
    retires older bytes after active page reads settle under the owner lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._references: dict[str, _ExportImage] = {}
        self._images: dict[
            tuple[tuple[str, str], tuple[tuple[str, ...] | None, tuple[str, ...] | None, int | None], str],
            _ExportImage,
        ] = {}
        self._frames: dict[str, tuple[int, str]] = {}
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            images = tuple(self._images.values())
            self._images.clear()
            self._references.clear()
            self._frames.clear()
            for image in images:
                image.directory.cleanup()

    def release(self, reference: str, principal: MutationPrincipal) -> bool:
        with self._lock:
            image = self._references.get(reference)
            if image is None:
                return False
            self._require_principal(image, principal)
            del self._references[reference]
            image.references -= 1
            if not image.references and image.retired_epoch is None:
                del self._images[(image.principal, image.scope, image.epoch)]
                image.directory.cleanup()
            return True

    def _observe_frame(self, epoch: str) -> None:
        from polylogue.archive.query.transaction import QueryContinuationStaleError

        identity, revision_text = epoch.rsplit(":", 1)
        revision = int(revision_text)
        latest = self._frames.get(identity)
        if latest is not None and revision < latest[0]:
            # An older pinned reader must not recreate bytes already proven
            # obsolete by a newer snapshot of the same physical User tier.
            raise QueryContinuationStaleError(issued_epoch=epoch, current_epoch=latest[1])
        if latest is None or revision > latest[0]:
            self._frames[identity] = (revision, epoch)
            for key, image in tuple(self._images.items()):
                image_identity, image_revision = image.epoch.rsplit(":", 1)
                if image_identity == identity and int(image_revision) < revision:
                    image.directory.cleanup()
                    image.retired_epoch = epoch
                    del self._images[key]

    @staticmethod
    def _require_principal(image: _ExportImage, principal: MutationPrincipal) -> None:
        if image.principal != (principal.actor_ref, principal.surface):
            raise AuthorizationMismatchError("assertion export does not belong to the authenticated principal")

    @staticmethod
    def _scope(payload: Mapping[str, object]) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, int | None]:
        kinds = cast(list[str] | None, payload.get("kinds"))
        statuses = cast(list[str] | None, payload.get("statuses"))
        return (
            None if kinds is None else tuple(sorted(set(kinds))),
            None if statuses is None else tuple(sorted(set(statuses))),
            cast(int | None, payload.get("limit")),
        )

    def _prepare(
        self,
        archive: ArchiveStore,
        scope: tuple[tuple[str, ...] | None, tuple[str, ...] | None, int | None],
        principal: MutationPrincipal,
        checkpoint: Callable[[], None],
        epoch: str,
    ) -> str:
        from polylogue.storage.sqlite.archive_tiers.user_write import (
            assertion_envelope_to_payload,
            iter_assertions_for_export,
        )
        from polylogue.storage.sqlite.connection_profile import readonly_temp_staging

        key = ((principal.actor_ref, principal.surface), scope, epoch)
        image = self._images.get(key)
        if image is not None:
            reference = f"assertion-export:{uuid4().hex}"
            image.references += 1
            self._references[reference] = image
            return reference
        directory = tempfile.TemporaryDirectory(prefix="polylogue-assertion-export-")
        count = 0
        try:
            with (
                readonly_temp_staging(archive._conn, temp_store="FILE"),
                closing(sqlite3.connect(Path(directory.name) / "rows.db")) as rows,
            ):
                rows.execute("CREATE TABLE export_rows (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL) STRICT")
                kinds, statuses, limit = scope
                with closing(
                    iter_assertions_for_export(
                        archive._conn, kinds=kinds, statuses=statuses, limit=limit, schema="user_tier"
                    )
                ) as selected:
                    for row in selected:
                        checkpoint()
                        rows.execute(
                            "INSERT INTO export_rows VALUES (?, ?)",
                            (
                                count,
                                json.dumps(
                                    assertion_envelope_to_payload(row), ensure_ascii=False, separators=(",", ":")
                                ),
                            ),
                        )
                        count += 1
                rows.commit()
            checkpoint()
            reference = f"assertion-export:{uuid4().hex}"
            image = _ExportImage(directory, (principal.actor_ref, principal.surface), scope, epoch, count, references=1)
            self._images[key] = image
            self._references[reference] = image
            return reference
        except BaseException:
            directory.cleanup()
            raise

    def page(
        self,
        payload: Mapping[str, object],
        *,
        archive: ArchiveStore,
        principal: MutationPrincipal,
        checkpoint: Callable[[], None],
    ) -> dict[str, object]:
        from polylogue.archive.query.transaction import QueryContinuationStaleError
        from polylogue.surfaces.outcome import decide_outcome

        archive.require_attached_user_tier()
        checkpoint()
        scope = self._scope(payload)
        reference = payload.get("selection_ref")
        offset = int(cast(int, payload.get("offset", 0)))
        with self._lock:
            if self._closed:
                raise InterruptedError("assertion export owner is closed")
            checkpoint()
            current = assertion_export_epoch(archive)
            self._observe_frame(current)
            if reference is None:
                if offset:
                    raise ValueError("continued assertion export requires its owned selection")
                reference = self._prepare(archive, scope, principal, checkpoint, current)
            if not isinstance(reference, str):
                raise ValueError("invalid assertion export selection")
            image = self._references.get(reference)
            if image is None:
                raise ValueError("assertion export selection is unavailable")
            self._require_principal(image, principal)
            if scope != image.scope:
                raise ValueError("assertion export selection parameters changed")
            try:
                if image.retired_epoch is not None or current != image.epoch:
                    raise QueryContinuationStaleError(
                        issued_epoch=image.epoch, current_epoch=image.retired_epoch or current
                    )
                checkpoint()
                size = int(cast(int, payload.get("page_size", 256)))
                start = min(offset, image.count)
                end = min(start + size, image.count)
                with closing(sqlite3.connect(image.path)) as rows:
                    items = []
                    for (value,) in rows.execute(
                        "SELECT payload FROM export_rows WHERE ordinal >= ? AND ordinal < ? ORDER BY ordinal",
                        (start, end),
                    ):
                        checkpoint()
                        items.append(json.loads(value))
                next_offset = offset + len(items)
                result = {
                    "items": items,
                    "total": image.count,
                    "offset": offset,
                    "next_offset": next_offset if next_offset < image.count else None,
                    "snapshot_epoch": image.epoch,
                    "selection_ref": reference,
                    "outcome": decide_outcome(matched=image.count).to_dict(),
                }
                return result
            except BaseException:
                self.release(reference, principal)
                raise
