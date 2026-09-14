"""A durable witness that the index tier's blob-owning population is current.

``index.db`` is the only owner of ``attachments.blob_hash``, and it is a
*rebuildable* tier: a wipe or an index-generation swap replaces it with a
schema-current, row-empty file. Blob GC reads that file's silence as "nothing
references these bytes", so the first pass after a wipe unlinks every blob whose
only liveness surface was the index.

The index tier cannot witness its own loss -- the evidence is exactly what was
removed -- and nothing inside ``source.db`` distinguishes an index-only
attachment payload from a blob no surface ever claimed: on the live archive all
1,240 index-only attachment hashes carry no ``blob_refs`` row, no direct source
owner and no publication receipt. The witness therefore lives beside the bytes
it protects, in the blob namespace, which no tier reset touches.

Semantics, deliberately narrow:

* GC records the identity of the index tier file it read and how many distinct
  blob hashes that tier owned.
* A population that falls *within one tier file* is an ordinary deletion and
  re-anchors the witness; GC keeps collecting.
* A population that falls *across a tier-file replacement* is the wipe/rebuild
  shape. Until the replacement tier owns at least as many blobs as its
  predecessor did, GC may not read its silence as proof, and every candidate
  whose liveness only the index could decide is blocked.

An archive that has never run a GC pass against a materialized index has no
witness and is evaluated exactly as before; this file only ever removes
authority that GC itself previously observed.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.blob_liveness import index_tier_blob_population
from polylogue.storage.blob_store import INDEX_LIVENESS_WATERMARK_FILENAME

#: Lives in the blob namespace root, beside ``.polylogue-blob-namespace``.
#: ``_candidate_blobs`` only walks two-hex shard directories, so a dotfile here
#: is never mistaken for a blob. The name itself is owned by ``blob_store``,
#: which decides what may sit in that root: ``BlobStore.iter_namespace``
#: reports every unnamed entry as a critical invalid-namespace finding, so the
#: two must not be able to drift apart.
WATERMARK_FILENAME = INDEX_LIVENESS_WATERMARK_FILENAME

_FORMAT = "polylogue.index-liveness-watermark.v1"


@dataclass(frozen=True, slots=True)
class IndexLivenessWatermark:
    """One observation of the index tier's blob-owning population."""

    tier_identity: str
    blob_population: int
    observed_at_ms: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": _FORMAT,
                "tier_identity": self.tier_identity,
                "blob_population": self.blob_population,
                "observed_at_ms": self.observed_at_ms,
            },
            sort_keys=True,
        )


def index_tier_identity(index_path: Path) -> str | None:
    """Return a stable identity for the index tier file, following symlinks.

    A promoted generation is reached through a symlink, so the identity must be
    the resolved file's ``(st_dev, st_ino)``: a rebuild that swaps the pointer
    to a freshly created generation is a different file, which is precisely the
    transition this module refuses to trust.
    """

    try:
        stat = os.stat(index_path)
    except OSError:
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


def read_index_liveness_watermark(blob_root: Path) -> IndexLivenessWatermark | None:
    """Return the recorded witness, or ``None`` when there is none to trust."""

    try:
        payload = json.loads((blob_root / WATERMARK_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        return None
    identity = payload.get("tier_identity")
    population = payload.get("blob_population")
    observed_at_ms = payload.get("observed_at_ms")
    if not isinstance(identity, str) or not isinstance(population, int) or not isinstance(observed_at_ms, int):
        return None
    if isinstance(population, bool) or population < 0:
        return None
    return IndexLivenessWatermark(identity, population, observed_at_ms)


def _fsync_directory(path: Path) -> None:
    """Persist a rename in ``path`` so the witness survives a crash."""

    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _record(blob_root: Path, watermark: IndexLivenessWatermark) -> None:
    target = blob_root / WATERMARK_FILENAME
    temporary = blob_root / f"{WATERMARK_FILENAME}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(watermark.to_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except OSError:
        # The witness is an accelerator for retention, never a precondition for
        # it: failing to record leaves the previous (more conservative) witness
        # in place and never widens what GC may delete.
        with contextlib.suppress(OSError):
            temporary.unlink()


def index_liveness_authority_blocker(
    *,
    blob_root: Path,
    index_path: Path,
    index_conn: sqlite3.Connection | None,
    record: bool = True,
) -> str | None:
    """Return why the index tier cannot decide absence, or ``None``.

    ``record`` is false for dry runs, which must not leave state behind.
    """

    if index_conn is None:
        return None
    identity = index_tier_identity(index_path)
    if identity is None:
        return None
    # Unreadability is not swallowed here: the reference-tier preflight already
    # proved this tier opens, so a failure now is a real fault and must fail
    # loud rather than degrade into "the index has nothing to say".
    population = index_tier_blob_population(index_conn)
    observed = IndexLivenessWatermark(identity, population, int(time.time() * 1000))
    previous = read_index_liveness_watermark(blob_root)
    if previous is None or previous.tier_identity == identity or population >= previous.blob_population:
        if record:
            _record(blob_root, observed)
        return None
    return (
        f"index tier at {index_path} is a replacement file owning {population} blob(s) where the "
        f"previous index tier owned {previous.blob_population}; its silence cannot prove a blob is "
        "unreferenced until it is materialized again"
    )


__all__ = [
    "WATERMARK_FILENAME",
    "IndexLivenessWatermark",
    "index_liveness_authority_blocker",
    "index_tier_identity",
    "read_index_liveness_watermark",
]
