"""Read-only conservation of the blob namespace and durable references."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import click

from polylogue.daemon.backup import _source_blob_reservations, _source_recoverability_proofs
from polylogue.storage.blob_integrity import project_source_blob_liveness
from polylogue.storage.blob_store import BlobNamespaceEntryKind, BlobNamespaceIssue, BlobStore


@dataclass(frozen=True, slots=True)
class BlobConservationReport:
    """A complete, non-mutating comparison of files and authoritative refs."""

    archive_root: str
    referenced_blobs: int
    present_blobs: int
    orphan_blobs: int
    dangling_references: int
    recoverable_references: int
    reserved_blobs: int
    invalid_namespace_entries: int
    staged_in_flight: int
    orphan_sample: tuple[str, ...] = ()
    dangling_sample: tuple[str, ...] = ()
    recoverable_sample: tuple[str, ...] = ()
    invalid_sample: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.orphan_blobs == 0 and self.dangling_references == 0 and self.invalid_namespace_entries == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "archive_root": self.archive_root,
            "referenced_blobs": self.referenced_blobs,
            "present_blobs": self.present_blobs,
            "orphan_blobs": self.orphan_blobs,
            "dangling_references": self.dangling_references,
            "recoverable_references": self.recoverable_references,
            "reserved_blobs": self.reserved_blobs,
            "invalid_namespace_entries": self.invalid_namespace_entries,
            "staged_in_flight": self.staged_in_flight,
            "orphan_sample": list(self.orphan_sample),
            "dangling_sample": list(self.dangling_sample),
            "recoverable_sample": list(self.recoverable_sample),
            "invalid_sample": list(self.invalid_sample),
        }


def check_blob_conservation(archive_root: Path, *, sample_size: int = 20) -> BlobConservationReport:
    """Compare canonical files with the descriptor-owned live projection.

    Both SQLite tiers are opened immutable/read-only. The backup module's
    existing source replay prover is the only recoverability implementation.
    Staging entries are reported separately and are never treated as blobs.
    """
    root = archive_root.resolve()
    source_db = root / "source.db"
    index_db = root / "index.db"
    store = BlobStore(root / "blob")
    projection = project_source_blob_liveness(source_db, index_db=index_db, immutable=True)
    if projection.blockers:
        raise RuntimeError("blob conservation projection blocked: " + "; ".join(projection.blockers))

    entries = tuple(store.iter_namespace())
    present = {entry.hash_hex for entry in entries if entry.kind is BlobNamespaceEntryKind.BLOB and entry.hash_hex}
    referenced = set(projection.live_hashes)
    reservations = _source_blob_reservations(source_db)
    protected = referenced | reservations
    missing = protected - present
    source_hashes = set().union(
        *(set(hashes) for owner, hashes in projection.owner_hashes if owner.startswith("source.db."))
    )
    unproven: list[dict[str, str]] = []
    proofs = _source_recoverability_proofs(
        source_db,
        root=root,
        missing_hashes=missing & source_hashes,
        unproven=unproven,
    )
    recoverable = {str(row["blob_hash"]) for row in proofs}
    invalid = tuple(
        entry.relative_path
        for entry in entries
        if entry.kind is not BlobNamespaceEntryKind.BLOB and entry.issue is not BlobNamespaceIssue.STAGED_WORK_FILE
    )
    staged = tuple(entry.relative_path for entry in entries if entry.issue is BlobNamespaceIssue.STAGED_WORK_FILE)
    limit = max(0, sample_size)
    return BlobConservationReport(
        archive_root=str(root),
        referenced_blobs=len(referenced),
        present_blobs=len(present),
        orphan_blobs=len(present - protected),
        dangling_references=len(missing - recoverable),
        recoverable_references=len(recoverable),
        reserved_blobs=len(reservations),
        invalid_namespace_entries=len(invalid),
        staged_in_flight=len(staged),
        orphan_sample=tuple(sorted(present - protected)[:limit]),
        dangling_sample=tuple(sorted(missing - recoverable)[:limit]),
        recoverable_sample=tuple(sorted(recoverable)[:limit]),
        invalid_sample=tuple(sorted(invalid)[:limit]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify two-direction blob/reference conservation without mutation.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = check_blob_conservation(args.archive_root, sample_size=args.sample_size)
    if args.as_json:
        print(json.dumps(report.to_dict(), sort_keys=True))
    else:
        print(f"Blob conservation: {'PASS' if report.ok else 'FAIL'}")
        print(f"Referenced: {report.referenced_blobs:,}; present: {report.present_blobs:,}")
        print(
            f"Orphans: {report.orphan_blobs:,}; dangling: {report.dangling_references:,}; recoverable: {report.recoverable_references:,}; reserved: {report.reserved_blobs:,}"
        )
        print(f"Invalid namespace: {report.invalid_namespace_entries:,}; staged in-flight: {report.staged_in_flight:,}")
    return 0 if report.ok else 1


@click.command("blob-conservation")
@click.option("--archive-root", type=click.Path(path_type=Path), required=True)
@click.option("--sample-size", type=int, default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def blob_conservation_command(archive_root: Path, sample_size: int, as_json: bool) -> None:
    """Run the read-only blob conservation check from the operations CLI."""
    args = ["--archive-root", str(archive_root), "--sample-size", str(sample_size)]
    if as_json:
        args.append("--json")
    raise click.exceptions.Exit(main(args))


__all__ = ["BlobConservationReport", "blob_conservation_command", "check_blob_conservation", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
