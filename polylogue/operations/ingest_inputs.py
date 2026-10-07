"""Freeze and enumerate machine-ingest inputs on the admitted compute lane."""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.sources.import_preflight import ImportPreflightResult

from polylogue.pipeline.services.acquisition_records import make_raw_record, pending_pre_parse_raw_admission_request
from polylogue.sources.retained_acquisition import iter_retained_source_records
from polylogue.sources.source_staging import (
    SourceInputBinding,
    bind_source_input,
    bind_staged_member,
    read_staging_receipt,
    write_bound_input,
)
from polylogue.sources.sqlite_export import SourceBytePage, source_byte_page
from polylogue.sources.sqlite_snapshot import is_sqlite_path, snapshot_sqlite_to_blob
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    RawAdmissionPlan,
    plan_raw_admission,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    CapturedSourceInputIdentity,
    FrozenSourceInput,
    RetainedSourceInput,
    SourceItemAdmission,
    source_item_id,
)
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    _close_failed_native_construction,
    open_scratch_connection,
    readonly_connection_context,
    retained_native_sql_owners_for_lifetime,
)


@contextmanager
def spool_connection(path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """Own one private spool transaction and its exact artifact through settlement."""
    if read_only:
        with readonly_connection_context(path, validate_schema=False, lifetime_dependencies=(path,)) as conn:
            yield conn
        return
    owner = open_scratch_connection(path, lifetime_dependencies=(path,))
    try:
        conn = owner.require_connection()
        conn.execute("PRAGMA journal_mode=DELETE").close()
        conn.execute("PRAGMA temp_store=FILE").close()
        with conn:
            yield conn
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


def unlink_spool(path: Path) -> None:
    """Remove only an operation's private artifact after its native owners retire."""
    retained = retained_native_sql_owners_for_lifetime(path)
    if retained:
        raise NativeConnectionSettlementError(retained[0], RuntimeError("private spool remains under native custody"))
    path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class PreparedSourceRecord:
    record: RawSessionRecord
    admission: RawAdmissionPlan
    member: SourceItemAdmission
    member_count: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedSourceMemberDisposition:
    source_generation_id: str
    source_item_id: str
    entry_ordinal: int
    member_name: str
    disposition: str
    diagnostic: str
    member_count: int


def discover_ingest_input_spool(path: Path, *, source_path: str | None, check_stop: Callable[[], None]) -> Path:
    """Sort the physical denominator on disk before retaining any input.

    Each row pairs the physical file with the logical path its raws are keyed
    on. ``source_path`` names the caller's original input: a file's logical
    path is ``source_path`` itself, and a directory member's is its relative
    path under ``source_path``, so a staged copy is keyed where the caller's
    material lives rather than where it was staged. Without ``source_path``
    the physical path is the logical one.
    """
    fd, name = tempfile.mkstemp(prefix="polylogue-ingest-paths-", suffix=".sqlite", dir=os.environ.get("TMPDIR"))
    os.close(fd)
    spool = Path(name)
    try:
        with spool_connection(spool) as conn:
            conn.execute(
                "CREATE TABLE paths(coordinate TEXT PRIMARY KEY, physical TEXT NOT NULL, logical TEXT NOT NULL, "
                "captured_member TEXT) "
                "WITHOUT ROWID"
            )
            conn.execute("CREATE TABLE receipt(body TEXT NOT NULL)")

            def spool_member(member: dict[str, object]) -> None:
                check_stop()
                captured = CapturedSourceInputIdentity.from_dict(member["identity"])
                conn.execute(
                    "INSERT INTO paths VALUES (?, ?, ?, ?)",
                    (
                        member["coordinate"],
                        str(path / str(member["relative_path"])),
                        captured.semantic_source_path,
                        json.dumps(member, ensure_ascii=True),
                    ),
                )

            receipt = read_staging_receipt(path, on_member=spool_member, check_stop=check_stop)
            if receipt is not None:
                if source_path is not None and source_path != receipt["header"]["source_path"]:
                    raise ValueError("ingest declaration differs from its captured staging receipt")
                conn.execute("INSERT INTO receipt VALUES (?)", (json.dumps(receipt, ensure_ascii=True),))
                return spool
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                conn.execute(
                    "INSERT INTO paths VALUES (?, ?, ?, NULL)", ("input:0", str(path), source_path or str(path))
                )
            elif stat.S_ISDIR(mode):
                conn.execute(
                    "CREATE TABLE directories(relative TEXT PRIMARY KEY, device INTEGER NOT NULL, inode INTEGER NOT NULL) "
                    "WITHOUT ROWID"
                )
                root_info = path.lstat()
                conn.execute("INSERT INTO directories VALUES ('', ?, ?)", (root_info.st_dev, root_info.st_ino))
                while True:
                    check_stop()
                    with closing(
                        conn.execute("SELECT relative, device, inode FROM directories ORDER BY relative LIMIT 1")
                    ) as pending:
                        directory = pending.fetchone()
                    if directory is None:
                        break
                    relative_directory, device, inode = directory
                    current_directory = path / relative_directory
                    descriptor = os.open(current_directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        opened = os.fstat(descriptor)
                        if (opened.st_dev, opened.st_ino) != (device, inode):
                            raise OSError(
                                errno.ESTALE, "ingest directory changed during enumeration", str(current_directory)
                            )
                        with os.scandir(descriptor) as entries:
                            for entry in entries:
                                check_stop()
                                relative = Path(relative_directory) / entry.name
                                candidate = path / relative
                                info = entry.stat(follow_symlinks=False)
                                if stat.S_ISDIR(info.st_mode):
                                    conn.execute(
                                        "INSERT INTO directories VALUES (?, ?, ?)",
                                        (str(relative), info.st_dev, info.st_ino),
                                    )
                                elif stat.S_ISREG(info.st_mode):
                                    logical = (
                                        str(Path(source_path) / relative) if source_path is not None else str(candidate)
                                    )
                                    conn.execute(
                                        "INSERT INTO paths VALUES (?, ?, ?, NULL)",
                                        (str(relative), str(candidate), logical),
                                    )
                                else:
                                    raise ValueError("ingest inputs must be regular files, not links or special files")
                    finally:
                        os.close(descriptor)
                    conn.execute("DELETE FROM directories WHERE relative=?", (relative_directory,))
            else:
                raise ValueError("ingest input must be a regular file or directory")
            if conn.execute("SELECT 1 FROM paths LIMIT 1").fetchone() is None:
                raise ValueError("ingest input contains no physical files")
        return spool
    except BaseException:
        unlink_spool(spool)
        raise


@contextmanager
def _bind_ingest_input_row(
    row: tuple[str, str, str, str | None],
    receipt: dict[str, object] | None,
    *,
    byte_page: SourceBytePage,
) -> Iterator[SourceInputBinding]:
    _coordinate, physical_name, logical_path, captured_member = row
    physical = Path(physical_name)
    with ExitStack() as owners:
        if captured_member is not None:
            if receipt is None:
                raise ValueError("staged member lacks its authenticated directory receipt")
            member = json.loads(captured_member)
            relative = Path(member["relative_path"])
            staged_root = physical.parents[len(relative.parts) - 1]
            binding = owners.enter_context(bind_staged_member(staged_root, member, receipt))
        else:
            # The page's live reader proves the binding; a fresh worker
            # process per input made a 10,000-input ingest spawn 10,000.
            binding = owners.enter_context(bind_source_input(physical, byte_page=byte_page))
        if logical_path != str(physical) and not binding.staged:
            raise ValueError("staged input lacks its captured original source identity")
        yield binding


def preflight_ingest_input(
    path: Path, *, check_stop: Callable[[], None]
) -> tuple[ImportPreflightResult, dict[str, object] | None]:
    """Classify the same authenticated, paged denominator retention consumes."""
    from polylogue.sources.import_preflight import preflight_import_bindings

    spool = discover_ingest_input_spool(path, source_path=None, check_stop=check_stop)
    try:
        with spool_connection(spool, read_only=True) as conn:
            receipt_row = conn.execute("SELECT body FROM receipt").fetchone()
            receipt = None if receipt_row is None else json.loads(receipt_row[0])
            first_input = conn.execute("SELECT coordinate, physical FROM paths ORDER BY coordinate LIMIT 1").fetchone()
        declaration = None if receipt is None else receipt["header"]
        single_file = (
            declaration["input_kind"] == "file" if declaration is not None else first_input == ("input:0", str(path))
        )
        source_path = str(path) if declaration is None else declaration["source_path"]

        def members() -> Generator[tuple[SourceInputBinding, str], None, None]:
            after = ""
            while True:
                with spool_connection(spool, read_only=True) as conn:
                    rows = conn.execute(
                        "SELECT coordinate, physical, logical, captured_member FROM paths "
                        "WHERE coordinate > ? ORDER BY coordinate LIMIT 256",
                        (after,),
                    ).fetchall()
                if not rows:
                    return
                with source_byte_page() as reader:
                    for row in rows:
                        check_stop()
                        with _bind_ingest_input_row(row, receipt, byte_page=reader) as binding:
                            yield binding, binding.source_path.name if row[0] == "input:0" else row[0]
                        after = row[0]

        with closing(members()) as bound_members:
            result = preflight_import_bindings(
                bound_members,
                source_path=source_path,
                single_file=single_file,
                check_stop=check_stop,
            )
        return result, declaration
    finally:
        unlink_spool(spool)


def retain_input_page(
    spool: Path,
    *,
    after_coordinate: str | None,
    publisher: ArchiveBlobPublisher,
    check_stop: Callable[[], None],
) -> tuple[FrozenSourceInput, ...]:
    """One compute-phase page; its SQLite connection never crosses threads."""
    with spool_connection(spool, read_only=True) as conn:
        rows = conn.execute(
            "SELECT coordinate, physical, logical, captured_member FROM paths WHERE coordinate > ? ORDER BY coordinate LIMIT 256",
            (after_coordinate or "",),
        ).fetchall()
        receipt_row = conn.execute("SELECT body FROM receipt").fetchone()
        receipt = None if receipt_row is None else json.loads(receipt_row[0])
    batch: list[FrozenSourceInput] = []
    with source_byte_page() as reader:
        for coordinate, physical_name, logical_path, captured_member in rows:
            check_stop()
            physical = Path(physical_name)
            with _bind_ingest_input_row(
                (coordinate, physical_name, logical_path, captured_member), receipt, byte_page=reader
            ) as binding:
                captured_identity = binding.captured_identity
                if is_sqlite_path(binding.source_path):
                    blob_hash = snapshot_sqlite_to_blob(
                        physical, publisher, heartbeat=check_stop, source_binding=binding
                    ).blob_hash
                else:
                    accepted_revision: dict[str, object] = {}

                    def retain_bytes(
                        destination: IO[bytes],
                        *,
                        revision: dict[str, object] = accepted_revision,
                        accepted: SourceInputBinding = binding,
                    ) -> None:
                        revision.update(write_bound_input(accepted, destination, reader=reader))

                    blob_hash, blob_size = publisher.write_from_writer(retain_bytes, heartbeat=check_stop)
                    if (accepted_revision["content_revision"], accepted_revision["size_bytes"]) != (
                        blob_hash,
                        blob_size,
                    ):
                        raise ValueError("retained input differs from its proved descriptor bytes")
            publication_id = publisher.receipt_id(blob_hash)
            if publication_id is None:
                raise RuntimeError("retained input has no publication reservation identity")
            batch.append(
                FrozenSourceInput(
                    str(coordinate),
                    captured_identity.semantic_source_path,
                    blob_hash,
                    publication_id,
                    captured_identity,
                )
            )
    return tuple(batch)


def enumerate_ingest_input(
    item: FrozenSourceInput | RetainedSourceInput,
    *,
    source_generation_id: str,
    enumeration_fingerprint: str,
    publisher: ArchiveBlobPublisher,
    acquired_at_ms: int,
    check_stop: Callable[[], None],
    source_name: str | None = None,
) -> Generator[PreparedSourceRecord | PreparedSourceMemberDisposition, None, None]:
    """Yield canonical admission plans; only normal exhaustion closes the item."""
    item_id = source_item_id(
        source_generation_id=source_generation_id,
        logical_coordinate=item.coordinate,
        addressing_mode="physical-file-v1",
    )
    acquired_at = datetime.fromtimestamp(acquired_at_ms / 1000, UTC).isoformat()
    blob_size = publisher.blob_path(item.blob_hash).stat().st_size
    for retained in iter_retained_source_records(
        enumeration_fingerprint=enumeration_fingerprint,
        source_path=item.source_path,
        blob_hash=item.blob_hash,
        blob_size=blob_size,
        blob_store=publisher,
        source_name=source_name,
        captured_identity=item.captured_identity,
        on_member_disposition=lambda *_fields: None,
    ):
        check_stop()
        if retained.member_disposition is not None:
            if retained.entry_ordinal is None or retained.member_name is None or retained.diagnostic is None:
                raise ValueError("retained member disposition lacks its exact central-directory identity")
            yield PreparedSourceMemberDisposition(
                source_generation_id,
                item_id,
                retained.entry_ordinal,
                retained.member_name,
                retained.member_disposition,
                retained.diagnostic,
                retained.member_count or 0,
            )
            continue
        if retained.data is None:
            raise ValueError("retained source record lacks its raw payload")
        record = make_raw_record(
            retained.data,
            "machine-ingest",
            blob_root=publisher.root,
            blob_store=publisher,
            acquired_at=acquired_at,
        )
        if retained.raw_id is not None:
            record = record.model_copy(
                update={"raw_id": retained.raw_id, "addressing_mode": retained.data.addressing_mode}
            )
        plan = plan_raw_admission(pending_pre_parse_raw_admission_request(record))
        yield PreparedSourceRecord(
            record,
            plan,
            SourceItemAdmission(
                source_generation_id,
                item_id,
                retained.coordinate,
                retained.entry_ordinal,
                retained.split_index,
                retained.data.addressing_mode.value if retained.data.addressing_mode is not None else None,
                retained.data.content_identity,
            ),
            retained.member_count,
        )
    check_stop()
