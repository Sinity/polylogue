"""Original input provenance and publication for privately staged sources."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from polylogue.core.provider_identity import captured_hermes_profile_key

if TYPE_CHECKING:
    from polylogue.sources.parsers.hermes_identity import CapturedHermesProfile
    from polylogue.sources.sqlite_export import BinaryWriteSink, SourceBytePage
    from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

_STAGING_METADATA_SUFFIX = ".polylogue-import"


def _staging_receipt_member(value: Any) -> dict[str, Any]:
    from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

    if (
        not isinstance(value, dict)
        or set(value)
        != {"type", "coordinate", "relative_path", "identity", "file_identity", "content_kind", "content_revision"}
        or value["type"] != "member"
    ):
        raise ValueError("invalid staged input member")
    relative = value["relative_path"]
    if (
        not isinstance(relative, str)
        or not relative
        or "\0" in relative
        or Path(relative).is_absolute()
        or Path(relative).as_posix() != relative
        or any(part in {".", ".."} for part in relative.split("/"))
        or not isinstance(value["coordinate"], str)
        or not value["coordinate"]
        or value["content_kind"] not in {"bytes", "sqlite"}
        or not isinstance(value["content_revision"], str)
        or len(value["content_revision"]) != 64
        or any(character not in "0123456789abcdef" for character in value["content_revision"])
        or not isinstance(value["file_identity"], list)
        or len(value["file_identity"]) != 3
        or any(type(component) is not int for component in value["file_identity"])
        or value["file_identity"][2] != stat.S_IFREG
    ):
        raise ValueError("invalid staged input member coordinate or material")
    CapturedSourceInputIdentity.from_dict(value["identity"])
    return value


def _read_staging_receipt_in_worker(request: dict[str, Any], sink: BinaryWriteSink) -> dict[str, Any]:
    """Authenticate one outside receipt while streaming its members to the spool."""
    from polylogue.sources.sqlite_export import _control_bytes, _identity, _named_identity

    directory = request["directory"]
    name = request["metadata_name"]
    expected = tuple(request["metadata_identity"])
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if _identity(before) != expected or not stat.S_ISREG(before.st_mode):
            raise OSError(errno.ESTALE, "staging receipt changed before reading", name)
        digest = hashlib.sha256()
        first = stream.readline()
        digest.update(first)
        header = json.loads(first)
        if (
            not isinstance(header, dict)
            or set(header) != {"type", "version", "input_kind", "source_path", "source_name", "root_identity"}
            or header["type"] != "header"
            or header["version"] != 3
            or header["input_kind"] not in {"file", "directory"}
            or not isinstance(header["source_path"], str)
            or not Path(header["source_path"]).is_absolute()
            or not isinstance(header["source_name"], str)
            or not header["source_name"]
            or header["root_identity"] != request["identities"][""]
        ):
            raise ValueError("invalid staged input receipt header")
        count = 0
        for line in stream:
            digest.update(line)
            value = json.loads(line)
            if isinstance(value, dict) and value.get("type") == "end":
                if set(value) != {"type", "count"} or type(value["count"]) is not int or value["count"] != count:
                    raise ValueError("invalid staged input receipt completion")
                if stream.read(1):
                    raise ValueError("staged input receipt has trailing material")
                break
            member = _staging_receipt_member(value)
            if header["input_kind"] == "file" and member["identity"]["semantic_source_path"] != header["source_path"]:
                raise ValueError("staged member differs from its captured declaration")
            if header["input_kind"] == "file":
                if count or member["coordinate"] != "input:0" or "/" in member["relative_path"]:
                    raise ValueError("invalid single-file staging receipt")
            elif member["coordinate"] != member["relative_path"]:
                raise ValueError("staged directory member coordinate differs")
            sink.write(_control_bytes(member) + b"\n")
            count += 1
        else:
            raise ValueError("staged input receipt is incomplete")
        if not count:
            raise ValueError("staged input receipt has no members")
        after = os.fstat(stream.fileno())
        observation = [before.st_ctime_ns, before.st_mtime_ns, before.st_size]
        if _identity(after) != expected or [after.st_ctime_ns, after.st_mtime_ns, after.st_size] != observation:
            raise OSError(errno.ESTALE, "staging receipt changed while reading", name)
    if _named_identity(directory, name) != expected:
        raise OSError(errno.ESTALE, "staging receipt changed before settlement", name)
    return {
        "header": header,
        "provenance": {
            "name": name,
            "identity": list(expected),
            "digest": digest.hexdigest(),
            "observation": observation,
        },
        "count": count,
    }


def read_staging_receipt(
    staged: Path, *, on_member: Callable[[dict[str, Any]], None], check_stop: Callable[[], None]
) -> dict[str, Any] | None:
    """Stream an operation's authenticated outside receipt without a parent file FD."""
    from polylogue.sources.sqlite_export import _exchange_source_worker, _identity, _named_identity

    flags = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(staged.parent, flags)
    try:
        name = staging_metadata_path(staged).name
        try:
            metadata_identity = _named_identity(directory, name)
        except FileNotFoundError:
            return None
        except OSError as exc:
            # A receipt name beyond the filesystem's name limit cannot exist:
            # receipts belong to short operation slots, never to a payload file
            # whose own valid name already uses the whole limit.
            if exc.errno != errno.ENAMETOOLONG:
                raise
            return None
        root = _identity(os.stat(staged.name, dir_fd=directory, follow_symlinks=False))
        if root[2] != stat.S_IFDIR:
            raise ValueError("staged receipt requires its private directory slot")

        class ReceiptSink:
            def __init__(self) -> None:
                self.pending = bytearray()

            def write(self, data: bytes) -> int:
                check_stop()
                self.pending.extend(data)
                while (boundary := self.pending.find(b"\n")) >= 0:
                    member = _staging_receipt_member(json.loads(self.pending[:boundary]))
                    del self.pending[: boundary + 1]
                    on_member(member)
                return len(data)

        sink = ReceiptSink()
        result = _exchange_source_worker(
            {
                "operation": "staging_receipt",
                "source": str(staged),
                "directory": directory,
                "metadata_directory": directory,
                "identities": {"": list(root)},
                "metadata_name": name,
                "metadata_identity": list(metadata_identity),
            },
            sink,
        )
        if sink.pending or set(result) != {"header", "provenance", "count"}:
            raise OSError(errno.EPROTO, "invalid staging receipt settlement")
        _verify_staging_metadata_name(directory, result["provenance"])
        if _identity(staged.lstat()) != root:
            raise OSError(errno.ESTALE, "staged input slot changed", str(staged))
        check_stop()
        return result
    finally:
        os.close(directory)


def probe_zip_container(path: Path) -> bool:
    """Prove ZIP shape on an accepted FD without releasing parent DB locks."""
    from polylogue.sources.sqlite_export import _exchange_source_worker, _identity, _named_identity

    physical = path.resolve(strict=True)
    directory = os.open(
        physical.parent,
        getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        # A raw stat: ``_named_identity`` refuses any non-regular file itself,
        # which would turn this "not a ZIP container" answer into an error.
        main = _identity(os.stat(physical.name, dir_fd=directory, follow_symlinks=False))
        if main[2] != stat.S_IFREG:
            return False
        parent = _identity(os.fstat(directory))
        result = _exchange_source_worker(
            {
                "operation": "zip_container",
                "source": str(physical),
                "directory": directory,
                "metadata_directory": directory,
                "identities": {"": main},
            }
        )
        if set(result) != {"zip"} or type(result["zip"]) is not bool:
            raise OSError(errno.EPROTO, "invalid ZIP container proof")
        if _identity(os.fstat(directory)) != parent or _named_identity(directory, physical.name) != main:
            raise OSError(errno.ESTALE, "ZIP container changed during inspection", str(path))
        return result["zip"]
    finally:
        os.close(directory)


def _probe_zip_container_in_worker(request: dict[str, Any]) -> dict[str, bool]:
    from polylogue.sources.sqlite_export import _identity, _named_identity

    source = Path(request["source"])
    directory = request["directory"]
    expected = tuple(request["identities"][""])
    descriptor = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if _identity(before) != expected or not stat.S_ISREG(before.st_mode):
            raise OSError(errno.ESTALE, "ZIP candidate differs from its accepted input", str(source))
        try:
            with zipfile.ZipFile(stream):
                result = True
        except zipfile.BadZipFile:
            result = False
        after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or _named_identity(directory, source.name) != expected:
            raise OSError(errno.ESTALE, "ZIP candidate changed during inspection", str(source))
    return {"zip": result}


@dataclass(frozen=True, slots=True)
class StagedInputMember:
    """One member proved by the operation-owned outside staging receipt."""

    identity: CapturedSourceInputIdentity
    file_identity: tuple[int, int, int]
    metadata_anchor: int
    provenance: dict[str, Any]
    content_kind: str
    content_revision: str

    def __post_init__(self) -> None:
        identity_fields: tuple[object, ...] = self.file_identity
        if (
            len(identity_fields) != 3
            or any(type(value) is not int for value in identity_fields)
            or self.file_identity[2] != stat.S_IFREG
            or self.content_kind not in {"bytes", "sqlite"}
            or len(self.content_revision) != 64
            or any(value not in "0123456789abcdef" for value in self.content_revision)
            or self.provenance.get("identity") is None
        ):
            raise ValueError("invalid captured staging member proof")
        _verify_staging_metadata_name(self.metadata_anchor, self.provenance)


@dataclass(frozen=True, slots=True)
class SourceInputBinding:
    """Declaration and provenance for the exact main inode the reader must open."""

    source: Path
    physical_path: Path
    source_path: Path
    identity_path: Path
    captured_profile_key: str
    captured_profile_root: Path
    captured_profile_source_path: Path
    parent_anchor: int
    metadata_anchor: int
    main_identity: tuple[int, int]
    provenance: dict[str, Any] | None
    staged: bool
    expected_content_kind: str | None = None
    expected_content_revision: str | None = None

    @property
    def captured_identity(self) -> CapturedSourceInputIdentity:
        """Persist the declaration and profile proved by this accepted input."""
        from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

        return CapturedSourceInputIdentity(
            canonical_source_path=str(self.identity_path),
            semantic_source_path=str(self.source_path),
            profile_root=str(self.captured_profile_root),
            profile_key=self.captured_profile_key,
            profile_source_path=str(self.captured_profile_source_path),
        )


@contextmanager
def bind_staged_member(
    staged_root: Path, member: dict[str, Any], receipt: dict[str, Any]
) -> Iterator[SourceInputBinding]:
    """Open the receipt's exact member through its accepted private directory."""
    from polylogue.sources.sqlite_export import _identity
    from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

    member = _staging_receipt_member(member)
    relative = Path(member["relative_path"])
    flags = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as owners:
        metadata_anchor = os.open(staged_root.parent, flags)
        owners.callback(os.close, metadata_anchor)
        parent_anchor = os.open(staged_root.name, flags, dir_fd=metadata_anchor)
        owners.callback(os.close, parent_anchor)
        if list(_identity(os.fstat(parent_anchor))) != receipt["header"]["root_identity"]:
            raise OSError(errno.ESTALE, "staged input root differs from its captured receipt", str(staged_root))
        for component in relative.parts[:-1]:
            parent_anchor = os.open(component, flags, dir_fd=parent_anchor)
            owners.callback(os.close, parent_anchor)
        staged_member = StagedInputMember(
            CapturedSourceInputIdentity.from_dict(member["identity"]),
            tuple(member["file_identity"]),
            metadata_anchor,
            receipt["provenance"],
            member["content_kind"],
            member["content_revision"],
        )
        with bind_source_input(
            staged_root / relative,
            parent_anchor=parent_anchor,
            semantic_parent=Path(staged_member.identity.semantic_source_path).parent,
            staged_input=staged_member,
        ) as binding:
            yield binding


@contextmanager
def bind_source_input(
    path: Path,
    *,
    parent_anchor: int | None = None,
    semantic_parent: Path | None = None,
    staged_input: StagedInputMember | None = None,
    captured_profile: CapturedHermesProfile | None = None,
    byte_page: SourceBytePage | None = None,
) -> Iterator[SourceInputBinding]:
    """Capture provenance once without opening an ordinary parent database FD.

    A caller binding a page of inputs lends its live ``byte_page``: that
    reader proves the binding instead of a fresh process per input.
    """
    from polylogue.sources.sqlite_export import _exchange_source_worker, _identity

    path = path.absolute()
    physical_path = path.resolve(strict=True) if parent_anchor is None else path
    if parent_anchor is not None and semantic_parent is None:
        raise OSError(errno.EINVAL, "anchored source input requires its captured semantic parent", str(path))
    declared_parent = path.parent.resolve(strict=True) if parent_anchor is None else semantic_parent
    assert declared_parent is not None
    semantic_path = declared_parent / path.name
    parent = physical_path.parent
    descriptor = (
        os.open(parent, getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW)
        if parent_anchor is None
        else os.dup(parent_anchor)
    )
    try:
        metadata_descriptor = (
            os.dup(staged_input.metadata_anchor)
            if staged_input is not None
            else os.open(
                declared_parent,
                getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            if parent_anchor is None
            else os.dup(parent_anchor)
        )
    except BaseException:
        os.close(descriptor)
        raise
    try:
        if parent_anchor is None and _identity(os.fstat(descriptor)) != _identity(physical_path.parent.stat()):
            raise OSError(errno.ESTALE, "source input parent changed", str(path))
        if staged_input is None and _identity(os.fstat(metadata_descriptor)) != _identity(path.parent.stat()):
            raise OSError(errno.ESTALE, "declared input parent changed", str(path))
        # A raw stat, so a FIFO or device reaches this binding's own typed
        # refusal instead of the SQLite exporter's.
        main = _identity(os.stat(physical_path.name, dir_fd=descriptor, follow_symlinks=False))
        if main[2] != stat.S_IFREG:
            raise OSError(errno.ESTALE, "source input must be a regular file", str(path))
        if staged_input is not None and main != staged_input.file_identity:
            raise OSError(errno.ESTALE, "staged input changed before binding", str(path))
        request = {
            "operation": "binding",
            "source": str(physical_path),
            "directory": descriptor,
            "metadata_directory": metadata_descriptor,
            "metadata_name": path.name,
            "semantic_source": str(semantic_path),
            "identities": {"": main},
            "staged_input": None
            if staged_input is None
            else {
                "identity": staged_input.identity.to_dict(),
                "provenance": staged_input.provenance,
            },
        }
        result = _exchange_source_worker(request) if byte_page is None else byte_page.exchange(request, None)
        if (
            set(result) != {"source_path", "provenance", "staged", "profile"}
            or not isinstance(result["source_path"], str)
            or not Path(result["source_path"]).is_absolute()
            or (result["provenance"] is not None and not isinstance(result["provenance"], dict))
            or type(result["staged"]) is not bool
        ):
            raise OSError(errno.EPROTO, "invalid source input binding result")
        provenance = result["provenance"]
        if staged_input is not None and (
            provenance is None or provenance.get("name") != staged_input.provenance["name"]
        ):
            raise OSError(errno.EPROTO, "invalid source input binding metadata name")
        _verify_staging_metadata_name(metadata_descriptor, provenance)
        if result["staged"] != (provenance is not None):
            raise OSError(errno.EPROTO, "inconsistent source input binding provenance")
        if not result["staged"] and Path(result["source_path"]) != semantic_path:
            raise OSError(errno.EPROTO, "inconsistent source input binding coordinate")
        actual = (
            path.stat()
            if parent_anchor is None
            else os.stat(physical_path.name, dir_fd=descriptor, follow_symlinks=False)
        )
        if (actual.st_dev, actual.st_ino) != main[:2]:
            raise OSError(errno.ESTALE, "declared input root changed", str(path))
        from polylogue.sources.parsers.hermes_identity import capture_profile_namespace

        with ExitStack() as profile_stack:
            if result["staged"]:
                from polylogue.sources.parsers.hermes_identity import CapturedHermesProfile

                receipt = result["profile"]
                if (
                    not isinstance(receipt, dict)
                    or set(receipt)
                    != {
                        "source_path",
                        "identity_path",
                        "profile_root",
                        "profile_key",
                        "profile_source_path",
                    }
                    or any(not isinstance(value, str) for value in receipt.values())
                ):
                    raise OSError(errno.EPROTO, "invalid staged profile identity receipt")
                from polylogue.core.provider_identity import profile_root_for_artifact

                profile_root = Path(receipt["profile_root"])
                profile_source = Path(receipt["profile_source_path"])
                if (
                    not profile_root.is_absolute()
                    or not profile_source.is_absolute()
                    or not Path(receipt["identity_path"]).is_absolute()
                    or receipt["source_path"] != result["source_path"]
                    or profile_root_for_artifact(profile_source) != profile_root
                    or captured_hermes_profile_key(profile_root) != receipt["profile_key"]
                ):
                    raise OSError(errno.EPROTO, "inconsistent staged profile identity receipt")
                profile = CapturedHermesProfile(
                    Path(receipt["profile_root"]), receipt["profile_key"], Path(receipt["profile_source_path"])
                )
                identity_path = Path(receipt["identity_path"])
            else:
                profile = captured_profile or profile_stack.enter_context(
                    capture_profile_namespace(path, metadata_descriptor)
                )
                identity_path = (
                    physical_path
                    if parent_anchor is None or captured_profile is not None
                    else declared_parent / path.name
                )
            yield SourceInputBinding(
                path,
                physical_path,
                Path(result["source_path"]),
                identity_path,
                profile.key,
                profile.root,
                profile.source_path,
                descriptor,
                metadata_descriptor,
                main[:2],
                provenance,
                result["staged"],
                None if staged_input is None else staged_input.content_kind,
                None if staged_input is None else staged_input.content_revision,
            )
    finally:
        os.close(metadata_descriptor)
        os.close(descriptor)


def stage_source_input(source: Path, staging_root: Path, *, check_stop: Callable[[], None]) -> Path:
    """Capture into one private slot and publish its streamed outside receipt last."""
    from polylogue.core.durable_fs import sync_directory
    from polylogue.core.provider_identity import profile_root_for_artifact
    from polylogue.sources.parsers.hermes_identity import (
        CapturedHermesProfile,
        capture_profile_namespace,
    )
    from polylogue.sources.sqlite_export import _control_bytes, _identity
    from polylogue.sources.sqlite_snapshot import _snapshot_sqlite_database_bound, is_sqlite_path

    declared = Path(os.path.abspath(source.expanduser()))
    physical = declared.resolve(strict=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    slot = Path(tempfile.mkdtemp(prefix="input-", dir=staging_root))
    metadata = staging_metadata_path(slot)
    try:
        metadata_descriptor, temporary_metadata_name = tempfile.mkstemp(prefix=".receipt-", dir=staging_root)
    except BaseException:
        shutil.rmtree(slot)
        raise
    temporary_metadata = Path(temporary_metadata_name)
    count = 0
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    def copy_member(binding: SourceInputBinding, relative: Path, receipt: Any, *, file_input: bool) -> None:
        nonlocal count
        check_stop()
        destination = slot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if is_sqlite_path(binding.source_path):
            # SQLite's journal suffix and control names must not lengthen a
            # valid payload basename. Keep control files outside the member
            # tree so every member name remains available. Rename only after
            # the connection and descriptor proof have settled.
            backup_descriptor, backup_name = tempfile.mkstemp(prefix=".db-", dir=staging_root)
            os.close(backup_descriptor)
            backup_path = Path(backup_name)
            try:
                accepted = _snapshot_sqlite_database_bound(
                    binding.source, backup_path, source_binding=binding, heartbeat=check_stop
                )
                os.replace(backup_path, destination)
                sync_directory(destination.parent)
                if staging_root != destination.parent:
                    sync_directory(staging_root)
            finally:
                backup_path.unlink(missing_ok=True)
            material = {
                "file_identity": accepted["database_identity"] + [stat.S_IFREG],
                "content_revision": accepted["logical_revision"],
            }
            content_kind = "sqlite"
        else:
            material = copy_bound_input(binding, destination, heartbeat=check_stop)
            content_kind = "bytes"
        member = {
            "type": "member",
            "coordinate": "input:0" if file_input else relative.as_posix(),
            "relative_path": relative.as_posix(),
            "identity": binding.captured_identity.to_dict(),
            "file_identity": material["file_identity"],
            "content_kind": content_kind,
            "content_revision": material["content_revision"],
        }
        receipt.write(_control_bytes(_staging_receipt_member(member)) + b"\n")
        count += 1
        check_stop()

    def walk(
        directory: int,
        physical_directory: Path,
        semantic_directory: Path,
        namespace: CapturedHermesProfile,
        relative: Path,
        ancestry: frozenset[tuple[int, int]],
        receipt: Any,
    ) -> None:
        # Directory aliases are accepted once. The parent descriptor and
        # captured namespace remain authority after a later alias retarget.
        with os.scandir(directory) as entries:
            for entry in entries:
                check_stop()
                before = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                member_relative = relative / entry.name
                semantic = semantic_directory / entry.name
                if stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                    child = os.open(entry.name, flags & ~os.O_NOFOLLOW, dir_fd=directory)
                    try:
                        opened = os.fstat(child)
                        named = os.stat(entry.name, dir_fd=directory)
                        identity = opened.st_dev, opened.st_ino
                        if identity != (named.st_dev, named.st_ino):
                            raise OSError(errno.ESTALE, "input directory alias changed during binding", str(semantic))
                        if identity in ancestry:
                            raise ValueError("input directory aliases contain a cycle")
                        child_physical = (physical_directory / entry.name).resolve(strict=True)
                        named_physical = child_physical.stat()
                        if identity != (named_physical.st_dev, named_physical.st_ino):
                            raise OSError(errno.ESTALE, "input directory physical coordinate changed", str(semantic))
                        declared_profile = profile_root_for_artifact(semantic / ".input")
                        if declared_profile == profile_root_for_artifact(semantic_directory / ".input"):
                            child_namespace = CapturedHermesProfile(
                                namespace.root, namespace.key, namespace.source_path.parent / entry.name / ".input"
                            )
                        else:
                            child_namespace = CapturedHermesProfile(
                                child_physical, captured_hermes_profile_key(child_physical), child_physical / ".input"
                            )
                        walk(
                            child,
                            child_physical,
                            semantic,
                            child_namespace,
                            member_relative,
                            ancestry | {identity},
                            receipt,
                        )
                    finally:
                        os.close(child)
                elif stat.S_ISREG(before.st_mode):
                    member_profile = CapturedHermesProfile(
                        namespace.root, namespace.key, namespace.source_path.parent / entry.name
                    )
                    with bind_source_input(
                        physical_directory / entry.name,
                        parent_anchor=directory,
                        semantic_parent=physical_directory,
                        captured_profile=member_profile,
                    ) as binding:
                        if binding.main_identity != (before.st_dev, before.st_ino):
                            raise OSError(errno.ESTALE, "enumerated input member changed", str(semantic))
                        copy_member(binding, member_relative, receipt, file_input=False)
                else:
                    raise ValueError("input members must be regular files or directory aliases")

    try:
        with os.fdopen(metadata_descriptor, "wb") as receipt:
            os.fchmod(receipt.fileno(), 0o600)
            root_identity = list(_identity(slot.lstat()))
            if physical.is_dir():
                directory = os.open(physical, flags)
                try:
                    with capture_profile_namespace(declared / ".input", directory) as namespace:
                        semantic_root = namespace.source_path.parent
                        receipt.write(
                            _control_bytes(
                                {
                                    "type": "header",
                                    "version": 3,
                                    "input_kind": "directory",
                                    "source_path": str(semantic_root),
                                    "source_name": declared.name,
                                    "root_identity": root_identity,
                                }
                            )
                            + b"\n"
                        )
                        opened = os.fstat(directory)
                        walk(
                            directory,
                            physical,
                            semantic_root,
                            namespace,
                            Path(),
                            frozenset({(opened.st_dev, opened.st_ino)}),
                            receipt,
                        )
                finally:
                    os.close(directory)
            else:
                with bind_source_input(declared) as binding:
                    receipt.write(
                        _control_bytes(
                            {
                                "type": "header",
                                "version": 3,
                                "input_kind": "file",
                                "source_path": str(binding.source_path),
                                "source_name": declared.name,
                                "root_identity": root_identity,
                            }
                        )
                        + b"\n"
                    )
                    copy_member(binding, Path(declared.name), receipt, file_input=True)
            if not count:
                raise ValueError("input contains no physical files")
            receipt.write(_control_bytes({"type": "end", "count": count}) + b"\n")
            receipt.flush()
            os.fsync(receipt.fileno())
        sync_directory(slot)
        os.replace(temporary_metadata, metadata)
        sync_directory(staging_root)
        check_stop()
        return slot
    except BaseException:
        # Only generated operation-owned paths enter cleanup. Source paths
        # and caller-supplied destinations are never removal targets.
        metadata.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        shutil.rmtree(slot)
        raise


def _read_bound_input_in_worker(request: dict[str, Any], sink: BinaryWriteSink | None) -> dict[str, Any]:
    """Stream ordinary bytes through the fresh reader's descriptor and ACK owner."""
    from polylogue.sources.sqlite_export import _identity, _named_identity, _WorkerSink

    def progress() -> None:
        if request.get("progress"):
            _WorkerSink().write(b"")

    source = Path(request["source"])
    expected = tuple(request["identities"][""])
    directory = request["directory"]
    if request.get("expected_content_kind") not in {None, "bytes"}:
        raise OSError(errno.ESTALE, "staged source is not the accepted byte input", str(source))
    descriptor = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    digest = hashlib.sha256()
    size = 0
    preflight = None
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if _identity(before) != expected or not stat.S_ISREG(before.st_mode):
            raise OSError(errno.ESTALE, "source differs from its accepted byte input", str(source))
        _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
        if request["operation"] == "preflight_bytes":
            from polylogue.sources.import_preflight import _preflight_handle

            class ProgressInput:
                def read(self, size: int = -1) -> bytes:
                    progress()
                    return stream.read(size)

                def seek(self, offset: int, whence: int = 0) -> int:
                    progress()
                    return stream.seek(offset, whence)

                def tell(self) -> int:
                    return stream.tell()

                def seekable(self) -> bool:
                    return True

            preflight = _preflight_handle(
                cast(BinaryIO, ProgressInput()), Path(request["semantic_source"]), check_stop=progress
            )
            stream.seek(0)
        while chunk := stream.read(1024 * 1024):
            progress()
            digest.update(chunk)
            size += len(chunk)
            if sink is not None:
                sink.write(chunk)
        after = os.fstat(stream.fileno())
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or size != before.st_size
            or _named_identity(directory, source.name) != expected
        ):
            raise OSError(errno.ESTALE, "source changed while retaining the accepted input", str(source))
        _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
        if request.get("expected_content_revision") not in {None, digest.hexdigest()}:
            raise OSError(errno.ESTALE, "staged bytes differ from the accepted input", str(source))
    result: dict[str, Any] = {
        "content_revision": digest.hexdigest(),
        "size_bytes": size,
        "file_observation": [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns],
    }
    if preflight is not None:
        result["preflight"] = preflight.to_dict()
    return result


def _bound_byte_request(binding: SourceInputBinding, operation: str) -> dict[str, Any]:
    from polylogue.sources.sqlite_export import _named_identity

    main = _named_identity(binding.parent_anchor, binding.physical_path.name)
    if main[:2] != binding.main_identity:
        raise OSError(errno.ESTALE, "anchored source changed before reading", str(binding.source))
    return {
        "operation": operation,
        "source": str(binding.physical_path),
        "semantic_source": str(binding.source_path),
        "directory": binding.parent_anchor,
        "metadata_directory": binding.metadata_anchor,
        "provenance": binding.provenance,
        "identities": {"": main},
        "expected_content_kind": binding.expected_content_kind,
        "expected_content_revision": binding.expected_content_revision,
    }


def preflight_bound_bytes(
    binding: SourceInputBinding,
    *,
    check_stop: Callable[[], None],
) -> dict[str, Any]:
    """Return shape evidence only after the accepted byte reader has settled."""
    from polylogue.sources.sqlite_export import _ProgressSink, source_byte_page

    request = _bound_byte_request(binding, "preflight_bytes")
    request["progress"] = True
    with source_byte_page() as reader:
        result = reader.exchange(request, _ProgressSink(check_stop))
    _validate_byte_settlement(binding, result, extra_fields={"preflight"})
    if not isinstance(result["preflight"], dict):
        raise OSError(errno.EPROTO, "invalid source preflight evidence")
    _verify_staging_metadata_name(binding.metadata_anchor, binding.provenance)
    return result["preflight"]


def write_bound_input(
    binding: SourceInputBinding, destination: BinaryWriteSink, *, reader: SourceBytePage
) -> dict[str, Any]:
    """Deliver one proved observation through its caller-owned byte page."""
    try:
        result = reader.exchange(_bound_byte_request(binding, "bytes"), destination)
        _validate_byte_settlement(binding, result)
        _verify_staging_metadata_name(binding.metadata_anchor, binding.provenance)
        return result
    except BaseException as error:
        reader.reject(error)
        raise


def _validate_byte_settlement(
    binding: SourceInputBinding,
    result: dict[str, Any],
    *,
    extra_fields: set[str] | None = None,
) -> None:
    """Require the same proved byte observation for capture and preflight."""
    if (
        set(result) != ({"content_revision", "size_bytes", "file_observation"} | (extra_fields or set()))
        or not isinstance(result["content_revision"], str)
        or len(result["content_revision"]) != 64
        or any(value not in "0123456789abcdef" for value in result["content_revision"])
        or type(result["size_bytes"]) is not int
        or result["size_bytes"] < 0
        or not isinstance(result["file_observation"], list)
        or len(result["file_observation"]) != 5
        or any(type(value) is not int for value in result["file_observation"])
        or tuple(result["file_observation"][:2]) != binding.main_identity
        or result["file_observation"][2] != result["size_bytes"]
    ):
        raise OSError(errno.EPROTO, "invalid source byte settlement")


def _copy_bound_input_in_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Copy byte input in the isolated owner, never closing its FD in a reader process."""
    from polylogue.core.durable_fs import clone_or_copy_replace
    from polylogue.sources.sqlite_export import _identity, _named_identity, _WorkerSink

    def progress() -> None:
        if request.get("progress"):
            _WorkerSink().write(b"")

    source = Path(request["source"])
    destination = Path(request["destination"])
    expected = tuple(request["identities"][""])
    directory = request["directory"]
    descriptor = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as accepted:
        before = os.fstat(accepted.fileno())
        if _identity(before) != expected or not stat.S_ISREG(before.st_mode):
            raise OSError(errno.ESTALE, "source differs from its accepted byte input", str(source))
        _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
        publication: dict[str, Any] = {}

        def digest_prefix(fd: int) -> str:
            digest = hashlib.sha256()
            offset = 0
            while offset < before.st_size:
                progress()
                chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
                if not chunk:
                    raise OSError(errno.ESTALE, "source prefix ended during copying", str(source))
                digest.update(chunk)
                offset += len(chunk)
            return digest.hexdigest()

        def prove_candidate(candidate: int) -> None:
            # A concurrent append does not change the captured prefix. A
            # reflink may contain that append, so pin the candidate to the
            # descriptor's accepted length before publication.
            os.ftruncate(candidate, before.st_size)
            revision = digest_prefix(candidate)
            if revision != digest_prefix(accepted.fileno()):
                raise OSError(errno.ESTALE, "source prefix changed during copying", str(source))
            after = os.fstat(accepted.fileno())
            if after.st_size < before.st_size or (
                after.st_size == before.st_size
                and (after.st_mtime_ns, after.st_ctime_ns) != (before.st_mtime_ns, before.st_ctime_ns)
            ):
                raise OSError(errno.ESTALE, "source changed during copying", str(source))
            if _named_identity(directory, source.name) != expected:
                raise OSError(errno.ESTALE, "anchored source changed during copying", str(source))
            _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
            os.fsync(candidate)
            publication.update(
                content_revision=revision,
                size_bytes=before.st_size,
                file_identity=list(_identity(os.fstat(candidate))),
            )

        class ProgressReader:
            def fileno(self) -> int:
                return accepted.fileno()

            def read(self, size: int = -1) -> bytes:
                progress()
                return accepted.read(size)

        progress()
        clone_or_copy_replace(cast(BinaryIO, ProgressReader()), destination, before_publish=prove_candidate)
        if list(_identity(destination.lstat())) != publication["file_identity"]:
            raise OSError(errno.ESTALE, "private copy changed before settlement", str(destination))
        return publication


def copy_bound_input(
    binding: SourceInputBinding, destination: Path, *, heartbeat: Callable[[], None] | None = None
) -> dict[str, Any]:
    """Copy the accepted byte source while retaining its immutable original receipt."""
    from polylogue.sources.sqlite_export import _exchange_source_worker, _named_identity, _ProgressSink

    main = _named_identity(binding.parent_anchor, binding.physical_path.name)
    if main[:2] != binding.main_identity:
        raise OSError(errno.ESTALE, "anchored source changed before copying", str(binding.source))
    result = _exchange_source_worker(
        {
            "operation": "copy",
            "progress": heartbeat is not None,
            "source": str(binding.physical_path),
            "destination": str(destination),
            "directory": binding.parent_anchor,
            "metadata_directory": binding.metadata_anchor,
            "provenance": binding.provenance,
            "identities": {"": main},
        },
        None if heartbeat is None else _ProgressSink(heartbeat),
    )
    if (
        set(result) != {"content_revision", "size_bytes", "file_identity"}
        or not isinstance(result["content_revision"], str)
        or len(result["content_revision"]) != 64
        or any(value not in "0123456789abcdef" for value in result["content_revision"])
        or type(result["size_bytes"]) is not int
        or result["size_bytes"] < 0
        or not isinstance(result["file_identity"], list)
        or len(result["file_identity"]) != 3
        or any(type(value) is not int for value in result["file_identity"])
    ):
        raise OSError(errno.EPROTO, "invalid source copy settlement")
    _verify_staging_metadata_name(binding.metadata_anchor, binding.provenance)
    return result


def _verify_staging_metadata_name(directory: int, expected: dict[str, Any] | None) -> None:
    """Check currency of the original fully validated outside receipt.

    The captured digest, inode and observation arrive only from the completed
    receipt reader through its owned spool/proof. Later reads check that same
    receipt has stayed unchanged; they never reinterpret new metadata or
    establish byte authority from a stat. Payload readers still prove each
    member independently. No ordinary metadata FD belongs to the parent.
    """
    from polylogue.sources.sqlite_export import _identity

    if expected is None:
        return
    identity = expected.get("identity")
    if (
        set(expected) != {"name", "identity", "digest", "observation"}
        or not isinstance(expected.get("name"), str)
        or expected["name"] in {"", ".", ".."}
        or "\0" in expected["name"]
        or Path(expected["name"]).name != expected["name"]
        or not isinstance(identity, list)
        or len(identity) != 3
        or any(type(value) is not int for value in identity)
        or identity[2] != stat.S_IFREG
        or not isinstance(expected.get("observation"), list)
        or len(expected["observation"]) != 3
        or any(type(value) is not int for value in expected["observation"])
        or not isinstance(expected.get("digest"), str)
        or len(expected["digest"]) != 64
        or any(character not in "0123456789abcdef" for character in expected["digest"])
    ):
        raise OSError(errno.EPROTO, "invalid staging metadata proof")
    current = os.stat(expected["name"], dir_fd=directory, follow_symlinks=False)
    if (
        list(_identity(current)) != identity
        or [current.st_ctime_ns, current.st_mtime_ns, current.st_size] != expected["observation"]
    ):
        raise OSError(errno.ESTALE, "staging metadata changed", expected["name"])


def staging_metadata_path(staged_path: Path) -> Path:
    """Return the sole outside receipt for an operation-owned directory slot."""
    return staged_path.with_name(f"{staged_path.name}{_STAGING_METADATA_SUFFIX}")
