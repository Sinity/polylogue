"""Member-level source continuity checks used by source-authority coordinators."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import overload


class SourceContinuityError(ValueError):
    """A source declaration, manifest, or continuity receipt is unsafe."""


class SourceRole(StrEnum):
    IMMUTABLE_EXPORT = "immutable-export"
    ARCHIVE_MEMBER = "archive-member"
    APPEND_JSONL = "append-jsonl"
    REWRITE_JSONL = "rewrite-leading-jsonl"
    MUTABLE_SQLITE = "mutable-sqlite"
    SPOOL = "spool"
    QUEUE = "queue"
    ATTACHMENT = "attachment"
    SIDECAR = "sidecar"
    PROVIDER_CACHE = "provider-cache"
    DIRECTORY = "directory"


class FrontierState(StrEnum):
    """Observation state for one configured source root."""

    PRESENT = "present"
    VALID_EMPTY = "valid-empty"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    source_id: str
    role: SourceRole
    root: Path
    mutable: bool = False
    layout_name: str | None = None
    exclude_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise SourceContinuityError("source_id must be non-empty")
        mutable_roles = {
            SourceRole.APPEND_JSONL,
            SourceRole.REWRITE_JSONL,
            SourceRole.MUTABLE_SQLITE,
            SourceRole.SPOOL,
            SourceRole.QUEUE,
            SourceRole.SIDECAR,
            SourceRole.PROVIDER_CACHE,
        }
        if self.role in mutable_roles and not self.mutable:
            raise SourceContinuityError(f"mutable source {self.source_id} must be marked mutable")


@dataclass(frozen=True, slots=True)
class FrontierMember:
    """One independently enumerated source member at a stable cut."""

    source_id: str
    coordinate: str
    identity: str
    content_sha256: str
    size: int
    logical_sha256: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.source_id, self.coordinate, self.identity, self.content_sha256


class _FrontierMemberStore:
    """Private SQLite spool for member rows; queries fetch bounded pages."""

    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="polylogue-source-frontier-")
        os.chmod(self._directory.name, 0o700)
        self.connection = sqlite3.connect(Path(self._directory.name) / "members.sqlite3")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute(
            """CREATE TABLE members (
                   ordinal INTEGER PRIMARY KEY,
                   source_id TEXT NOT NULL, coordinate TEXT NOT NULL,
                   identity TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                   size INTEGER NOT NULL CHECK(size >= 0), logical_sha256 TEXT,
                   source_path TEXT NOT NULL, physical_identity TEXT NOT NULL UNIQUE,
                   UNIQUE(source_id, coordinate, identity, content_sha256)
               )"""
        )
        self.count = 0
        self.byte_count = 0
        self.closed = False

    def add(self, member: FrontierMember, source_path: Path, physical_identity: tuple[int, ...]) -> None:
        try:
            self.connection.execute(
                "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.count,
                    member.source_id,
                    member.coordinate,
                    member.identity,
                    member.content_sha256,
                    member.size,
                    member.logical_sha256,
                    str(source_path),
                    json.dumps(physical_identity, separators=(",", ":")),
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "physical_identity" in str(exc):
                raise SourceContinuityError("source frontier contains duplicate physical member identity") from exc
            raise SourceContinuityError("source frontier contains duplicate members") from exc
        self.count += 1
        self.byte_count += member.size

    def rows(self) -> Iterator[tuple[str, str, str, str, int, str | None, str, str]]:
        cursor = self.connection.execute(
            """SELECT source_id, coordinate, identity, content_sha256, size, logical_sha256,
                      source_path, physical_identity FROM members ORDER BY ordinal"""
        )
        while page := cursor.fetchmany(512):
            for source_id, coordinate, identity, content, size, logical, source_path, physical_identity in page:
                yield (
                    str(source_id),
                    str(coordinate),
                    str(identity),
                    str(content),
                    int(size),
                    None if logical is None else str(logical),
                    str(source_path),
                    str(physical_identity),
                )

    def verify_integrity(self, declaration_ids: set[str]) -> None:
        """Check the captured spool without rebuilding its member set in memory."""
        if self.closed:
            raise SourceContinuityError("source frontier member store is closed")
        row = self.connection.execute(
            """SELECT COUNT(*), COALESCE(SUM(size), 0), MIN(ordinal), MAX(ordinal),
                      COUNT(DISTINCT source_id), COUNT(DISTINCT physical_identity)
               FROM members"""
        ).fetchone()
        if row is None:
            raise SourceContinuityError("source frontier member store is unavailable")
        count, byte_count, minimum, maximum, source_count, physical_count = row
        count = int(count)
        byte_count = int(byte_count)
        source_count = int(source_count)
        physical_count = int(physical_count)
        if count != self.count or byte_count != self.byte_count:
            raise SourceContinuityError(
                "source frontier member store totals changed "
                f"(captured={self.count}/{self.byte_count}, stored={count}/{byte_count})"
            )
        if count and (int(minimum) != 0 or int(maximum) != count - 1):
            raise SourceContinuityError("source frontier member ordinals are not contiguous")
        if physical_count != count:
            raise SourceContinuityError("source frontier contains duplicate physical member identity")
        if source_count > len(declaration_ids):
            raise SourceContinuityError("source frontier contains an undeclared source ID")
        undeclared = self.connection.execute(
            "SELECT 1 FROM members WHERE source_id NOT IN (" + ",".join("?" for _ in declaration_ids) + ") LIMIT 1",
            tuple(sorted(declaration_ids)),
        ).fetchone()
        if undeclared is not None:
            raise SourceContinuityError("source frontier contains an undeclared source ID")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        self._directory.cleanup()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class _FrontierMembers(Sequence[FrontierMember]):
    def __init__(self, store: _FrontierMemberStore) -> None:
        self._store = store

    def __len__(self) -> int:
        return self._store.count

    def __iter__(self) -> Iterator[FrontierMember]:
        for source_id, coordinate, identity, content, size, logical, _path, _physical in self._store.rows():
            yield FrontierMember(
                str(source_id),
                str(coordinate),
                str(identity),
                str(content),
                int(size),
                None if logical is None else str(logical),
            )

    @overload
    def __getitem__(self, index: int) -> FrontierMember: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FrontierMember, ...]: ...

    def __getitem__(self, index: int | slice) -> FrontierMember | tuple[FrontierMember, ...]:
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self._store.connection.execute(
            "SELECT source_id, coordinate, identity, content_sha256, size, logical_sha256 FROM members WHERE ordinal=?",
            (index,),
        ).fetchone()
        assert row is not None
        return FrontierMember(
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            None if row[5] is None else str(row[5]),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(self[index] == other[index] for index in range(len(self)))

    def copy_to(self, connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS temp._polylogue_source_frontier_member")
        connection.execute(
            "CREATE TEMP TABLE _polylogue_source_frontier_member "
            "(ordinal INTEGER PRIMARY KEY, source_id TEXT NOT NULL, coordinate TEXT NOT NULL, identity TEXT NOT NULL, "
            "source_path TEXT NOT NULL, content_sha256 TEXT NOT NULL)"
        )
        connection.execute("DROP TABLE IF EXISTS temp._polylogue_source_frontier_path")
        connection.execute(
            "CREATE TEMP TABLE _polylogue_source_frontier_path (source_path TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        cursor = self._store.connection.execute(
            "SELECT ordinal, source_id, coordinate, identity, source_path, content_sha256 FROM members ORDER BY ordinal"
        )
        while page := cursor.fetchmany(512):
            connection.executemany("INSERT INTO temp._polylogue_source_frontier_member VALUES (?, ?, ?, ?, ?, ?)", page)
            connection.executemany(
                "INSERT OR IGNORE INTO temp._polylogue_source_frontier_path VALUES (?)",
                ((row[4],) for row in page),
            )
        connection.execute(
            "CREATE INDEX temp._polylogue_source_frontier_member_owner "
            "ON _polylogue_source_frontier_member(source_path, content_sha256)"
        )

    def close(self) -> None:
        self._store.close()


@dataclass(frozen=True, slots=True)
class SourceFrontier:
    """Complete configured-source denominator used by conservation checks."""

    declarations: tuple[SourceDeclaration, ...]
    members: Sequence[FrontierMember]
    root_states: Mapping[str, FrontierState]
    blockers: tuple[str, ...]
    frontier_sha256: str

    @property
    def item_count(self) -> int:
        return len(self.members)

    @property
    def byte_count(self) -> int:
        if isinstance(self.members, _FrontierMembers):
            return self.members._store.byte_count
        return sum(member.size for member in self.members)

    def close(self) -> None:
        if isinstance(self.members, _FrontierMembers):
            self.members.close()

    def copy_members_to(self, connection: sqlite3.Connection) -> None:
        if not isinstance(self.members, _FrontierMembers):
            raise SourceContinuityError("source frontier member store is unavailable")
        self.members.copy_to(connection)

    def __enter__(self) -> SourceFrontier:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def complete(self) -> bool:
        return not self.blockers and all(state is not FrontierState.UNAVAILABLE for state in self.root_states.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "declarations": [
                {
                    "source_id": declaration.source_id,
                    "role": declaration.role.value,
                    "root": str(declaration.root),
                    "mutable": declaration.mutable,
                    "layout_name": declaration.layout_name,
                    "exclude_coordinates": list(declaration.exclude_coordinates),
                }
                for declaration in self.declarations
            ],
            "frontier_sha256": self.frontier_sha256,
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "complete": self.complete,
            "blockers": list(self.blockers),
            "root_states": {key: value.value for key, value in sorted(self.root_states.items())},
            "members": [
                {
                    "source_id": member.source_id,
                    "coordinate": member.coordinate,
                    "identity": member.identity,
                    "content_sha256": member.content_sha256,
                    "size": member.size,
                    "logical_sha256": member.logical_sha256,
                }
                for member in self.members
            ],
        }

    def verify_integrity(self) -> None:
        declaration_ids = {declaration.source_id for declaration in self.declarations}
        if len(declaration_ids) != len(self.declarations):
            raise SourceContinuityError("source frontier contains duplicate source IDs")
        if set(self.root_states) != declaration_ids:
            raise SourceContinuityError("source frontier root states do not cover declarations")
        if not isinstance(self.members, _FrontierMembers):
            raise SourceContinuityError("source frontier member store is unavailable")
        self.members._store.verify_integrity(declaration_ids)
        expected = _frontier_digest(self.declarations, self.members, self.root_states, self.blockers)
        if expected != self.frontier_sha256:
            raise SourceContinuityError("source frontier integrity check failed")


def _frontier_digest(
    declarations: Sequence[SourceDeclaration],
    members: Iterable[FrontierMember],
    root_states: Mapping[str, FrontierState],
    blockers: Sequence[str],
) -> str:
    """Hash the canonical frontier JSON while keeping one member row at a time."""
    digest = hashlib.sha256()
    digest.update(b'{"blockers":')
    digest.update(json.dumps(list(blockers), separators=(",", ":")).encode())
    digest.update(b',"declarations":')
    digest.update(
        json.dumps(
            [
                (d.source_id, d.role.value, str(d.root), d.mutable, d.layout_name, d.exclude_coordinates)
                for d in declarations
            ],
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b',"members":[')
    for index, member in enumerate(members):
        if index:
            digest.update(b",")
        digest.update(
            json.dumps(
                (
                    member.source_id,
                    member.coordinate,
                    member.identity,
                    member.content_sha256,
                    member.size,
                    member.logical_sha256,
                ),
                separators=(",", ":"),
            ).encode()
        )
    digest.update(b'],"root_states":')
    digest.update(
        json.dumps(sorted((key, value.value) for key, value in root_states.items()), separators=(",", ":")).encode()
    )
    digest.update(b"}")
    return digest.hexdigest()


_CAPTURED_INODE = re.compile(r"dev:(\d+):ino:(\d+)")


def build_source_frontier(
    declarations: Iterable[SourceDeclaration], *, initial_refusals: Mapping[str, str] | None = None
) -> SourceFrontier:
    """Enumerate every configured root, retaining unavailable roots as blockers."""
    rows = tuple(declarations)
    if not rows:
        raise SourceContinuityError("source frontier declaration is empty")
    if len({row.source_id for row in rows}) != len(rows):
        raise SourceContinuityError("source frontier contains duplicate source IDs")
    if set(initial_refusals or ()) - {row.source_id for row in rows}:
        raise SourceContinuityError("source frontier refusal names an undeclared root")
    if len({Path(row.root).resolve(strict=False) for row in rows}) != len(rows):
        raise SourceContinuityError("source frontier contains duplicate roots")
    from polylogue.sources.source_snapshot import (
        SourceSnapshotError,
        bind_source_observation,
        iter_observe_source_members,
    )

    store = _FrontierMemberStore()
    members = _FrontierMembers(store)
    states: dict[str, FrontierState] = {}
    blockers: list[str] = []
    try:
        for declaration in rows:
            initial_refusal = (initial_refusals or {}).get(declaration.source_id)
            if initial_refusal is not None:
                # The configured member split was refused before binding.
                # Later readability cannot complete that earlier denominator.
                states[declaration.source_id] = FrontierState.UNAVAILABLE
                blockers.append(f"unavailable:{declaration.source_id}:{declaration.root}:{initial_refusal}")
                continue
            before_count = store.count
            try:
                binding = bind_source_observation(declaration)
                root_is_directory = binding.root_identity.kind == "directory"
                disappeared: Path | None = None
                for item in iter_observe_source_members(binding):
                    identity: tuple[int, ...]
                    if declaration.role is SourceRole.ARCHIVE_MEMBER:
                        device, inode, _ctime, offset = item.identity.split(":")
                        identity = (int(device), int(inode), int(offset))
                        _archive_name, separator, archive_member = item.coordinate.partition("!")
                        if not separator or not archive_member:
                            raise SourceContinuityError(
                                f"archive frontier member has no member coordinate: {item.source_id}:{item.coordinate}"
                            )
                        source_path = Path(f"{declaration.root}:{archive_member}")
                    else:
                        captured = _CAPTURED_INODE.search(item.identity)
                        if captured is not None:
                            identity = (int(captured.group(1)), int(captured.group(2)))
                        else:
                            # SQLite logical identities are not physical; stat
                            # the corresponding original member for alias proof.
                            member_path = (
                                declaration.root if not root_is_directory else declaration.root / item.coordinate
                            )
                            try:
                                info = member_path.stat()
                            except OSError:
                                disappeared = member_path
                                break
                            identity = (info.st_dev, info.st_ino)
                        source_path = declaration.root / item.coordinate if root_is_directory else declaration.root
                    store.add(
                        FrontierMember(
                            item.source_id,
                            item.coordinate,
                            item.identity,
                            item.content_sha256,
                            item.size_bytes,
                            item.content_sha256 if declaration.role is SourceRole.MUTABLE_SQLITE else None,
                        ),
                        source_path,
                        identity,
                    )
                if disappeared is not None:
                    raise SourceSnapshotError(f"source member disappeared after observation: {disappeared}")
            except SourceContinuityError:
                raise
            except (OSError, SourceSnapshotError, ValueError) as exc:
                # journal_mode=OFF keeps this private spool fast but does not
                # roll back writes to a savepoint. Remove only this root's
                # partial rows explicitly before recording its unavailable
                # state.
                store.connection.execute("DELETE FROM members WHERE source_id = ?", (declaration.source_id,))
                store.count = before_count
                store.byte_count = int(
                    store.connection.execute("SELECT COALESCE(SUM(size), 0) FROM members").fetchone()[0]
                )
                states[declaration.source_id] = FrontierState.UNAVAILABLE
                blockers.append(f"unavailable:{declaration.source_id}:{declaration.root}:{exc}")
                continue
            states[declaration.source_id] = (
                FrontierState.PRESENT if store.count > before_count else FrontierState.VALID_EMPTY
            )
        store.connection.commit()
        digest = _frontier_digest(rows, members, states, blockers)
        return SourceFrontier(rows, members, states, tuple(blockers), digest)
    except BaseException:
        store.close()
        raise


def configured_source_frontier(archive_root: Path) -> SourceFrontier:
    """Observe the configured provider layouts and Polylogue capture spools.

    The denominator is assembled from the same resolved source roots used by
    daemon intake, before consulting any acquisition ledger. Hook carriers,
    hook pending envelopes, and browser captures are explicit source roots.
    """
    from polylogue.config import resolve_runtime_config
    from polylogue.paths import archive_root as configured_archive_root
    from polylogue.sources.hooks import hook_spool_sources
    from polylogue.sources.origin_specs import database_capability_for_provider, pre_acquisition_path_exclusion
    from polylogue.sources.source_layout import source_layout_for
    from polylogue.sources.source_walk import layout_source_candidates

    runtime = resolve_runtime_config()
    expected_archive = Path(archive_root).expanduser().resolve(strict=False)
    if Path(configured_archive_root()).expanduser().resolve(strict=False) != expected_archive:
        raise SourceContinuityError("archive root differs from the resolved source-spool configuration")

    rows: list[SourceDeclaration] = []
    initial_refusals: dict[str, str] = {}
    source_paths = getattr(runtime, "source_paths", None)
    canonical_paths = (
        ("claude-code", "claude_code"),
        ("claude-code-todos", "claude_code_todos"),
        ("claude-code-history", "claude_code_history"),
        ("codex", "codex"),
        ("codex-state", "codex_state"),
        ("codex-memories", "codex_memories"),
        ("gemini-cli", "gemini_cli"),
        ("hermes", "hermes"),
        ("antigravity", "antigravity"),
        ("antigravity-cli", "antigravity_cli"),
        ("browser-capture", "browser_capture"),
        ("inbox", "inbox"),
    )
    candidates: dict[str, Path] = {}
    if source_paths is not None:
        candidates.update(
            (name, Path(value))
            for name, attribute in canonical_paths
            if (value := getattr(source_paths, attribute, None)) is not None
        )
    runtime_source_names = {source.name for source in runtime.sources}
    for source in runtime.sources:
        if source.name != "hooks" and source.path is not None:
            candidates.setdefault(source.name, Path(source.path))

    for name, path in candidates.items():
        if name == "hooks":
            # The complete hook root is declared below so pending envelopes
            # and provider carriers share the same spool owner.
            continue
        root_observed = True
        try:
            path.lstat()
        except FileNotFoundError:
            root_observed = False
            # Resolved canonical paths are optional until present. A path
            # already admitted by the runtime/watch declarations remains in
            # the denominator even if it disappeared before this observation.
            if name not in runtime_source_names:
                continue
            initial_refusals[f"configured:{name}"] = "declared source disappeared before member selection"
        except OSError as exc:
            root_observed = False
            initial_refusals[f"configured:{name}"] = str(exc)
        layout_name = None if name == "claude-code-history" else name
        role = SourceRole.APPEND_JSONL if name == "claude-code-history" else SourceRole.DIRECTORY
        try:
            # Keep an unreadable declaration for the frontier owner to mark
            # unavailable. A second successful type probe cannot authorize
            # the provisional database/exclusion walk after failed custody.
            is_directory = root_observed and path.is_dir()
        except OSError:
            is_directory = False
        if not is_directory and layout_name is not None and name not in dict(canonical_paths):
            layout_name = None
        sqlite_paths: list[Path] = []
        database_paths: list[Path] = []
        path_exclusions: list[Path] = []
        if layout_name is not None and is_directory:
            layout = source_layout_for(layout_name)
            provider = layout.provider
            capability = database_capability_for_provider(provider) if provider is not None else None
            declared_members = {item.filename for item in capability.members} if capability is not None else set()
            if provider is not None:
                for member in layout_source_candidates(layout_name, path):
                    if not member.is_file():
                        continue
                    if member.name in declared_members:
                        database_paths.append(member)
                    elif pre_acquisition_path_exclusion(provider, member) is not None:
                        path_exclusions.append(member)
            if declared_members:
                sqlite_paths = [
                    member
                    for member in database_paths
                    if capability is not None
                    and (rule := capability.member(member.name)) is not None
                    and rule.disposition != "out-of-scope"
                ]
        # Admitted databases have separate logical declarations; declared
        # out-of-scope projections are not acquisition obligations at all.
        excluded = tuple(sorted(member.relative_to(path).as_posix() for member in (*database_paths, *path_exclusions)))
        rows.append(
            SourceDeclaration(
                f"configured:{name}",
                role,
                path,
                True,
                layout_name,
                excluded,
            )
        )
        rows.extend(
            SourceDeclaration(
                f"configured:{name}:sqlite:{member.relative_to(path).as_posix()}",
                SourceRole.MUTABLE_SQLITE,
                member,
                True,
            )
            for member in sqlite_paths
        )
    observed_spools: list[tuple[Path, tuple[int, int]]] = []
    for spec in hook_spool_sources():
        spool_root = Path(spec.root)
        try:
            spool_identity = spool_root.lstat()
        except OSError as exc:
            raise SourceContinuityError(f"hook spool root is unavailable: {spool_root}: {exc}") from exc
        if not stat.S_ISDIR(spool_identity.st_mode):
            raise SourceContinuityError(f"hook spool root is not a directory: {spool_root}")
        observed_spools.append((spool_root, (spool_identity.st_dev, spool_identity.st_ino)))
        for carrier_provider in ("claude-code", "codex", "hermes"):
            carrier_root = spool_root / "carriers" / carrier_provider
            if _source_path_present(carrier_root):
                rows.append(
                    SourceDeclaration(
                        f"{spec.source_id}:carrier:{carrier_provider}",
                        SourceRole.SPOOL,
                        carrier_root,
                        True,
                        f"{carrier_provider}-hooks",
                    )
                )
        pending = spool_root / "pending"
        if _source_path_present(pending):
            rows.append(SourceDeclaration(f"{spec.source_id}:pending", SourceRole.SPOOL, pending, True))
    unique: dict[str, SourceDeclaration] = {row.source_id: row for row in rows}
    frontier = build_source_frontier(unique.values(), initial_refusals=initial_refusals)
    for spool_root, expected in observed_spools:
        try:
            after_identity = spool_root.lstat()
        except OSError as exc:
            raise SourceContinuityError(f"hook spool root disappeared during observation: {spool_root}: {exc}") from exc
        if (after_identity.st_dev, after_identity.st_ino) != expected:
            raise SourceContinuityError(f"hook spool root identity changed during observation: {spool_root}")
    return frontier


def _source_path_present(path: Path) -> bool:
    """Treat only a definite missing optional child as absent."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # Keep permission and I/O failures in the declaration set so the
        # frontier records the child as UNAVAILABLE instead of hiding it.
        return True
    return True


def canonical_source_declarations(
    *,
    configured: Iterable[SourceDeclaration] = (),
    hook_primary: Path | None = None,
    hook_legacy: Iterable[Path] = (),
    restored_spools: Iterable[Path] = (),
    browser_queue: Path | None = None,
    attachments: Path | None = None,
    sidecars: Iterable[Path] = (),
    provider_caches: Iterable[Path] = (),
    exports: Iterable[Path] = (),
    live_sources: Iterable[Path] = (),
) -> tuple[SourceDeclaration, ...]:
    """Assemble the single typed declaration for all source families."""
    rows = list(configured)
    if hook_primary is not None:
        rows.append(SourceDeclaration("hooks-primary", SourceRole.SPOOL, hook_primary, True))
    rows.extend(
        SourceDeclaration(f"hooks-legacy-{n}", SourceRole.SPOOL, path, True) for n, path in enumerate(hook_legacy)
    )
    rows.extend(
        SourceDeclaration(f"restored-spool-{n}", SourceRole.SPOOL, path, True) for n, path in enumerate(restored_spools)
    )
    if browser_queue is not None:
        rows.append(SourceDeclaration("browser-queue", SourceRole.QUEUE, browser_queue, True))
    if attachments is not None:
        rows.append(SourceDeclaration("attachments", SourceRole.ATTACHMENT, attachments))
    rows.extend(SourceDeclaration(f"sidecar-{n}", SourceRole.SIDECAR, path, True) for n, path in enumerate(sidecars))
    rows.extend(
        SourceDeclaration(f"provider-cache-{n}", SourceRole.PROVIDER_CACHE, path, True)
        for n, path in enumerate(provider_caches)
    )
    rows.extend(SourceDeclaration(f"export-{n}", SourceRole.IMMUTABLE_EXPORT, path) for n, path in enumerate(exports))
    rows.extend(
        SourceDeclaration(f"live-source-{n}", SourceRole.DIRECTORY, path, True) for n, path in enumerate(live_sources)
    )
    ids = [row.source_id for row in rows]
    roots = [Path(row.root).resolve(strict=False) for row in rows]
    if len(ids) != len(set(ids)):
        raise SourceContinuityError("duplicate source_id in canonical source declaration")
    if len(roots) != len(set(roots)):
        raise SourceContinuityError("duplicate root in canonical source declaration")
    # Resolved roots are for duplicate detection only; the declaration keeps the
    # configured absolute path so a symlinked root still reaches the source
    # snapshot's fail-closed root refusal instead of being replaced by its target.
    return tuple(
        SourceDeclaration(
            row.source_id,
            row.role,
            Path(row.root).absolute(),
            row.mutable,
            row.layout_name,
            row.exclude_coordinates,
        )
        for row in rows
    )
