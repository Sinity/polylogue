"""Replayable, run-scoped observation storage for schema inference."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal, Self, TypeAlias, overload

from polylogue.core.json import JSONValue
from polylogue.paths import cache_home
from polylogue.schemas.generation.models import _UnitMembership
from polylogue.schemas.observation import SchemaUnit

ObservationTerminalStatus: TypeAlias = Literal[
    "included",
    "intentionally_excluded",
    "decode_failed",
    "unsupported",
    "quarantined",
]

_JOURNAL_FORMAT = 1
_DEFAULT_STALE_AGE_S = 24 * 60 * 60
_ROOT_MODE = 0o700
_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class ObservationTerminal:
    """One raw artifact's terminal schema-observation outcome."""

    raw_id: str
    status: ObservationTerminalStatus
    artifact_kind: str | None
    source_path: str | None
    reason: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_json(value: bytes) -> JSONValue:
    decoded = json.loads(value)
    if decoded is None or isinstance(decoded, str | int | float | bool | list | dict):
        return decoded
    raise TypeError(f"Observation journal contains a non-JSON value: {type(decoded).__name__}")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _prepare_root(
    root: Path,
    *,
    forbidden_roots: Iterable[Path],
    tighten_permissions: bool,
) -> Path:
    resolved = root.expanduser().resolve(strict=False)
    for forbidden in forbidden_roots:
        forbidden_resolved = forbidden.expanduser().resolve(strict=False)
        if _is_within(resolved, forbidden_resolved):
            raise ValueError(f"Observation journal root is inside a forbidden data root: {resolved}")

    if resolved.exists():
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Observation journal root is not a real directory: {resolved}")
    else:
        resolved.mkdir(mode=_ROOT_MODE, parents=True)

    metadata = resolved.lstat()
    if metadata.st_uid != os.geteuid():
        raise PermissionError(f"Observation journal root is not owned by the current user: {resolved}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        if not tighten_permissions:
            raise PermissionError(f"Observation journal root permissions are broader than 0700: {resolved}")
        resolved.chmod(_ROOT_MODE)
    return resolved


def _delete_journal_files(path: Path) -> None:
    for candidate in (
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
        path,
    ):
        candidate.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _journal_owner_pid(path: Path) -> int | None:
    parts = path.stem.split("-", maxsplit=2)
    if len(parts) != 3 or parts[0] != "run":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def recover_stale_journals(
    root: Path | None = None,
    *,
    now_s: float | None = None,
    minimum_age_s: float = _DEFAULT_STALE_AGE_S,
) -> list[Path]:
    """Remove journals whose owning process is dead and age exceeds the guard."""
    journal_root = (root or cache_home() / "schema-observation-journals").expanduser().resolve(strict=False)
    if not journal_root.is_dir():
        return []
    current_time = now_s if now_s is not None else __import__("time").time()
    removed: list[Path] = []
    for path in sorted(journal_root.glob("run-*-*.sqlite3")):
        owner_pid = _journal_owner_pid(path)
        if owner_pid is None or _pid_is_alive(owner_pid):
            continue
        try:
            age = current_time - path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < minimum_age_s:
            continue
        _delete_journal_files(path)
        removed.append(path)
    return removed


class ObservationJournal:
    """A replayable SQLite spool owned by one schema-generation run."""

    def __init__(self, *, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        root: Path | None = None,
        forbidden_roots: Iterable[Path] = (),
        tighten_permissions: bool = True,
        owner_pid: int | None = None,
    ) -> Self:
        """Create a permission-restricted journal without touching archive tiers."""
        journal_root = _prepare_root(
            root or cache_home() / "schema-observation-journals",
            forbidden_roots=forbidden_roots,
            tighten_permissions=tighten_permissions,
        )
        pid = owner_pid if owner_pid is not None else os.getpid()
        descriptor, raw_path = tempfile.mkstemp(
            dir=journal_root,
            prefix=f"run-{pid}-{uuid.uuid4().hex}-",
            suffix=".sqlite3",
        )
        path = Path(raw_path)
        try:
            os.fchmod(descriptor, _FILE_MODE)
        finally:
            os.close(descriptor)

        try:
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;
                PRAGMA temp_store = FILE;
                PRAGMA cache_size = -65536;

                CREATE TABLE journal_meta (
                    format_version INTEGER NOT NULL,
                    owner_pid INTEGER NOT NULL
                ) STRICT;

                CREATE TABLE units (
                    unit_id INTEGER PRIMARY KEY,
                    cluster_payload_json BLOB NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    session_id TEXT,
                    raw_id TEXT,
                    source_path TEXT,
                    bundle_scope TEXT,
                    observed_at TEXT,
                    exact_structure_id TEXT NOT NULL,
                    profile_tokens_json BLOB NOT NULL,
                    profile_family_id TEXT,
                    package_family_id TEXT
                ) STRICT;

                CREATE TABLE samples (
                    unit_id INTEGER NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    sample_json BLOB NOT NULL,
                    PRIMARY KEY (unit_id, position)
                ) STRICT, WITHOUT ROWID;

                CREATE TABLE artifact_terminals (
                    raw_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN (
                        'included', 'intentionally_excluded', 'decode_failed', 'unsupported', 'quarantined'
                    )),
                    artifact_kind TEXT,
                    source_path TEXT,
                    reason TEXT
                ) STRICT, WITHOUT ROWID;

                CREATE INDEX units_artifact_kind_idx ON units(artifact_kind, unit_id);
                CREATE INDEX units_bundle_scope_idx ON units(bundle_scope, unit_id);
                CREATE INDEX units_profile_family_idx ON units(profile_family_id, unit_id);
                CREATE INDEX units_package_family_idx ON units(package_family_id, artifact_kind, unit_id);
                """
            )
            connection.execute(
                "INSERT INTO journal_meta(format_version, owner_pid) VALUES (?, ?)",
                (_JOURNAL_FORMAT, pid),
            )
            connection.commit()
        except BaseException:
            _delete_journal_files(path)
            raise
        return cls(path=path, connection=connection)

    def append_unit(self, unit: SchemaUnit, *, retain_cluster_payload: bool = True) -> int:
        """Append one canonical unit and its independently replayable samples."""
        cursor = self._connection.execute(
            """
            INSERT INTO units(
                cluster_payload_json, artifact_kind, session_id, raw_id, source_path,
                bundle_scope, observed_at, exact_structure_id, profile_tokens_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _canonical_json(unit.cluster_payload if retain_cluster_payload else {}),
                unit.artifact_kind,
                unit.session_id,
                unit.raw_id,
                unit.source_path,
                unit.bundle_scope,
                unit.observed_at,
                unit.exact_structure_id,
                _canonical_json(list(unit.profile_tokens)),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("Observation journal did not return a unit identity")
        unit_id = cursor.lastrowid
        self._connection.executemany(
            "INSERT INTO samples(unit_id, position, sample_json) VALUES (?, ?, ?)",
            ((unit_id, position, _canonical_json(sample)) for position, sample in enumerate(unit.schema_samples)),
        )
        return unit_id

    def record_terminal(
        self,
        *,
        raw_id: str,
        status: ObservationTerminalStatus,
        artifact_kind: str | None,
        source_path: str | None,
        reason: str | None,
    ) -> None:
        """Record exactly one terminal inclusion/exclusion outcome for an artifact."""
        self._connection.execute(
            """
            INSERT INTO artifact_terminals(raw_id, status, artifact_kind, source_path, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (raw_id, status, artifact_kind, source_path, reason),
        )

    def iter_units(self) -> Iterator[SchemaUnit]:
        """Replay units in stable ingestion order without retaining the corpus."""
        for _unit_id, unit in self.iter_identified_units():
            yield unit

    def _unit_from_row(self, row: sqlite3.Row, *, include_cluster_payload: bool = True) -> SchemaUnit:
        sample_values = []
        for sample_row in self._connection.execute(
            "SELECT sample_json FROM samples WHERE unit_id = ? ORDER BY position",
            (row["unit_id"],),
        ):
            sample = _decode_json(sample_row["sample_json"])
            if not isinstance(sample, dict):
                raise TypeError("Observation journal schema sample is not an object")
            sample_values.append(sample)
        profile_tokens = _decode_json(row["profile_tokens_json"])
        if not isinstance(profile_tokens, list) or not all(isinstance(item, str) for item in profile_tokens):
            raise TypeError("Observation journal profile tokens are not a string list")
        return SchemaUnit(
            cluster_payload=_decode_json(row["cluster_payload_json"]) if include_cluster_payload else {},
            schema_samples=sample_values,
            artifact_kind=str(row["artifact_kind"]),
            session_id=row["session_id"],
            raw_id=row["raw_id"],
            source_path=row["source_path"],
            bundle_scope=row["bundle_scope"],
            observed_at=row["observed_at"],
            exact_structure_id=str(row["exact_structure_id"]),
            profile_tokens=tuple(str(item) for item in profile_tokens),
        )

    def iter_identified_units(self) -> Iterator[tuple[int, SchemaUnit]]:
        """Replay journal identities and units without retaining prior rows."""
        for row in self._connection.execute("SELECT * FROM units ORDER BY unit_id"):
            yield int(row["unit_id"]), self._unit_from_row(row)

    def assign_profile_family(self, unit_id: int, profile_family_id: str) -> None:
        self._connection.execute(
            "UPDATE units SET profile_family_id = ? WHERE unit_id = ?",
            (profile_family_id, unit_id),
        )

    def normalize_profile_families(self, replacements: dict[str, str]) -> None:
        changed = [(source, target) for source, target in replacements.items() if source != target]
        if not changed:
            return
        self._connection.execute(
            "CREATE TEMP TABLE profile_family_replacements(source TEXT PRIMARY KEY, target TEXT NOT NULL) STRICT"
        )
        try:
            self._connection.executemany(
                "INSERT INTO profile_family_replacements(source, target) VALUES (?, ?)",
                changed,
            )
            self._connection.execute(
                """
                UPDATE units
                SET profile_family_id = (
                    SELECT target
                    FROM profile_family_replacements
                    WHERE source = units.profile_family_id
                )
                WHERE profile_family_id IN (SELECT source FROM profile_family_replacements)
                """
            )
        finally:
            self._connection.execute("DROP TABLE profile_family_replacements")

    def assign_package_family(self, unit_id: int, package_family_id: str) -> None:
        self._connection.execute(
            "UPDATE units SET package_family_id = ? WHERE unit_id = ?",
            (package_family_id, unit_id),
        )

    def memberships(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> JournalMemberships:
        return JournalMemberships(
            self,
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )

    def _membership_where(
        self,
        *,
        profile_family_id: str | None,
        package_family_id: str | None,
        artifact_kind: str | None,
    ) -> tuple[str, list[str]]:
        predicates = ["profile_family_id IS NOT NULL"]
        parameters: list[str] = []
        for column, value in (
            ("profile_family_id", profile_family_id),
            ("package_family_id", package_family_id),
            ("artifact_kind", artifact_kind),
        ):
            if value is not None:
                predicates.append(f"{column} = ?")
                parameters.append(value)
        return " AND ".join(predicates), parameters

    def iter_identified_memberships(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
        scope_order: bool = False,
    ) -> Iterator[tuple[int, _UnitMembership]]:
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        order = (
            "COALESCE(bundle_scope, raw_id, source_path, profile_family_id || ':' || artifact_kind || ':' || exact_structure_id), "
            "COALESCE(observed_at, ''), COALESCE(source_path, ''), profile_family_id, unit_id"
            if scope_order
            else "unit_id"
        )
        query = f"SELECT * FROM units WHERE {where} ORDER BY {order}"
        for row in self._connection.execute(query, parameters):
            yield (
                int(row["unit_id"]),
                _UnitMembership(
                    unit=self._unit_from_row(row, include_cluster_payload=False),
                    profile_family_id=str(row["profile_family_id"]),
                ),
            )

    def membership_count(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> int:
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        query = f"SELECT COUNT(*) FROM units WHERE {where}"
        return int(self._connection.execute(query, parameters).fetchone()[0])

    def iter_terminals(self) -> Iterator[ObservationTerminal]:
        """Replay terminal artifact outcomes in stable raw identity order."""
        for row in self._connection.execute("SELECT * FROM artifact_terminals ORDER BY raw_id"):
            yield ObservationTerminal(
                raw_id=str(row["raw_id"]),
                status=row["status"],
                artifact_kind=row["artifact_kind"],
                source_path=row["source_path"],
                reason=row["reason"],
            )

    @property
    def unit_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM units").fetchone()[0])

    @property
    def sample_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def terminal_counts(self) -> dict[str, int]:
        return dict(
            Counter(
                {
                    str(row[0]): int(row[1])
                    for row in self._connection.execute(
                        "SELECT status, COUNT(*) FROM artifact_terminals GROUP BY status"
                    )
                }
            )
        )

    def close(self) -> None:
        """Close SQLite and remove every journal sidecar exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.commit()
        finally:
            try:
                self._connection.close()
            finally:
                _delete_journal_files(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class JournalMemberships(Sequence[_UnitMembership]):
    """Replayable membership view backed by one observation journal."""

    def __init__(
        self,
        journal: ObservationJournal,
        *,
        profile_family_id: str | None,
        package_family_id: str | None,
        artifact_kind: str | None,
    ) -> None:
        self._journal = journal
        self._profile_family_id = profile_family_id
        self._package_family_id = package_family_id
        self._artifact_kind = artifact_kind

    def __len__(self) -> int:
        return self._journal.membership_count(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def __iter__(self) -> Iterator[_UnitMembership]:
        for _unit_id, membership in self._journal.iter_identified_memberships(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        ):
            yield membership

    @overload
    def __getitem__(self, index: int) -> _UnitMembership: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_UnitMembership]: ...

    def __getitem__(self, index: int | slice) -> _UnitMembership | Sequence[_UnitMembership]:
        if isinstance(index, slice):
            return list(self)[index]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0:
            raise IndexError(index)
        try:
            return next(islice(iter(self), normalized, normalized + 1))
        except StopIteration as error:
            raise IndexError(index) from error


__all__ = [
    "ObservationJournal",
    "JournalMemberships",
    "ObservationTerminal",
    "ObservationTerminalStatus",
    "recover_stale_journals",
]
