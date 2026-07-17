"""Replayable, run-scoped observation storage for schema inference."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import stat
import tempfile
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import FrameType
from typing import Literal, NoReturn, Self, overload

from polylogue.core.json import JSONDocument, JSONDocumentList, JSONValue, json_document
from polylogue.paths import cache_home
from polylogue.schemas.generation.models import _ProfileSummary, _UnitMembership
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.observation_models import ObservationTerminalStatus

_JOURNAL_FORMAT = 1
_DEFAULT_STALE_AGE_S = 24 * 60 * 60
_ROOT_MODE = 0o700
_FILE_MODE = 0o600
_COMMIT_UNIT_LIMIT = 1_024
_COMMIT_PAYLOAD_BYTES_LIMIT = 32 * 1024 * 1024
_DISTINCT_UNIT_COLUMNS = frozenset({"bundle_scope", "exact_structure_id", "profile_family_id"})
DistinctUnitColumn = Literal["bundle_scope", "exact_structure_id", "profile_family_id"]
_SCOPE_KEY_SQL = (
    "COALESCE(bundle_scope, raw_id, source_path, "
    "profile_family_id || ':' || artifact_kind || ':' || exact_structure_id)"
)


@dataclass(frozen=True, slots=True)
class ObservationTerminal:
    """One raw artifact's terminal schema-observation outcome."""

    raw_id: str
    status: ObservationTerminalStatus
    artifact_kind: str | None
    source_path: str | None
    reason: str | None


def _canonical_json(value: object) -> bytes:
    """Serialize provider-derived JSON without rejecting valid decoded escapes.

    Provider JSON may legally contain a lone UTF-16 surrogate escape.  Python
    preserves that code unit in ``str`` but cannot UTF-8 encode it directly.
    ASCII JSON escaping preserves the exact JSON value for replay while keeping
    the private journal valid UTF-8 bytes.
    """
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


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
        self._pending_unit_count = 0
        self._pending_payload_bytes = 0
        self._previous_signal_handlers: dict[signal.Signals, object] = {}

    def _record_pending_write(self, payload_bytes: int = 0) -> None:
        """Bound one private WAL transaction without changing observations.

        The journal is only a run-scoped staging spool: nothing becomes a
        generated package until the enclosing generation returns successfully.
        Committing batches therefore changes neither output atomicity nor
        evidence semantics, but prevents a full archive from becoming one
        uncheckpointable WAL transaction.
        """
        self._pending_payload_bytes += payload_bytes
        if self._pending_unit_count >= _COMMIT_UNIT_LIMIT or self._pending_payload_bytes >= _COMMIT_PAYLOAD_BYTES_LIMIT:
            self.flush()

    def flush(self) -> None:
        """Commit pending private journal evidence before a replay phase."""
        self._connection.commit()
        self._pending_unit_count = 0
        self._pending_payload_bytes = 0

    def _handle_termination(self, signum: int, frame: FrameType | None) -> NoReturn:
        """Turn ordinary termination into stack unwinding through ``__exit__``."""
        del frame
        raise SystemExit(128 + signum)

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGTERM, signal.SIGHUP):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_termination)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_signal_handlers.items():
            if isinstance(handler, signal.Handlers) or callable(handler):
                signal.signal(signum, handler)
            else:
                signal.signal(signum, signal.SIG_DFL)
        self._previous_signal_handlers.clear()

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
                    schema_sample_count INTEGER NOT NULL,
                    profile_family_id TEXT,
                    package_family_id TEXT,
                    package_selected INTEGER NOT NULL DEFAULT 0 CHECK (package_selected IN (0, 1))
                ) STRICT;

                CREATE TABLE samples (
                    unit_id INTEGER NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    sample_json BLOB NOT NULL,
                    PRIMARY KEY (unit_id, position)
                ) STRICT, WITHOUT ROWID;

                CREATE TABLE profile_summaries (
                    artifact_kind TEXT NOT NULL,
                    profile_tokens_json BLOB NOT NULL,
                    dominant_keys_json BLOB NOT NULL,
                    sample_count INTEGER NOT NULL,
                    schema_sample_count INTEGER NOT NULL,
                    profile_family_id TEXT,
                    PRIMARY KEY (artifact_kind, profile_tokens_json)
                ) STRICT, WITHOUT ROWID;

                CREATE TABLE profile_summary_paths (
                    artifact_kind TEXT NOT NULL,
                    profile_tokens_json BLOB NOT NULL,
                    source_path TEXT NOT NULL,
                    PRIMARY KEY (artifact_kind, profile_tokens_json, source_path),
                    FOREIGN KEY (artifact_kind, profile_tokens_json)
                        REFERENCES profile_summaries(artifact_kind, profile_tokens_json)
                        ON DELETE CASCADE
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
                CREATE INDEX units_profile_family_idx ON units(profile_family_id, artifact_kind, unit_id);
                CREATE INDEX units_package_family_idx ON units(package_family_id, package_selected, artifact_kind, unit_id);
                CREATE INDEX profile_summaries_family_idx
                    ON profile_summaries(profile_family_id, artifact_kind, profile_tokens_json);
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
        cluster_payload_json = _canonical_json(unit.cluster_payload if retain_cluster_payload else {})
        profile_tokens_json = _canonical_json(list(unit.profile_tokens))
        sample_payload_bytes = 0

        def sample_rows() -> Iterator[tuple[int, bytes]]:
            nonlocal sample_payload_bytes
            for position, sample in enumerate(unit.schema_samples):
                sample_json = _canonical_json(sample)
                sample_payload_bytes += len(sample_json)
                yield position, sample_json

        cursor = self._connection.execute(
            """
            INSERT INTO units(
                cluster_payload_json, artifact_kind, session_id, raw_id, source_path,
                bundle_scope, observed_at, exact_structure_id, profile_tokens_json,
                schema_sample_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_payload_json,
                unit.artifact_kind,
                unit.session_id,
                unit.raw_id,
                unit.source_path,
                unit.bundle_scope,
                unit.observed_at,
                unit.exact_structure_id,
                profile_tokens_json,
                len(unit.schema_samples),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("Observation journal did not return a unit identity")
        unit_id = cursor.lastrowid
        self._connection.executemany(
            "INSERT INTO samples(unit_id, position, sample_json) VALUES (?, ?, ?)",
            ((unit_id, position, sample_json) for position, sample_json in sample_rows()),
        )
        self._pending_unit_count += 1
        self._record_pending_write(len(cluster_payload_json) + len(profile_tokens_json) + sample_payload_bytes)
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
        self._record_pending_write(
            sum(len(value.encode("utf-8")) for value in (raw_id, artifact_kind, source_path, reason) if value)
        )

    def observe_profile_summary(self, unit: SchemaUnit, *, dominant_keys: Sequence[str]) -> None:
        """Merge one profile observation into a disk-backed exact summary."""
        profile_tokens_json = _canonical_json(list(unit.profile_tokens))
        self._connection.execute(
            """
            INSERT INTO profile_summaries(
                artifact_kind, profile_tokens_json, dominant_keys_json,
                sample_count, schema_sample_count
            ) VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(artifact_kind, profile_tokens_json) DO UPDATE SET
                sample_count = sample_count + 1,
                schema_sample_count = schema_sample_count + excluded.schema_sample_count
            """,
            (
                unit.artifact_kind,
                profile_tokens_json,
                _canonical_json(list(dominant_keys)[:20]),
                len(unit.schema_samples),
            ),
        )
        if unit.source_path:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO profile_summary_paths(
                    artifact_kind, profile_tokens_json, source_path
                )
                SELECT ?, ?, ?
                WHERE (
                    SELECT COUNT(*) FROM profile_summary_paths
                    WHERE artifact_kind = ? AND profile_tokens_json = ?
                ) < 5
                """,
                (
                    unit.artifact_kind,
                    profile_tokens_json,
                    unit.source_path,
                    unit.artifact_kind,
                    profile_tokens_json,
                ),
            )
        self._record_pending_write(
            len(profile_tokens_json) + sum(len(key.encode("utf-8")) for key in dominant_keys[:20])
        )

    def iter_profile_summaries(self) -> Iterator[_ProfileSummary]:
        """Replay exact summaries in deterministic clustering priority order."""
        query = """
            SELECT * FROM profile_summaries
            ORDER BY
                CASE artifact_kind
                    WHEN 'session_document' THEN 120
                    WHEN 'session_record_stream' THEN 120
                    WHEN 'subagent_session_stream' THEN 90
                    ELSE 0
                END DESC,
                sample_count DESC,
                schema_sample_count DESC,
                artifact_kind,
                profile_tokens_json
        """
        for row in self._connection.execute(query):
            profile_tokens = _decode_json(row["profile_tokens_json"])
            dominant_keys = _decode_json(row["dominant_keys_json"])
            if not isinstance(profile_tokens, list) or not all(isinstance(item, str) for item in profile_tokens):
                raise TypeError("Observation journal profile summary tokens are not a string list")
            if not isinstance(dominant_keys, list) or not all(isinstance(item, str) for item in dominant_keys):
                raise TypeError("Observation journal dominant keys are not a string list")
            paths = [
                str(path_row[0])
                for path_row in self._connection.execute(
                    """
                    SELECT source_path FROM profile_summary_paths
                    WHERE artifact_kind = ? AND profile_tokens_json = ?
                    ORDER BY source_path
                    """,
                    (row["artifact_kind"], row["profile_tokens_json"]),
                )
            ]
            yield _ProfileSummary(
                artifact_kind=str(row["artifact_kind"]),
                profile_tokens=tuple(str(item) for item in profile_tokens),
                dominant_keys=[str(item) for item in dominant_keys],
                sample_count=int(row["sample_count"]),
                schema_sample_count=int(row["schema_sample_count"]),
                representative_paths=paths,
            )

    def assign_profile_summary_family(self, summary: _ProfileSummary, profile_family_id: str) -> None:
        self._connection.execute(
            """
            UPDATE profile_summaries SET profile_family_id = ?
            WHERE artifact_kind = ? AND profile_tokens_json = ?
            """,
            (profile_family_id, summary.artifact_kind, _canonical_json(list(summary.profile_tokens))),
        )

    def apply_profile_summary_families_to_units(self) -> None:
        """Resolve every unit through its exact disk-backed profile summary."""
        self._connection.execute(
            """
            UPDATE units SET profile_family_id = (
                SELECT profile_family_id FROM profile_summaries
                WHERE profile_summaries.artifact_kind = units.artifact_kind
                  AND profile_summaries.profile_tokens_json = units.profile_tokens_json
            )
            """
        )

    def iter_units(self) -> Iterator[SchemaUnit]:
        """Replay units in stable ingestion order without retaining the corpus."""
        for _unit_id, unit in self.iter_identified_units():
            yield unit

    def _unit_from_row(
        self,
        row: sqlite3.Row,
        *,
        include_cluster_payload: bool = True,
        include_samples: bool = True,
        replayed_samples: JSONDocumentList | None = None,
    ) -> SchemaUnit:
        sample_values: JSONDocumentList = replayed_samples or []
        if include_samples and replayed_samples is None:
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

    def iter_identified_unit_metadata(self) -> Iterator[tuple[int, SchemaUnit, int]]:
        """Replay unit metadata and exact sample counts without decoding samples."""
        for row in self._connection.execute("SELECT * FROM units ORDER BY unit_id"):
            yield (
                int(row["unit_id"]),
                self._unit_from_row(row, include_cluster_payload=False, include_samples=False),
                int(row["schema_sample_count"]),
            )

    def iter_identified_membership_metadata(self) -> Iterator[tuple[int, _UnitMembership, int]]:
        """Replay assigned membership metadata with exact sample counts."""
        for row in self._connection.execute("SELECT * FROM units WHERE profile_family_id IS NOT NULL ORDER BY unit_id"):
            yield (
                int(row["unit_id"]),
                _UnitMembership(
                    unit=self._unit_from_row(row, include_cluster_payload=False, include_samples=False),
                    profile_family_id=str(row["profile_family_id"]),
                ),
                int(row["schema_sample_count"]),
            )

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
                UPDATE profile_summaries
                SET profile_family_id = (
                    SELECT target
                    FROM profile_family_replacements
                    WHERE source = profile_summaries.profile_family_id
                )
                WHERE profile_family_id IN (SELECT source FROM profile_family_replacements)
                """
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
            "UPDATE units SET package_family_id = ?, package_selected = 1 WHERE unit_id = ?",
            (package_family_id, unit_id),
        )

    def assign_canonical_package_families(self, anchor_kinds: frozenset[str]) -> dict[str, int]:
        """Assign one canonical unit per scope/kind/structure in SQLite.

        Package assembly used to collect every unit in a scope into Python merely
        to remove duplicate structural observations and choose a sole anchor.
        The journal already owns the necessary identity columns, so retain that
        relation here and expose only the resulting replay views.
        """
        if not anchor_kinds:
            raise ValueError("Package assignment needs at least one anchor artifact kind")
        placeholders = ", ".join("?" for _ in anchor_kinds)
        anchor_parameters = tuple(sorted(anchor_kinds))
        scope_key = _SCOPE_KEY_SQL
        self._connection.execute("UPDATE units SET package_family_id = NULL, package_selected = 0")
        self._connection.execute(
            f"""
            WITH canonical AS (
                SELECT
                    unit_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY {scope_key}, artifact_kind, exact_structure_id
                        ORDER BY COALESCE(observed_at, ''), COALESCE(source_path, ''), unit_id
                    ) AS row_number
                FROM units
                WHERE profile_family_id IS NOT NULL
            )
            UPDATE units
            SET package_family_id = profile_family_id, package_selected = 1
            WHERE unit_id IN (
                SELECT unit_id FROM canonical
                WHERE row_number = 1 AND artifact_kind IN ({placeholders})
            )
            """,
            anchor_parameters,
        )
        self._connection.execute(
            f"""
            WITH canonical AS (
                SELECT
                    unit_id,
                    artifact_kind,
                    {scope_key} AS scope_key,
                    ROW_NUMBER() OVER (
                        PARTITION BY {scope_key}, artifact_kind, exact_structure_id
                        ORDER BY COALESCE(observed_at, ''), COALESCE(source_path, ''), unit_id
                    ) AS row_number
                FROM units
                WHERE profile_family_id IS NOT NULL
            ), scope_anchors AS (
                SELECT
                    {scope_key} AS scope_key,
                    MIN(package_family_id) AS package_family_id,
                    COUNT(DISTINCT package_family_id) AS family_count
                FROM units
                WHERE package_selected = 1 AND artifact_kind IN ({placeholders})
                GROUP BY {scope_key}
            )
            UPDATE units
            SET package_family_id = (
                SELECT package_family_id FROM scope_anchors
                WHERE scope_anchors.scope_key = {scope_key}
            ), package_selected = 1
            WHERE unit_id IN (
                SELECT canonical.unit_id
                FROM canonical JOIN scope_anchors USING(scope_key)
                WHERE canonical.row_number = 1
                  AND canonical.artifact_kind NOT IN ({placeholders})
                  AND scope_anchors.family_count = 1
            )
            """,
            (*anchor_parameters, *anchor_parameters),
        )
        rows = self._connection.execute(
            f"""
            WITH canonical AS (
                SELECT
                    artifact_kind,
                    {scope_key} AS scope_key,
                    ROW_NUMBER() OVER (
                        PARTITION BY {scope_key}, artifact_kind, exact_structure_id
                        ORDER BY COALESCE(observed_at, ''), COALESCE(source_path, ''), unit_id
                    ) AS row_number
                FROM units
                WHERE profile_family_id IS NOT NULL
            ), scope_anchors AS (
                SELECT {scope_key} AS scope_key, COUNT(DISTINCT package_family_id) AS family_count
                FROM units
                WHERE package_selected = 1 AND artifact_kind IN ({placeholders})
                GROUP BY {scope_key}
            )
            SELECT canonical.artifact_kind, COUNT(*) AS count
            FROM canonical LEFT JOIN scope_anchors USING(scope_key)
            WHERE canonical.row_number = 1
              AND canonical.artifact_kind NOT IN ({placeholders})
              AND COALESCE(scope_anchors.family_count, 0) != 1
            GROUP BY canonical.artifact_kind
            """,
            (*anchor_parameters, *anchor_parameters),
        )
        return {str(row["artifact_kind"]): int(row["count"]) for row in rows}

    def iter_package_family_ids(self) -> Iterator[str]:
        for row in self._connection.execute(
            """
            SELECT DISTINCT package_family_id FROM units
            WHERE package_selected = 1 AND package_family_id IS NOT NULL
            ORDER BY package_family_id
            """
        ):
            yield str(row[0])

    def package_metadata(self, package_family_id: str) -> tuple[str | None, str | None, list[str]]:
        """Return bounded metadata for one already-assigned output package."""
        where = "package_selected = 1 AND package_family_id = ?"
        first_seen, last_seen = self._connection.execute(
            f"SELECT MIN(observed_at), MAX(observed_at) FROM units WHERE {where}", (package_family_id,)
        ).fetchone()
        paths = [
            str(row[0])
            for row in self._connection.execute(
                f"""
                SELECT DISTINCT source_path FROM units
                WHERE {where} AND source_path IS NOT NULL AND source_path != ''
                ORDER BY source_path LIMIT 5
                """,
                (package_family_id,),
            )
        ]
        return (
            str(first_seen) if first_seen is not None else None,
            str(last_seen) if last_seen is not None else None,
            paths,
        )

    def memberships(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
        include_samples: bool = True,
    ) -> JournalMemberships:
        return JournalMemberships(
            self,
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
            include_samples=include_samples,
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
        if package_family_id is not None:
            predicates.append("package_selected = 1")
        return " AND ".join(predicates), parameters

    def iter_identified_memberships(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
        scope_order: bool = False,
        include_samples: bool = True,
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
        if include_samples:
            yield from self._iter_joined_memberships(where, parameters)
            return
        query = f"SELECT * FROM units WHERE {where} ORDER BY {order}"
        for row in self._connection.execute(query, parameters):
            yield (
                int(row["unit_id"]),
                _UnitMembership(
                    unit=self._unit_from_row(
                        row,
                        include_cluster_payload=False,
                        include_samples=include_samples,
                    ),
                    profile_family_id=str(row["profile_family_id"]),
                ),
            )

    def _iter_joined_memberships(
        self,
        where: str,
        parameters: list[str],
    ) -> Iterator[tuple[int, _UnitMembership]]:
        """Replay samples with their units in one ordered SQLite scan."""
        joined_query = (
            "SELECT units.*, samples.position AS replay_position, "
            "samples.sample_json AS replay_sample_json "
            "FROM units JOIN samples ON samples.unit_id = units.unit_id "
            f"WHERE {where} ORDER BY samples.unit_id, samples.position"
        )
        current_row: sqlite3.Row | None = None
        current_unit_id: int | None = None
        sample_values: JSONDocumentList = []
        for row in self._connection.execute(joined_query, parameters):
            unit_id = int(row["unit_id"])
            if current_unit_id is not None and unit_id != current_unit_id and current_row is not None:
                yield (
                    current_unit_id,
                    _UnitMembership(
                        unit=self._unit_from_row(
                            current_row,
                            include_cluster_payload=False,
                            replayed_samples=sample_values,
                        ),
                        profile_family_id=str(current_row["profile_family_id"]),
                    ),
                )
                sample_values = []
            current_row = row
            current_unit_id = unit_id
            encoded_sample = row["replay_sample_json"]
            if encoded_sample is not None:
                sample = _decode_json(encoded_sample)
                if not isinstance(sample, dict):
                    raise TypeError("Observation journal schema sample is not an object")
                sample_values.append(sample)
        if current_unit_id is not None and current_row is not None:
            yield (
                current_unit_id,
                _UnitMembership(
                    unit=self._unit_from_row(
                        current_row,
                        include_cluster_payload=False,
                        replayed_samples=sample_values,
                    ),
                    profile_family_id=str(current_row["profile_family_id"]),
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

    def membership_sample_count(
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
        query = f"SELECT COUNT(*) FROM samples JOIN units USING(unit_id) WHERE {where}"
        return int(self._connection.execute(query, parameters).fetchone()[0])

    def iter_distinct_membership_values(
        self,
        column: DistinctUnitColumn,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> Iterator[str]:
        """Stream one exact metadata dimension without a Python dedupe set."""
        if column not in _DISTINCT_UNIT_COLUMNS:
            raise ValueError(f"Unsupported journal distinct column: {column}")
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        query = (
            f'SELECT DISTINCT "{column}" FROM units '
            f'WHERE {where} AND "{column}" IS NOT NULL AND "{column}" != ? '
            f'ORDER BY "{column}"'
        )
        for row in self._connection.execute(query, [*parameters, ""]):
            yield str(row[0])

    def distinct_membership_count(
        self,
        column: DistinctUnitColumn,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> int:
        """Count an exact metadata dimension in SQLite rather than retaining it."""
        if column not in _DISTINCT_UNIT_COLUMNS:
            raise ValueError(f"Unsupported journal distinct column: {column}")
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        query = (
            f'SELECT COUNT(DISTINCT "{column}") FROM units WHERE {where} AND "{column}" IS NOT NULL AND "{column}" != ?'
        )
        return int(self._connection.execute(query, [*parameters, ""]).fetchone()[0])

    def iter_distinct_membership_scope_keys(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> Iterator[str]:
        """Stream the exact package scope identity used by package assembly."""
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        query = f"SELECT DISTINCT {_SCOPE_KEY_SQL} AS scope_key FROM units WHERE {where} ORDER BY scope_key"
        for row in self._connection.execute(query, parameters):
            yield str(row[0])

    def distinct_membership_scope_count(
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
        query = f"SELECT COUNT(DISTINCT {_SCOPE_KEY_SQL}) FROM units WHERE {where}"
        return int(self._connection.execute(query, parameters).fetchone()[0])

    def iter_membership_session_ids(
        self,
        *,
        profile_family_id: str | None = None,
        package_family_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> Iterator[str | None]:
        """Replay one session identity per sample without decoding sample JSON."""
        where, parameters = self._membership_where(
            profile_family_id=profile_family_id,
            package_family_id=package_family_id,
            artifact_kind=artifact_kind,
        )
        query = (
            "SELECT units.session_id FROM samples JOIN units USING(unit_id) "
            f"WHERE {where} ORDER BY units.unit_id, samples.position"
        )
        for row in self._connection.execute(query, parameters):
            yield row[0]

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

    def terminal_summary(self) -> JSONDocument:
        """Return privacy-safe typed loss counts without raw identities or paths."""
        reasons = {
            str(row[0]): int(row[1])
            for row in self._connection.execute(
                "SELECT reason, COUNT(*) FROM artifact_terminals WHERE reason IS NOT NULL GROUP BY reason"
            )
        }
        return json_document(
            {
                "total": int(self._connection.execute("SELECT COUNT(*) FROM artifact_terminals").fetchone()[0]),
                "status_counts": self.terminal_counts(),
                "reason_counts": reasons,
            }
        )

    def close(self) -> None:
        """Close SQLite and remove every journal sidecar exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            try:
                self._connection.close()
            finally:
                try:
                    _delete_journal_files(self.path)
                finally:
                    self._restore_signal_handlers()

    def __enter__(self) -> Self:
        self._install_signal_handlers()
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
        include_samples: bool = True,
    ) -> None:
        self._journal = journal
        self._profile_family_id = profile_family_id
        self._package_family_id = package_family_id
        self._artifact_kind = artifact_kind
        self._include_samples = include_samples

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
            include_samples=self._include_samples,
        ):
            yield membership

    @property
    def sample_count(self) -> int:
        return self._journal.membership_sample_count(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def iter_session_ids(self) -> Iterator[str | None]:
        return self._journal.iter_membership_session_ids(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def iter_distinct_values(self, column: DistinctUnitColumn) -> Iterator[str]:
        return self._journal.iter_distinct_membership_values(
            column,
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def distinct_count(self, column: DistinctUnitColumn) -> int:
        return self._journal.distinct_membership_count(
            column,
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def iter_scope_keys(self) -> Iterator[str]:
        return self._journal.iter_distinct_membership_scope_keys(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def scope_count(self) -> int:
        return self._journal.distinct_membership_scope_count(
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
        )

    def metadata(self) -> JournalMemberships:
        return JournalMemberships(
            self._journal,
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=self._artifact_kind,
            include_samples=False,
        )

    def for_artifact(self, artifact_kind: str) -> JournalMemberships:
        if self._artifact_kind is not None and self._artifact_kind != artifact_kind:
            raise ValueError(f"Membership view is already filtered to {self._artifact_kind}")
        return JournalMemberships(
            self._journal,
            profile_family_id=self._profile_family_id,
            package_family_id=self._package_family_id,
            artifact_kind=artifact_kind,
            include_samples=self._include_samples,
        )

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
    "DistinctUnitColumn",
    "ObservationJournal",
    "JournalMemberships",
    "ObservationTerminal",
    "ObservationTerminalStatus",
    "recover_stale_journals",
]
