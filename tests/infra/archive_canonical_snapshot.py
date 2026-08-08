"""Canonical semantic snapshots for real archive-route comparisons.

The comparator is deliberately a read-only adapter around the production
SQLite archive and ``ArchiveStore`` read surfaces.  It compares semantic rows
and public projections, rather than database bytes or generation-local files.
Only fields named in :data:`RUN_LOCAL_NORMALIZATION_ALLOWLIST` are omitted;
provider identity, semantic timestamps, provenance, authority, and result
states remain part of the comparison.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

SqlValue = str | int | float | None
FactRow = tuple[SqlValue, ...]


@dataclass(frozen=True, slots=True)
class RelationSnapshot:
    """A deterministic projection of one table or view."""

    database: str
    relation: str
    exists: bool
    columns: tuple[str, ...]
    rows: tuple[FactRow, ...]

    @property
    def relation_key(self) -> tuple[str, str]:
        return self.database, self.relation


@dataclass(frozen=True, slots=True)
class CanonicalArchiveSnapshot:
    """The semantic archive state used by every route-equivalence proof."""

    canonical_rows: tuple[RelationSnapshot, ...]
    provenance: tuple[RelationSnapshot, ...]
    authority: tuple[RelationSnapshot, ...]
    user_state: tuple[RelationSnapshot, ...]
    links: tuple[RelationSnapshot, ...]
    attachments: tuple[RelationSnapshot, ...]
    derived_views: tuple[RelationSnapshot, ...]
    public_projections: tuple[tuple[str, object], ...]


# This is intentionally a field-level allowlist, not a rule such as "ignore
# all timestamps".  Acquisition, authored-content, authority, and evidence
# timestamps are semantic facts and must survive route comparison.
RUN_LOCAL_NORMALIZATION_ALLOWLIST: Mapping[str, frozenset[str]] = {
    "index.insight_materialization": frozenset({"materialized_at_ms"}),
    "index.session_links": frozenset({"observed_at_ms", "resolved_at_ms"}),
    "index.session_profiles": frozenset({"materialized_at", "priced_at_ms"}),
    "index.session_latency_profiles": frozenset({"materialized_at"}),
    "index.session_tag_rollups": frozenset({"materialized_at"}),
    "index.threads": frozenset({"materialized_at"}),
    "index.fts_freshness_state": frozenset({"checked_at"}),
    "source.raw_authority_verdicts": frozenset({"computed_at_ms"}),
    "ops.convergence_debt": frozenset({"debt_id", "created_at_ms", "updated_at_ms", "next_retry_at"}),
}

# Absolute paths are run-local when they point inside a temporary archive.
# They are retained as archive-relative paths so a route comparison still
# catches a wrong source file while ignoring each run's temporary root.
RUN_LOCAL_PATH_ALLOWLIST: Mapping[str, frozenset[str]] = {
    "source.raw_sessions": frozenset({"source_path"}),
    "source.raw_artifacts": frozenset({"source_path"}),
    "source.raw_live_source_reconciliation_receipts": frozenset({"source_path", "backup_manifest_path"}),
    "source.raw_append_chain_backfill_receipts": frozenset({"source_path", "backup_manifest_path"}),
    "source.raw_membership_writeback_receipts": frozenset({"backup_manifest_path"}),
    "source.raw_byte_duplicate_supersession_receipts": frozenset({"backup_manifest_path"}),
    "source.raw_quarantine_group_dedup_receipts": frozenset({"source_path", "backup_manifest_path"}),
    "source.raw_failure_disposition_receipts": frozenset({"source_path", "backup_manifest_path"}),
    "source.blob_refs": frozenset({"source_path"}),
}

PUBLIC_RUN_LOCAL_FIELDS = frozenset({"materialized_at", "priced_at"})

# The categories are intentionally explicit.  A newly relevant relation must
# be named here and reviewed, rather than silently disappearing from a proof.
_RELATION_GROUPS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "canonical_rows": (
        ("index", "sessions"),
        ("index", "messages"),
        ("index", "blocks"),
        ("index", "web_content_constructs"),
    ),
    "provenance": (
        ("index", "session_events"),
        ("index", "session_agent_policies"),
        ("index", "session_working_dirs"),
        ("index", "session_refs"),
        ("index", "session_repos"),
        ("index", "session_commits"),
        ("index", "session_provider_usage_events"),
        ("index", "session_model_usage"),
        ("index", "file_edits"),
        ("index", "paste_spans"),
        ("source", "raw_artifacts"),
        ("source", "raw_hook_events"),
    ),
    "authority": (
        ("index", "raw_revision_heads"),
        ("source", "raw_sessions"),
        ("source", "raw_capture_observations"),
        ("source", "raw_session_memberships"),
        ("source", "raw_membership_census"),
        ("source", "raw_authority_parser_census"),
        ("source", "raw_authority_censuses"),
        ("source", "raw_authority_plans"),
        ("source", "raw_authority_census_plans"),
        ("source", "raw_authority_census_post_plans"),
        ("source", "raw_authority_blockers"),
        ("source", "raw_authority_verdicts"),
        ("source", "excised_content"),
        ("source", "raw_live_source_reconciliation_receipts"),
        ("source", "raw_membership_writeback_receipts"),
        ("source", "raw_append_chain_backfill_receipts"),
        ("source", "raw_byte_duplicate_supersession_receipts"),
        ("source", "raw_non_session_duplicate_exclusion_receipts"),
        ("source", "raw_quarantine_group_dedup_receipts"),
        ("source", "raw_unknown_export_reclassification_receipts"),
        ("source", "raw_failure_disposition_receipts"),
        ("source", "blob_refs"),
        ("source", "verified_blob_receipts"),
    ),
    "user_state": (
        ("user", "assertions"),
        ("user", "queries"),
        ("user", "query_names"),
        ("user", "result_sets"),
        ("user", "result_set_members"),
        ("user", "query_edges"),
        ("user", "retained_query_runs"),
        ("user", "query_evaluation_receipts"),
        ("user", "watched_query_baselines"),
        ("user", "result_set_holdout_policies"),
        ("user", "holdout_access_receipts"),
        ("user", "annotation_schemas"),
        ("user", "annotation_batches"),
        ("user", "user_settings"),
        ("user", "context_deliveries"),
    ),
    "links": (
        ("index", "session_links"),
        ("index", "action_pairs"),
        ("index", "delegation_facts"),
        ("index", "work_evidence_edges"),
        ("index", "work_evidence_nodes"),
        ("index", "work_evidence_graphs"),
    ),
    "attachments": (
        ("index", "attachments"),
        ("index", "attachment_refs"),
        ("index", "attachment_native_ids"),
    ),
    "derived_views": (
        ("index", "session_profiles"),
        ("index", "session_latency_profiles"),
        ("index", "session_work_events"),
        ("index", "session_phases"),
        ("index", "threads"),
        ("index", "thread_sessions"),
        ("index", "session_tags"),
        ("index", "session_tag_rollups"),
        ("index", "repos"),
        ("index", "repo_checkouts"),
        ("index", "insight_materialization"),
        ("index", "actions"),
        ("index", "delegations"),
        ("index", "fts_freshness_state"),
        ("ops", "convergence_debt"),
    ),
}

# These are physical maintenance relations, not semantic archive state.  The
# exclusion is explicit so adding one here is visible in code review.
NON_COMPARABLE_RELATIONS: Mapping[str, str] = {
    "index.messages_fts": "compared through public search projections",
    "index.messages_fts_identity": "compared through indexed block identity projections",
    "index.blocks_command_trigram": "compared through action/public projections",
    "index.session_work_events_fts": "compared through public work-event projections",
    "index.query_unit_frame_state": "cursor invalidation is route history, not archive state",
    "source.raw_revision_applications": "attempt receipt ids and timestamps are run-local",
}


def capture_canonical_snapshot(
    archive_root: Path,
    *,
    session_ids: Sequence[str] | None = None,
    search_queries: Sequence[str] = (),
) -> CanonicalArchiveSnapshot:
    """Capture semantic rows and representative public reads without writes."""

    root = archive_root.expanduser().resolve()
    connections = _open_connections(root)
    try:
        raw_identity_map = _raw_identity_map(connections.get("source"))
        sections = {
            name: tuple(
                _capture_relation(
                    connections.get(database),
                    database,
                    relation,
                    root,
                    raw_identity_map=raw_identity_map,
                )
                for database, relation in relations
            )
            for name, relations in _RELATION_GROUPS.items()
        }
        ids = tuple(session_ids) if session_ids is not None else _session_ids(connections["index"])
        effective_search_queries = tuple(search_queries) or _default_search_queries(connections["index"])
        public = _capture_public_projections(root, ids, effective_search_queries)
        return CanonicalArchiveSnapshot(
            canonical_rows=sections["canonical_rows"],
            provenance=sections["provenance"],
            authority=sections["authority"],
            user_state=sections["user_state"],
            links=sections["links"],
            attachments=sections["attachments"],
            derived_views=sections["derived_views"],
            public_projections=public,
        )
    finally:
        for connection in connections.values():
            connection.close()


def archive_snapshot(
    archive_root: Path,
    *,
    session_ids: Sequence[str] | None = None,
    search_queries: Sequence[str] = (),
) -> CanonicalArchiveSnapshot:
    """Compatibility name for the shared canonical snapshot capture."""

    return capture_canonical_snapshot(archive_root, session_ids=session_ids, search_queries=search_queries)


def diff_canonical_snapshots(expected: CanonicalArchiveSnapshot, actual: CanonicalArchiveSnapshot) -> tuple[str, ...]:
    """Return named semantic differences, preserving the first useful detail."""

    differences: list[str] = []
    for section in (
        "canonical_rows",
        "provenance",
        "authority",
        "user_state",
        "links",
        "attachments",
        "derived_views",
        "public_projections",
    ):
        expected_value = getattr(expected, section)
        actual_value = getattr(actual, section)
        if expected_value == actual_value:
            continue
        if section == "public_projections":
            differences.append(f"public_projections: {_value_difference(expected_value, actual_value)}")
            continue
        expected_relations = {relation.relation_key: relation for relation in expected_value}
        actual_relations = {relation.relation_key: relation for relation in actual_value}
        for key in sorted(set(expected_relations) | set(actual_relations)):
            left = expected_relations.get(key)
            right = actual_relations.get(key)
            if left != right:
                differences.append(f"{section}/{key}: {_relation_difference(left, right)}")
    return tuple(differences)


def assert_canonical_snapshots_equal(expected: CanonicalArchiveSnapshot, actual: CanonicalArchiveSnapshot) -> None:
    """Assert equality with a category/relation diagnostic for red mutations."""

    differences = diff_canonical_snapshots(expected, actual)
    if differences:
        detail = "\n".join(f"  {difference}" for difference in differences[:8])
        raise AssertionError(f"canonical archive snapshots differ:\n{detail}")


def assert_canonical_archives_equal(
    expected_root: Path,
    actual_root: Path,
    *,
    session_ids: Sequence[str] | None = None,
    search_queries: Sequence[str] = (),
) -> None:
    """Capture and compare two archive roots through production read seams."""

    expected = capture_canonical_snapshot(expected_root, session_ids=session_ids, search_queries=search_queries)
    actual = capture_canonical_snapshot(actual_root, session_ids=session_ids, search_queries=search_queries)
    assert_canonical_snapshots_equal(expected, actual)


def assert_archives_equivalent(expected: object, actual: object) -> None:
    """Accept either archive roots or harness objects exposing ``.root``."""

    expected_root = _archive_root(expected)
    actual_root = _archive_root(actual)
    assert_canonical_archives_equal(expected_root, actual_root)


def _open_connections(root: Path) -> dict[str, sqlite3.Connection]:
    paths = {name: root / f"{name}.db" for name in ("index", "source", "user", "ops")}
    connections: dict[str, sqlite3.Connection] = {}
    try:
        for name, path in paths.items():
            if path.exists():
                connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                connection.row_factory = sqlite3.Row
                connections[name] = connection
        if "index" not in connections:
            raise FileNotFoundError(f"archive index database not found: {root / 'index.db'}")
        return connections
    except Exception:
        for connection in connections.values():
            connection.close()
        raise


def _capture_relation(
    connection: sqlite3.Connection | None,
    database: str,
    relation: str,
    root: Path,
    *,
    raw_identity_map: Mapping[str, str],
) -> RelationSnapshot:
    key = f"{database}.{relation}"
    if connection is None:
        return RelationSnapshot(database, relation, False, (), ())
    kind_row = connection.execute(
        "SELECT type FROM sqlite_master WHERE name = ? COLLATE BINARY", (relation,)
    ).fetchone()
    if kind_row is None:
        return RelationSnapshot(database, relation, False, (), ())
    columns = tuple(str(row[1]) for row in connection.execute(f'PRAGMA table_info("{relation}")'))
    omitted = RUN_LOCAL_NORMALIZATION_ALLOWLIST.get(key, frozenset())
    unknown = omitted - set(columns)
    if unknown:
        raise AssertionError(f"run-local normalization names missing columns in {key}: {sorted(unknown)}")
    selected = tuple(column for column in columns if column not in omitted)
    if not selected:
        raise AssertionError(f"canonical relation {key} has no stable columns")
    quoted = ", ".join(f'"{column}"' for column in selected)
    rows = [
        tuple(
            _normalize_value(database, relation, column, value, root, raw_identity_map=raw_identity_map)
            for column, value in zip(selected, row, strict=True)
        )
        for row in connection.execute(f'SELECT {quoted} FROM "{relation}"')
    ]
    return RelationSnapshot(database, relation, True, selected, tuple(sorted(rows, key=repr)))


def _capture_public_projections(
    root: Path, session_ids: Sequence[str], search_queries: Sequence[str]
) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
        for session_id in ids:
            values.extend(
                (
                    (f"summary:{session_id}", _freeze_public(archive.read_summary(session_id))),
                    (f"tree:{session_id}", _freeze_public(archive.get_session_tree(session_id))),
                    (f"profile:{session_id}", _freeze_public(archive.get_session_profile_insight(session_id))),
                    (f"latency:{session_id}", _freeze_public(archive.get_session_latency_profile_insight(session_id))),
                    (f"work-events:{session_id}", _freeze_public(archive.get_session_work_event_insights(session_id))),
                    (f"phases:{session_id}", _freeze_public(archive.get_session_phase_insights(session_id))),
                )
            )
        values.extend(
            (
                ("actions", _freeze_public(archive.query_session_actions(ids, limit=100_000))),
                ("threads", _freeze_public(archive.list_thread_insights(limit=None))),
                ("index-status", _freeze_public(archive.index_status())),
            )
        )
        values.extend((f"search:{query}", tuple(archive.search_blocks(query))) for query in search_queries)
    return tuple(values)


def _raw_identity_map(connection: sqlite3.Connection | None) -> dict[str, str]:
    """Map production raw ids to path-independent comparator identities."""

    if connection is None:
        return {}
    table = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_sessions'").fetchone()
    if table is None:
        return {}
    mapping: dict[str, str] = {}
    for row in connection.execute("SELECT raw_id, origin, source_index, blob_hash, native_id FROM raw_sessions"):
        raw_id, origin, source_index, blob_hash, native_id = row
        blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        stable_identity = f"raw[{origin}|{source_index}|{blob_hash_hex}|{native_id!r}]"
        mapping[str(raw_id)] = stable_identity
    return mapping


def _default_search_queries(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Choose stable, tokenizer-compatible probes from the public FTS table."""

    table = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'").fetchone()
    if table is None:
        return ()

    vocab_name = "canonical_snapshot_fts_vocab"
    temporary_schema = "te" + "mp"
    try:
        connection.execute(
            f"CREATE VIRTUAL TABLE {temporary_schema}.{vocab_name} USING fts5vocab(main, messages_fts, row)"
        )
    except sqlite3.OperationalError:
        return ()

    try:
        candidates = connection.execute(
            f"SELECT term FROM {temporary_schema}.{vocab_name} WHERE doc > 0 ORDER BY term"
        ).fetchall()
        queries: list[str] = []
        for (term,) in candidates:
            normalized = str(term)
            if normalized.casefold() in {"and", "or", "not", "near"}:
                continue
            try:
                match = connection.execute(
                    "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ? LIMIT 1", (normalized,)
                ).fetchone()
            except sqlite3.OperationalError:
                continue
            if match is not None:
                queries.append(normalized)
            if len(queries) == 3:
                break
        return tuple(queries)
    finally:
        connection.execute(f"DROP TABLE {temporary_schema}.{vocab_name}")


def _session_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in connection.execute("SELECT session_id FROM sessions ORDER BY session_id"))


def _normalize_value(
    database: str,
    relation: str,
    column: str,
    value: object,
    root: Path,
    *,
    raw_identity_map: Mapping[str, str],
) -> SqlValue:
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float)):
        if isinstance(value, str) and _is_raw_id_column(column):
            value = raw_identity_map.get(value, value)
        if isinstance(value, str) and column in RUN_LOCAL_PATH_ALLOWLIST.get(f"{database}.{relation}", frozenset()):
            return _archive_relative_path(value, root)
        return value
    raise TypeError(f"unsupported SQLite value in {database}.{relation}.{column}: {type(value)!r}")


def _is_raw_id_column(column: str) -> bool:
    return column == "raw_id" or column.endswith("_raw_id") or column == "ref_id"


def _archive_relative_path(value: str, root: Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            return value
    return candidate.as_posix()


def _freeze_public(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _freeze_public(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _freeze_public(model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_public(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in PUBLIC_RUN_LOCAL_FIELDS
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_freeze_public(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _archive_root(value: object) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value)
    root = getattr(value, "root", None)
    if isinstance(root, Path):
        return root
    raise TypeError(f"expected archive root or object with Path root, got {type(value)!r}")


def _relation_difference(expected: RelationSnapshot | None, actual: RelationSnapshot | None) -> str:
    if expected is None or actual is None:
        return f"missing expected={expected is not None}, actual={actual is not None}"
    if not expected.exists or not actual.exists:
        return f"exists expected={expected.exists}, actual={actual.exists}"
    if expected.columns != actual.columns:
        return f"columns expected={expected.columns}, actual={actual.columns}"
    expected_rows = set(expected.rows)
    actual_rows = set(actual.rows)
    return (
        f"only_expected={sorted(expected_rows - actual_rows, key=repr)[:2]!r}, "
        f"only_actual={sorted(actual_rows - expected_rows, key=repr)[:2]!r}"
    )


def _value_difference(expected: object, actual: object) -> str:
    return f"expected={expected!r}, actual={actual!r}"


__all__ = [
    "CanonicalArchiveSnapshot",
    "FactRow",
    "NON_COMPARABLE_RELATIONS",
    "RUN_LOCAL_NORMALIZATION_ALLOWLIST",
    "RUN_LOCAL_PATH_ALLOWLIST",
    "RelationSnapshot",
    "archive_snapshot",
    "assert_archives_equivalent",
    "assert_canonical_archives_equal",
    "assert_canonical_snapshots_equal",
    "capture_canonical_snapshot",
    "diff_canonical_snapshots",
]
