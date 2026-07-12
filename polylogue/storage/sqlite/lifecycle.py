"""Declared, disposable fast-forward plans for rebuildable SQLite tiers.

Index-tier schema versions remain rebuildable derived state.  A declaration in
this module is not a migration chain: it is a narrowly scoped, version-pair
proof that a clone may be brought forward without raw replay.  Any transition
whose result depends on parser semantics is explicitly routed to reprocess or
full rebuild instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DerivedDeltaClass(StrEnum):
    """Meaning of one derived-tier schema delta."""

    CONSTRAINT_ONLY = "constraint-only"
    VIEW_ONLY = "view-only"
    INDEX_ONLY = "index-only"
    FTS_REINDEX = "fts-reindex"
    SEMANTIC_REPARSE = "semantic-reparse"


class FastForwardOperationKind(StrEnum):
    """Generated-SQL operation backed by canonical index DDL."""

    REPLACE_TABLE = "replace-table"
    REPLACE_VIEW = "replace-view"
    CREATE_INDEX = "create-index"
    REBUILD_FTS = "rebuild-fts"


@dataclass(frozen=True, slots=True)
class FastForwardOperation:
    """One canonical object operation in a clone-only fast-forward plan."""

    name: str
    kind: FastForwardOperationKind
    objects: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class IndexDeltaDeclaration:
    """The declared meaning and SQL surface of one index schema version."""

    version: int
    classes: tuple[DerivedDeltaClass, ...]
    operations: tuple[FastForwardOperation, ...] = ()

    @property
    def requires_semantic_reparse(self) -> bool:
        return DerivedDeltaClass.SEMANTIC_REPARSE in self.classes


@dataclass(frozen=True, slots=True)
class IndexFastForwardPlan:
    """A disposable plan for one exact source/target version pair."""

    source_version: int
    target_version: int
    declarations: tuple[IndexDeltaDeclaration, ...]

    @property
    def requires_semantic_reparse(self) -> bool:
        return any(declaration.requires_semantic_reparse for declaration in self.declarations)

    @property
    def eligible_for_sql_fast_forward(self) -> bool:
        return bool(self.declarations) and not self.requires_semantic_reparse

    @property
    def canonical_objects(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            dict.fromkeys(
                object_ref
                for declaration in self.declarations
                for operation in declaration.operations
                for object_ref in operation.objects
            )
        )


# Deliberately bounded: v32 is the oldest observed live generation for which
# we retain a clone-proof plan.  Earlier versions continue to rebuild from raw
# evidence rather than silently acquiring an unsupported upgrade path.
INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR = 32

INDEX_DELTA_DECLARATIONS: tuple[IndexDeltaDeclaration, ...] = (
    IndexDeltaDeclaration(
        version=33,
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v33-insight-check",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "insight_materialization"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=34,
        classes=(DerivedDeltaClass.INDEX_ONLY, DerivedDeltaClass.VIEW_ONLY),
        operations=(
            FastForwardOperation(
                name="v34-index-and-delegations",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_web_constructs_message"),),
            ),
            FastForwardOperation(
                name="v34-index-and-delegations",
                kind=FastForwardOperationKind.REPLACE_VIEW,
                objects=(("view", "delegations"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=35,
        classes=(DerivedDeltaClass.FTS_REINDEX,),
        operations=(
            FastForwardOperation(
                name="v35-messages-fts",
                kind=FastForwardOperationKind.REBUILD_FTS,
                objects=(
                    ("table", "messages_fts"),
                    ("trigger", "messages_fts_ai"),
                    ("trigger", "messages_fts_ad"),
                    ("trigger", "messages_fts_au"),
                ),
            ),
            FastForwardOperation(
                name="v35-insight-fts",
                kind=FastForwardOperationKind.REBUILD_FTS,
                objects=(
                    ("table", "session_work_events_fts"),
                    ("trigger", "session_work_events_fts_ai"),
                    ("trigger", "session_work_events_fts_ad"),
                    ("trigger", "session_work_events_fts_au"),
                    ("table", "threads_fts"),
                    ("trigger", "threads_fts_ai"),
                    ("trigger", "threads_fts_ad"),
                    ("trigger", "threads_fts_au"),
                ),
            ),
        ),
    ),
)


def index_delta_declaration_report(current_version: int) -> dict[str, object]:
    """Return the static declaration coverage used by the schema-policy lint."""
    versions = tuple(declaration.version for declaration in INDEX_DELTA_DECLARATIONS)
    expected = tuple(range(INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR + 1, current_version + 1))
    missing = tuple(version for version in expected if version not in versions)
    duplicates = tuple(sorted({version for version in versions if versions.count(version) > 1}))
    invalid = tuple(
        declaration.version
        for declaration in INDEX_DELTA_DECLARATIONS
        if declaration.version > current_version or not declaration.classes
    )
    return {
        "compatibility_floor": INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR,
        "declared_versions": versions,
        "missing_versions": missing,
        "duplicate_versions": duplicates,
        "invalid_versions": invalid,
        "ok": not missing and not duplicates and not invalid,
    }


def index_fast_forward_plan(source_version: int, target_version: int) -> IndexFastForwardPlan | None:
    """Build a contiguous SQL plan, or ``None`` when rebuild/reprocess is required."""
    if source_version < INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR or source_version >= target_version:
        return None
    declarations = tuple(
        declaration
        for declaration in INDEX_DELTA_DECLARATIONS
        if source_version < declaration.version <= target_version
    )
    if tuple(declaration.version for declaration in declarations) != tuple(
        range(source_version + 1, target_version + 1)
    ):
        return None
    plan = IndexFastForwardPlan(source_version, target_version, declarations)
    return plan if plan.eligible_for_sql_fast_forward else None


__all__ = [
    "DerivedDeltaClass",
    "FastForwardOperation",
    "FastForwardOperationKind",
    "INDEX_DELTA_DECLARATIONS",
    "INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR",
    "IndexDeltaDeclaration",
    "IndexFastForwardPlan",
    "index_delta_declaration_report",
    "index_fast_forward_plan",
]
