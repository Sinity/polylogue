"""Declared, disposable fast-forward plans for rebuildable SQLite tiers.

Index-tier schema versions remain rebuildable derived state.  A declaration in
this module is not a migration chain: it is a narrowly scoped, version-pair
proof that a clone may be brought forward without raw replay.  Any transition
whose result depends on parser semantics is explicitly routed to reprocess or
full rebuild instead.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict


class DerivedDeltaClass(StrEnum):
    """Meaning of one derived-tier schema delta."""

    CONSTRAINT_ONLY = "constraint-only"
    VIEW_ONLY = "view-only"
    INDEX_ONLY = "index-only"
    FTS_REINDEX = "fts-reindex"
    CACHE_REMOVAL = "cache-removal"
    SEMANTIC_REPARSE = "semantic-reparse"


class FastForwardOperationKind(StrEnum):
    """Generated-SQL operation backed by canonical index DDL."""

    REPLACE_TABLE = "replace-table"
    REPLACE_VIEW = "replace-view"
    CREATE_INDEX = "create-index"
    REBUILD_FTS = "rebuild-fts"
    DROP_TABLE = "drop-table"


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
        return (
            bool(self.declarations)
            and not self.requires_semantic_reparse
            and all(declaration.classes for declaration in self.declarations)
            and all(declaration.operations for declaration in self.declarations)
        )

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

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return the stable executor-stage order declared by this plan."""
        return tuple(
            dict.fromkeys(operation.name for declaration in self.declarations for operation in declaration.operations)
        )


class IndexDeltaDeclarationReport(TypedDict):
    """Static coverage result consumed by the schema-version policy lint."""

    compatibility_floor: int
    declared_versions: tuple[int, ...]
    missing_versions: tuple[int, ...]
    duplicate_versions: tuple[int, ...]
    invalid_versions: tuple[int, ...]
    ok: bool


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
    IndexDeltaDeclaration(
        version=36,
        # Origin gained the internal ``beads-issue`` token.  This is a
        # constraint-only widening: existing parsed rows do not change, but
        # the two tables that embed Origin CHECKs must be copy-forwarded on a
        # clone.  The archive schema-forward actuator owns that exact
        # operation; it never reparses raw evidence or rebuilds FTS.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v36-origin-checks",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(
                    ("table", "sessions"),
                    ("table", "session_links"),
                ),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=37,
        # Run projection, observed-event, and context-snapshot rows are now
        # derived on read from normalized source evidence. Removing their
        # rebuildable caches is a clone-safe deletion with no raw reparse.
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="v37-drop-run-projection-caches",
                kind=FastForwardOperationKind.DROP_TABLE,
                objects=(
                    ("table", "session_runs"),
                    ("table", "session_observed_events"),
                    ("table", "session_context_snapshots"),
                ),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=38,
        classes=(DerivedDeltaClass.INDEX_ONLY, DerivedDeltaClass.VIEW_ONLY),
        operations=(
            FastForwardOperation(
                name="v38-action-and-delegation-projections",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "action_pairs"), ("table", "delegation_facts")),
            ),
            FastForwardOperation(
                name="v38-action-and-delegation-projections",
                kind=FastForwardOperationKind.REPLACE_VIEW,
                objects=(("view", "actions"), ("view", "delegations")),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=39,
        # Work topology derives from admitted provider facts. Reprocess/rebuild
        # rather than clone-forward so a graph is never retained without the
        # source evidence its nodes and edges cite.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=41,
        # action_pairs stops materializing tool_input/output_text text
        # copies (polylogue-2i2w) -- every tool interaction existed ~3x on
        # disk (blocks + action_pairs + FTS), and the copy turned a
        # per-session delete into scattered overflow-page random IO. The
        # surviving join/rank/outcome columns are copy-forward safe (no
        # parser semantics involved): a clone drops the two columns from
        # action_pairs and the actions VIEW is replaced to re-join blocks
        # for the dropped text at read time, so every reader keeps
        # byte-identical payloads without any reparse of raw evidence.
        classes=(DerivedDeltaClass.CACHE_REMOVAL, DerivedDeltaClass.VIEW_ONLY),
        operations=(
            FastForwardOperation(
                name="v41-action-pairs-text-copy-removal",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "action_pairs"),),
            ),
            FastForwardOperation(
                name="v41-action-pairs-text-copy-removal",
                kind=FastForwardOperationKind.REPLACE_VIEW,
                objects=(("view", "actions"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=42,
        # The writer stops materializing `token_count`/`message_usage`/
        # `agent_policy`/`agent_message` rows into `session_events`
        # (polylogue-bo9n zero-evidence-loss filtering) -- each is fully
        # re-derivable from a sibling typed table already written at the
        # same commit (`session_provider_usage_events`, `session_agent_policies`)
        # or from a twin `ParsedMessage`, so no unique evidence is lost. This
        # is a writer-materialization change, not a DDL change (session_events
        # keeps its existing columns) -- there is no declared clone-safe SQL
        # delta because a fast-forward would require re-deriving which
        # already-persisted rows a fresh parse would have skipped; existing
        # index tiers rebuild from source evidence instead
        # (`polylogue ops reset --index && polylogued run`).
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=43,
        # Adds the messages_fts_identity rowid/block_id ledger
        # (polylogue-1xc.12) and refreshes the messages_fts trigger bodies to
        # also maintain it. Every ledgered field (block_id, source_hash,
        # recipe_id) is derivable from already-persisted blocks columns
        # (block_id, content_hash) plus the FTS_MESSAGES_IDENTITY_RECIPE_ID
        # constant -- no raw reparse needed, so this is a clone-safe rebuild
        # of a derived surface, the same shape as v35's FTS tokenizer
        # reindex.
        classes=(DerivedDeltaClass.FTS_REINDEX,),
        operations=(
            FastForwardOperation(
                name="v43-messages-fts-identity",
                kind=FastForwardOperationKind.REBUILD_FTS,
                objects=(
                    ("table", "messages_fts_identity"),
                    ("trigger", "messages_fts_ai"),
                    ("trigger", "messages_fts_ad"),
                    ("trigger", "messages_fts_au"),
                ),
            ),
        ),
    ),
)


def resolve_canonical_index_objects(objects: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], str]:
    """Resolve the exact canonical ``CREATE`` statement for each declared object.

    Built from a scratch in-memory connection executing the current
    ``INDEX_DDL`` so the resolved text is always the live canonical shape --
    the same source of truth ``devtools/index_fast_forward.py``'s offline
    clone actuator uses (``_canonical_schema``). Lives here (not in the
    executor module under ``storage/sqlite/archive_tiers/``) so it stays a
    read-only declaration-resolution helper, never itself an index-tier
    mutation the writer-module inventory has to account for.
    """
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX])
        resolved: dict[tuple[str, str], str] = {}
        for object_type, name in objects:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, name),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError(
                    f"canonical index DDL is missing {object_type} {name!r}; fast-forward cannot proceed"
                )
            resolved[(object_type, name)] = str(row[0])
        return resolved
    finally:
        conn.close()


def index_delta_declaration_report(current_version: int) -> IndexDeltaDeclarationReport:
    """Return the static declaration coverage used by the schema-policy lint."""
    versions = tuple(declaration.version for declaration in INDEX_DELTA_DECLARATIONS)
    expected = tuple(range(INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR + 1, current_version + 1))
    missing = tuple(version for version in expected if version not in versions)
    duplicates = tuple(sorted({version for version in versions if versions.count(version) > 1}))
    invalid = tuple(
        declaration.version
        for declaration in INDEX_DELTA_DECLARATIONS
        if declaration.version > current_version
        or not declaration.classes
        or (not declaration.requires_semantic_reparse and not declaration.operations)
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
        sorted(
            (
                declaration
                for declaration in INDEX_DELTA_DECLARATIONS
                if source_version < declaration.version <= target_version
            ),
            key=lambda declaration: declaration.version,
        )
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
    "IndexDeltaDeclarationReport",
    "IndexFastForwardPlan",
    "index_delta_declaration_report",
    "index_fast_forward_plan",
    "resolve_canonical_index_objects",
]
