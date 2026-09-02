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
from typing import Literal, TypedDict


class DerivedDeltaClass(StrEnum):
    """Meaning of one derived-tier schema delta."""

    CONSTRAINT_ONLY = "constraint-only"
    VIEW_ONLY = "view-only"
    INDEX_ONLY = "index-only"
    FTS_REINDEX = "fts-reindex"
    CACHE_REMOVAL = "cache-removal"
    SEMANTIC_REPARSE = "semantic-reparse"
    # polylogue-9rw0.1: a clone-safe DDL shape (additive nullable columns or
    # tables) whose new VALUES come from parser/materializer semantics, not
    # from existing columns. Unlike SEMANTIC_REPARSE, the raw evidence never
    # needs re-reading in full: a bounded, declared reprocess scope (see
    # ``TargetedReprocessScope``) names exactly which already-persisted
    # sessions must be re-materialized to populate the new surface. The v44
    # witness (sessions.title_ref/title_confidence, 3,201 of 18,871 Codex
    # sessions) is the measured case this exists to express: previously the
    # only truthful class was SEMANTIC_REPARSE, which routed every archive
    # through a full raw replay to backfill two nullable columns on one
    # origin's sessions.
    SHAPE_FORWARD_TARGETED_REPROCESS = "shape-forward-targeted-reprocess"


@dataclass(frozen=True, slots=True)
class SameVersionSchemaVariant:
    """A declared object-definition variant retained by a derived tier."""

    introduced_version: int
    object_names: tuple[tuple[str, str], ...]
    transformation: Literal["remove_fts_bulk_guard"]

    def __post_init__(self) -> None:
        if self.introduced_version < 1 or not self.object_names:
            raise ValueError("SameVersionSchemaVariant requires a version and objects")


INDEX_SAME_VERSION_SCHEMA_VARIANTS: tuple[SameVersionSchemaVariant, ...] = (
    SameVersionSchemaVariant(
        introduced_version=63,
        object_names=(
            ("trigger", "blocks_command_trigram_ai"),
            ("trigger", "blocks_command_trigram_ad"),
            ("trigger", "blocks_command_trigram_au"),
        ),
        transformation="remove_fts_bulk_guard",
    ),
)


def same_version_schema_variants(version: int) -> tuple[SameVersionSchemaVariant, ...]:
    """Return compatibility variants retained by a generation version."""
    return tuple(variant for variant in INDEX_SAME_VERSION_SCHEMA_VARIANTS if version >= variant.introduced_version)


@dataclass(frozen=True, slots=True)
class TargetedReprocessScope:
    """A bounded reprocess scope over already-persisted sessions, as data.

    Mirrors the ``origin``/``session_ids`` dimensions of
    ``polylogue.maintenance.scope.MaintenanceScopeFilter`` -- the vocabulary
    every other maintenance backfill already scopes by -- without importing
    that module here: ``lifecycle.py`` is deliberately free of any
    non-stdlib import so the schema-versioning policy lint, the fast-forward
    executor, and unit tests can all evaluate it from the thinnest possible
    surface. A caller that already imports ``polylogue.maintenance`` can
    losslessly round-trip an instance into a ``MaintenanceScopeFilter``.

    At least one dimension must be set -- an empty scope would silently mean
    "every session", which is exactly the full-corpus cost this delta class
    exists to avoid.
    """

    origin: str | None = None
    session_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.origin is None and not self.session_ids:
            raise ValueError("TargetedReprocessScope must declare origin and/or session_ids")

    def matching_predicate_sql(self) -> tuple[str, tuple[object, ...]]:
        """Return a ``sessions``-table WHERE fragment and its bound params."""
        clauses: list[str] = []
        params: list[object] = []
        if self.origin is not None:
            clauses.append("origin = ?")
            params.append(self.origin)
        if self.session_ids:
            placeholders = ", ".join("?" for _ in self.session_ids)
            clauses.append(f"session_id IN ({placeholders})")
            params.extend(self.session_ids)
        return " AND ".join(clauses), tuple(params)


CanaryChangeOperation = Literal["added", "removed", "changed"]
CanaryChangeScope = Literal["row", "schema"]


@dataclass(frozen=True, slots=True)
class ExpectedCanaryChange:
    """One row- or schema-difference shape a semantic delta may authorize.

    ``row`` signatures describe values produced by replay.  ``schema``
    signatures describe the DDL shape itself (for example, an additive column
    that is absent from a genuinely older generation).  Keeping the two
    namespaces explicit prevents a broad row declaration from silently
    authorizing a schema difference.
    """

    table: str
    operations: tuple[CanaryChangeOperation, ...]
    columns: tuple[str, ...]
    scope: CanaryChangeScope = "row"

    def __post_init__(self) -> None:
        if not self.table or not self.operations or not self.columns:
            raise ValueError("ExpectedCanaryChange requires table, operations, and columns")
        if self.scope not in ("row", "schema"):
            raise ValueError("ExpectedCanaryChange scope must be 'row' or 'schema'")


class FastForwardOperationKind(StrEnum):
    """Generated-SQL operation backed by canonical index DDL."""

    REPLACE_TABLE = "replace-table"
    REPLACE_VIEW = "replace-view"
    CREATE_INDEX = "create-index"
    REBUILD_FTS = "rebuild-fts"
    DROP_TABLE = "drop-table"
    REPAIR_ORPHAN_ATTACHMENT_NATIVE_IDS = "repair-orphan-attachment-native-ids"


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
    # Required exactly when SHAPE_FORWARD_TARGETED_REPROCESS is in ``classes``
    # (enforced by ``__post_init__``): the exact bounded scope a fast-forward
    # of this version must enqueue for reprocessing, expressed as data.
    reprocess_scope: TargetedReprocessScope | None = None
    expected_canary_changes: tuple[ExpectedCanaryChange, ...] = ()

    def __post_init__(self) -> None:
        declares_class = DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS in self.classes
        has_scope = self.reprocess_scope is not None
        if declares_class != has_scope:
            raise ValueError(
                f"index delta v{self.version}: SHAPE_FORWARD_TARGETED_REPROCESS and reprocess_scope "
                f"must be declared together (class present={declares_class!r}, scope present={has_scope!r})"
            )
        if self.expected_canary_changes and not (self.requires_semantic_reparse or self.requires_targeted_reprocess):
            raise ValueError(f"index delta v{self.version}: expected canary changes require semantic work")

    @property
    def requires_semantic_reparse(self) -> bool:
        return DerivedDeltaClass.SEMANTIC_REPARSE in self.classes

    @property
    def requires_targeted_reprocess(self) -> bool:
        return DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS in self.classes


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
    def pending_reprocess_scopes(self) -> tuple[tuple[int, TargetedReprocessScope], ...]:
        """Return the (version, scope) pairs this plan's execution must enqueue.

        Non-empty exactly when the plan crosses a SHAPE_FORWARD_TARGETED_REPROCESS
        declaration -- the executor consumes this to enqueue real, bounded
        reprocess debt after the clone-safe DDL lands, instead of silently
        treating the shape-only copy-forward as complete.
        """
        return tuple(
            (declaration.version, declaration.reprocess_scope)
            for declaration in self.declarations
            if declaration.requires_targeted_reprocess and declaration.reprocess_scope is not None
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
            FastForwardOperation(
                name="v37-repair-orphan-attachment-native-ids",
                kind=FastForwardOperationKind.REPAIR_ORPHAN_ATTACHMENT_NATIVE_IDS,
                objects=(("table", "attachment_native_ids"),),
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
        version=40,
        # Adds query_unit_frame_state (polylogue-z9gh.9, PR #3068): a
        # singleton, trigger-maintained epoch counter that composes with the
        # user-tier assertion epoch to bind query-unit continuation resumes
        # to the exact archive snapshot they were issued against. Insert,
        # update, and delete triggers on session_links, sessions, messages,
        # blocks, session_tags, session_profiles, and delegation_facts each
        # bump the counter -- every table already exists at v39 and gains
        # only a maintenance trigger, so no row is reparsed or reshaped. The
        # table itself starts at epoch=0 via INSERT OR IGNORE. A clone can
        # therefore create the table and its 21 triggers verbatim from
        # canonical INDEX_DDL with no semantic reparse of raw evidence.
        classes=(DerivedDeltaClass.INDEX_ONLY,),
        operations=(
            FastForwardOperation(
                name="v40-query-unit-frame-state",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(
                    ("table", "query_unit_frame_state"),
                    ("trigger", "query_unit_frame_session_links_insert"),
                    ("trigger", "query_unit_frame_session_links_update"),
                    ("trigger", "query_unit_frame_session_links_delete"),
                    ("trigger", "query_unit_frame_sessions_insert"),
                    ("trigger", "query_unit_frame_sessions_update"),
                    ("trigger", "query_unit_frame_sessions_delete"),
                    ("trigger", "query_unit_frame_messages_insert"),
                    ("trigger", "query_unit_frame_messages_update"),
                    ("trigger", "query_unit_frame_messages_delete"),
                    ("trigger", "query_unit_frame_blocks_insert"),
                    ("trigger", "query_unit_frame_blocks_update"),
                    ("trigger", "query_unit_frame_blocks_delete"),
                    ("trigger", "query_unit_frame_session_tags_insert"),
                    ("trigger", "query_unit_frame_session_tags_update"),
                    ("trigger", "query_unit_frame_session_tags_delete"),
                    ("trigger", "query_unit_frame_session_profiles_insert"),
                    ("trigger", "query_unit_frame_session_profiles_update"),
                    ("trigger", "query_unit_frame_session_profiles_delete"),
                    ("trigger", "query_unit_frame_delegation_facts_insert"),
                    ("trigger", "query_unit_frame_delegation_facts_update"),
                    ("trigger", "query_unit_frame_delegation_facts_delete"),
                ),
            ),
        ),
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
    IndexDeltaDeclaration(
        version=44,
        # Adds sessions.title_ref/title_confidence (polylogue-ih67 AC#5). The
        # DDL surface is two nullable columns on an 18,871-row table -- trivial
        # to copy forward -- but their VALUES come from the v44 Codex title
        # resolver (thread name -> authored history -> first HUMAN_AUTHORED
        # message), so a shape-only fast-forward would leave every row NULL
        # while a cold rebuild populates them.
        #
        # polylogue-9rw0.1: re-declared under SHAPE_FORWARD_TARGETED_REPROCESS
        # instead of SEMANTIC_REPARSE. The DDL still copy-forwards generically
        # via the shared-columns REPLACE_TABLE path below (existing rows keep
        # every column they already have; the two new columns start NULL,
        # same as any additive-column REPLACE_TABLE); what changes is that the
        # declaration now also states, as data, exactly which already-
        # persisted sessions need re-materializing to populate them --
        # origin=codex-session, the full 100% of the affected population
        # measured in the bead witness -- so the executor enqueues bounded
        # per-session reprocess debt instead of the archive silently
        # promoting to v44 with every title_ref left NULL, or an operator
        # having to reach for a full `polylogue ops reset --index` replay of
        # all 41,363 raws to backfill two columns on one origin.
        classes=(DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS,),
        operations=(
            FastForwardOperation(
                name="v44-title-ref-shape",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "sessions"),),
            ),
        ),
        reprocess_scope=TargetedReprocessScope(origin="codex-session"),
        expected_canary_changes=(
            # A genuine pre-v44 generation reports the additive DDL as schema
            # differences, while a current-shape fixture reports populated row
            # values.  Keep both signatures explicit; a row signature must not
            # waive a schema difference (or vice versa).
            ExpectedCanaryChange(
                table="sessions",
                operations=("added",),
                columns=("title_ref", "title_confidence"),
                scope="schema",
            ),
            ExpectedCanaryChange(
                table="sessions",
                operations=("changed",),
                columns=("title_ref", "title_confidence"),
                scope="row",
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=45,
        # polylogue-1vpm.7: delegation_facts_source stops pairing a dispatch
        # to a resolved child by cardinality-gated ordinal position (rank N
        # dispatch <-> rank N child) and instead joins on identity -- a
        # trivial 1:1 cohort (no second candidate on either side), or,
        # non-trivially, provider-asserted content equality between the
        # dispatch's own instruction text and the child's first turn.
        # 'ambiguous' is retired from the mapping_state vocabulary. Every
        # input this recomputation reads (actions, session_links, messages,
        # blocks) is already fully parsed and persisted in index.db -- no
        # RAW evidence is reparsed. But delegation_facts is a materialized
        # TABLE (not the view), and _apply_replace_table's generic copy-
        # forward only carries EXISTING rows onto a same-shaped schema
        # (no column added or dropped here); it cannot re-derive VALUES
        # from the new join logic. A plain REPLACE_TABLE fast-forward would
        # therefore silently retain stale, ordinal-paired rows -- including
        # 'ambiguous' ones the whole point of this delta is to make
        # unreachable. As in v42/v44, DerivedDeltaClass has no category for
        # "clone-safe shape, values via targeted reprocess of already-
        # persisted derived rows" (tracked by polylogue-9rw0.1);
        # SEMANTIC_REPARSE is the honest, conservative classification until
        # that vocabulary gap is closed, and it keeps the existing
        # `polylogue ops reset --index && polylogued run` full-rebuild
        # behaviour rather than risk a fast-forwarded clone disagreeing with
        # a cold rebuild.
        #
        # v45 also folds in the session_provider_usage_events.payload_json
        # retirement: the column is dropped and its 8 billing-provenance
        # keys (estimated_cost_usd, actual_cost_usd, cost_status,
        # cost_source, pricing_version, billing_provider, billing_base_url,
        # billing_mode) become nullable typed columns on the same table
        # (archive_tiers/index.py). Considered in isolation that delta is
        # clone-safe -- the values already live in the row as JSON, the
        # backfill is `json_extract(payload_json, '$.key')` against
        # already-persisted rows, not a raw reparse -- and would deserve its
        # own CONSTRAINT_ONLY declaration with a REPLACE_TABLE operation.
        # But IndexDeltaDeclaration is one classification per *version*, not
        # per DDL object, and v45 already carries the delegation_facts
        # SEMANTIC_REPARSE above. Declaring this version
        # `(CONSTRAINT_ONLY, SEMANTIC_REPARSE)` would not change behaviour
        # (`requires_semantic_reparse` short-circuits `eligible_for_sql_
        # fast_forward` the moment SEMANTIC_REPARSE is present) but would
        # misstate the version as partially fast-forwardable when it is
        # not: every archive touching v45 goes through
        # `polylogue ops reset --index && polylogued run` regardless of
        # which of the two DDL changes triggered it. The truthful
        # declaration for the whole version therefore stays
        # `(SEMANTIC_REPARSE,)`; the CONSTRAINT_ONLY-shaped sub-delta is
        # recorded here in prose so a future split (polylogue-9rw0.1) can
        # recover it once per-object declarations exist.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=46,
        # polylogue-2qx.4: the unread-wire batch (polylogue-cgfy/cuxz.8/9x22
        # field-landing decisions), landed as ONE bump covering every field
        # rather than one per origin -- see index.py's v46 header comment for
        # the full per-field list (messages.stop_reason,
        # blocks.tool_result_outcome_unknown_reason, sessions.display_name,
        # sessions.run_settings_json, session_links.parent_tool_use_block_id,
        # and the new file_edits/session_refs tables).
        #
        # Every new column's/table's VALUES depend on parser semantics to
        # populate honestly -- a shape-only copy-forward would leave every
        # row/table empty, exactly the v42/v44/v45 precedent recorded above.
        # As in those versions, DerivedDeltaClass has no category for
        # "clone-safe shape, values via targeted reprocess"
        # (tracked by polylogue-9rw0.1); SEMANTIC_REPARSE is the honest,
        # conservative classification, and it keeps the existing
        # `polylogue ops reset --index && polylogued run` full-rebuild
        # behaviour.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=47,
        # polylogue-o4j2: sessions.pending_drafts_json -- aistudio-drive
        # pendingInputs draft text, moved off the session_events axis (see
        # index.py's v47 header comment). Values depend on parser semantics
        # (the new column is populated only by re-parsing the drive.py
        # payload), so a shape-only copy-forward would leave every row NULL
        # -- the same v42/v44/v45/v46 precedent. SEMANTIC_REPARSE routes
        # through `polylogue ops reset --index && polylogued run`.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=48,
        # polylogue-lzh8: retroactive declaration for 1e0246d77 (#3088,
        # "admit Claude Workflow artifacts through OriginSpec", merged
        # 2026-07-18), which changed origin_specs.py's artifact rules to
        # correctly classify workflow_run_snapshot / workflow_journal /
        # agent_sidecar_meta / adopt_manifest paths as parse_policy="fact"
        # (never session-parsed) without declaring an INDEX_SCHEMA_VERSION
        # bump. There is no DDL surface to fast-forward: this is a pure
        # values/classification change over already-persisted rows, the
        # same v42/v44/v45/v46 "SEMANTIC_REPARSE with no clone-safe SQL
        # delta" shape. See index.py's v48 header comment for the measured
        # live-impact counts (172 zero-message sessions from the four
        # workflow-artifact kinds, all acquired before the fix reached the
        # deployed daemon build). Resolving these existing rows requires
        # `polylogue ops reset --index && polylogued run` -- deliberately
        # NOT executed by this declaration; see the accompanying PR body for
        # the explicit operator-facing instruction.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=49,
        # polylogue-gt1z + polylogue-shnc, batched (see index.py's v49 header
        # comment for the full rationale): sessions.reported_cost_usd (exact
        # provider-reported session cost total) plus two new
        # session_model_usage CHECK constraints enforcing the
        # 'priced'/'origin_reported' provenance invariant. Values/repairs
        # depend on parser and pricing semantics, so a shape-only
        # copy-forward would leave every row NULL/still-violating -- the same
        # v42/v44/v45/v46/v47/v48 precedent. SEMANTIC_REPARSE routes through
        # `polylogue ops reset --index && polylogued run`.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=50,
        # polylogue-vf9x: adds blocks.signature (nullable TEXT) and fixes two
        # independent reasoning/thinking-content-loss defects (see index.py's
        # v50 header comment for the full writeup):
        #   - Claude Code: an `if text:` guard in base_support.py's
        #     `content_blocks_from_segments` dropped every THINKING segment
        #     whose wire body is empty-string-plus-signature-only -- the
        #     shape Claude has shipped since ~2026-06 -- silently zeroing
        #     `thinking_count` for every affected session.
        #   - Codex: standalone `reasoning` response_item records were read
        #     only by the generic session_event compactor (no
        #     `reasoning`-specific branch existed), so summary/content text
        #     was never read into any surface at all.
        # Both changes require re-parsing already-acquired raw evidence to
        # recover the previously-dropped/discarded text, so this is the same
        # v42/v44/v45/v46/v48/v49 "values depend on parser semantics" shape --
        # not a free clone-safe fast-forward, even though `signature` is a
        # real additive column. `polylogue ops reset --index && polylogued
        # run` is required to recover historical thinking/reasoning content;
        # deliberately NOT executed by this declaration.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=51,
        # polylogue-u6tl: delegation_facts.mapping_state and .result_status
        # gain CHECK constraints GENERATED from their existing typed
        # counterparts (DelegationMappingState / DelegationResultStatus in
        # archive_tiers/archive.py) via the `literal_check` generator, which
        # CLAUDE.md documents by name as the mechanism keeping typing.Literal
        # types and their SQL CHECK lists in lockstep -- and which had zero
        # call sites until now. No column is added and no existing value
        # changes: this only tightens a CHECK the single writer
        # (delegation_facts_insert_sql) already satisfied. Measured on the
        # live archive before declaring: 0 rows outside either vocabulary, so
        # a copy-forward rejects nothing. CONSTRAINT_ONLY per the v33/v36
        # precedent (adding/widening a CHECK over unchanged values), NOT
        # SEMANTIC_REPARSE -- no reparse is required for this to take effect.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v51-delegation-facts-check",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "delegation_facts"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=52,
        # polylogue-rlvj: fts_freshness_state gains a table-level CHECK
        # (state='ready' implies missing_rows=0 AND excess_rows=0 AND
        # duplicate_rows=0 AND source_rows=indexed_rows) -- see index.py's
        # v52 header comment for the live incident (a targeted repair's
        # scoped verdict overwriting the global ledger row with a false
        # 'ready'). No row's values need to change for archives that were
        # never poisoned by the bug: every writer that already reaches
        # state='ready' already writes balanced counters, so this is a pure
        # constraint widening, the same v33/v36/v51 precedent. Archives that
        # WERE poisoned by the bug carry a row that violates the new CHECK;
        # the fast-forward executor's `_REPLACE_TABLE_SANITIZERS` entry
        # downgrades that row to 'stale' before the copy runs instead of
        # letting the migration abort.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v52-fts-freshness-check",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "fts_freshness_state"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=53,
        # polylogue-jc4q: changes how a Claude Code resume/fork/usage-limit
        # boundary carryover derives its identity (sources/dispatch.py,
        # sources/parsers/claude/code_parser.py) -- see
        # INDEX_SCHEMA_VERSION's v53 comment (archive_tiers/index.py) for the
        # full writeup. This changes `sessions.native_id` (generated from
        # `provider_session_id`) and `session_links` lineage edges for
        # already-acquired raw evidence, so it requires re-parsing, not a
        # clone-safe fast-forward. SEMANTIC_REPARSE routes through `polylogue
        # ops reset --index && polylogued run`, deliberately NOT executed by
        # this declaration.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=54,
        # PR #3537: tightened claude.looks_like_ai (sources/parsers/claude/
        # ai_parser.py) to require a genuine chat_messages entry (role/sender
        # + text/content) rather than a bare chat_messages key -- see
        # INDEX_SCHEMA_VERSION's v54 comment (archive_tiers/index.py) for the
        # full writeup. Moves the classifier's admission decision for
        # already-acquired raw evidence, so it requires re-parsing, not a
        # clone-safe fast-forward. SEMANTIC_REPARSE routes through `polylogue
        # ops reset --index && polylogued run`, deliberately NOT executed by
        # this declaration.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=55,
        # polylogue-0cn3 / polylogue-5dfu: derived-tier vocabulary cleanup --
        # see INDEX_SCHEMA_VERSION's v55 comment (archive_tiers/index.py) for
        # the full writeup. Drops the title_source/title_ref/title_confidence
        # COALESCE ratchet in write.py's session upsert, deletes
        # TitleSource.UNKNOWN/USER and LinkType.REPAIRED, and narrows
        # TopologyEdgeStatus from 4 declared members to the 2 ever storable
        # (REPAIRED/QUARANTINED). A live archive can carry
        # title_source='unknown' or link_type='repaired' rows the narrowed
        # CHECKs would reject on a plain copy-forward, and the write path's
        # title_source computation genuinely changes (no longer ever
        # recomputes 'unknown') -- SEMANTIC_REPARSE, not a free fast-forward.
        # `polylogue ops reset --index && polylogued run` is required,
        # deliberately NOT executed by this declaration.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=56,
        # polylogue-fuky: the writer stops materializing `agent_reasoning`
        # rows into `session_events` for Codex sessions -- confirmed
        # duplicate of the paired `reasoning` record's already-materialized
        # THINKING-block message, see INDEX_SCHEMA_VERSION's v56 comment
        # (archive_tiers/index.py) for the full writeup. Matches the v42
        # `agent_message` precedent: a writer-materialization change, not a
        # DDL change (session_events keeps its existing columns) -- no
        # declared clone-safe SQL delta because a fast-forward would require
        # re-deriving which already-persisted rows a fresh parse would have
        # skipped; existing index tiers rebuild from source evidence instead
        # (`polylogue ops reset --index && polylogued run`).
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=57,
        # The one-shot sidecar-purge receipt has no remaining consumer. Its
        # removal is a clone-safe rebuildable-cache deletion.
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="v57-drop-agent-meta-sidecar-purge-receipts",
                kind=FastForwardOperationKind.DROP_TABLE,
                objects=(("table", "agent_meta_sidecar_purge_receipts"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=58,
        # polylogue-omsw: `archive.artifact_taxonomy.classify_artifact`/
        # `classify_artifact_path` changed classification decisions for
        # identical input bytes -- see INDEX_SCHEMA_VERSION's v58 comment
        # (archive_tiers/index.py) for the full writeup and measured
        # live-impact counts. Same v42/v44/v45/v46/v48 "SEMANTIC_REPARSE,
        # no clone-safe SQL delta" shape: no DDL change on any table, so
        # there is nothing here for a fast-forward to do. Resolving already-
        # materialized wrong-classification rows requires `polylogue ops
        # reset --index && polylogued run` -- deliberately NOT executed by
        # this declaration.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=59,
        # polylogue-4987i: Claude Code session_events ordering is now
        # deterministic and chunk-count-invariant -- see INDEX_SCHEMA_VERSION's
        # v59 comment (archive_tiers/index.py) for the full writeup, root
        # cause, and live spot-check evidence. No DDL change on any table
        # (content_hash values shift for a subset of already-materialized
        # rows, not the schema), so there is nothing here for a fast-forward
        # to do. Resolving already-materialized sessions with the old order
        # requires `polylogue ops reset --index && polylogued run`.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=60,
        # polylogue-taj0o Stage 2: Claude Code eager and streaming parsing
        # are unified into one incremental multi-way merge -- see
        # INDEX_SCHEMA_VERSION's v60 comment (archive_tiers/index.py) for
        # the full writeup, root cause (the eager/streaming primary-group
        # definition conflict), and scope. No DDL change on any table, so
        # there is nothing here for a fast-forward to do. Resolving
        # already-materialized sessions affected by the identity-definition
        # change requires `polylogue ops reset --index && polylogued run`.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=61,
        # polylogue-resk: session_model_usage.priced_with/priced_at_ms are
        # dropped (write-only outside tests -- see INDEX_SCHEMA_VERSION's v61
        # comment, archive_tiers/index.py, for the measured evidence) and the
        # CHECK that referenced priced_with is narrowed to just cost_usd. A
        # dead-column removal like v33/v36/v38/v41/v44: existing rows
        # copy-forward on every surviving column via the fast-forward
        # executor's REPLACE_TABLE path, no raw reparse. price_catalogs
        # (the FK target) is dropped separately via the index-tier
        # same-version benign-DDL registry (index_convergence.py) since a
        # whole-table drop is already idempotent without a version bump.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v61-drop-session-model-usage-pricing-columns",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "session_model_usage"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=62,
        # polylogue-664l: batches two column-level dead-weight removals from
        # the 2026-07-31 producer/consumer audit -- see INDEX_SCHEMA_VERSION's
        # v62 comment (archive_tiers/index.py) for the full writeup and
        # re-audit evidence for all four originally-named findings.
        #  - session_provider_usage_events drops 9 write-only columns
        #    (model_context_window + 8 Hermes billing-provenance columns).
        #    No reader ever selected them; the remaining columns (the
        #    last_*/total_* token counters every production reader actually
        #    uses) are unaffected -- CACHE_REMOVAL, the v41 `action_pairs`
        #    text-copy-removal precedent.
        #  - attachment_native_ids.id_kind's CHECK drops the 'source' member,
        #    which has zero producers anywhere in the repo, so zero live rows
        #    can exist with that value on any real archive -- CONSTRAINT_ONLY,
        #    the v33/v36/v51/v52 "tightening over unchanged values" precedent.
        # Both REPLACE_TABLE operations below use the generic shared-column
        # copy-forward (`_apply_replace_table`): it naturally drops columns
        # the canonical DDL no longer declares while preserving every
        # remaining column's values byte-for-byte, so this whole version is
        # eligible for fast-forward with no reprocess debt.
        classes=(DerivedDeltaClass.CACHE_REMOVAL, DerivedDeltaClass.CONSTRAINT_ONLY),
        operations=(
            FastForwardOperation(
                name="v62-provider-usage-events-billing-columns",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "session_provider_usage_events"),),
            ),
            # _apply_replace_table's DROP TABLE step also drops every index
            # defined on the table; re-declare them explicitly instead of
            # relying on runtime_indexes.py's narrower ensure-on-connect list
            # (which only covers the source_message/time_model pair, not the
            # base session_id/position index).
            FastForwardOperation(
                name="v62-provider-usage-events-billing-columns",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_session_provider_usage_events_session"),),
            ),
            FastForwardOperation(
                name="v62-attachment-native-ids-source-kind",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "attachment_native_ids"),),
            ),
            FastForwardOperation(
                name="v62-attachment-native-ids-source-kind",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_attachment_native_ids_native"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=63,
        # polylogue-eizc: drops `threads_fts`, a maintained FTS surface
        # with triggers and rebuild/repair/freshness machinery but no
        # application-layer consumers. Its only MATCH reader had zero
        # production callers, while the live thread-search path already
        # uses a LIKE scan. This is a clone-safe CACHE_REMOVAL delta, so
        # fast-forward drops the table and its triggers without raw replay.
        # `blocks_command_trigram` was kept at v63 because affordance usage
        # had a real production consumer for it; that devtools consumer
        # (devtools/affordance_usage.py) was deleted 2026-08-25
        # (polylogue-9m6ry) with no replacement, so its continued
        # justification is now open -- see the comment in
        # archive_tiers/index.py.
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="v63-drop-threads-fts",
                kind=FastForwardOperationKind.DROP_TABLE,
                objects=(
                    ("trigger", "threads_fts_ai"),
                    ("trigger", "threads_fts_ad"),
                    ("trigger", "threads_fts_au"),
                    ("table", "threads_fts"),
                ),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=64,
        # The new session stamps are derived from parser/lowering semantics.
        # Existing rows require raw replay for trustworthy values, so nullable
        # DDL cannot take the clone-safe fast-forward route.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=65,
        # polylogue-cuxz.5: `actions.result_state` is a CASE projection over
        # existing action_pairs columns. Replacing this derived view exposes
        # no-result separately from an existing result with unknown outcome;
        # it neither changes parser semantics nor materialized values.
        classes=(DerivedDeltaClass.VIEW_ONLY,),
        operations=(
            FastForwardOperation(
                name="v65-actions-result-state",
                kind=FastForwardOperationKind.REPLACE_VIEW,
                objects=(("view", "actions"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=66,
        # Native and positional message identities now use disjoint tagged
        # namespaces. Existing generated ids are still valid read values, but
        # every newly written message and dependent reference must be produced
        # by raw replay under the new expression.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=67,
        # Parser-authored title provenance changes stored session semantics.
        # Existing untyped title text cannot distinguish provider evidence
        # from page-title and native-id fallbacks, so only raw replay can
        # produce trustworthy title_source values.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=68,
        # polylogue-2tfug: ApplicationDecision gains REPARSE_REAFFIRMATION, and
        # raw_revision_applications.decision's CHECK is generated from that
        # enum, so the constraint widens. Purely permissive: every value the
        # old CHECK admitted the new one still admits, no stored row changes
        # meaning, and no parser semantics are involved.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v68-reparse-reaffirmation-decision",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "raw_revision_applications"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=69,
        # session_links.status gains TopologyEdgeStatus.AUTHORITY_CONTRADICTED
        # (polylogue-foee): a parser-inferred parent edge that acquired Codex
        # hook evidence contradicts is now excluded from composition and
        # retained as typed evidence, and the authoritative edge is written in
        # its place. Widening the generated CHECK is inert for existing rows,
        # but the new state is only PRODUCED by re-deriving edges against the
        # durable hook spool during raw replay -- an existing generation
        # cannot fast-forward into it, and its lineage projections would keep
        # composing through parents the hook evidence overrules. That makes
        # this parser-semantics-dependent, not a constraint-only widening.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=70,
        # polylogue-terra-p1: raw_revision_applications now persists the
        # accepted frontier proven by each application receipt. Existing rows
        # cannot be populated from raw_revision_heads without confusing the
        # current mutable head with historical application evidence, so this
        # schema change requires semantic raw replay rather than a clone-safe
        # table copy-forward.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=71,
        # n6 n6rkz: raw application decision ids now bind all immutable
        # accepted-evidence fields, and the semantic duplicate index follows
        # that same shape. Existing rows need raw replay to receive the new
        # identities without treating a mutable head as historical evidence.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=72,
        # FTS freshness evidence is derived metadata. A replace-table
        # fast-forward retains the last measured counters while the sanitizer
        # downgrades legacy count-only READY rows before the stricter CHECK.
        classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
        operations=(
            FastForwardOperation(
                name="v72-fts-freshness-exact-evidence",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "fts_freshness_state"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=73,
        # polylogue-ulm94: lineage branch points gain an identity-free
        # message-content witness and provider usage totals preserve omitted
        # values as NULL. Both values depend on parser/write semantics and
        # existing rows must be regenerated from raw evidence.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=74,
        # polylogue-h6r: work-evidence nodes now retain the provider-reported
        # run role used to form WorkerProfileRef groupings. Older rows have no
        # role evidence, so they must be regenerated by the derived rebuild.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=75,
        # polylogue-jfeaz.1: session_profiles no longer persists usage,
        # pricing, monetary, or provenance mirrors.  Those facts are read
        # through the canonical session_model_usage projection; old profile
        # rows cannot be converted into that authority, so rebuild the
        # derived tier from durable evidence.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=76,
        # Compaction boundary ranges and summary identities are parser-derived
        # values. Existing index rows require fresh re-ingest.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=77,
        # The attachment sweep counts live refs per attachment_id, and a
        # session delete sweeps every attachment the session touched. Without
        # this index each count scans attachment_refs whole, which puts a
        # per-attachment full scan inside the bulk delete path.
        classes=(DerivedDeltaClass.INDEX_ONLY,),
        operations=(
            FastForwardOperation(
                name="v77-attachment-ref-attachment-index",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_attachment_refs_attachment"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=78,
        # Message identity provenance is assigned by the parsed-session
        # lowering path. Existing rows require replay because native ids that
        # collide are deliberately downgraded to positional identity.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=79,
        # Global recent-session listings order without an origin predicate.
        # The existing origin-leading index cannot satisfy that global ORDER BY.
        classes=(DerivedDeltaClass.INDEX_ONLY,),
        operations=(
            FastForwardOperation(
                name="v79-global-session-sort-index",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_sessions_sort_key"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=80,
        # Session kind is parser-derived admission state. Existing index rows
        # need semantic replay to recover it from source evidence.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=81,
        # Existing rows require regeneration because the old cost label
        # cannot distinguish provider dollars from catalog computation.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=82,
        # Session insight cost provenance now describes the source of
        # aggregate dollars. Existing derived rows need semantic reprocessing
        # to replace the old provider-reported label where catalog pricing was
        # used.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=83,
        # polylogue-xd0ha: deliberate unknown outcomes are now
        # admitted and their parser reasons are preserved in the canonical
        # projection. Existing rows require fresh parser replay.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        # Delegation cost estimation now treats provider-reported dollars as
        # exact even when catalog pricing is unavailable. Existing derived
        # rows need the updated view definition.
        version=84,
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=85,
        # Attachment direction and producer provenance depend on each origin's
        # structured parser evidence. A shape-only copy-forward would mint
        # unsupported values, so existing rows require semantic reparse.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=90,
        # Canonical provider identity claims and parent-side dispatch
        # observations change the normalized topology reconstructed from raw
        # source; existing indexes must be replayed.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=86,
        # ``threads.dominant_repo_id`` had no writer or reader. Replacing the
        # projection table preserves every surviving derived value while
        # removing the unused column.
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="v86-drop-threads-dominant-repo-id",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "threads"),),
            ),
            FastForwardOperation(
                name="v86-drop-threads-dominant-repo-id",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_threads_time"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=87,
        # Collapse redundant derived relations while retaining bounded action
        # and delegation maintenance required by archive-scale reads.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=88,
        # The retired sidecar purge receipt was already removed from every
        # open archive by same-version convergence. Remove its stale fresh-DDL
        # declaration without rebuilding derived data.
        classes=(DerivedDeltaClass.CACHE_REMOVAL,),
        operations=(
            FastForwardOperation(
                name="v88-drop-agent-meta-sidecar-purge-receipts",
                kind=FastForwardOperationKind.DROP_TABLE,
                objects=(("table", "agent_meta_sidecar_purge_receipts"),),
            ),
        ),
    ),
    IndexDeltaDeclaration(
        version=89,
        # Semantic content identity now follows the complete parser field
        # partition. Existing rows need parser replay to regenerate hashes.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
    IndexDeltaDeclaration(
        version=85,
        # The session-insight materialization ledger is retired. Existing
        # derived rows must be regenerated through ordinary convergence from
        # source-bound row provenance; a clone-safe fast-forward cannot remove
        # the lifecycle dependency from already-derived state.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
    ),
)


def resolve_canonical_index_objects(objects: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], str]:
    """Resolve the exact canonical ``CREATE`` statement for each declared object.

    Built from a scratch in-memory connection executing the current
    ``INDEX_DDL`` so the resolved text is always the live canonical shape.
    Lives here (not in the executor module under
    ``storage/sqlite/archive_tiers/``) so it stays a read-only
    declaration-resolution helper, never itself an index-tier mutation the
    writer-module inventory has to account for.
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


def invalid_canary_change_declarations(
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> tuple[tuple[int, str], ...]:
    """Return packaged canary signatures that cannot match canonical index DDL.

    Expected signatures are executable comparator authority, not free-form
    labels.  Validate their table/column vocabulary against the canonical
    index schema so a typo cannot silently produce an inert expectation (or an
    over-broad declaration that waives an unrelated difference).
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX])
        invalid: list[tuple[int, str]] = []
        for declaration in declarations:
            for change in declaration.expected_canary_changes:
                columns: set[str] = set()
                try:
                    rows = connection.execute(f'PRAGMA table_xinfo("{change.table.replace(chr(34), chr(34) * 2)}")')
                    columns = {str(row[1]) for row in rows if int(row[6]) not in (1, 2)}
                except sqlite3.OperationalError:
                    pass
                if not columns:
                    invalid.append((declaration.version, f"unknown table {change.table!r}"))
                    continue
                missing = sorted(set(change.columns).difference(columns))
                if missing:
                    invalid.append((declaration.version, f"unknown column(s) on {change.table}: {', '.join(missing)}"))
        return tuple(invalid)
    finally:
        connection.close()


def index_delta_declaration_report(current_version: int) -> IndexDeltaDeclarationReport:
    """Return the static declaration coverage used by the schema-policy lint."""
    versions = tuple(declaration.version for declaration in INDEX_DELTA_DECLARATIONS)
    expected = tuple(range(INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR + 1, current_version + 1))
    missing = tuple(version for version in expected if version not in versions)
    duplicates = tuple(sorted({version for version in versions if versions.count(version) > 1}))
    invalid_signature_versions = {version for version, _reason in invalid_canary_change_declarations()}
    invalid = tuple(
        declaration.version
        for declaration in INDEX_DELTA_DECLARATIONS
        if declaration.version > current_version
        or not declaration.classes
        or (not declaration.requires_semantic_reparse and not declaration.operations)
        or (declaration.requires_targeted_reprocess and declaration.reprocess_scope is None)
        or declaration.version in invalid_signature_versions
    )
    return {
        "compatibility_floor": INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR,
        "declared_versions": versions,
        "missing_versions": missing,
        "duplicate_versions": duplicates,
        "invalid_versions": invalid,
        "ok": not missing and not duplicates and not invalid,
    }


def index_fast_forward_plan(
    source_version: int,
    target_version: int,
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> IndexFastForwardPlan | None:
    """Build a contiguous SQL plan, or ``None`` when rebuild/reprocess is required.

    Args:
        source_version: Starting version.
        target_version: Target version.
        declarations: Custom declaration table for testing. Uses INDEX_DELTA_DECLARATIONS if None.
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS

    if source_version < INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR or source_version >= target_version:
        return None
    filtered_declarations = tuple(
        sorted(
            (declaration for declaration in declarations if source_version < declaration.version <= target_version),
            key=lambda declaration: declaration.version,
        )
    )
    if tuple(declaration.version for declaration in filtered_declarations) != tuple(
        range(source_version + 1, target_version + 1)
    ):
        return None
    plan = IndexFastForwardPlan(source_version, target_version, filtered_declarations)
    return plan if plan.eligible_for_sql_fast_forward else None


def get_latest_sql_fast_forwardable_version(
    target_version: int | None = None,
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> int | None:
    """Return the highest version that CAN fast-forward to the given target.

    Useful for tests that need a source version that will actually fast-forward
    without a rebuild. Returns None if no version below the target is fast-forwardable.

    Args:
        target_version: Target version. Defaults to max of declarations.
        declarations: Custom declaration table for testing. Uses INDEX_DELTA_DECLARATIONS if None.
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS
    if target_version is None:
        target_version = max(d.version for d in declarations)

    for version in range(target_version - 1, INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR - 1, -1):
        plan = index_fast_forward_plan(version, target_version, declarations)
        if plan is not None and plan.eligible_for_sql_fast_forward:
            return version
    return None


def get_semantic_reparse_blocking_version_pair(
    target_version: int | None = None,
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> tuple[int, int] | None:
    """Return a (source, target) pair where a SEMANTIC_REPARSE delta blocks the fast-forward.

    Useful for tests that want to verify that semantic-reparse gaps correctly require
    a rebuild. Returns None if all deltas to the target can be fast-forwarded.

    Args:
        target_version: Target version. Defaults to max of declarations.
        declarations: Custom declaration table for testing. Uses INDEX_DELTA_DECLARATIONS if None.
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS
    if target_version is None:
        target_version = max(d.version for d in declarations)

    for source in range(INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR, target_version):
        plan = index_fast_forward_plan(source, target_version, declarations)
        if plan is not None and plan.requires_semantic_reparse:
            return (source, target_version)
    return None


def index_delta_declarations_between(
    source_version: int,
    target_version: int,
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> tuple[IndexDeltaDeclaration, ...]:
    """Return every declaration crossed by ``source_version -> target_version``.

    Unlike :func:`index_fast_forward_plan` this deliberately does not require a
    contiguous, SQL-fast-forwardable run.  The reindex canary exists precisely
    for the semantic-reparse case, where no fast-forward plan is available and
    the declarations are the only statement of what the rebuild is supposed to
    change.  Gaps are reported separately by
    :func:`undeclared_index_delta_versions` rather than silently widening what
    a canary may call expected.
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS
    return tuple(
        sorted(
            (declaration for declaration in declarations if source_version < declaration.version <= target_version),
            key=lambda declaration: declaration.version,
        )
    )


def undeclared_index_delta_versions(
    source_version: int,
    target_version: int,
    declarations: tuple[IndexDeltaDeclaration, ...] | None = None,
) -> tuple[int, ...]:
    """Return crossed versions that ship no declaration at all.

    A crossed version with no declaration can authorize nothing, so anything it
    changes surfaces as UNEXPECTED.  Callers report these versions so the gap is
    visible evidence instead of an unexplained pile of unexpected rows.
    """
    if declarations is None:
        declarations = INDEX_DELTA_DECLARATIONS
    declared = {declaration.version for declaration in declarations}
    return tuple(version for version in range(source_version + 1, target_version + 1) if version not in declared)


__all__ = [
    "CanaryChangeOperation",
    "CanaryChangeScope",
    "DerivedDeltaClass",
    "ExpectedCanaryChange",
    "FastForwardOperation",
    "FastForwardOperationKind",
    "INDEX_DELTA_DECLARATIONS",
    "INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR",
    "IndexDeltaDeclaration",
    "IndexDeltaDeclarationReport",
    "IndexFastForwardPlan",
    "INDEX_SAME_VERSION_SCHEMA_VARIANTS",
    "SameVersionSchemaVariant",
    "TargetedReprocessScope",
    "get_latest_sql_fast_forwardable_version",
    "get_semantic_reparse_blocking_version_pair",
    "index_delta_declaration_report",
    "index_delta_declarations_between",
    "invalid_canary_change_declarations",
    "index_fast_forward_plan",
    "resolve_canonical_index_objects",
    "same_version_schema_variants",
    "undeclared_index_delta_versions",
]
