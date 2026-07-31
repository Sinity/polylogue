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
        # while a cold rebuild populates them. That divergence is exactly what
        # DerivedDeltaClass cannot currently express: there is no class for
        # "additive columns, clone-safe shape, values via targeted reprocess".
        # SEMANTIC_REPARSE is therefore the truthful classification today, and
        # it keeps the existing full-rebuild behaviour rather than silently
        # promoting a generation whose new columns disagree with a cold
        # rebuild. polylogue-9rw0.1 owns extending the vocabulary so this exact
        # delta class becomes a bounded shape fast-forward plus a Codex-scoped
        # reprocess instead of a full-corpus replay.
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
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


__all__ = [
    "DerivedDeltaClass",
    "FastForwardOperation",
    "FastForwardOperationKind",
    "INDEX_DELTA_DECLARATIONS",
    "INDEX_FAST_FORWARD_COMPATIBILITY_FLOOR",
    "IndexDeltaDeclaration",
    "IndexDeltaDeclarationReport",
    "IndexFastForwardPlan",
    "get_latest_sql_fast_forwardable_version",
    "get_semantic_reparse_blocking_version_pair",
    "index_delta_declaration_report",
    "index_fast_forward_plan",
    "resolve_canonical_index_objects",
]
