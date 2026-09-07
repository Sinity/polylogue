"""Executable declarations for the query-contract differential.

The differential is generated from these declarations, never from a
hand-written per-unit test list: every unit in
:data:`~polylogue.archive.query.metadata.QUERY_UNIT_DESCRIPTORS`, every
terminal pipeline stage, every session-``with`` read projection and every
ref-shaped field on a list-emitted row payload is either covered by a law or
carries an explicit exemption naming why. A new unit, stage or ref field with
no declaration fails :mod:`tests.unit.archive.query.test_query_law_harness`.

Nothing here executes a query. :mod:`tests.infra.query_differential` runs the
laws across surfaces and :mod:`tests.infra.query_census` measures them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from polylogue.archive.query.metadata import (
    PROJECTION_QUERY_UNITS,
    QUERY_UNIT_DESCRIPTORS,
    QueryUnitDescriptor,
    QueryUnitName,
)
from polylogue.scenarios.workload import BudgetMeasure, BudgetSemantics, WorkloadBudget

SurfaceName = Literal["python", "cli", "http", "mcp"]
SURFACE_NAMES: tuple[SurfaceName, ...] = ("python", "cli", "http", "mcp")

#: Terminal pipeline stage kinds the executed AST may emit. Mirrors the
#: ``kind`` enum published in ``QueryUnitEnvelope.pipeline_stages``; a stage
#: added there without a law or an exemption here fails the harness.
PipelineStageKind = Literal[
    "session_scope",
    "sort",
    "limit",
    "offset",
    "group",
    "count",
    "agg",
    "transform",
    "terminal",
]
PIPELINE_STAGE_KINDS: tuple[PipelineStageKind, ...] = get_args(PipelineStageKind)


@dataclass(frozen=True, slots=True)
class QueryLaw:
    """One metamorphic or differential law over the terminal query route.

    ``anti_vacuity`` names the mutation that must turn the law red. The
    mutants in :mod:`tests.infra.query_differential` are keyed on these
    ids, so a law whose stated mutation is not actually wired is visible.
    """

    law_id: str
    summary: str
    anti_vacuity: str
    #: ``True`` when the law compares two executions of the same surface
    #: (metamorphic); ``False`` when it compares surfaces (differential).
    metamorphic: bool = True


QUERY_LAWS: tuple[QueryLaw, ...] = (
    QueryLaw(
        "predicate-commutativity",
        "`U where A AND B` selects exactly the rows `U where B AND A` selects, in the same order.",
        "lowering that folds one conjunct into a text filter applied after ordering",
    ),
    QueryLaw(
        "predicate-idempotence",
        "`U where A AND A` selects exactly the rows `U where A` selects.",
        "a predicate lowered as a repeated join rather than a repeated restriction",
    ),
    QueryLaw(
        "page-concatenation",
        "Concatenated pages of size k enumerate each logical member exactly once and equal the unpaged page.",
        "an offset applied to the fetch window but not to the emitted rows",
    ),
    QueryLaw(
        "limit-monotonicity",
        "The rows of `limit=n` are a prefix of the rows of `limit=n+m` for the same request.",
        "a limit pushed below a sort, so a smaller page selects different rows",
    ),
    QueryLaw(
        "group-count-population",
        "Grouped counts sum to the population of matching rows at the grouped grain.",
        "a group-by lowered over the page rather than the match set",
    ),
    QueryLaw(
        "structured-plan-equivalence",
        "Re-executing the typed predicate AST of a DSL expression returns the same page.",
        "execution reading the raw expression text instead of the compiled predicate",
    ),
    QueryLaw(
        "ref-canonicalization",
        "Every list-emitted ref parses as its declared ref family and re-formats identically.",
        "a row payload emitting a provider-native or non-canonical ref string",
    ),
    QueryLaw(
        "ref-detail-closure",
        "Every list-emitted archive ref resolves to the detail record it names.",
        "a list route emitting an id shape the detail route cannot resolve",
    ),
    QueryLaw(
        "continuation-progress",
        "Following continuations enumerates each member exactly once and terminates.",
        "a continuation that replays its own offset, or never clears has_next",
        metamorphic=False,
    ),
    QueryLaw(
        "cancellation-halts-work",
        "A cancelled read never starts, or is interrupted inside SQLite, and completes its cleanup either way.",
        "a cancellation flag observed only after the result page is built",
    ),
    QueryLaw(
        "cross-surface-selection",
        "Every surface returns the same selection, order, totals and paging facts for one request.",
        "a surface-local filter or sort applied outside the shared query route",
        metamorphic=False,
    ),
    QueryLaw(
        "cross-surface-typing",
        "Every surface emits the same public row type and field vocabulary for one request.",
        "a surface projecting a private column or dropping a declared public field",
        metamorphic=False,
    ),
    QueryLaw(
        "cross-surface-error",
        "An unusable expression fails on every surface with the same refusal class.",
        "a surface degrading a compile error into an empty successful page",
        metamorphic=False,
    ),
    QueryLaw(
        "unknown-outcome-agreement",
        "Structural tool outcomes and error facts agree across surfaces and are never inferred from prose.",
        "a surface reading `unknown` tool outcome as success",
        metamorphic=False,
    ),
)

QUERY_LAWS_BY_ID: Mapping[str, QueryLaw] = {law.law_id: law for law in QUERY_LAWS}


@dataclass(frozen=True, slots=True)
class LawExemption:
    """One declared gap in the generated law matrix."""

    law_id: str
    unit: QueryUnitName | None = None
    surface: SurfaceName | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("a law exemption must state its reason")
        if self.law_id not in QUERY_LAWS_BY_ID:
            raise ValueError(f"exemption names an unknown law: {self.law_id!r}")


QUERY_LAW_EXEMPTIONS: tuple[LawExemption, ...] = (
    LawExemption(
        "group-count-population",
        unit="run",
        reason="the run descriptor declares no aggregate group fields, so no grouped page exists to sum",
    ),
    LawExemption(
        "group-count-population",
        unit="context-snapshot",
        reason="the context-snapshot descriptor declares no aggregate group fields",
    ),
    LawExemption(
        "continuation-progress",
        surface="cli",
        reason=(
            "the CLI emits pages through query_unit_rows without a QueryTransaction, so it has no "
            "continuation token; its paging is covered by page-concatenation over explicit offsets"
        ),
    ),
)


def law_applies(law_id: str, *, unit: QueryUnitName | None = None, surface: SurfaceName | None = None) -> bool:
    """Return whether a law is in force for a unit/surface pair."""

    for exemption in QUERY_LAW_EXEMPTIONS:
        if exemption.law_id != law_id:
            continue
        if exemption.unit is not None and exemption.unit != unit:
            continue
        if exemption.surface is not None and exemption.surface != surface:
            continue
        return False
    return True


def exemption_reason(law_id: str, *, unit: QueryUnitName | None = None, surface: SurfaceName | None = None) -> str:
    """Return the declared reason a law does not apply, for receipts."""

    for exemption in QUERY_LAW_EXEMPTIONS:
        if exemption.law_id != law_id:
            continue
        if exemption.unit is not None and exemption.unit != unit:
            continue
        if exemption.surface is not None and exemption.surface != surface:
            continue
        return exemption.reason
    return ""


#: Stage kinds whose behaviour a law directly constrains. A stage kind absent
#: here needs a :data:`PIPELINE_STAGE_EXEMPTIONS` entry.
PIPELINE_STAGE_LAWS: Mapping[PipelineStageKind, tuple[str, ...]] = {
    "session_scope": ("predicate-commutativity", "predicate-idempotence", "structured-plan-equivalence"),
    "sort": ("limit-monotonicity", "page-concatenation"),
    "limit": ("limit-monotonicity",),
    "offset": ("page-concatenation",),
    "group": ("group-count-population",),
    "count": ("group-count-population",),
    "terminal": ("cross-surface-selection",),
}

PIPELINE_STAGE_EXEMPTIONS: Mapping[PipelineStageKind, str] = {
    "agg": (
        "agg reducers run in Python over a capped row fetch and publish their own exactness flag; "
        "their sampling contract is owned by the agg-metric tests, not by a page-level law"
    ),
    "transform": (
        "the transform stage only renames emitted fields; the underlying selection is already "
        "constrained by every row-level law"
    ),
}


@dataclass(frozen=True, slots=True)
class UnitIdentity:
    """Stable per-unit row identity used to compare selections semantically."""

    unit: QueryUnitName
    fields: tuple[str, ...]
    reason: str = ""


UNIT_IDENTITIES: tuple[UnitIdentity, ...] = (
    UnitIdentity("message", ("message_id",)),
    UnitIdentity("action", ("tool_use_block_id",)),
    UnitIdentity("block", ("block_id",)),
    UnitIdentity("assertion", ("assertion_id",)),
    UnitIdentity(
        "file",
        ("session_id", "path"),
        reason="a file row is a per-session path rollup and carries no single generated id",
    ),
    UnitIdentity("run", ("run_ref",)),
    UnitIdentity("observed-event", ("event_ref",)),
    UnitIdentity("context-snapshot", ("snapshot_ref",)),
    UnitIdentity("delegation", ("delegation_ref",)),
)

UNIT_IDENTITY_BY_UNIT: Mapping[QueryUnitName, UnitIdentity] = {identity.unit: identity for identity in UNIT_IDENTITIES}


@dataclass(frozen=True, slots=True)
class UnitProbe:
    """The generated request shape one unit's laws are executed with.

    ``scope`` selects the whole corpus at this unit's grain and ``narrowing``
    selects a strict, non-empty subset of it, so a conjunction law cannot hold
    vacuously by both conjuncts selecting the same rows. ``group_field`` is the
    aggregate group key the population law sums over; ``None`` requires a
    ``group-count-population`` exemption for the unit.
    """

    unit: QueryUnitName
    source: str
    scope: str
    narrowing: str
    group_field: str | None

    @property
    def scoped_expression(self) -> str:
        return f"{self.source} where {self.scope}"

    def conjunction(self, first: str, second: str) -> str:
        return f"{self.source} where {first} AND {second}"


#: The corpus origin every probe scopes on. Declared here so the corpus and
#: the probes cannot drift apart silently.
PROBE_SCOPE = "session.origin:claude-code-session"

UNIT_PROBES: tuple[UnitProbe, ...] = (
    UnitProbe("message", "messages", PROBE_SCOPE, "role:assistant", "role"),
    UnitProbe("action", "actions", PROBE_SCOPE, "tool:Bash", "tool"),
    UnitProbe("block", "blocks", PROBE_SCOPE, "type:tool_use", "type"),
    UnitProbe("assertion", "assertions", PROBE_SCOPE, "kind:tag", "kind"),
    UnitProbe("file", "files", PROBE_SCOPE, "tool:Edit", "path"),
    UnitProbe("run", "runs", PROBE_SCOPE, "session.messages:>4", None),
    UnitProbe("observed-event", "observed-events", PROBE_SCOPE, "tool:Bash", "kind"),
    UnitProbe("context-snapshot", "context-snapshots", PROBE_SCOPE, "boundary:session_start", None),
    UnitProbe("delegation", "delegations", PROBE_SCOPE, "mapping_state:unresolved", "mapping_state"),
)

UNIT_PROBE_BY_UNIT: Mapping[QueryUnitName, UnitProbe] = {probe.unit: probe for probe in UNIT_PROBES}

#: An expression no unit can compile. Every surface must refuse it identically.
UNCOMPILABLE_EXPRESSION = "messages where nosuchfield:1"


RefShape = Literal["object-ref", "evidence-ref", "archive-id"]
RefTarget = Literal["session", "message", "block", "action", "none"]


@dataclass(frozen=True, slots=True)
class RefFamily:
    """One list-emitted reference family on a terminal row payload."""

    unit: QueryUnitName
    field: str
    shape: RefShape
    target: RefTarget
    many: bool = False
    #: ``True`` when the ref names an archive record the detail route must
    #: resolve. ``False`` for refs into projections with no detail read.
    detail_closure: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.detail_closure and self.target == "none":
            raise ValueError(f"{self.unit}.{self.field}: detail closure requires an archive target")
        if not self.detail_closure and not self.reason:
            raise ValueError(f"{self.unit}.{self.field}: a ref without detail closure must state why")


REF_FAMILIES: tuple[RefFamily, ...] = (
    RefFamily("message", "message_id", "archive-id", "message", detail_closure=True),
    RefFamily("message", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily("action", "message_id", "archive-id", "message", detail_closure=True),
    RefFamily("action", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily("action", "tool_use_block_id", "archive-id", "block", detail_closure=True),
    RefFamily(
        "action",
        "tool_result_block_id",
        "archive-id",
        "block",
        reason="null for an unpaired tool use; the missing-result population is asserted separately",
    ),
    RefFamily(
        "action",
        "followup_message_ref",
        "archive-id",
        "message",
        reason="a followup pointer is optional evidence, absent whenever no followup was classified",
    ),
    RefFamily("block", "block_id", "archive-id", "block", detail_closure=True),
    RefFamily("block", "message_id", "archive-id", "message", detail_closure=True),
    RefFamily("block", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "assertion",
        "assertion_id",
        "archive-id",
        "none",
        reason="the durable user-tier assertion id; assertions are read through their claim listing, not a detail route",
    ),
    RefFamily(
        "assertion",
        "target_ref",
        "object-ref",
        "none",
        reason="an assertion may target any public object kind, including kinds with no archive record",
    ),
    RefFamily(
        "assertion",
        "scope_ref",
        "object-ref",
        "none",
        reason="scope is an optional narrowing ref, frequently absent",
    ),
    RefFamily(
        "assertion",
        "author_ref",
        "object-ref",
        "none",
        reason="the author is an actor identity, not an archive record",
    ),
    RefFamily(
        "assertion",
        "evidence_refs",
        "evidence-ref",
        "none",
        many=True,
        reason="assertion evidence may cite material outside the index tier",
    ),
    RefFamily("file", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "file",
        "first_message_id",
        "archive-id",
        "message",
        reason="null when the path rollup has no first-touch message evidence",
    ),
    RefFamily(
        "file",
        "first_tool_use_block_id",
        "archive-id",
        "block",
        reason="null when the path was observed without a tool-use block",
    ),
    RefFamily(
        "file",
        "last_tool_use_block_id",
        "archive-id",
        "block",
        reason="null when the path was observed without a tool-use block",
    ),
    RefFamily("run", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "run",
        "run_ref",
        "object-ref",
        "none",
        reason="a run is a projection identity compiled from the session, not a stored record",
    ),
    RefFamily("run", "parent_run_ref", "object-ref", "none", reason="absent for a root run"),
    RefFamily("run", "agent_ref", "object-ref", "none", reason="absent when no agent identity was observed"),
    RefFamily("run", "lineage_refs", "object-ref", "none", many=True, reason="empty for a run with no lineage"),
    RefFamily(
        "run",
        "context_snapshot_ref",
        "object-ref",
        "none",
        reason="absent when the run has no boundary snapshot",
    ),
    RefFamily(
        "run",
        "transcript_ref",
        "evidence-ref",
        "none",
        reason="a transcript pointer addresses acquired source bytes, not an index record",
    ),
    RefFamily("run", "evidence_refs", "evidence-ref", "none", many=True, reason="empty when no evidence was cited"),
    RefFamily("observed-event", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "observed-event",
        "event_ref",
        "object-ref",
        "none",
        reason="an observed event is a projection identity, not a stored record",
    ),
    RefFamily("observed-event", "subject_ref", "object-ref", "none", reason="absent for a subjectless event"),
    RefFamily(
        "observed-event",
        "object_refs",
        "object-ref",
        "none",
        many=True,
        reason="object refs address tool calls and external identities, not index records",
    ),
    RefFamily(
        "observed-event",
        "evidence_refs",
        "evidence-ref",
        "none",
        many=True,
        reason="empty when the event carries no cited evidence",
    ),
    RefFamily("context-snapshot", "session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "context-snapshot",
        "snapshot_ref",
        "object-ref",
        "none",
        reason="a context snapshot is a projection identity, not a stored record",
    ),
    RefFamily(
        "context-snapshot",
        "run_ref",
        "object-ref",
        "none",
        reason="the owning run is itself a projection identity",
    ),
    RefFamily(
        "context-snapshot",
        "segment_refs",
        "object-ref",
        "none",
        many=True,
        reason="segments are projection identities within the snapshot",
    ),
    RefFamily(
        "context-snapshot",
        "evidence_refs",
        "evidence-ref",
        "none",
        many=True,
        reason="empty when the snapshot cites no evidence",
    ),
    RefFamily(
        "delegation",
        "delegation_ref",
        "object-ref",
        "none",
        reason="a delegation ref addresses the view row, whose two id shapes have no single detail read",
    ),
    RefFamily("delegation", "parent_session_id", "archive-id", "session", detail_closure=True),
    RefFamily(
        "delegation",
        "child_session_id",
        "archive-id",
        "session",
        reason="null for an unresolved dispatch with no observed child",
    ),
    RefFamily(
        "delegation",
        "instruction_message_id",
        "archive-id",
        "message",
        reason="null for an edge-only delegation with no parent-side dispatch",
    ),
    RefFamily(
        "delegation",
        "instruction_tool_use_block_id",
        "archive-id",
        "block",
        reason="null for an edge-only delegation with no parent-side dispatch",
    ),
    RefFamily(
        "delegation",
        "artifact_block_id",
        "archive-id",
        "block",
        reason="null when the dispatch produced no observed result block",
    ),
    RefFamily(
        "delegation",
        "evidence_refs",
        "archive-id",
        "none",
        many=True,
        reason="delegation evidence is emitted as bare kind-prefixed ids over instruction and artifact blocks",
    ),
)

REF_FAMILIES_BY_UNIT: Mapping[QueryUnitName, tuple[RefFamily, ...]] = {
    descriptor.unit: tuple(family for family in REF_FAMILIES if family.unit == descriptor.unit)
    for descriptor in QUERY_UNIT_DESCRIPTORS
}

#: Row payload fields whose name looks ref-shaped but which carry no
#: reference. Each needs a reason so a genuinely new ref cannot hide here.
NON_REF_FIELDS: Mapping[str, str] = {
    "assertion.key": "an opaque assertion key, not a reference",
    "delegation.instruction_sha256": "a content digest of the instruction text",
    "delegation.artifact_sha256": "a content digest of the artifact text",
    "run.native_session_id": "the provider-native id retained for provenance, deliberately not a public ref",
    "run.native_parent_session_id": "the provider-native parent id retained for provenance",
}


def ref_suspected_fields(descriptor: QueryUnitDescriptor) -> tuple[str, ...]:
    """Return payload fields whose shape claims to be a reference.

    The rule is deliberately syntactic: anything named ``*_ref``/``*_refs`` or
    ending in a generated archive identity suffix. A payload gaining such a
    field without a :data:`REF_FAMILIES` or :data:`NON_REF_FIELDS` entry is a
    new, undeclared ref family.
    """

    from polylogue.surfaces import payloads as surface_payloads

    model = getattr(surface_payloads, descriptor.payload_model, None)
    fields = getattr(model, "model_fields", {})
    suspected = []
    for name in fields:
        if name.endswith(("_ref", "_refs")) or name.endswith(("_id", "_ids")):
            suspected.append(name)
    return tuple(sorted(suspected))


@dataclass(frozen=True, slots=True)
class ProjectionCoverage:
    """One session-``with <unit>`` read projection and its covering law."""

    unit: QueryUnitName
    law_id: str


PROJECTION_COVERAGE: tuple[ProjectionCoverage, ...] = tuple(
    ProjectionCoverage(unit, "cross-surface-selection") for unit in sorted(PROJECTION_QUERY_UNITS)
)


ScanDisposition = Literal["expected", "classified"]


@dataclass(frozen=True, slots=True)
class ScanAllowance:
    """One declared full scan or materialization in a census family's plan."""

    family_id: str
    detail: str
    disposition: ScanDisposition
    owner: str
    reason: str


@dataclass(frozen=True, slots=True)
class CensusFamily:
    """One declared query family measured by the workload census.

    ``slo_owner`` names the read surface in ``docs/plans/slo-catalog.yaml``
    that owns this family's latency budget; the resource budgets below are
    the deterministic half of the same envelope and gate regressions.
    """

    family_id: str
    unit: QueryUnitName
    expression: str
    selectivity: Literal["exact-one", "observed-p50", "observed-p99"]
    slo_owner: str
    #: SQL that computes the same logical identity set with the cheapest
    #: correct primitive. The census compares the routed answer against it.
    cheapest_primitive_sql: str
    cheapest_primitive_identity: str
    #: The lowered restriction this family's selectivity depends on. The census
    #: requires it to appear in the routed SQL *before* the first ``ORDER BY``:
    #: a selective predicate applied only after a global window or group is the
    #: shape that made a one-coordinator page read the whole archive.
    pushdown_marker: str
    budgets: tuple[WorkloadBudget, ...]
    scan_allowances: tuple[ScanAllowance, ...] = ()


def _regression_gate(measure: BudgetMeasure, maximum: float) -> WorkloadBudget:
    return WorkloadBudget(measure=measure, maximum=maximum, semantics=BudgetSemantics.REGRESSION_GATE, phase="query")


def _measure_only(measure: BudgetMeasure) -> WorkloadBudget:
    # A wall-clock or RSS ceiling on a shared, contended host is a coin flip;
    # these are recorded as evidence and gated only by the deterministic
    # SQLite work and response-size measures above.
    return WorkloadBudget(
        measure=measure,
        maximum=_MEASURE_ONLY_CEILING,
        semantics=BudgetSemantics.MEASURE_ONLY,
        phase="query",
    )


#: A measure-only budget still needs a maximum to be expressible; this one is
#: far above any observation and never decides a verdict.
_MEASURE_ONLY_CEILING = 1e15


_MEASURE_ONLY_MEASURES: tuple[BudgetMeasure, ...] = (
    BudgetMeasure.WALL_MS,
    BudgetMeasure.CPU_MS,
    BudgetMeasure.PEAK_RSS_BYTES,
    BudgetMeasure.ANON_BYTES,
    BudgetMeasure.FILE_CACHE_BYTES,
    BudgetMeasure.SWAP_BYTES,
    BudgetMeasure.TEMP_STORAGE_BYTES,
    BudgetMeasure.READ_IO_BYTES,
    BudgetMeasure.WRITE_IO_BYTES,
)


def _family_budgets(*, vm_steps: int, response_bytes: int) -> tuple[WorkloadBudget, ...]:
    return (
        _regression_gate(BudgetMeasure.SQLITE_VM_STEPS, vm_steps),
        _regression_gate(BudgetMeasure.RESPONSE_BYTES, response_bytes),
        *(_measure_only(measure) for measure in _MEASURE_ONLY_MEASURES),
    )


CENSUS_FAMILIES: tuple[CensusFamily, ...] = (
    CensusFamily(
        family_id="query:messages:origin-scope",
        unit="message",
        expression="messages where session.origin:claude-code-session",
        selectivity="observed-p99",
        slo_owner="query",
        cheapest_primitive_sql=(
            "SELECT m.message_id FROM messages m JOIN sessions s ON s.session_id = m.session_id "
            "WHERE s.origin = 'claude-code-session'"
        ),
        cheapest_primitive_identity="message_id",
        pushdown_marker="WHERE s.origin IN ('claude-code-session')",
        budgets=_family_budgets(vm_steps=4_000_000, response_bytes=4_000_000),
        scan_allowances=(
            ScanAllowance(
                family_id="query:messages:origin-scope",
                detail="SCAN sqlite_master",
                disposition="expected",
                owner="polylogue.storage.sqlite.connection_profile",
                reason="opening a tier reads its schema inventory once to validate identity and find the FTS table",
            ),
            ScanAllowance(
                family_id="query:messages:origin-scope",
                detail="USE TEMP B-TREE FOR ORDER BY",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason=(
                    "the stable order is a COALESCE across the unit and its owning session, which no single "
                    "index covers; the sort is over the bounded page, not the archive"
                ),
            ),
            ScanAllowance(
                family_id="query:messages:origin-scope",
                detail="SCAN (subquery-1)",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason="the repo rollup is a correlated per-row subquery over one session's repo edges",
            ),
            ScanAllowance(
                family_id="query:messages:origin-scope",
                detail="SCAN ordered",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason="the message text is concatenated from one message's own ordered blocks",
            ),
        ),
    ),
    CensusFamily(
        family_id="query:actions:tool-scope",
        unit="action",
        expression="actions where tool:Workflow",
        selectivity="observed-p50",
        slo_owner="query",
        cheapest_primitive_sql=(
            "SELECT u.block_id FROM blocks u WHERE u.block_type = 'tool_use' AND lower(u.tool_name) = 'workflow'"
        ),
        cheapest_primitive_identity="tool_use_block_id",
        pushdown_marker="WHERE lower(a.tool_name) = 'workflow'",
        budgets=_family_budgets(vm_steps=4_000_000, response_bytes=2_000_000),
        scan_allowances=(
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="SCAN sqlite_master",
                disposition="expected",
                owner="polylogue.storage.sqlite.connection_profile",
                reason="opening a tier reads its schema inventory once to validate identity and find the FTS table",
            ),
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="USE TEMP B-TREE FOR ORDER BY",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason=(
                    "the stable order is a COALESCE across the unit and its owning session, which no single "
                    "index covers; the sort is over the bounded page, not the archive"
                ),
            ),
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="SCAN ap USING INDEX sqlite_autoindex_action_pairs_1",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.index",
                reason="the actions view pairs tool_use with tool_result through the action_pairs relation",
            ),
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="SCAN a",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason=(
                    "`a` is the selected_actions CTE, already restricted by the pushed-down tool predicate "
                    "and the page limit, not the archive-wide action relation"
                ),
            ),
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="SCAN ordered",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason="follow-up classification reads one selected action's own message tail",
            ),
            ScanAllowance(
                family_id="query:actions:tool-scope",
                detail="USE TEMP B-TREE FOR LAST TERM OF ORDER BY",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason="the follow-up lookups tie-break on message_id, which the position index does not carry",
            ),
        ),
    ),
    CensusFamily(
        family_id="query:delegations:coordinator-scope",
        unit="delegation",
        expression="delegations where session.origin:claude-code-session",
        selectivity="exact-one",
        slo_owner="query",
        cheapest_primitive_sql=(
            "SELECT COALESCE('delegation:' || d.instruction_tool_use_block_id, "
            "'delegation:edge:' || d.parent_session_id || '::' || d.child_session_id) "
            "FROM delegation_facts d JOIN sessions s ON s.session_id = d.parent_session_id "
            "WHERE s.origin = 'claude-code-session'"
        ),
        cheapest_primitive_identity="delegation_ref",
        pushdown_marker="WHERE s.origin IN ('claude-code-session')",
        budgets=_family_budgets(vm_steps=4_000_000, response_bytes=2_000_000),
        scan_allowances=(
            ScanAllowance(
                family_id="query:delegations:coordinator-scope",
                detail="SCAN sqlite_master",
                disposition="expected",
                owner="polylogue.storage.sqlite.connection_profile",
                reason="opening a tier reads its schema inventory once to validate identity and find the FTS table",
            ),
            ScanAllowance(
                family_id="query:delegations:coordinator-scope",
                detail="USE TEMP B-TREE FOR ORDER BY",
                disposition="expected",
                owner="polylogue.storage.sqlite.archive_tiers.archive_query_reads",
                reason=(
                    "the stable order is a COALESCE across the unit and its owning session, which no single "
                    "index covers; the sort is over the bounded page, not the archive"
                ),
            ),
        ),
    ),
)

CENSUS_FAMILIES_BY_ID: Mapping[str, CensusFamily] = {family.family_id: family for family in CENSUS_FAMILIES}


@dataclass(frozen=True, slots=True)
class ShapeAnchor:
    """One corpus shape target quoted from an archive-composition profile.

    ``distribution_ref`` is the key the production profile builder
    (:func:`polylogue.schemas.generation.archive_workload_profile.build_archive_workload_profile`)
    emits for the same dimension, so a recorded profile can replace these
    declared values without changing the generator. The committed values are
    declared anchors: no operator archive profile is tracked in this
    repository.
    """

    distribution_ref: str
    dimension: str
    p50: float
    p95: float
    maximum: float


CORPUS_SHAPE_ANCHORS: tuple[ShapeAnchor, ...] = (
    ShapeAnchor("index.session_shapes.message_count", "messages-per-session", 4, 12, 24),
    ShapeAnchor("index.action_shapes.tool_uses_per_session", "tool-uses-per-session", 2, 5, 8),
    ShapeAnchor("index.action_shapes.tool_results_per_session", "tool-results-per-session", 2, 5, 9),
    ShapeAnchor("index.topology.children_per_parent", "children-per-parent", 1, 3, 4),
    ShapeAnchor("source.blob_size", "largest-block-bytes", 512, 8_192, 65_536),
)

CORPUS_SHAPE_ANCHORS_BY_DIMENSION: Mapping[str, ShapeAnchor] = {
    anchor.dimension: anchor for anchor in CORPUS_SHAPE_ANCHORS
}


PathologyName = Literal[
    "duplicate-tool-result",
    "missing-tool-result",
    "late-tool-result",
    "wide-lineage",
    "deep-lineage",
    "active-growth",
    "large-payload",
    "low-selectivity",
    "high-selectivity",
    "mandate-incident",
]
REQUIRED_PATHOLOGIES: tuple[PathologyName, ...] = get_args(PathologyName)


__all__ = [
    "CENSUS_FAMILIES",
    "CENSUS_FAMILIES_BY_ID",
    "CORPUS_SHAPE_ANCHORS",
    "CORPUS_SHAPE_ANCHORS_BY_DIMENSION",
    "NON_REF_FIELDS",
    "PIPELINE_STAGE_EXEMPTIONS",
    "PROBE_SCOPE",
    "PIPELINE_STAGE_KINDS",
    "PIPELINE_STAGE_LAWS",
    "PROJECTION_COVERAGE",
    "QUERY_LAWS",
    "QUERY_LAWS_BY_ID",
    "QUERY_LAW_EXEMPTIONS",
    "REF_FAMILIES",
    "REF_FAMILIES_BY_UNIT",
    "REQUIRED_PATHOLOGIES",
    "SURFACE_NAMES",
    "UNCOMPILABLE_EXPRESSION",
    "UNIT_IDENTITIES",
    "UNIT_IDENTITY_BY_UNIT",
    "UNIT_PROBES",
    "UNIT_PROBE_BY_UNIT",
    "CensusFamily",
    "LawExemption",
    "PathologyName",
    "PipelineStageKind",
    "ProjectionCoverage",
    "QueryLaw",
    "RefFamily",
    "ScanAllowance",
    "ShapeAnchor",
    "SurfaceName",
    "UnitIdentity",
    "UnitProbe",
    "exemption_reason",
    "law_applies",
    "ref_suspected_fields",
]
