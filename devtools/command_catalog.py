"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"

CATEGORY_ORDER: tuple[str, ...] = (
    "core",
    "verification",
    "generated surfaces",
    "schema",
    "benchmarking",
    "archive",
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    category: str
    description: str
    module: str
    entrypoint: str = "main"
    json_flag: bool = False
    #: Flags surfaced in this command's own ``--help`` and forwarded verbatim.
    flags: tuple[tuple[str, str], ...] = ()
    use_when: str | None = None
    examples: tuple[str, ...] = ()
    featured: bool = False

    @property
    def command_path(self) -> tuple[str, ...]:
        return tuple(part for part in self.name.split(" ") if part)

    @property
    def invocation(self) -> str:
        return control_plane_command(*self.command_path)

    @property
    def argv(self) -> tuple[str, ...]:
        return control_plane_argv(*self.command_path)

    def resolve_main(self) -> CommandMain:
        module = importlib.import_module(self.module)
        entrypoint = getattr(module, self.entrypoint)
        if not callable(entrypoint):
            raise TypeError(f"{self.module}.{self.entrypoint} is not callable")

        def _main(argv: list[str] | None = None) -> int:
            result = entrypoint(argv)
            if not isinstance(result, int):
                raise TypeError(f"{self.module}.{self.entrypoint} returned {type(result).__name__}, expected int")
            return result

        return _main

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["invocation"] = self.invocation
        data["argv"] = list(self.argv)
        return data


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "status",
        "core",
        "Render the devshell status view.",
        "devtools.project_motd",
        json_flag=True,
        use_when="Check repo state, generated-surface drift, and the next default verification steps.",
        examples=("devtools status", "devtools status --json", "devtools status --verify-generated"),
        featured=True,
    ),
    CommandSpec(
        "test",
        "core",
        "Run focused pytest selections or inspect full-run timing outliers.",
        "devtools.run_tests",
        use_when=(
            "Run a specific test file, directory, or -k/-m selection in the inner loop, or inspect the latest "
            "full-run timing receipts, without invoking raw pytest."
        ),
        examples=(
            "devtools test tests/unit/pipeline",
            "devtools test -k hybrid",
            "devtools test tests/unit/storage -x",
            "devtools test --outliers 20",
        ),
        featured=True,
    ),
    CommandSpec(
        "why",
        "core",
        "Explain the most recent verification run, or where verification time went.",
        "devtools.why",
        json_flag=True,
        use_when="A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.",
        examples=(
            "devtools why",
            "devtools why --history 24",
            "devtools why --run 20260817T213631Z-2709409-d5c6e72c",
        ),
        featured=True,
    ),
    CommandSpec(
        "schema new",
        "schema",
        "Scaffold a typed declaration, adapter stub, contract skeleton, and landing plan.",
        "devtools.scaffold",
        json_flag=True,
        flags=(
            ("--list", "List the declared scaffold families."),
            ("--dry-run", "Print the plan without writing any file."),
            ("--reuse-family", "Join an existing declaration family instead of starting a new one."),
            ("--justification", "Record why a new family or durable object is needed."),
            ("--validate", "Report actionable diagnostics for the family's live owning registry."),
        ),
        use_when=(
            "Adding an MCP tool, daemon route, or other declaration-backed extension: the scaffold asks the five "
            "compatibility questions, refuses an unjustified new family or durable object, and generates a "
            "compiling declaration plus the exact steps and generated-surface commands needed to land it."
        ),
        examples=(
            "devtools schema new --list",
            "devtools schema new mcp-tool recall --identity mcp-tool:recall --lifecycle registered-handler-retained "
            "--authority mcp-capability:read --access-result-shape query:exhaustive_page:envelope "
            "--durability transport-adapter --dry-run",
            "devtools schema new --validate mcp-tool",
        ),
    ),
    CommandSpec(
        "cache gc",
        "core",
        "Preview or apply age-gated GC for the shared seeded-archive fixture cache.",
        "devtools.seeded_archive_cache_gc",
        json_flag=True,
        flags=(("--apply", "Apply the previewed collection instead of only reporting it."),),
        use_when=(
            "Maintain the reusable NVMe seeded-artifact cache from the generated default, named-workload, and "
            "benchmark reachability inventory. Preview is the default; pass --apply explicitly after reviewing "
            "the bounded receipt."
        ),
        examples=("devtools cache gc --json", "devtools cache gc --apply --json"),
    ),
    CommandSpec(
        "verify",
        "verification",
        "Run every quick gate, then a bounded affected selection or the explicit complete test corpus.",
        "devtools.verify",
        json_flag=True,
        flags=(
            ("--quick", "Run the static gates only."),
            ("--all", "Run the static gates plus the complete test corpus."),
        ),
        use_when="Run the gates and bounded affected tests locally before pushing. --quick stops at static gates; --all runs the complete corpus at the explicit master/corpus boundary. Unknown or oversized affected plans are refused before pytest and name the count, reason, and next boundary.",
        examples=("devtools verify", "devtools verify --quick", "devtools verify --all"),
        featured=True,
    ),
    CommandSpec(
        "verify provider-completeness",
        "verification",
        "Report provider/importer package completeness from OriginSpec declarations.",
        "devtools.verify_provider_completeness",
        json_flag=True,
        flags=(("--check", "Fail when an accepted package has missing or partial required evidence."),),
        use_when=(
            "Check detector, parser, fixture, schema, query, read, explain, privacy, and documentation evidence "
            "before accepting a source or citing origin coverage."
        ),
        examples=(
            "devtools verify provider-completeness --json",
            "devtools verify provider-completeness --origin codex-session --check",
        ),
    ),
    CommandSpec(
        "verify api-parity",
        "verification",
        "Check CLI/MCP/Python semantic-operation parity and the library documentation.",
        "devtools.verify_api_parity",
        json_flag=True,
        flags=(("--check", "Exit non-zero when any parity or documentation finding is reported."),),
        use_when=(
            "After changing the Python facade, an MCP tool declaration, or docs/library-api.md, to prove every "
            "public callable is still bound by a semantic operation or an explicit exclusion and that the "
            "documented calls still match the live signatures."
        ),
        examples=(
            "devtools verify api-parity",
            "devtools verify api-parity --json",
        ),
    ),
    CommandSpec(
        "verify cli-acceptance",
        "verification",
        "Render, lint and measure the public CLI acceptance surface.",
        "devtools.cli_acceptance",
        json_flag=True,
        flags=(
            ("--skip-benchmarks", "Skip the latency lane instead of running the SLO benchmarks."),
            ("--include-lab", "Include the lab-tier latency rows (cold start, warm status, concurrent reads)."),
        ),
        use_when=(
            "Demonstrate CLI usability, accessibility and latency in one run: the gallery of rendered "
            "output, its accessibility lint, the typed-termination scenarios, and the declared "
            "latency budgets."
        ),
        examples=(
            "devtools verify cli-acceptance --skip-benchmarks",
            "devtools verify cli-acceptance --gallery-out .cache/cli-gallery.md --skip-benchmarks",
            "devtools verify cli-acceptance --include-lab --json",
        ),
    ),
    CommandSpec(
        "gate",
        "verification",
        "Run one named invariant check.",
        "devtools.gate",
        use_when="Run a single gate in isolation, or list the declared gates and which of them verify --quick runs.",
        examples=("devtools gate --list", "devtools gate layering", "devtools gate mypy"),
        featured=True,
    ),
    CommandSpec(
        "render",
        "generated surfaces",
        "Refresh or verify one generated repository surface, or all of them.",
        "devtools.render_all",
        flags=(("--check", "Exit non-zero when a selected surface is out of sync."),),
        use_when="Refresh or verify generated repo surfaces after changing docs, CLI help, declarations, or agent memory.",
        examples=("devtools render all", "devtools render all --check", "devtools render cli-reference"),
        featured=True,
    ),
    CommandSpec(
        "scenario",
        "verification",
        "Run a named archive verification scenario.",
        "devtools.verification_scenario",
        json_flag=True,
        use_when="Run a named archive verification scenario through the direct CLI path.",
        examples=(
            "devtools scenario list",
            "devtools scenario run archive-smoke --tier 0",
            "devtools scenario run storage-correctness --report-dir .cache/storage-correctness-report --json",
        ),
    ),
    CommandSpec(
        "smoke",
        "verification",
        "Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow.",
        "devtools.deployment_smoke",
        json_flag=True,
        use_when=(
            "After a system rebuild or before live UI probing, verify that the systemwide "
            "polylogue/polylogued binaries, loopback daemon routes, browser-capture receiver, "
            "and browser-capture archive materialization match the expected deployed surface."
        ),
        examples=("devtools smoke", "devtools smoke --json"),
    ),
    CommandSpec(
        "schema list",
        "schema",
        "List committed schema packages, versions, and evidence manifests.",
        "devtools.schema_inspect",
        entrypoint="list_main",
        json_flag=True,
        use_when="Inspect committed provider schema package catalogs without presenting them as normal archive usage.",
        examples=("devtools schema list --provider chatgpt --json",),
    ),
    CommandSpec(
        "schema closure",
        "schema",
        "Report which source files feed the derived schema identity.",
        "devtools.schema_closure",
        json_flag=True,
        use_when=(
            "Before merging a substrate change, to learn whether it moves the derived schema identity -- "
            "membership follows the import graph, so a third of the package qualifies and no directory rule "
            "describes it."
        ),
        examples=(
            "devtools schema closure polylogue/daemon/write_coordinator.py",
            "devtools schema closure",
        ),
    ),
    CommandSpec(
        "schema compare",
        "schema",
        "Compare two committed schema package versions for a provider.",
        "devtools.schema_inspect",
        entrypoint="compare_main",
        json_flag=True,
        use_when="Review schema package drift between committed versions.",
        examples=("devtools schema compare --provider chatgpt --from v1 --to v2 --markdown",),
    ),
    CommandSpec(
        "schema explain",
        "schema",
        "Explain a committed package element schema with evidence and annotations.",
        "devtools.schema_inspect",
        entrypoint="explain_main",
        json_flag=True,
        use_when="Inspect schema package annotations, semantic roles, and review evidence.",
        examples=("devtools schema explain --provider chatgpt --version latest --verbose",),
    ),
    CommandSpec(
        "schema generate",
        "schema",
        "Generate provider schema packages and optional evidence clusters.",
        "devtools.schema_generate",
        json_flag=True,
        use_when="Refresh provider schema package artifacts from archive observations outside the archive CLI.",
        examples=("devtools schema generate --provider chatgpt --cluster",),
    ),
    CommandSpec(
        "schema commit",
        "schema",
        "Persist a real full-corpus schema generation into committed provider packages.",
        "devtools.schema_commit",
        json_flag=True,
        use_when=(
            "Actually regenerate and write `polylogue/schemas/providers/<provider>/versions/...` from the live "
            "archive -- `schema generate` only ever previews and never writes committed package files."
        ),
        examples=(
            "devtools schema commit --provider chatgpt --full-corpus --dry-run",
            "devtools schema commit --provider chatgpt --full-corpus",
        ),
    ),
    CommandSpec(
        "schema frontier",
        "schema",
        "Declare, record and check the schema-source frontier.",
        "devtools.schema_frontier",
        json_flag=True,
        use_when=(
            "Before and after any provider schema generation run: the frontier names every declared source root "
            "with its exclusions and records each root's admitted membership, so a moved, emptied or mutated root "
            "fails the check instead of silently narrowing the sample set."
        ),
        examples=(
            "devtools schema frontier",
            "devtools schema frontier --list",
            "devtools schema frontier --record --subject gemini-cli",
            "devtools schema frontier --verify-content --json",
        ),
    ),
    CommandSpec(
        "schema reconcile",
        "schema",
        "Account for every declared schema subject after a generation pass.",
        "devtools.schema_reconcile",
        json_flag=True,
        use_when=(
            "After a provider schema generation pass: derives the provider denominator from the schema-subject "
            "and OriginSpec declarations rather than from the run's own output, so a subject the pass never "
            "reached is recorded as not-run instead of vanishing, and binds the matrix to the baseline digest, "
            "code revision, generator semantics and resolved inference configuration."
        ),
        examples=(
            "devtools schema reconcile --receipts /realm/tmp/work/schema-run/receipts",
            "devtools schema reconcile --receipts ./receipts --write polylogue/schemas/providers/provider-matrix.json",
        ),
    ),
    CommandSpec(
        "schema promote",
        "schema",
        "Promote a schema evidence cluster into a registered package version.",
        "devtools.schema_promote",
        json_flag=True,
        use_when="Turn reviewed schema evidence clusters into committed provider schema packages.",
        examples=("devtools schema promote --provider chatgpt --cluster chatgpt-message-v2",),
    ),
    CommandSpec(
        "schema parser-diff",
        "schema",
        "List observed provider wire keys that no parser references.",
        "devtools.schema_parser_diff",
        json_flag=True,
        use_when=(
            "Scope a parser batch by evidence before a rebuild: ranks every schema key nothing reads by how "
            "many records actually carry it. Output is a triage queue, not a verdict -- parser-side matching "
            "is name-based, so read the parser before acting on a row."
        ),
        examples=(
            "devtools schema parser-diff",
            "devtools schema parser-diff --provider codex --min-encountered 1000",
        ),
    ),
    CommandSpec(
        "bench pipeline",
        "benchmarking",
        "Run typed pipeline probes against synthetic, staged, or archive-subset inputs.",
        "devtools.pipeline_probe",
        use_when="Run real pipeline stages and optionally capture emitted summaries as regression cases.",
        examples=(
            "devtools bench pipeline --provider chatgpt --stage parse",
            "devtools bench pipeline --input-mode archive-subset --capture-regression live-parse-drift",
        ),
    ),
    CommandSpec(
        "bench parser-census",
        "benchmarking",
        "Parse a recorded source denominator with no archive and diff the result against the last census.",
        "devtools.parser_census",
        json_flag=True,
        flags=(
            ("--no-baseline", "Record the census without comparing it to the previous one."),
            ("--no-write", "Do not persist the census."),
        ),
        use_when=(
            "Prove a parser change against the real corpus instead of fixtures alone: exits non-zero on a new "
            "parse failure, a member the denominator stopped naming, an origin that stopped classifying, or a "
            "digest that moved with identical bytes and an unchanged parser fingerprint."
        ),
        examples=(
            "devtools bench parser-census --source claude-code=tests/fixtures/corpus --no-baseline",
            "devtools bench parser-census --subject codex --workers 8",
        ),
    ),
    CommandSpec(
        "bench slo",
        "benchmarking",
        "Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements.",
        "devtools.verify_slos",
        json_flag=True,
        use_when=(
            "Confirm read-surface and interactive (daemon query / completion / cold CLI / ingest-to-searchable) "
            "latencies stay within their declared SLOs. Exits non-zero when any measured surface exceeds its budget."
        ),
        examples=("devtools bench slo", "devtools bench slo --json", "devtools bench slo --skip-benchmarks --json"),
    ),
    CommandSpec(
        "bench collection",
        "benchmarking",
        "Measure what a pytest selection costs to collect, before any test runs.",
        "devtools.collection_cost",
        json_flag=True,
        flags=(("--budget-mib", "Exit 3 when the collection peak exceeds this many MiB."),),
        use_when=(
            "Reproduce the per-worker collection cost a width is bounded by, before and after a change, on one head."
        ),
        examples=(
            "devtools bench collection --json",
            "devtools bench collection --budget-mib 430",
            "devtools bench collection tests/unit/devtools/",
        ),
    ),
    CommandSpec(
        "bench baseline",
        "benchmarking",
        "List or record committed measurement receipts under tests/benchmarks/baselines/.",
        "devtools.measurement_receipts",
        json_flag=True,
        # ``--record RECEIPT`` and ``--reason TEXT`` take values, and a declared
        # flag is always compiled as a Click boolean (``devtools/click_dispatch.py``
        # ``_make_command``), which would swallow the value. They forward through
        # the pass-through argv instead and are documented in the examples.
        flags=(("--list", "List the committed measurement baselines."),),
        use_when=(
            "Put a measurement where the next reader finds it with "
            "`--record <receipt> --reason <why>`. The static gates already commit their "
            "baselines; this is the measurement half. It is explained-not-ratcheted -- a number that moved "
            "is recorded with its reason, because a measurement legitimately moves with the host -- and it "
            "is not a gate, so nothing runs it per PR. The scalar regression ratchet with tolerances stays "
            "in tests/benchmarks/floors.json."
        ),
        examples=(
            "devtools bench baseline --list",
            "devtools bench baseline --record .cache/measurements/finished-build-sealed-516-raw.json "
            '--reason "first recorded arm"',
        ),
    ),
    CommandSpec(
        "bench memory",
        "benchmarking",
        "Measure query-memory envelopes on generated fixtures.",
        "devtools.query_memory_budget",
        use_when="Assert memory budgets around a concrete query or archive-facing command.",
        examples=("devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze",),
    ),
    CommandSpec(
        "bench query-envelope",
        "benchmarking",
        "Measure repeated incident-scale query RSS, PSS, swap, and temp envelopes.",
        "devtools.query_execution_envelope",
        json_flag=False,
        use_when="Run the opt-in live archive proof for repeated aggregate query_units calls and emit a receipt.",
        examples=(
            "devtools bench query-envelope --archive-root /path/to/archive --receipt .cache/query-envelope.json",
        ),
    ),
    CommandSpec(
        "archive lineage-validation",
        "archive",
        "Validate lineage-count evidence before citing archive counts externally.",
        "devtools.lineage_validation",
        json_flag=True,
        use_when=(
            "Before publishing archive session/message/cardinality numbers, emit exact physical/logical counts, "
            "session-link inheritance rollups, branch-point integrity checks, and sampled composed-read proof "
            "from the active archive instead of relying on scratch SQL or planner-estimated diagnostics."
        ),
        examples=(
            "devtools archive lineage-validation --json",
            "devtools archive lineage-validation --sample-prefix-sharing 100 --json",
        ),
    ),
    CommandSpec(
        "archive tool-outcome-census",
        "archive",
        "Classify every archived tool result by origin, construct, outcome and unknown reason.",
        "devtools.tool_outcome_census",
        json_flag=True,
        use_when=(
            "Before accepting a rebuilt archive, prove the tool-outcome contract holds over the whole "
            "candidate: no unknown outcome without a reason, no known outcome carrying one, no reason "
            "an origin's parsers do not own, and no public projection that disagrees with the block."
        ),
        examples=(
            "devtools archive tool-outcome-census --archive-root /path/to/archive",
            "devtools archive tool-outcome-census --archive-root /path/to/archive --json",
        ),
    ),
    CommandSpec(
        "archive tool-pairing-census",
        "archive",
        "Classify every tool call/result pairing gap against declared evidence.",
        "devtools.tool_pairing_census",
        json_flag=True,
        use_when=(
            "Before claiming what an archive's unpaired tool calls and unmatched tool results mean. "
            "Reports the exact no-result denominator by origin, provider construct, transcript "
            "position, source survival and acquisition state, classifies every cohort, and prints "
            "the query plan and runtime it used."
        ),
        examples=(
            "devtools archive tool-pairing-census",
            "devtools archive tool-pairing-census --json",
            "devtools archive tool-pairing-census --archive-root /path/to/archive --no-source-check",
        ),
    ),
    CommandSpec(
        "archive continuity-evidence",
        "archive",
        "Replay continuity scenarios and verify their query routes are discoverable.",
        "devtools.continuity_evidence",
        use_when=(
            "Replay the continuity scenario catalog over MCP stdio JSON-RPC and cross-check "
            "its query routes against discovery. The default seeds the packaged synthetic corpus. "
            "A supplied --archive-root must be paired with the exact --catalog that describes it."
        ),
        examples=(
            "devtools archive continuity-evidence",
            "devtools archive continuity-evidence --output .cache/continuity-evidence.json",
        ),
    ),
    CommandSpec(
        "archive continuity-cold-model",
        "archive",
        "Grade cold, wire-only model plan formulation against the continuity registry.",
        "devtools.continuity_cold_model",
        json_flag=True,
        use_when=(
            "To ask whether a cold client could have formulated each continuity scenario's plan from "
            "sparse operator wording plus wire-captured discovery alone. Discovery pages the real "
            "explain tool to exhaustion over MCP stdio with no in-process registry fallback; the "
            "production replay stays the execution oracle and is graded on separate axes. The default "
            "scripted backend replays a recorded plan artifact and makes no network call."
        ),
        examples=(
            "devtools archive continuity-cold-model --plans tests/data/continuity/cold-model-plans.json",
            "devtools archive continuity-cold-model --plans plans.json --attempts 3 --required-passes 2",
        ),
    ),
)

COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


def command_name_from_tokens(tokens: Iterable[str], commands: Iterable[CommandSpec] = COMMAND_SPECS) -> str | None:
    """Resolve leading argv tokens to a registered command name."""
    token_tuple = tuple(tokens)
    if not token_tuple:
        return None
    by_path = {spec.command_path: spec.name for spec in commands}
    max_len = max((len(path) for path in by_path), default=0)
    for length in range(min(max_len, len(token_tuple)), 0, -1):
        candidate = token_tuple[:length]
        if candidate in by_path:
            return by_path[candidate]
    return None


def _flatten_argv_parts(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part for arg in args for part in arg.split(" ") if part)


def control_plane_command(*args: str) -> str:
    parts = [CONTROL_PLANE, *_flatten_argv_parts(args)]
    return " ".join(part for part in parts if part)


def control_plane_argv(*args: str) -> tuple[str, ...]:
    return tuple(part for part in (CONTROL_PLANE, *_flatten_argv_parts(args)) if part)


def featured_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> tuple[CommandSpec, ...]:
    return tuple(spec for spec in commands if spec.featured)


def grouped_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> OrderedDict[str, list[CommandSpec]]:
    grouped: OrderedDict[str, list[CommandSpec]] = OrderedDict((category, []) for category in CATEGORY_ORDER)
    for spec in commands:
        grouped.setdefault(spec.category, [])
        grouped[spec.category].append(spec)
    for _category, specs in grouped.items():
        specs.sort(key=lambda item: item.name)
    return OrderedDict((category, specs) for category, specs in grouped.items() if specs)


__all__ = [
    "CATEGORY_ORDER",
    "COMMANDS",
    "COMMAND_SPECS",
    "CONTROL_PLANE",
    "CommandMain",
    "CommandSpec",
    "command_name_from_tokens",
    "control_plane_argv",
    "control_plane_command",
    "featured_command_specs",
    "grouped_command_specs",
]
