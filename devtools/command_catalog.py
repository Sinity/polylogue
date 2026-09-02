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
        "Run the local verification baseline: every quick gate, then the selected or complete test corpus.",
        "devtools.verify",
        json_flag=True,
        flags=(
            ("--quick", "Run the static gates only."),
            ("--all", "Run the static gates plus the complete test corpus."),
        ),
        use_when="Run the gates and tests locally before pushing. --quick stops at the static gates; --all runs the complete corpus.",
        examples=("devtools verify", "devtools verify --quick", "devtools verify --all"),
        featured=True,
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
        "render schema-disposition",
        "generated surfaces",
        "Render the declaration-derived six-tier schema disposition artifacts.",
        "devtools.render_schema_disposition",
        examples=("devtools render schema-disposition", "devtools render schema-disposition --check"),
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
            "devtools scenario run rebuild-safety --report-dir .cache/rebuild-safety-report --json",
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
        "bench ingest-amplification",
        "benchmarking",
        "Measure deterministic per-tier ingest write amplification on a synthetic fixture.",
        "devtools.ingest_amplification_probe",
        json_flag=True,
        use_when=(
            "Establish or compare the post-fix baseline for daemon live-ingest write amplification. "
            "Drives the public batch-ingest path over a deterministic synthetic corpus in a temp dir "
            "and attributes bytes written per archive tier per append batch."
        ),
        examples=(
            "devtools bench ingest-amplification --json",
            "devtools bench ingest-amplification --batches 8 --seed 1851",
        ),
    ),
    CommandSpec(
        "bench ingest-throughput",
        "benchmarking",
        "Measure ingest wall-clock throughput on a synthetic fixture.",
        "devtools.ingest_throughput_probe",
        json_flag=True,
        use_when=(
            "Measure ingest wall-clock / throughput, the time-based counterpart to the "
            "bytes-based ingest-amplification probe. Wall-clock is host-variable: diagnostic and "
            "campaign-comparable, no CI thresholds."
        ),
        examples=(
            "devtools bench ingest-throughput --json",
            "devtools bench ingest-throughput --batches 20 --seed 2391",
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
        "bench concurrency",
        "benchmarking",
        "Run the managed bounded-compute scaling profile across representative workloads.",
        "devtools.concurrency_profile",
        json_flag=True,
        use_when=(
            "Compare bounded worker and admission configurations for tiny-file, ordinary, whale, mixed-ingest, "
            "derivation, and interactive-read workloads on the selected free-threaded runtime."
        ),
        examples=("devtools bench concurrency", "devtools bench concurrency --json"),
    ),
    CommandSpec(
        "bench cli-interaction",
        "benchmarking",
        "Run the complete installed CLI and direct typed-UDS interaction profile.",
        "devtools.cli_interaction_profile",
        use_when=(
            "Measure cold installed status and warm CLI/daemon interaction together, including completion, "
            "pagination, cancellation, fuzzy-launch declarations, and concurrent-read evidence."
        ),
        examples=("devtools bench cli-interaction",),
    ),
    CommandSpec(
        "bench daemon-operation",
        "benchmarking",
        "Run the installed CLI and direct typed-UDS daemon operation profile.",
        "devtools.daemon_performance_profile",
        use_when=(
            "Measure the daemon architecture on the production route: installed CLI status, typed UDS find/read, "
            "completion, concurrent reads, cancellation, and declared background workload denominators."
        ),
        examples=("devtools bench daemon-operation",),
    ),
    CommandSpec(
        "archive index-fast-forward",
        "archive",
        "Plan and prove a declared index fast-forward against retained raw replay.",
        "devtools.index_fast_forward",
        use_when=(
            "Advance a stopped index generation across a declared clone-safe schema gap. The actuator clones the "
            "active generation, applies lifecycle operations, proves a deterministic retained-raw sample through "
            "the production parser/materializer route, then atomically activates the proven generation."
        ),
        examples=(
            "devtools archive index-fast-forward prepare --archive-root /path/to/archive --receipt /path/to/receipt.json",
            "devtools archive index-fast-forward activate --receipt /path/to/receipt.json",
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
