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
    "generated surfaces",
    "release",
    "verification",
    "benchmarking",
    "workspace",
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    category: str
    description: str
    module: str
    entrypoint: str = "main"
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
        use_when="Check repo state, generated-surface drift, and the next default verification steps.",
        examples=("devtools status", "devtools status --json", "devtools status --verify-generated"),
        featured=True,
    ),
    CommandSpec(
        "why",
        "core",
        "Explain the most recent verification run, or where verification time went.",
        "devtools.why",
        use_when="A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.",
        examples=(
            "devtools why",
            "devtools why --history 24",
            "devtools why --run 20260817T213631Z-testmon-2709409-d5c6e72c",
        ),
        featured=True,
    ),
    CommandSpec(
        "render all",
        "generated surfaces",
        "Refresh or verify generated docs and agent files.",
        "devtools.render_all",
        use_when="Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.",
        examples=("devtools render all", "devtools render all --check"),
        featured=True,
    ),
    CommandSpec(
        "render agent-manual",
        "generated surfaces",
        "Render the declaration-generated six-tool agent manual and packaged integration assets.",
        "devtools.render_agent_manual",
        use_when="Refresh or check cold-start guidance after MCP, query, origin, recipe, or delivery changes.",
        examples=("devtools render agent-manual", "devtools render agent-manual --check"),
    ),
    CommandSpec(
        "render cli-reference",
        "generated surfaces",
        "Render docs/cli-reference.md from live CLI help.",
        "devtools.render_cli_reference",
    ),
    CommandSpec(
        "render cli-output-schemas",
        "generated surfaces",
        "Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/.",
        "devtools.render_cli_output_schemas",
        use_when=(
            "Refresh or verify published JSON Schemas after changing the surface payload models "
            "that back stable CLI JSON output (#1272)."
        ),
        examples=(
            "devtools render cli-output-schemas",
            "devtools render cli-output-schemas --check",
        ),
    ),
    CommandSpec(
        "render openapi",
        "generated surfaces",
        "Render docs/openapi/search.yaml from typed daemon query payload models.",
        "devtools.render_openapi",
        use_when=(
            "Refresh or verify the published OpenAPI schema for daemon HTTP query routes "
            "after changing a route handler or a shared surface payload model."
        ),
        examples=(
            "devtools render openapi",
            "devtools render openapi --check",
        ),
    ),
    CommandSpec(
        "render webui-design-system",
        "generated surfaces",
        "Render WebUI v2 CSS tokens, public badge contracts, and contrast evidence.",
        "devtools.render_webui_design_system",
        use_when=(
            "Refresh or verify browser design-system contracts after changing the Python theme "
            "palette, the public Origin enum, or evidence-state vocabulary."
        ),
        examples=(
            "devtools render webui-design-system",
            "devtools render webui-design-system --check",
        ),
    ),
    CommandSpec(
        "render webui-client",
        "generated surfaces",
        "Render the committed WebUI TypeScript client from docs/openapi/search.yaml.",
        "devtools.render_webui_client",
        use_when=(
            "Refresh or verify WebUI request/response types and continuation iterators after changing "
            "the generated daemon OpenAPI contract."
        ),
        examples=(
            "devtools render webui-client",
            "devtools render webui-client --check",
        ),
    ),
    CommandSpec(
        "render devtools-reference",
        "generated surfaces",
        "Render the command catalog inside docs/devtools.md.",
        "devtools.render_devtools_reference",
    ),
    CommandSpec(
        "render docs-surface",
        "generated surfaces",
        "Render docs/README.md and the README documentation table.",
        "devtools.render_docs_surface",
    ),
    CommandSpec(
        "render query-discovery",
        "generated surfaces",
        "Render parser-gated query discovery examples and result semantics into docs/search.md.",
        "devtools.render_query_discovery",
        use_when=(
            "Refresh or verify query examples after changing the expression grammar, query-unit metadata, "
            "result-semantics vocabulary, completions, or MCP cookbook recipes."
        ),
        examples=("devtools render query-discovery", "devtools render query-discovery --check"),
    ),
    CommandSpec(
        "render pages",
        "generated surfaces",
        "Build the GitHub Pages documentation site into .cache/site/.",
        "devtools.render_pages",
        use_when="Build or verify the full GitHub Pages documentation site after changing docs, templates, or design docs.",
        examples=("devtools render pages", "devtools render pages --check", "devtools render pages --serve"),
    ),
    CommandSpec(
        "render visual-tapes",
        "generated surfaces",
        "Write VHS tape files and optionally capture GIFs for the default visual evidence specs.",
        "devtools.render_visual_tapes",
        use_when="Regenerate the first-contact demo screencast media from the committed tape specs.",
        examples=(
            "devtools render visual-tapes",
            "devtools render visual-tapes --capture",
        ),
    ),
    CommandSpec(
        "verify",
        "verification",
        "Run the local verification baseline before pushing or creating a PR, including the required committed-schema privacy registry check.",
        "devtools.verify",
        use_when="Run format, lint, mypy, render all, committed-schema privacy, and test checks locally before pushing.",
        examples=("devtools verify", "devtools verify --quick"),
        featured=True,
    ),
    CommandSpec(
        "verify ci-commands",
        "verification",
        "Validate devtools invocations in structured CI run fields.",
        "devtools.verify_ci_commands",
        use_when="Catch CI scripts that reference a removed or misspelled devtools command.",
        examples=("devtools verify ci-commands", "devtools verify ci-commands --json"),
    ),
    CommandSpec(
        "verify reindex-packets",
        "verification",
        "Validate the current reindex execution packets from the external Beads blocks graph.",
        "devtools.reindex_packets",
        use_when=(
            "Before dispatching reindex work, recompute blocks-only closure, packet topology, conflicts, "
            "capability carriers, and launch readiness from current Beads. The report is read-only; apply "
            "authority remains unsupported until a coordinator evidence adapter exists."
        ),
        examples=(
            "devtools verify reindex-packets --enforce-readiness",
            "devtools verify reindex-packets --diagnostic --json",
        ),
    ),
    CommandSpec(
        "verify portfolio-frontier",
        "verification",
        "Validate complete Beads ambition, active-set, and execution-focus views.",
        "devtools.portfolio_frontier",
        use_when="Inspect the complete external Beads export; soft active-set bands diagnose growth but never truncate work.",
        examples=("devtools verify portfolio-frontier /path/to/issues.jsonl",),
    ),
    CommandSpec(
        "verify doc-commands",
        "verification",
        "Validate executable documentation examples against live command inventories.",
        "devtools.verify_doc_commands",
        use_when="Catch README and documentation examples that reference an unknown command path or flag.",
        examples=("devtools verify doc-commands", "devtools verify doc-commands --json"),
    ),
    CommandSpec(
        "verify corpus-fidelity",
        "verification",
        "Run the production corpus-fidelity acceptance gate against an archive root.",
        "devtools.corpus_fidelity",
        use_when=(
            "Run after a promoted index rebuild, alongside `polylogue ops maintenance verify-archive`, "
            "to prove source-backed corpus absence, attachment, and revision fidelity from the registry."
        ),
        examples=(
            "devtools verify corpus-fidelity --archive-root /path/to/archive",
            "devtools verify corpus-fidelity --archive-root /path/to/archive --json",
        ),
    ),
    CommandSpec(
        "verify semantic-fidelity",
        "verification",
        "Run the bounded production-route semantic contradiction and construct-flow census.",
        "devtools.semantic_fidelity",
        use_when="Check OriginSpec witnesses, detector precedence, dropped-construct mutations, and privacy-safe construct flow.",
        examples=(
            "devtools verify semantic-fidelity --json",
            "devtools verify semantic-fidelity --json --report .agent/reports/semantic-fidelity-v1.json",
        ),
    ),
    CommandSpec(
        "verify schema-inference-gate",
        "verification",
        "Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt.",
        "devtools.schema_inference_gate",
        use_when=(
            "Run before schema inference or the 818fy rebuild. Declare every external source root represented in "
            "source.db; the command scans those roots and runs BlobStore's full verifier without mutating the archive."
        ),
        examples=(
            "devtools verify schema-inference-gate --archive-root /path/to/archive "
            "--ground-truth-root codex-session=/path/to/codex --receipt /path/to/schema-inference-gate-receipt.json",
            "devtools verify schema-inference-gate --archive-root /path/to/archive "
            "--ground-truth-root codex-session=/path/to/codex --receipt /path/to/schema-inference-gate-receipt.json --json",
        ),
    ),
    CommandSpec(
        "verify provider-completeness",
        "verification",
        "Report provider/importer package completeness by origin and capture mode.",
        "devtools.provider_completeness",
        use_when=(
            "Inspect detector, parser, fixture, schema, docs, ImportExplain, and caveat coverage "
            "before claiming a provider/importer mode is product-ready."
        ),
        examples=(
            "devtools verify provider-completeness",
            "devtools verify provider-completeness --json",
            "devtools verify provider-completeness --origin codex-session --json",
            "devtools verify provider-completeness --check",
        ),
    ),
    CommandSpec(
        "test",
        "verification",
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
        "verify coverage",
        "verification",
        "Run pytest with the repository coverage floor from pyproject.toml.",
        "devtools.coverage_gate",
        use_when="Enforce the committed coverage ratchet locally or in CI without duplicating threshold values.",
        examples=(
            "devtools verify coverage",
            "devtools verify coverage --ignore-integration --term-missing",
            "devtools verify coverage -- --maxfail=1",
        ),
    ),
    CommandSpec(
        "verify mutation-freshness",
        "verification",
        "Verify executable mutation campaigns meet the selected freshness and kill-rate thresholds.",
        "devtools.verify_mutation_freshness",
        use_when=(
            "Enforce mutation campaign freshness and kill-rate thresholds after a rotating CI campaign "
            "has produced its local artifacts."
        ),
        examples=(
            "devtools verify mutation-freshness --enforce-kill-rate",
            "devtools verify mutation-freshness --strict --default-freshness-days 30",
        ),
    ),
    CommandSpec(
        "bench ingest-amplification",
        "benchmarking",
        "Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851).",
        "devtools.ingest_amplification_probe",
        use_when=(
            "Establish or compare the post-fix baseline for daemon live-ingest write amplification. "
            "Drives the public batch-ingest path over a deterministic synthetic corpus in a temp dir "
            "and attributes bytes written per archive tier (source/index/embeddings/user/ops) "
            "per append batch. Additive measurement only — does not touch production ingest logic."
        ),
        examples=(
            "devtools bench ingest-amplification",
            "devtools bench ingest-amplification --json",
            "devtools bench ingest-amplification --batches 8 --seed 1851",
        ),
    ),
    CommandSpec(
        "bench ingest-throughput",
        "benchmarking",
        "Measure ingest wall-clock throughput on a synthetic fixture.",
        "devtools.ingest_throughput_probe",
        use_when=(
            "Measure ingest wall-clock / throughput, the time-based counterpart to the "
            "bytes-based ingest-amplification probe. Drives the public batch-ingest path over "
            "a deterministic synthetic corpus in a temp dir and times each append batch, "
            "reporting messages/sessions per second and a per-batch-ms distribution. "
            "Wall-clock is host-variable: diagnostic and campaign-comparable, no CI thresholds. "
            "Additive measurement only — does not touch production ingest logic."
        ),
        examples=(
            "devtools bench ingest-throughput",
            "devtools bench ingest-throughput --json",
            "devtools bench ingest-throughput --batches 20 --seed 2391",
        ),
    ),
    CommandSpec(
        "verify read-surface",
        "verification",
        "Capture and compare archive read-surface snapshots.",
        "devtools.self_verify",
        use_when=(
            "Freeze archive read-surface behavior before archive work, then compare candidate "
            "archives against the captured envelope baseline."
        ),
        examples=(
            "devtools verify read-surface capture --out .local/self-verify/baseline.json",
            "devtools verify read-surface compare .local/self-verify/baseline.json .local/self-verify/candidate.json --json",
        ),
    ),
    CommandSpec(
        "workspace index-fast-forward",
        "workspace",
        "Plan and prove a declared index fast-forward against retained raw replay.",
        "devtools.index_fast_forward",
        use_when=(
            "Advance a stopped index generation across a declared clone-safe schema gap. The actuator clones the "
            "active generation, applies lifecycle operations, proves a deterministic retained-raw sample through "
            "the production parser/materializer route, then atomically activates the proven generation."
        ),
        examples=(
            "devtools workspace index-fast-forward prepare --archive-root /path/to/archive --receipt /path/to/receipt.json",
            "devtools workspace index-fast-forward activate --receipt /path/to/receipt.json",
        ),
    ),
    CommandSpec(
        "workspace seeded-archive-cache-gc",
        "workspace",
        "Preview or apply age-gated GC for the shared seeded-archive fixture cache.",
        "devtools.seeded_archive_cache_gc",
        use_when=(
            "Maintain the reusable NVMe seeded-artifact cache from the generated default, named-workload, and "
            "benchmark reachability inventory. Preview is the default; pass --apply explicitly after reviewing "
            "the bounded receipt. Active locks, leases, protected worktrees, corrupt evidence, and grace-period "
            "artifacts remain under the shared GC primitive."
        ),
        examples=(
            "devtools workspace seeded-archive-cache-gc --json",
            "devtools workspace seeded-archive-cache-gc --apply --json",
        ),
    ),
    CommandSpec(
        "workspace deployment-smoke",
        "workspace",
        "Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow.",
        "devtools.deployment_smoke",
        use_when=(
            "After a system rebuild or before live UI probing, verify that the systemwide "
            "polylogue/polylogued binaries, loopback daemon routes, browser-capture receiver, "
            "and browser-capture archive materialization match the expected deployed surface."
        ),
        examples=(
            "devtools workspace deployment-smoke",
            "devtools workspace deployment-smoke --json",
            "devtools workspace deployment-smoke --daemon-url http://127.0.0.1:8766 --receiver-url http://127.0.0.1:8765",
        ),
    ),
    CommandSpec(
        "workspace lineage-validation",
        "workspace",
        "Validate lineage-count evidence before citing archive counts externally.",
        "devtools.lineage_validation",
        use_when=(
            "Before publishing archive session/message/cardinality numbers, emit exact physical/logical counts, "
            "session-link inheritance rollups, branch-point integrity checks, and sampled composed-read proof "
            "from the active archive instead of relying on scratch SQL or planner-estimated diagnostics."
        ),
        examples=(
            "devtools workspace lineage-validation --json",
            "devtools workspace lineage-validation --sample-prefix-sharing 100 --json",
            "devtools workspace lineage-validation --out-dir .local/evidence/lineage-validation/current",
        ),
    ),
    CommandSpec(
        "workspace physical-identity-census",
        "workspace",
        "Census raw evidence hidden by origin/native session identity collapse.",
        "devtools.physical_identity_census",
        use_when="Before designing physical session identity changes, measure same-origin/native candidates and distinguishable family evidence.",
        examples=(
            "devtools workspace physical-identity-census --json",
            "devtools workspace physical-identity-census --out docs/evidence/physical-session-identity-census.json",
        ),
    ),
    CommandSpec(
        "verify agent-integration",
        "verification",
        "Verify manual compilation, parser examples, continuation, native delivery, packaging, and live cutover signatures.",
        "devtools.verify_agent_integration",
        use_when="Validate the six-tool manual or native integration; add --require-live after the MCP cutover lands.",
        examples=(
            "devtools verify agent-integration",
            "devtools verify agent-integration --json",
            "devtools verify agent-integration --require-live",
        ),
    ),
    CommandSpec(
        "bench slo",
        "benchmarking",
        "Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements.",
        "devtools.verify_slos",
        use_when=(
            "Run directly to confirm read-surface "
            "(query / reader / facets / context / cost) latencies stay within their declared SLOs. "
            "Exits non-zero when any measured surface exceeds its budget."
        ),
        examples=(
            "devtools bench slo",
            "devtools bench slo --json",
            "devtools bench slo --skip-benchmarks --json",
        ),
    ),
    CommandSpec(
        "bench daemon-operation",
        "benchmarking",
        "Run the installed CLI and direct typed-UDS daemon operation profile.",
        "devtools.daemon_performance_profile",
        use_when=(
            "Measure the daemon architecture on the production route: installed CLI status, typed UDS find/read, "
            "completion, concurrent reads, cancellation, and declared background workload denominators. "
            "The profile records runtime, queue, CPU/RSS, SQLite, writer-hold, first-byte/full-render, "
            "rows/bytes, and cancellation evidence where the route exposes it."
        ),
        examples=("devtools bench daemon-operation",),
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
        "verify schema-versioning",
        "verification",
        "Verify durable-tier migration and derived-tier rebuild boundaries.",
        "devtools.verify_schema_upgrade_lane",
        use_when=(
            "Enforce the policy boundary documented in docs/internals.md § "
            "'Schema Versioning Model'. Durable tiers use explicit additive "
            "migrations with a backup gate; derived tiers are rebuilt or "
            "blue-green replaced from source evidence."
        ),
        examples=("devtools verify schema-versioning", "devtools verify schema-versioning --json"),
    ),
    CommandSpec(
        "verify oracle-integrity",
        "verification",
        "Verify tests certify production-reachable code and never read ambient user paths.",
        "devtools.verify_oracle_integrity",
        use_when=(
            "Before a deletion sweep, and as a standing gate. Catches the two ways a green "
            "test can certify nothing: its entire target set is unreachable from any production "
            "entrypoint (dead-engine suites), or it reads a real ~/.codex / ~/.claude / /realm "
            "path instead of a fixture. Reachability seeds four root classes import edges "
            "miss -- Click lazy commands, `python -m` entrypoints, ancestor packages, and "
            "literal-container registries -- resolves facade re-exports per symbol, and "
            "flags module-level Path.home()/expanduser constants in polylogue/** that "
            "capture an ambient location at import time, "
            "because import edges alone under-report and this repo has four recorded wrong "
            "deletions derived from grep."
        ),
        examples=(
            "devtools verify oracle-integrity",
            "devtools verify oracle-integrity --ignore-baseline --json",
        ),
    ),
    CommandSpec(
        "verify consumer-reachability",
        "verification",
        "Require newly added modules, tables, and tools to have production consumers.",
        "devtools.consumer_reachability",
        use_when="Run the fail-closed incremental surface-consumer gate used by quick verification and pre-push.",
        examples=(
            "devtools verify consumer-reachability",
            "devtools verify consumer-reachability --base SHA --head SHA",
        ),
    ),
    CommandSpec(
        "verify definition-closure",
        "verification",
        "Evaluate representative definition-to-production closure policies as a bounded JSON matrix.",
        "devtools.definition_closure",
        use_when="Check that authoritative definitions have required production, lifecycle, contract, discovery, and real-route edges.",
        examples=("devtools verify definition-closure", "devtools verify definition-closure --json"),
    ),
    CommandSpec(
        "verify timestamp-doctrine",
        "verification",
        "Verify durable-tier DDL never stores a timestamp column as TEXT.",
        "devtools.verify_timestamp_doctrine",
        use_when=(
            "Enforce the time doctrine (UTC epoch-ms canon, docs/internals.md) at DDL-review "
            "time (cpf.1): a TEXT timestamp in source.db/user.db re-introduces tz-unknown "
            "ambiguity and lexicographic-vs-temporal sort divergence, and durable tiers need "
            "an explicit additive migration to fix later -- catching it before merge is orders "
            "cheaper than a copy-forward migration after."
        ),
        examples=("devtools verify timestamp-doctrine", "devtools verify timestamp-doctrine --json"),
    ),
    CommandSpec(
        "verify insight-honesty",
        "verification",
        "Verify every registered insight product is rigor-contracted or exempt.",
        "devtools.verify_insight_rigor_honesty",
        use_when=(
            "Enforce that polylogue.insights.registry.INSIGHT_REGISTRY and "
            "polylogue.insights.rigor's contract matrix/exemption list never drift apart "
            "(9e5.28) -- a registered product with neither a RigorContract nor a "
            "RIGOR_EXEMPT entry used to silently vanish from `polylogue ops insights audit` "
            "instead of showing as uncovered."
        ),
        examples=("devtools verify insight-honesty", "devtools verify insight-honesty --json"),
    ),
    CommandSpec(
        "release verify-distribution",
        "release",
        "Verify wheel/sdist installed artifacts expose only supported runtime entrypoints.",
        "devtools.verify_distribution_surface",
        use_when=(
            "Build wheel and sdist artifacts, rebuild a wheel from an unpacked sdist without .git, "
            "and smoke installed runtime console scripts."
        ),
        examples=("devtools release verify-distribution",),
    ),
    CommandSpec(
        "bench pipeline",
        "verification",
        "Run typed pipeline probes against synthetic, staged, or archive-subset inputs.",
        "devtools.pipeline_probe",
        use_when="Run real pipeline stages and optionally capture emitted summaries as regression cases.",
        examples=(
            "devtools bench pipeline --provider chatgpt --stage parse",
            "devtools bench pipeline --input-mode archive-subset --capture-regression live-parse-drift",
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
        "verify scenario",
        "verification",
        "Run a named archive verification scenario.",
        "devtools.verification_scenario",
        use_when="Run a named archive verification scenario through the direct CLI path.",
        examples=(
            "devtools verify scenario list",
            "devtools verify scenario run archive-smoke --tier 0",
            "devtools verify scenario run rebuild-safety --report-dir .cache/rebuild-safety-report --json",
            "devtools verify scenario run safety-case --report-dir .cache/safety-case --json",
        ),
    ),
    CommandSpec(
        "workspace schema list",
        "verification",
        "List committed schema packages, versions, and evidence manifests.",
        "devtools.schema_inspect",
        entrypoint="list_main",
        use_when="Inspect committed provider schema package catalogs without presenting them as normal archive usage.",
        examples=("devtools workspace schema list --provider chatgpt --json",),
    ),
    CommandSpec(
        "workspace schema compare",
        "verification",
        "Compare two committed schema package versions for a provider.",
        "devtools.schema_inspect",
        entrypoint="compare_main",
        use_when="Review schema package drift between committed versions in the lab surface.",
        examples=("devtools workspace schema compare --provider chatgpt --from v1 --to v2 --markdown",),
    ),
    CommandSpec(
        "workspace schema explain",
        "verification",
        "Explain a committed package element schema with evidence and annotations.",
        "devtools.schema_inspect",
        entrypoint="explain_main",
        use_when="Inspect schema package annotations, semantic roles, and review evidence from the lab surface.",
        examples=("devtools workspace schema explain --provider chatgpt --version latest --verbose",),
    ),
    CommandSpec(
        "workspace schema generate",
        "verification",
        "Generate provider schema packages and optional evidence clusters.",
        "devtools.schema_generate",
        use_when="Refresh provider schema package artifacts from archive observations outside the archive CLI.",
        examples=("devtools workspace schema generate --provider chatgpt --cluster",),
    ),
    CommandSpec(
        "workspace schema commit",
        "verification",
        "Persist a real full-corpus schema generation into committed provider packages.",
        "devtools.schema_commit",
        use_when=(
            "Actually regenerate and write `polylogue/schemas/providers/<provider>/versions/...` from the live "
            "archive -- 'lab schema generate' only ever previews and never writes committed package files."
        ),
        examples=(
            "devtools workspace schema commit --provider chatgpt --full-corpus --dry-run",
            "devtools workspace schema commit --provider chatgpt --full-corpus",
        ),
    ),
    CommandSpec(
        "workspace schema promote",
        "verification",
        "Promote a schema evidence cluster into a registered package version.",
        "devtools.schema_promote",
        use_when="Turn reviewed schema evidence clusters into committed provider schema packages.",
        examples=("devtools workspace schema promote --provider chatgpt --cluster chatgpt-message-v2",),
    ),
    CommandSpec(
        "verify schema-audit",
        "verification",
        "Run committed provider schema package quality checks.",
        "devtools.schema_audit",
        use_when="Check committed schema package quality gates without presenting them as normal archive usage.",
        examples=("devtools verify schema-audit --provider chatgpt --json",),
    ),
    CommandSpec(
        "workspace schema parser-diff",
        "verification",
        "List observed provider wire keys that no parser references.",
        "devtools.schema_parser_diff",
        use_when=(
            "Scope a parser batch by evidence before a rebuild: ranks every schema key nothing reads by how "
            "many records actually carry it. Output is a triage queue, not a verdict -- parser-side matching "
            "is name-based, so read the parser before acting on a row."
        ),
        examples=(
            "devtools workspace schema parser-diff",
            "devtools workspace schema parser-diff --provider codex --min-encountered 1000",
            "devtools workspace schema parser-diff --json",
        ),
    ),
    CommandSpec(
        "verify schema-roundtrip",
        "verification",
        "Verify committed provider schema packages reload and roundtrip cleanly.",
        "devtools.verify_schema_roundtrip",
        use_when=(
            "Close the schema inference-validation loop: package manifests must roundtrip through typed models, "
            "and every supported element schema must be reachable from the runtime registry."
        ),
        examples=(
            "devtools verify schema-roundtrip --provider chatgpt",
            "devtools verify schema-roundtrip --all --json",
        ),
    ),
    CommandSpec(
        "verify layering",
        "verification",
        "Check inter-package imports against declared layering rules from docs/plans/layering.yaml.",
        "devtools.verify_layering",
        use_when=(
            "Diagnose architecture drift: which files import across declared "
            "package boundaries. This runs in verify --quick."
        ),
        examples=("devtools verify layering", "devtools verify layering --json"),
    ),
    CommandSpec(
        "verify patterns",
        "verification",
        "Enforce AST-shape defect-family rules with shrinking grandfathered baselines.",
        "devtools.verify_patterns",
        use_when=(
            "Catch new instances of verified code-shape defect families while allowing only the committed "
            "file:line debt inventory to remain."
        ),
        examples=("devtools verify patterns", "devtools verify patterns --json"),
    ),
    CommandSpec(
        "release build-package",
        "release",
        "Build the default Nix package with the out-link under .local/result.",
        "devtools.build_package",
        use_when="Produce the Nix package artifact with its out-link kept under the repo-local output root.",
        examples=("devtools release build-package",),
    ),
    CommandSpec(
        "bench mutation",
        "benchmarking",
        "Run focused mutation campaigns with isolated execution and JSON artifacts.",
        "devtools.mutmut_campaign",
        use_when="Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.",
        examples=("devtools bench mutation list", "devtools bench mutation run filters"),
        featured=True,
    ),
    CommandSpec(
        "workspace continuity-evidence",
        "workspace",
        "Replay continuity scenarios and verify their query routes are discoverable.",
        "devtools.continuity_evidence",
        use_when=(
            "Replay the continuity scenario catalog over MCP stdio JSON-RPC and cross-check "
            "its query routes against discovery. The default seeds the packaged synthetic corpus. "
            "A supplied --archive-root must be paired with the exact --catalog that describes it; "
            "the runner rejects an unrelated live archive rather than applying synthetic oracles."
        ),
        examples=(
            "devtools workspace continuity-evidence",
            "devtools workspace continuity-evidence --output .cache/continuity-evidence.json",
            "devtools workspace continuity-evidence --archive-root /path/to/archive --catalog /path/to/catalog.json",
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
