"""Tiered registry for the public documentation surface.

The registry is intentionally explicit: rendering checks it against the docs
tree, so adding a Markdown page without choosing a reader-facing home fails
the generated-surface gate instead of silently creating another orphan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DocsTier = Literal["guide", "reference", "internals", "operations", "evidence", "design", "archive"]


@dataclass(frozen=True, slots=True)
class DocsEntry:
    """A documentation page with its reader-facing tier."""

    title: str
    path: str
    description: str
    tier: DocsTier


DOCS_TIER_TITLES: tuple[tuple[DocsTier, str], ...] = (
    ("guide", "Guides"),
    ("reference", "Reference"),
    ("internals", "Architecture and Internals"),
    ("operations", "Operations and Contributor Workflow"),
    ("evidence", "Demos, Evidence, and Product"),
    ("design", "Design Notes"),
    ("archive", "Historical Records and Generated Artifacts"),
)


def _entry(title: str, path: str, description: str, tier: DocsTier) -> DocsEntry:
    return DocsEntry(title, f"docs/{path}", description, tier)


DOCS_REFERENCE_ENTRIES: tuple[DocsEntry, ...] = (
    # Guides
    _entry(
        "Getting Started",
        "getting-started.md",
        "First archive, first query, and the next documentation steps.",
        "guide",
    ),
    _entry(
        "Installation",
        "installation.md",
        "Source checkout, Nix flake, and managed NixOS/Home Manager install paths.",
        "guide",
    ),
    _entry(
        "Onboarding",
        "onboarding.md",
        "First-run source detection, configuration, daemon startup, and recovery.",
        "guide",
    ),
    _entry(
        "Search & Query",
        "search.md",
        "Query grammar, retrieval lanes, ranking policy, and the typed SearchEnvelope contract.",
        "guide",
    ),
    _entry("Insights", "insights.md", "Available derived views and their evidence boundaries.", "guide"),
    _entry("Export", "export.md", "Query-set reads and renderer formats for local export packages.", "guide"),
    _entry(
        "Browser Capture",
        "browser-capture.md",
        "Local browser extension capture for ChatGPT and Claude.ai sessions.",
        "guide",
    ),
    _entry("Hooks", "hooks.md", "Hook-event capture, configuration, and event catalog.", "guide"),
    _entry("Cloud Agents", "cloud-agents.md", "Privacy and setup boundaries for cloud-agent work.", "guide"),
    _entry("Generate", "generate.md", "Synthetic archive generation, seed mode, and demo workflows.", "guide"),
    _entry(
        "Maintenance",
        "maintenance.md",
        "Operator guide for preview, plan, run, resume, and incident recovery.",
        "guide",
    ),
    _entry("Providers", "providers/README.md", "Provider-specific parsing and export-format notes.", "guide"),
    _entry("Provider Index", "providers/index.md", "Concise provider guide entrypoint.", "guide"),
    _entry("ChatGPT Provider", "providers/chatgpt.md", "ChatGPT export detection and parser notes.", "guide"),
    _entry("Claude.ai Provider", "providers/claude-ai.md", "Claude web export detection and parser notes.", "guide"),
    _entry(
        "Claude Code Provider", "providers/claude-code.md", "Claude Code session detection and parser notes.", "guide"
    ),
    _entry("Gemini Provider", "providers/gemini.md", "Gemini/Drive acquisition and parser notes.", "guide"),
    _entry("Codex Provider", "providers/openai-codex.md", "Codex session detection and parser notes.", "guide"),
    # Reference
    _entry("CLI Reference", "cli-reference.md", "Generated command reference from live help output.", "reference"),
    _entry("MCP Reference", "mcp-reference.md", "Generated MCP tool and contract reference.", "reference"),
    _entry("Library API", "library-api.md", "Async archive API, filters, and query patterns.", "reference"),
    _entry("MCP Integration", "mcp-integration.md", "Model Context Protocol server setup and usage.", "reference"),
    _entry(
        "Configuration", "configuration.md", "XDG paths, environment variables, and runtime configuration.", "reference"
    ),
    _entry("Glossary", "glossary.md", "Plain-language translation of the internal taxonomy.", "reference"),
    _entry(
        "Provider, Origin, and Source Identity",
        "provider-origin-identity.md",
        "Vocabulary map for provider-wire family, public origin, material source, and parser binding.",
        "reference",
    ),
    _entry(
        "Provider Package Completeness",
        "provider-completeness.md",
        "Readiness report for provider/importer package modes by origin and capture mode.",
        "reference",
    ),
    _entry(
        "Material Protocol v1", "material-protocol-v1.md", "Normalized-session interchange wire format.", "reference"
    ),
    _entry(
        "Schema Annotations", "schema-annotations.md", "Versioned annotation schemas and batch provenance.", "reference"
    ),
    _entry(
        "Query Identity",
        "query-identity.md",
        "Canonical query, query-run, and result-set reference identities.",
        "reference",
    ),
    _entry(
        "CLI Output Schemas", "schemas/cli-output/README.md", "Machine-readable CLI output schema catalog.", "reference"
    ),
    _entry(
        "Projection Render Specification",
        "projection-render-spec.md",
        "Projection and renderer contract for archive reads.",
        "reference",
    ),
    _entry(
        "Web Route Readiness States",
        "web-route-readiness-states.md",
        "Readiness-state semantics for web routes.",
        "reference",
    ),
    # Architecture and internals
    _entry("Architecture", "architecture.md", "System rings, ownership boundaries, and data flow.", "internals"),
    _entry(
        "Architecture Spine",
        "architecture-spine.md",
        "Target shape, guardrails, and major decisions with rejected alternatives.",
        "internals",
    ),
    _entry("Data Model", "data-model.md", "Archive entities, storage shape, and metadata rules.", "internals"),
    _entry("Schema", "schema.md", "Index and durable tier schema, FTS, vectors, and versioning.", "internals"),
    _entry("Internals", "internals.md", "Working implementation reference and debugging landmarks.", "internals"),
    _entry("Daemon", "daemon.md", "Daemon ownership, convergence, HTTP serving, and service operation.", "internals"),
    _entry(
        "Daemon Threat Model",
        "daemon-threat-model.md",
        "Local API assets, threats, mitigations, and roles.",
        "internals",
    ),
    _entry("Security", "security.md", "Security boundaries for local archives and readers.", "internals"),
    _entry(
        "Archive Backup",
        "archive-backup.md",
        "Archive-tier backup profiles, restore boundaries, and blob-GC safety rules.",
        "internals",
    ),
    _entry(
        "Cost Model", "cost-model.md", "Cost, usage, cache, and subscription-credit accounting semantics.", "internals"
    ),
    _entry("Agent Forensics", "agent-forensics.md", "Forensic investigation methods over agent work.", "internals"),
    _entry("Repository Layout", "repo-layout.md", "Every top-level entry and its purpose.", "internals"),
    # Operations
    _entry("Developer Tools", "devtools.md", "Generated surfaces, validation, and repo hygiene.", "operations"),
    _entry(
        "Branch-Local Development Loop",
        "dev-loop.md",
        "Daemon, web-shell, browser-capture, and extension debugging workflow.",
        "operations",
    ),
    _entry("Test Economics", "test-economics.md", "Test-selection and verification cost model.", "operations"),
    _entry(
        "Test Quality Workflows",
        "test-quality-workflows.md",
        "Executable mutation-campaign and benchmark registries.",
        "operations",
    ),
    _entry(
        "Visual Evidence",
        "visual-evidence.md",
        "Synthetic reader DOM/media evidence lanes and local screenshot boundaries.",
        "operations",
    ),
    _entry(
        "Release Checklist", "release.md", "Cut-time packaging, installed-artifact, and publish checks.", "operations"
    ),
    # Evidence and product
    _entry(
        "Demos and Proofs",
        "demos.md",
        "Reproducible proofs, construct-valid demo doctrine, and flagship demonstrations.",
        "evidence",
    ),
    _entry(
        "Proof Artifacts",
        "proof-artifacts.md",
        "Claim-to-proof map for public-facing demo and evidence claims.",
        "evidence",
    ),
    _entry(
        "Structured Failure Follow-Up",
        "findings/claim-vs-evidence.md",
        "Bounded field finding with oracle, sample frame, calibration, and caveats.",
        "evidence",
    ),
    _entry(
        "Polylogue on Sinex",
        "sinex-interop.md",
        "Current bridge, target authority split, and rebuild proof.",
        "evidence",
    ),
    _entry(
        "Insights Rigor Matrix",
        "insights-rigor-matrix.md",
        "Evidence strengths and limitations for insight families.",
        "evidence",
    ),
    _entry(
        "Query-Action Workflows",
        "product/workflows.md",
        "Executable product contract for workflows, affordances, completions, and golden paths.",
        "evidence",
    ),
    _entry(
        "Demo Corpus Construct Audit",
        "plans/demo-corpus-construct-audit.md",
        "Generated construct-coverage audit for the demo fixture world.",
        "evidence",
    ),
    _entry(
        "Release Readiness Gate",
        "plans/release-readiness-gate.md",
        "Externally presentable release gate and required proof contract.",
        "evidence",
    ),
    _entry(
        "Demo Packet v2", "examples/demo-packet-v2/README.md", "Worked private-data-free evidence packet.", "evidence"
    ),
    _entry(
        "Demo Tour Report",
        "examples/demo-tour/report.md",
        "Recorded output and receipts from the demo tour.",
        "evidence",
    ),
    _entry(
        "UVX Installation Proof",
        "examples/demo-tour/uvx-proof.md",
        "Recorded installation proof for the uvx distribution path.",
        "evidence",
    ),
    _entry(
        "Visual Tape Examples",
        "examples/visual-tapes/README.md",
        "Reader-evidence and visual-tape artifact catalog.",
        "evidence",
    ),
    _entry(
        "Example and Proof Index",
        "examples/README.md",
        "Index of recorded proof artifacts and worked examples.",
        "evidence",
    ),
    # Design
    _entry(
        "Design Direction",
        "design/README.md",
        "Beads-first design doctrine and standing domain-model references.",
        "design",
    ),
    _entry("Agent-First MCP", "design/agent-first-mcp.md", "Agent-facing MCP surface doctrine.", "design"),
    _entry(
        "Archive Storytelling",
        "design/archive-storytelling.md",
        "Narrative and artifact design for archives.",
        "design",
    ),
    _entry(
        "Hermes Archival Export Contract",
        "design/hermes-archival-export-contract.md",
        "Versioned Hermes session export schema, durable lifecycle-event spool, and snapshot reconciliation.",
        "design",
    ),
    _entry(
        "Browser Capture Redesign",
        "design/browser-capture-redesign/README.md",
        "Browser-capture redesign rationale and verification artifacts.",
        "design",
    ),
    _entry(
        "Incident 14:32 Proof World",
        "design/incident-1432-proof-world.md",
        "Deterministic demo corpus and anti-circularity rules.",
        "design",
    ),
    _entry("Project Memory", "design/project-memory.md", "Long-term memory model and product intent.", "design"),
    _entry(
        "Query-Action Workflows Design",
        "design/query-action-workflows.md",
        "Historical design pointer for the workflow contract.",
        "design",
    ),
    _entry(
        "Query Set Algebra", "design/query-set-algebra.md", "Set-composition semantics over query results.", "design"
    ),
    _entry(
        "Session Lineage Model",
        "design/session-lineage-model.md",
        "Fork, resume, compaction, and composition semantics.",
        "design",
    ),
    _entry(
        "Analysis Rigor",
        "design/analysis-rigor.md",
        "Rigor mechanisms for agent claims: population validity and comparative judgment.",
        "design",
    ),
    _entry("Second Brain", "design/second-brain.md", "Vision note for remembered work.", "design"),
    _entry("Time Machine", "design/time-machine.md", "Vision note for reconstructing work over time.", "design"),
    _entry("Whole Product", "design/whole-product.md", "Product vision and system relationships.", "design"),
    # Historical and generated material
    _entry(
        "Closed-Issue Workload Audit",
        "audits/2026-05-19-closed-issue-workload-audit.md",
        "Historical audit of closed-issue workload.",
        "archive",
    ),
    _entry(
        "Cross-Surface Coherence Audit",
        "audits/2026-05-20-cross-surface-coherence-audit.md",
        "Historical cross-surface coherence audit.",
        "archive",
    ),
    _entry("API Bypass Audit", "audits/2026-05-25-api-bypass-audit.md", "Historical audit of API bypasses.", "archive"),
    _entry(
        "Daemon Loop Lock-Starvation Map",
        "audits/2026-07-09-daemon-loop-lock-starvation-map.md",
        "Lock-starvation investigation record.",
        "archive",
    ),
    _entry(
        "Hash Boundary Census",
        "audits/2026-07-09-hash-boundary-census.md",
        "Hash-boundary investigation record.",
        "archive",
    ),
    _entry(
        "Race Window Audit", "audits/2026-07-09-race-window-audit.md", "Race-window investigation record.", "archive"
    ),
    _entry("Audit Record Index", "audits/README.md", "Index of dated investigation records.", "archive"),
    _entry(
        "1498 Cascade Retrospective",
        "retro/2026-05-24-1498-cascade.md",
        "Historical cascade incident retrospective.",
        "archive",
    ),
    _entry("Retrospective Index", "retro/README.md", "Index of historical incident retrospectives.", "archive"),
    _entry(
        "Query Pipeline Substrate Plan",
        "plans/query-pipeline-substrate.md",
        "Historical/active query pipeline design plan.",
        "archive",
    ),
    _entry(
        "Semantic Card Tool Map",
        "generated/semantic-card-tool-map.md",
        "Generated map from semantic cards to tools.",
        "archive",
    ),
    _entry("Topology Status", "topology-status.md", "Generated module-topology status dashboard.", "archive"),
)

REPO_GUIDE_ENTRIES: tuple[DocsEntry, ...] = (
    DocsEntry(
        "Contributing",
        "CONTRIBUTING.md",
        "Branching, issues, PRs, squash-merge history, and repo policy.",
        "operations",
    ),
    DocsEntry(
        "Testing", "TESTING.md", "Baseline test matrix, protected surfaces, and verification entrypoints.", "operations"
    ),
    DocsEntry("Agent Guide", "CLAUDE.md", "Agent memory and working rules.", "operations"),
)

README_DOC_TITLES: tuple[str, ...] = (
    "Getting Started",
    "Installation",
    "Demos and Proofs",
    "Proof Artifacts",
    "Architecture",
    "Search & Query",
    "CLI Reference",
    "MCP Integration",
    "Configuration",
    "Security",
    "Developer Tools",
    "Providers",
)

__all__ = [
    "DOCS_REFERENCE_ENTRIES",
    "DOCS_TIER_TITLES",
    "README_DOC_TITLES",
    "REPO_GUIDE_ENTRIES",
    "DocsEntry",
    "DocsTier",
]
