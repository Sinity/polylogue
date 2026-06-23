"""Shared registry for the public documentation surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocsEntry:
    """Single rendered documentation entry."""

    title: str
    path: str
    description: str


DOCS_REFERENCE_ENTRIES: tuple[DocsEntry, ...] = (
    DocsEntry(
        "Installation",
        "docs/installation.md",
        "Linux, Nix/NixOS, Homebrew, and container install/run paths.",
    ),
    DocsEntry("CLI Reference", "docs/cli-reference.md", "Generated command reference from live help output."),
    DocsEntry(
        "Search & Query",
        "docs/search.md",
        "Query grammar, retrieval lanes, ranking policy, and the typed SearchEnvelope contract.",
    ),
    DocsEntry(
        "Developer Tools",
        "docs/devtools.md",
        "`devtools` guide for generated surfaces, validation, and repo hygiene.",
    ),
    DocsEntry(
        "Branch-Local Development Loop",
        "docs/dev-loop.md",
        "Branch-local daemon, web-shell, browser-capture, and extension debugging workflow.",
    ),
    DocsEntry("Library API", "docs/library-api.md", "Async archive API, filters, and query patterns."),
    DocsEntry("Data Model", "docs/data-model.md", "Archive entities, storage shape, and metadata rules."),
    DocsEntry(
        "Provider, Origin, and Source Identity",
        "docs/provider-origin-identity.md",
        "Vocabulary map for provider-wire family, public origin, material source, capture mode, parser binding, and refs.",
    ),
    DocsEntry(
        "Provider Package Completeness",
        "docs/provider-completeness.md",
        "Readiness report for provider/importer package modes by origin and capture mode.",
    ),
    DocsEntry(
        "Archive Backup",
        "docs/archive-backup.md",
        "Archive-tier backup profiles, restore boundaries, and blob-GC safety rules.",
    ),
    DocsEntry("Configuration", "docs/configuration.md", "XDG paths, environment variables, and runtime configuration."),
    DocsEntry(
        "Daemon Threat Model",
        "docs/daemon-threat-model.md",
        "Local API threat model — assets, threats, mitigations, and API roles.",
    ),
    DocsEntry(
        "Repository Layout",
        "docs/repo-layout.md",
        "Every top-level entry and its purpose.",
    ),
    DocsEntry("Architecture", "docs/architecture.md", "System rings, ownership boundaries, and data flow."),
    DocsEntry(
        "Architecture Spine",
        "docs/architecture-spine.md",
        "Target shape, guardrails, and major decisions with rejected alternatives.",
    ),
    DocsEntry(
        "Execution Plan",
        "docs/execution-plan.md",
        "Current issue-driven sequencing plan for the remaining backlog.",
    ),
    DocsEntry(
        "Design Direction",
        "docs/design/README.md",
        "Historical design inputs and current guidance for using them without treating them as parallel roadmaps.",
    ),
    DocsEntry(
        "Query-Action Workflows",
        "docs/product/workflows.md",
        "Executable `find QUERY then ACTION` product contract for workflows, affordances, completions, and golden paths.",
    ),
    DocsEntry("Internals", "docs/internals.md", "Working implementation reference and debugging landmarks."),
    DocsEntry("MCP Integration", "docs/mcp-integration.md", "Model Context Protocol server setup and usage."),
    DocsEntry(
        "Browser Capture",
        "docs/browser-capture.md",
        "Local browser extension capture for ChatGPT and Claude.ai sessions.",
    ),
    DocsEntry("Generate", "docs/generate.md", "Synthetic archive generation, seed mode, and demo workflows."),
    DocsEntry(
        "Maintenance",
        "docs/maintenance.md",
        "Operator guide for preview/plan/run, resume, scope filters, and incident runbooks.",
    ),
    DocsEntry("Providers", "docs/providers/README.md", "Provider-specific parsing and export-format notes."),
    DocsEntry(
        "Test Quality Workflows",
        "docs/test-quality-workflows.md",
        "Generated validation lanes, mutation campaigns, and benchmark campaigns.",
    ),
    DocsEntry(
        "Release Readiness Gate",
        "docs/plans/release-readiness-gate.md",
        "Externally-presentable release gate, required checks, and release PR evidence contract.",
    ),
    DocsEntry("Release Checklist", "docs/release.md", "Cut-time packaging, installed-artifact, and publish checks."),
)

REPO_GUIDE_ENTRIES: tuple[DocsEntry, ...] = (
    DocsEntry("Contributing", "CONTRIBUTING.md", "Branching, issues, PRs, squash-merge history, and repo policy."),
    DocsEntry("Testing", "TESTING.md", "Baseline test matrix, protected surfaces, and verification entrypoints."),
    DocsEntry("Agent Guide", "CLAUDE.md", "Agent memory and working rules."),
)

README_DOC_TITLES: tuple[str, ...] = (
    "Architecture",
    "Installation",
    "Architecture Spine",
    "Execution Plan",
    "Design Direction",
    "Query-Action Workflows",
    "CLI Reference",
    "Search & Query",
    "Browser Capture",
    "Library API",
    "MCP Integration",
    "Configuration",
    "Provider, Origin, and Source Identity",
    "Provider Package Completeness",
    "Archive Backup",
    "Developer Tools",
    "Branch-Local Development Loop",
    "Providers",
)

__all__ = [
    "DOCS_REFERENCE_ENTRIES",
    "README_DOC_TITLES",
    "REPO_GUIDE_ENTRIES",
    "DocsEntry",
]
