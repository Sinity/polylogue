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
    DocsEntry("CLI Reference", "docs/cli-reference.md", "Generated command reference from live help output."),
    DocsEntry(
        "Developer Tools",
        "docs/devtools.md",
        "`devtools` guide for generated surfaces, validation, and repo hygiene.",
    ),
    DocsEntry("Library API", "docs/library-api.md", "Async archive API, filters, and query patterns."),
    DocsEntry("Data Model", "docs/data-model.md", "Archive entities, storage shape, and metadata rules."),
    DocsEntry("Configuration", "docs/configuration.md", "XDG paths, environment variables, and runtime configuration."),
    DocsEntry("Architecture", "docs/architecture.md", "System rings, ownership boundaries, and data flow."),
    DocsEntry("Internals", "docs/internals.md", "Working implementation reference and debugging landmarks."),
    DocsEntry("MCP Integration", "docs/mcp-integration.md", "Model Context Protocol server setup and usage."),
    DocsEntry("Generate", "docs/generate.md", "Synthetic archive generation, seed mode, and demo workflows."),
    DocsEntry("Providers", "docs/providers/README.md", "Provider-specific parsing and export-format notes."),
    DocsEntry(
        "Test Quality Workflows",
        "docs/test-quality-workflows.md",
        "Generated validation lanes, mutation campaigns, and benchmark campaigns.",
    ),
    DocsEntry(
        "Verification Catalog",
        "docs/verification-catalog.md",
        "Generated proof-obligation subjects, claims, runners, and catalog self-checks.",
    ),
    DocsEntry(
        "Verification Lab",
        "docs/verification-lab.md",
        "Accepted command-surface decision for proof catalog, routing, and evidence operators.",
    ),
)

REPO_GUIDE_ENTRIES: tuple[DocsEntry, ...] = (
    DocsEntry("Contributing", "CONTRIBUTING.md", "Branching, issues, PRs, squash-merge history, and repo policy."),
    DocsEntry("Testing", "TESTING.md", "Baseline test matrix, protected surfaces, and QA entrypoints."),
    DocsEntry("Agent Guide", "CLAUDE.md", "Agent memory and working rules."),
    DocsEntry("Local Cache Layout", ".cache/README.md", "Disposable cache roots chosen by the repo itself."),
    DocsEntry(
        "Local Working Outputs",
        ".local/README.md",
        "Untracked local outputs such as campaigns, showcases, and reports.",
    ),
)

README_DOC_TITLES: tuple[str, ...] = (
    "CLI Reference",
    "Configuration",
    "Library API",
    "Developer Tools",
    "Architecture",
    "Providers",
)

README_GUIDE_TITLES: tuple[str, ...] = (
    "Contributing",
    "Testing",
    "Agent Guide",
)


__all__ = [
    "DOCS_REFERENCE_ENTRIES",
    "README_DOC_TITLES",
    "README_GUIDE_TITLES",
    "REPO_GUIDE_ENTRIES",
    "DocsEntry",
]
