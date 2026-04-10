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
        "Developer Control Plane",
        "docs/devtools.md",
        "`python -m devtools` guide for generation, validation, and repo hygiene.",
    ),
    DocsEntry("Library API", "docs/library-api.md", "Async archive API, filters, and query patterns."),
    DocsEntry("Data Model", "docs/data-model.md", "Archive entities, storage shape, and metadata rules."),
    DocsEntry("Configuration", "docs/configuration.md", "XDG paths, environment variables, and runtime configuration."),
    DocsEntry("Architecture", "docs/architecture.md", "System rings, ownership boundaries, and data flow."),
    DocsEntry("Internals", "docs/internals.md", "Operator-facing implementation reference and debugging landmarks."),
    DocsEntry("MCP Integration", "docs/mcp-integration.md", "Model Context Protocol server setup and usage."),
    DocsEntry("Generate", "docs/generate.md", "Synthetic archive generation, seed mode, and demo workflows."),
    DocsEntry("Providers", "docs/providers/README.md", "Provider-specific parsing and export-format notes."),
    DocsEntry(
        "Test Quality Workflows",
        "docs/test-quality-workflows.md",
        "Generated validation lanes, mutation campaigns, and benchmark campaigns.",
    ),
    DocsEntry(
        "Mutation Testing Baseline",
        "docs/mutation-testing-baseline.md",
        "Mutation policy, baseline expectations, and operator workflow.",
    ),
)

REPO_GUIDE_ENTRIES: tuple[DocsEntry, ...] = (
    DocsEntry("Contributing", "CONTRIBUTING.md", "Branching, issues, PRs, squash-merge history, and repo policy."),
    DocsEntry("Testing", "TESTING.md", "Baseline test matrix, protected surfaces, and QA entrypoints."),
    DocsEntry("Agent Guide", "CLAUDE.md", "Root transclusion surface for repository-specific agent guidance."),
    DocsEntry("Local Cache Layout", ".cache/README.md", "Disposable cache roots chosen by the repo itself."),
    DocsEntry(
        "Local Working Outputs",
        ".local/README.md",
        "Meaningful but untracked local outputs such as campaigns, showcases, and proof bundles.",
    ),
)


__all__ = ["DOCS_REFERENCE_ENTRIES", "REPO_GUIDE_ENTRIES", "DocsEntry"]
