"""Exercise types and the full catalog for ``polylogue demo --showcase``."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Validation:
    """Expected outcome for an exercise."""

    exit_code: int = 0
    stdout_contains: tuple[str, ...] = ()
    stdout_not_contains: tuple[str, ...] = ()
    stdout_is_valid_json: bool = False
    stdout_min_lines: int | None = None
    custom: Callable[[str, int], str | None] | None = None  # (output, exit_code) -> error|None


@dataclass(frozen=True)
class Exercise:
    """A single showcase exercise — one CLI invocation with validation."""

    name: str  # Unique ID e.g. "query.list-json"
    group: str  # structural | sources | pipeline | query-read | query-write | subcommands | advanced
    description: str  # Human-readable, used in cookbook headings
    args: list[str] = field(default_factory=list)  # CLI args (without 'polylogue' prefix)
    validation: Validation = field(default_factory=Validation)
    needs_data: bool = False  # Requires populated database
    writes: bool = False  # Mutates state — skip in --live mode
    depends_on: str | None = None  # Exercise that must complete first
    output_ext: str = ".txt"  # .txt / .json / .md / .csv / .html / .org
    tier: int = 1  # Complexity tier: 0=fast/structural, 1=basic, 2=advanced
    env: str = "any"  # "any" | "seeded" (skip in live mode) | "live" (skip in seeded mode)
    timeout_s: float = 120.0  # Per-exercise timeout in seconds
    vhs_capture: bool = False  # Whether this exercise should be captured as VHS recording
    artifact_class: str = "text"  # "text" | "json" | "visual" | "bundle"
    capture_steps: tuple[str, ...] = ()  # Optional VHS interaction steps for complex scenarios


# =============================================================================
# Custom validators
# =============================================================================


def _is_integer(output: str, _exit_code: int) -> str | None:
    """Validate output is a single integer."""
    stripped = output.strip()
    if not stripped:
        return "output is empty"
    try:
        int(stripped)
    except ValueError:
        return f"expected integer, got: {stripped!r}"
    return None


def _is_valid_json_array(output: str, _exit_code: int) -> str | None:
    """Validate output is a valid JSON array."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"
    if not isinstance(data, list):
        return f"expected JSON array, got {type(data).__name__}"
    return None


def _each_line_valid_json(output: str, _exit_code: int) -> str | None:
    """Validate each non-empty line is valid JSON (for streaming JSONL output)."""
    for i, line in enumerate(output.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            return f"line {i} invalid JSON: {e}"
    return None


def _json_only_fields(*allowed: str) -> Callable[[str, int], str | None]:
    """Validate JSON array entries contain only the specified keys."""
    allowed_set = set(allowed)

    def check(output: str, _exit_code: int) -> str | None:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return f"invalid JSON: {e}"
        if not isinstance(data, list):
            return f"expected JSON array, got {type(data).__name__}"
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            extra = set(item.keys()) - allowed_set
            if extra:
                return f"item {i} has unexpected keys: {extra}"
        return None

    return check


# =============================================================================
# Exercise catalog
# =============================================================================

# Shorthand constructors
_V = Validation
_E = Exercise

EXERCISES: tuple[Exercise, ...] = (
    # =========================================================================
    # structural (8) — tier 0: no data, tests CLI plumbing, <1s each
    # =========================================================================
    _E("help-main", "structural", "Main help screen",
       ["--help"], _V(stdout_contains=("polylogue",)), tier=0, vhs_capture=True),
    _E("help-run", "structural", "Run subcommand help",
       ["run", "--help"], _V(stdout_contains=("--stage",)), tier=0),
    _E("help-check", "structural", "Check subcommand help",
       ["check", "--help"], _V(stdout_contains=("--repair",)), tier=0),
    _E("help-demo", "structural", "Demo subcommand help",
       ["demo", "--help"], _V(stdout_contains=("--seed",)), tier=0),
    _E("help-tags", "structural", "Tags subcommand help",
       ["tags", "--help"], _V(stdout_contains=("--json",)), tier=0),
    _E("help-embed", "structural", "Embed subcommand help",
       ["embed", "--help"], _V(stdout_contains=("--stats",)), tier=0),
    _E("help-site", "structural", "Site subcommand help",
       ["site", "--help"], _V(stdout_contains=("--output",)), tier=0),
    _E("version", "structural", "Version output",
       ["--version"], _V(stdout_contains=("polylogue",)), tier=0),

    # =========================================================================
    # sources (3) — tier 0: no data, list sources and completions
    # =========================================================================
    _E("sources-list", "sources", "List configured sources",
       ["sources"], tier=0),
    _E("sources-json", "sources", "List sources as JSON",
       ["sources", "--json"], _V(stdout_is_valid_json=True), tier=0),
    _E("completions-bash", "sources", "Bash shell completions",
       ["completions", "--shell", "bash"], _V(stdout_contains=("complete",)), tier=0),

    # =========================================================================
    # pipeline (5 seeded + 1 live) — seeded exercises write, live preview is read-only
    # =========================================================================
    # Preview is read-only (no DB mutations): runs in both seeded and live modes
    _E("run-preview", "pipeline", "Preview pipeline plan (seeded fixtures)",
       ["run", "--preview", "--source", "inbox"], writes=False, env="seeded", tier=1,
       vhs_capture=True),
    _E("run-preview-live", "pipeline", "Preview pipeline against live sources",
       ["run", "--preview"], writes=False, env="live", tier=1,
       timeout_s=600.0,  # preview of real data can take minutes
       validation=_V(stdout_contains=("Preview",))),
    _E("run-all", "pipeline", "Run full pipeline (seeded fixtures)",
       ["run", "--source", "inbox"], writes=True, env="seeded", tier=2),
    _E("run-stage-render", "pipeline", "Run render stage only",
       ["run", "--stage", "render", "--source", "inbox"], writes=True,
       env="seeded", tier=2, depends_on="run-all"),
    _E("run-stage-index", "pipeline", "Run index stage only",
       ["run", "--stage", "index", "--source", "inbox"], writes=True,
       env="seeded", tier=2, depends_on="run-all"),

    # =========================================================================
    # query-read — tier 1: basic queries requiring seeded/live data
    # =========================================================================
    _E("stats-default", "query-read", "Default archive statistics",
       [], _V(stdout_contains=("Archive",)), needs_data=True, tier=1, vhs_capture=True),
    _E("stats-verbose", "query-read", "Verbose statistics",
       ["-v"], needs_data=True, tier=1),
    _E("query-list", "query-read", "List conversations",
       ["--list"], _V(stdout_min_lines=1), needs_data=True, tier=1, vhs_capture=True),
    _E("query-list-json", "query-read", "List conversations as JSON",
       ["--list", "-f", "json"], _V(custom=_is_valid_json_array),
       needs_data=True, output_ext=".json", tier=1),
    _E("query-list-csv", "query-read", "List conversations as CSV",
       ["--list", "-f", "csv"], _V(stdout_contains=("id",)),
       needs_data=True, output_ext=".csv", tier=1),
    _E("query-list-yaml", "query-read", "List conversations as YAML",
       ["--list", "-f", "yaml"], _V(stdout_contains=("provider:",)),
       needs_data=True, tier=1),
    _E("query-count", "query-read", "Count conversations",
       ["--count"], _V(custom=_is_integer), needs_data=True, tier=1),
    _E("query-latest", "query-read", "Show latest conversation",
       ["--latest"], _V(stdout_min_lines=1), needs_data=True, tier=1),
    _E("query-latest-json", "query-read", "Latest conversation as JSON",
       ["--latest", "-f", "json"], _V(stdout_is_valid_json=True),
       needs_data=True, output_ext=".json", tier=1),
    _E("query-latest-md", "query-read", "Latest conversation as Markdown",
       ["--latest", "-f", "markdown"], _V(stdout_contains=("#",)),
       needs_data=True, output_ext=".md", tier=1, vhs_capture=True),
    _E("query-latest-html", "query-read", "Latest conversation as HTML",
       ["--latest", "-f", "html"], _V(stdout_contains=("<",)),
       needs_data=True, output_ext=".html", tier=1),
    _E("query-latest-plaintext", "query-read", "Latest conversation as plaintext",
       ["--latest", "-f", "plaintext"], _V(stdout_min_lines=1),
       needs_data=True, tier=1),
    _E("query-latest-obsidian", "query-read", "Latest conversation as Obsidian note",
       ["--latest", "-f", "obsidian"], _V(stdout_contains=("---",)),
       needs_data=True, output_ext=".md", tier=1),
    _E("query-latest-org", "query-read", "Latest conversation as Org-mode",
       ["--latest", "-f", "org"], _V(stdout_contains=("#+TITLE",)),
       needs_data=True, output_ext=".org", tier=1),
    _E("query-latest-csv", "query-read", "Latest conversation as CSV",
       ["--latest", "-f", "csv"], _V(stdout_contains=("conversation_id",)),
       needs_data=True, output_ext=".csv", tier=1),
    _E("query-filter-provider", "query-read", "Filter by provider",
       ["-p", "chatgpt", "--list"], needs_data=True, tier=1),
    _E("query-filter-since", "query-read", "Filter by date",
       ["--since", "2020-01-01", "--count"], _V(custom=_is_integer),
       needs_data=True, tier=1),
    _E("query-sort-tokens", "query-read", "Sort by token count",
       ["--sort", "tokens", "--list", "-n", "3"], needs_data=True, tier=1),
    _E("query-sort-messages", "query-read", "Sort by message count",
       ["--sort", "messages", "--list", "-n", "3"], needs_data=True, tier=1),
    _E("query-stats", "query-read", "Conversation statistics",
       ["--stats"], needs_data=True, tier=1),
    _E("query-stats-by-provider", "query-read", "Statistics by provider",
       ["--stats-by", "provider"], needs_data=True, tier=1),
    _E("query-stats-by-month", "query-read", "Statistics by month",
       ["--stats-by", "month"], needs_data=True, tier=1),
    _E("query-stream-latest", "query-read", "Stream latest conversation",
       ["--stream", "--latest"], _V(stdout_min_lines=1), needs_data=True, tier=1),
    _E("query-stream-json", "query-read", "Stream latest as JSON lines",
       ["--stream", "--latest", "-f", "json"], _V(custom=_each_line_valid_json),
       needs_data=True, output_ext=".jsonl", tier=1),
    _E("query-dialogue-only", "query-read", "Latest with dialogue only",
       ["--latest", "-d"], needs_data=True, tier=1),

    # =========================================================================
    # query-write (8) — tier 2: mutate state, seeded only
    # =========================================================================
    _E("count-baseline", "query-write", "Baseline count before mutations",
       ["--count"], _V(custom=_is_integer),
       needs_data=True, writes=True, env="seeded", tier=2),
    _E("tag-add", "query-write", "Add tag to latest conversation",
       ["--latest", "--add-tag", "showcase-test"],
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="count-baseline"),
    _E("tag-verify", "query-write", "Verify tag was added",
       ["-t", "showcase-test", "--count"], _V(custom=_is_integer),
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="tag-add"),
    _E("tags-after-add", "query-write", "List tags after add",
       ["tags"],
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="tag-add"),
    _E("set-meta", "query-write", "Set metadata on conversation",
       ["--latest", "--set", "note", "exercise"],
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="count-baseline"),
    _E("delete-dry-run", "query-write", "Dry-run delete",
       ["-p", "chatgpt", "--latest", "--delete", "--dry-run"],
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="count-baseline"),
    _E("delete-one", "query-write", "Delete one conversation",
       ["-p", "chatgpt", "-n", "1", "--delete", "--force"],
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="count-baseline"),
    _E("count-decreased", "query-write", "Verify count decreased after delete",
       ["--count"], _V(custom=_is_integer),
       needs_data=True, writes=True, env="seeded", tier=2, depends_on="delete-one"),

    # =========================================================================
    # subcommands (6) — tier 1: read-only health/stats; tier 2: site generation
    # =========================================================================
    _E("check-health", "subcommands", "Health check",
       ["check"], _V(stdout_contains=("ok",)), needs_data=True, tier=1, vhs_capture=True),
    _E("check-json", "subcommands", "Health check as JSON",
       ["check", "--json"], _V(stdout_is_valid_json=True),
       needs_data=True, output_ext=".json", tier=1),
    _E("check-verbose", "subcommands", "Verbose health check",
       ["check", "-v"], needs_data=True, tier=1),
    _E("embed-stats", "subcommands", "Embedding statistics",
       ["embed", "--stats"], needs_data=True, tier=1),
    _E("tags-json", "subcommands", "Tags as JSON",
       ["tags", "--json"], _V(stdout_is_valid_json=True),
       needs_data=True, output_ext=".json", tier=1),
    _E("site-generate", "subcommands", "Generate static site",
       ["site"], _V(stdout_contains=("Building site",)),
       needs_data=True, writes=True, env="seeded", tier=2),

    # =========================================================================
    # advanced (7) — tier 2: complex queries, multi-flag combinations
    # =========================================================================
    _E("combined-filters", "advanced", "Combined provider + date filter",
       ["-p", "chatgpt", "--since", "2020-01-01", "--list", "-f", "json"],
       _V(custom=_is_valid_json_array), needs_data=True, output_ext=".json", tier=2),
    _E("reverse-sort", "advanced", "Reverse sort by date",
       ["--sort", "date", "--reverse", "--list", "-n", "3"],
       needs_data=True, tier=2),
    _E("exclude-provider", "advanced", "Exclude a provider",
       ["--exclude-provider", "gemini", "--count"], _V(custom=_is_integer),
       needs_data=True, tier=2),
    _E("fields-selector", "advanced", "Select specific JSON fields",
       ["--list", "-f", "json", "--fields", "id,title,provider"],
       _V(custom=_json_only_fields("id", "title", "provider")),
       needs_data=True, output_ext=".json", tier=2),
    _E("transform-strip", "advanced", "Strip tool calls from output",
       ["--latest", "--transform", "strip-all"],
       needs_data=True, tier=2, timeout_s=300.0),
    _E("sample-random", "advanced", "Random sample of conversations",
       ["--sample", "2", "--list"], needs_data=True, tier=2),
    _E("query-search-term", "advanced", "Full-text search",
       ["debug", "--list"], needs_data=True, tier=2),
)

EXERCISE_INDEX: dict[str, Exercise] = {e.name: e for e in EXERCISES}

GROUPS: tuple[str, ...] = (
    "structural", "sources", "pipeline", "query-read",
    "query-write", "subcommands", "advanced",
)


def exercises_by_group() -> dict[str, list[Exercise]]:
    """Return exercises grouped by their group name, in catalog order."""
    result: dict[str, list[Exercise]] = {g: [] for g in GROUPS}
    for ex in EXERCISES:
        result[ex.group].append(ex)
    return result


def vhs_exercises() -> list[Exercise]:
    """Return only exercises where vhs_capture is True."""
    return [e for e in EXERCISES if e.vhs_capture]


def topological_order(exercises: list[Exercise]) -> list[Exercise]:
    """Sort exercises respecting depends_on ordering."""
    index = {e.name: e for e in exercises}
    visited: set[str] = set()
    result: list[Exercise] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        ex = index.get(name)
        if ex is None:
            return
        if ex.depends_on and ex.depends_on in index:
            visit(ex.depends_on)
        visited.add(name)
        result.append(ex)

    for ex in exercises:
        visit(ex.name)
    return result
