"""CLI output assurance registry (#1272).

For every primary subcommand of `polylogue`, declare the output
assurance contract it carries:

* ``json_contract`` — does the command emit a stable JSON shape that
  external consumers may bind to (validated by ``tests/unit/cli/
  test_cli_output_schemas.py`` against the schemas published under
  ``docs/schemas/cli-output/``)?
* ``snapshot`` — is the human-readable output pinned by a syrupy
  snapshot under ``tests/unit/cli/__snapshots__/``?
* ``plain`` — does the command honour ``POLYLOGUE_FORCE_PLAIN=1`` /
  ``--plain`` (no rich formatting; deterministic for snapshotting)?
* ``streaming`` — does the command support NDJSON streaming via
  ``--format ndjson``?
* ``schema`` — the published JSON Schema file name (under
  ``docs/schemas/cli-output/``) that describes the JSON output rows,
  if any.

The matrix is enumerated by
``tests/unit/cli/test_cli_output_assurance_matrix.py`` so that adding
a primary subcommand without declaring a contract fails CI.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputAssurance:
    """Output assurance declaration for one CLI command."""

    command: str
    family: str
    json_contract: bool
    snapshot: bool
    plain: bool
    streaming: bool
    schema: str | None = None
    notes: str | None = None


# Command families — keep aligned with `polylogue --help`.
QUERY_FAMILY = "query"
LIST_FAMILY = "list"
SEARCH_FAMILY = "search"
STATS_FAMILY = "stats"
INSIGHTS_FAMILY = "insights"
DAEMON_FAMILY = "daemon"
CONFIG_FAMILY = "config"
MAINTENANCE_FAMILY = "maintenance"
SHELL_FAMILY = "shell"
INGEST_FAMILY = "import"


# Every primary subcommand of `polylogue` declares its assurance posture
# here. ``polylogue <verb>`` that is just a query-mode verb (list, show,
# count, stats, open, delete) lives under the query/list/search/stats
# families. Top-level utility commands live under their own families.
ASSURANCE_REGISTRY: tuple[OutputAssurance, ...] = (
    # --- Query mode (default) + structural verbs ---------------------------
    OutputAssurance(
        command="list",
        family=LIST_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=True,
        schema="session-list-row.schema.json",
        notes="--format json|ndjson|yaml|csv all backed by SessionListRowPayload.",
    ),
    OutputAssurance(
        command="search",
        family=SEARCH_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=True,
        schema="session-search-hit.schema.json",
        notes=(
            "Search is the default query mode: bare `polylogue <terms>` "
            "with --format json|ndjson uses SessionSearchHitPayload."
        ),
    ),
    OutputAssurance(
        command="stats",
        family=STATS_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=True,
        schema=None,
        notes=(
            "--format json emits an array of dimension rows; the row shape "
            "depends on --by but is always {dimension-key, count, ...}."
        ),
    ),
    OutputAssurance(
        command="count",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Single integer to stdout; no JSON envelope.",
    ),
    OutputAssurance(
        command="read",
        family=QUERY_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=True,
        notes=(
            "Router verb: --view selects renderer; --to selects destination. "
            "--view messages --format json emits a messages JSON envelope; "
            "--all --format ndjson streams one session JSON per line."
        ),
    ),
    OutputAssurance(
        command="delete",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Side-effect command; --dry-run prints affected IDs.",
    ),
    OutputAssurance(
        command="mark",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Mutates user metadata (tags, notes, user-state marks) on matched sessions; side-effect command.",
    ),
    OutputAssurance(
        command="tags",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Tag-management subgroup; mutation-side commands.",
    ),
    OutputAssurance(
        command="tutorial",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes=(
            "Interactive first-run walkthrough; output is operator-facing prose and not contracted as machine-readable."
        ),
    ),
    OutputAssurance(
        command="analyze",
        family=QUERY_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes=(
            "Analytics over the matched result set: wraps the stats and facets "
            "surfaces; ``--by`` groups by dimension and ``--format json`` emits "
            "structured aggregates."
        ),
    ),
    OutputAssurance(
        command="resume",
        family=QUERY_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    OutputAssurance(
        command="resume-candidates",
        family=QUERY_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Ranks read-pull resume candidates; --format json emits a typed success envelope.",
    ),
    OutputAssurance(
        command="feedback",
        family=QUERY_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="record/list/clear subgroup; --machine wraps output in MachineSuccessPayload.",
    ),
    OutputAssurance(
        command="context-pack",
        family=QUERY_FAMILY,
        json_contract=True,
        snapshot=False,
        plain=True,
        streaming=False,
        notes="Provenance-rich JSON document; downstream agent surface.",
    ),
    # --- Insights ----------------------------------------------------------
    OutputAssurance(
        command="insights",
        family=INSIGHTS_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Rebuild/inspect derived insights; subcommands emit insight-specific JSON.",
    ),
    OutputAssurance(
        command="cost",
        family=INSIGHTS_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    OutputAssurance(
        command="diagnostics",
        family=INSIGHTS_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Temporal session diagnostics; JSON output for downstream analytics.",
    ),
    OutputAssurance(
        command="user-state",
        family=INSIGHTS_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    # --- Daemon / runtime --------------------------------------------------
    OutputAssurance(
        command="status",
        family=DAEMON_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Daemon and archive status; --json flag emits structured payload.",
    ),
    OutputAssurance(
        command="dashboard",
        family=DAEMON_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Side-effect command; opens local dashboard.",
    ),
    OutputAssurance(
        command="import",
        family=INGEST_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="--machine emits success/error envelope per import run.",
    ),
    OutputAssurance(
        command="embed",
        family=INGEST_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Embedding pipeline status; --json emits structured progress.",
    ),
    # --- Config / setup ----------------------------------------------------
    OutputAssurance(
        command="config",
        family=CONFIG_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="--format json emits resolved settings dictionary.",
    ),
    OutputAssurance(
        command="init",
        family=CONFIG_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Writes a polylogue.toml; --machine emits success/error envelope.",
    ),
    OutputAssurance(
        command="auth",
        family=CONFIG_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    # --- Shell integration ------------------------------------------------
    OutputAssurance(
        command="completions",
        family=SHELL_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Emits shell-specific completion script; covered by completion contract tests.",
    ),
    # --- Maintenance / repair ---------------------------------------------
    OutputAssurance(
        command="doctor",
        family=MAINTENANCE_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Archive health check; --json emits structured report.",
    ),
    OutputAssurance(
        command="maintenance",
        family=MAINTENANCE_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    OutputAssurance(
        command="backup",
        family=MAINTENANCE_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Side-effect command; prints destination path.",
    ),
    OutputAssurance(
        command="reset",
        family=MAINTENANCE_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
    ),
    OutputAssurance(
        command="schema",
        family=MAINTENANCE_FAMILY,
        json_contract=True,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="schema audit/inspect subgroup; --json emits structured audit.",
    ),
    OutputAssurance(
        command="recent",
        family=LIST_FAMILY,
        json_contract=True,
        snapshot=False,
        plain=True,
        streaming=True,
        notes="Lists the latest sessions; --format json|ndjson|... like list, streaming-friendly.",
    ),
    OutputAssurance(
        command="paths",
        family=CONFIG_FAMILY,
        json_contract=True,
        snapshot=False,
        plain=True,
        streaming=False,
        notes="Prints canonical archive paths; --format json emits a structured path map.",
    ),
    OutputAssurance(
        command="context",
        family=INSIGHTS_FAMILY,
        json_contract=True,
        snapshot=False,
        plain=True,
        streaming=False,
        notes="Composes a context preamble from archive objects; compose emits JSON.",
    ),
    OutputAssurance(
        command="commands",
        family=SHELL_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Discovery listing of command groups; plain deterministic output.",
    ),
    OutputAssurance(
        command="blackboard",
        family=MAINTENANCE_FAMILY,
        json_contract=False,
        snapshot=True,
        plain=True,
        streaming=False,
        notes="Agent-addressable notes group (list/add/clear); plain output.",
    ),
)


ASSURANCE_BY_COMMAND: dict[str, OutputAssurance] = {entry.command: entry for entry in ASSURANCE_REGISTRY}


FAMILY_ORDER: tuple[str, ...] = (
    QUERY_FAMILY,
    LIST_FAMILY,
    SEARCH_FAMILY,
    STATS_FAMILY,
    INSIGHTS_FAMILY,
    DAEMON_FAMILY,
    INGEST_FAMILY,
    CONFIG_FAMILY,
    SHELL_FAMILY,
    MAINTENANCE_FAMILY,
)


def entries_by_family() -> dict[str, tuple[OutputAssurance, ...]]:
    """Group the registry by family, preserving FAMILY_ORDER."""
    grouped: dict[str, list[OutputAssurance]] = {family: [] for family in FAMILY_ORDER}
    for entry in ASSURANCE_REGISTRY:
        grouped.setdefault(entry.family, []).append(entry)
    return {family: tuple(sorted(rows, key=lambda r: r.command)) for family, rows in grouped.items() if rows}


def _bool_cell(value: bool) -> str:
    return "yes" if value else "no"


def render_matrix_markdown() -> str:
    """Render the assurance matrix as a Markdown table grouped by family."""
    lines = [
        "## Output Assurance Matrix",
        "",
        (
            "Per-command output assurance contract (#1272). For every primary "
            "`polylogue` subcommand we declare whether it has a stable JSON "
            "shape, whether the human surface is pinned by a snapshot test, "
            "whether `--plain` is honoured for deterministic output, and "
            "whether NDJSON streaming via `--format ndjson` is supported."
        ),
        "",
        (
            "Published JSON Schemas live under "
            "[`docs/schemas/cli-output/`](./schemas/cli-output/README.md) and "
            "are regenerated by `devtools render-cli-output-schemas`."
        ),
        "",
    ]
    for family, rows in entries_by_family().items():
        lines.extend([f"### Family: `{family}`", ""])
        lines.extend(
            [
                "| Command | JSON contract | Snapshot | `--plain` | NDJSON | Schema | Notes |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for entry in rows:
            schema_cell = f"[`{entry.schema}`](./schemas/cli-output/{entry.schema})" if entry.schema else "—"
            notes_cell = entry.notes or ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{entry.command}`",
                        _bool_cell(entry.json_contract),
                        _bool_cell(entry.snapshot),
                        _bool_cell(entry.plain),
                        _bool_cell(entry.streaming),
                        schema_cell,
                        notes_cell,
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def primary_commands() -> Iterable[str]:
    """Iterate every primary command tracked by the assurance registry."""
    return (entry.command for entry in ASSURANCE_REGISTRY)


__all__ = [
    "ASSURANCE_BY_COMMAND",
    "ASSURANCE_REGISTRY",
    "FAMILY_ORDER",
    "OutputAssurance",
    "entries_by_family",
    "primary_commands",
    "render_matrix_markdown",
]
