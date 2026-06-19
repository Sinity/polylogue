"""Render JSON Schema artifacts for stable CLI output payloads.

For each Pydantic surface payload model that backs a stable CLI JSON
output surface (#1272), emit a JSON Schema file under
``docs/schemas/cli-output/``. External tooling — shell pipelines,
LLM-based agents, downstream validators — can use these as the
contract for ``polylogue <verb> --format json`` output.

Each schema is generated from the model's ``model_json_schema()``
output. The output file name encodes the CLI surface it belongs to,
not the Python class name.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.archive.query.metadata import terminal_query_source_list
from polylogue.surfaces.payloads import (
    ImportExplainPayload,
    MachineErrorPayload,
    MachineSuccessPayload,
    MutationResultPayload,
    QueryErrorPayload,
    QueryUnitEnvelope,
    SearchEnvelope,
    SessionListRowPayload,
    SessionMessageRowPayload,
    SessionMessagesResponsePayload,
    SessionNeighborCandidatePayload,
    SessionSearchHitPayload,
    SessionSummaryPayload,
)

DEFAULT_OUTPUT_DIR = Path("docs/schemas/cli-output")
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True, slots=True)
class CliOutputSchema:
    """A CLI output surface that has a published JSON Schema contract."""

    name: str
    title: str
    description: str
    model: type[BaseModel]
    surfaces: tuple[str, ...]


SCHEMAS: tuple[CliOutputSchema, ...] = (
    CliOutputSchema(
        name="session-list-row",
        title="Session List Row",
        description=(
            "One row of `polylogue read --all --format json` output. The read-all path "
            "emits an array of these objects; the `ndjson` format emits one per "
            "line as results stream."
        ),
        model=SessionListRowPayload,
        surfaces=(
            "polylogue read --all --format json",
            "polylogue read --all --format ndjson",
            "polylogue read --all --format yaml",
        ),
    ),
    CliOutputSchema(
        name="session-summary",
        title="Session Summary",
        description=("Compact session identity payload used by analyze group-by and embedded inside search hits."),
        model=SessionSummaryPayload,
        surfaces=(
            "polylogue analyze --format json (rows)",
            "polylogue --format json <query> (hits[].session)",
        ),
    ),
    CliOutputSchema(
        name="session-message-row",
        title="Session Message Row",
        description=(
            "One message row emitted by `polylogue read --view messages --format ndjson` "
            "and embedded inside the finite messages response."
        ),
        model=SessionMessageRowPayload,
        surfaces=(
            "polylogue read --view messages --format ndjson",
            "polylogue read --view messages --format json (messages[])",
        ),
    ),
    CliOutputSchema(
        name="session-messages-response",
        title="Session Messages Response",
        description=("Finite response for `polylogue read --view messages --format json`."),
        model=SessionMessagesResponsePayload,
        surfaces=("polylogue read --view messages --format json",),
    ),
    CliOutputSchema(
        name="session-search-hit",
        title="Session Search Hit",
        description=(
            "A search match: session identity plus evidence (match "
            "surface, retrieval lane, snippet, score). Default query-mode "
            "JSON output uses this shape; the `ndjson` mode emits one per "
            "line as hits arrive."
        ),
        model=SessionSearchHitPayload,
        surfaces=(
            "polylogue --format json <query>",
            "polylogue --format ndjson <query>",
        ),
    ),
    CliOutputSchema(
        name="search-envelope",
        title="Search Envelope",
        description=(
            "Typed ranked-result envelope (#1266) shared across CLI JSON, "
            "MCP, Python API, and daemon HTTP. Wraps the per-hit "
            "`SessionSearchHit` array with `total`, `limit`, "
            "`offset`, `next_cursor`, `query`, `retrieval_lane`, and the "
            "`ranking_policy`/`ranking_policy_version` declaration."
        ),
        model=SearchEnvelope,
        surfaces=(
            "polylogue --format json <query>",
            "GET /api/sessions?query=...",
        ),
    ),
    CliOutputSchema(
        name="query-unit-envelope",
        title="Query Unit Envelope",
        description=(
            "Terminal row envelope for explicit "
            f"`{terminal_query_source_list()} where ...` "
            "query-unit expressions. "
            "Shared by CLI JSON output, Python API, MCP, and daemon HTTP."
        ),
        model=QueryUnitEnvelope,
        surfaces=(
            "polylogue --format json messages where ...",
            "polylogue --format json actions where ...",
            "polylogue --format json blocks where ...",
            "polylogue --format json assertions where ...",
            "polylogue --format json observed-events where ...",
            "polylogue --format json context-snapshots where ...",
            "Polylogue.query_units(...)",
            "MCP query_units",
            "GET /api/query-units?expression=...",
        ),
    ),
    CliOutputSchema(
        name="import-explain",
        title="Import Explain Payload",
        description=(
            "Finite explanation for `polylogue import PATH --explain --format json`: "
            "detector/parser decisions, artifact taxonomy, produced-row counts, "
            "skips, caveats, and redacted source refs."
        ),
        model=ImportExplainPayload,
        surfaces=(
            "polylogue import PATH --explain --format json",
            "polylogue import PATH --explain --format ndjson (entries)",
        ),
    ),
    CliOutputSchema(
        name="session-neighbor-candidate",
        title="Session Neighbor Candidate",
        description=(
            "Semantic-neighbor candidate for the `read --view neighbors` surface: session identity plus per-reason "
            "evidence and rank."
        ),
        model=SessionNeighborCandidatePayload,
        surfaces=("polylogue read --view neighbors --format json",),
    ),
    CliOutputSchema(
        name="mutation-result",
        title="Mutation Result",
        description=(
            "Shared result envelope for CLI, MCP, API, and daemon mutation "
            "surfaces. Used by tag, metadata, delete, and related bulk "
            "mutation actions to report status, affected counts, and "
            "optional session ids."
        ),
        model=MutationResultPayload,
        surfaces=(
            "polylogue find <query> then delete --dry-run",
            "polylogue find <query> then delete --yes",
            "MCP mutation tools",
            "daemon mutation endpoints",
        ),
    ),
    CliOutputSchema(
        name="machine-error",
        title="Machine Error Envelope",
        description=(
            "Standard CLI machine-readable error envelope. Emitted by any "
            "command that ran with `--machine` or otherwise opts into a "
            "JSON error surface."
        ),
        model=MachineErrorPayload,
        surfaces=("polylogue * --machine (error path)",),
    ),
    CliOutputSchema(
        name="machine-success",
        title="Machine Success Envelope",
        description=(
            "Standard CLI machine-readable success envelope. Emitted by "
            "commands that ran with `--machine` and produced structured "
            "output."
        ),
        model=MachineSuccessPayload,
        surfaces=("polylogue * --machine (success path)",),
    ),
    CliOutputSchema(
        name="query-error",
        title="Query Error Payload",
        description=(
            "Shared typed error payload for daemon HTTP, MCP, and query/read "
            "surfaces that need field-addressable machine-readable failures."
        ),
        model=QueryErrorPayload,
        surfaces=(
            "GET /api/sessions?query=... (error path)",
            "daemon query/read error responses",
            "MCP query/read error responses",
        ),
    ),
)


def _build_schema(entry: CliOutputSchema) -> str:
    """Return the rendered JSON Schema body for one CLI surface."""
    schema = entry.model.model_json_schema()
    schema["$schema"] = SCHEMA_DIALECT
    schema["$id"] = f"https://polylogue.dev/schemas/cli-output/{entry.name}.schema.json"
    schema["title"] = entry.title
    schema["description"] = (
        entry.description
        + "\n\nGenerated from "
        + f"`polylogue.surfaces.payloads.{entry.model.__name__}` "
        + "by `devtools render cli-output-schemas`. Do not edit by hand."
    )
    schema["x-polylogue-cli-surfaces"] = list(entry.surfaces)
    schema["x-polylogue-source-model"] = entry.model.__name__
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def _build_index(output_dir: Path) -> str:
    """Return the README/index body listing the published schemas."""
    relative_output = output_dir.as_posix()
    parts = [
        f"<!-- Generated by `{control_plane_command('render cli-output-schemas')}`. -->",
        "",
        "# CLI Output Schemas",
        "",
        (
            "Published JSON Schema contracts for `polylogue` CLI surfaces "
            "that emit machine-readable output. Use these to validate piped "
            "output in shell scripts, LLM tool-use harnesses, or any "
            "downstream consumer that depends on a stable shape."
        ),
        "",
        f"All files in `{relative_output}/` are generated. To regenerate or check sync, run:",
        "",
        "```bash",
        "devtools render cli-output-schemas         # write",
        "devtools render cli-output-schemas --check # CI sync check",
        "```",
        "",
        "## Schemas",
        "",
        "| File | Surfaces | Source model |",
        "| --- | --- | --- |",
    ]
    for entry in SCHEMAS:
        surfaces = "<br>".join(f"`{s}`" for s in entry.surfaces)
        parts.append(
            f"| [`{entry.name}.schema.json`](./{entry.name}.schema.json) | {surfaces} | `{entry.model.__name__}` |"
        )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_all(output_dir: Path, *, check: bool) -> int:
    """Render every schema and the index. Returns 0 on success, 1 on drift."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drift: list[str] = []
    expected_files: set[Path] = set()

    for entry in SCHEMAS:
        target = output_dir / f"{entry.name}.schema.json"
        expected_files.add(target)
        body = _build_schema(entry)
        if check:
            try:
                current = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                current = ""
            if current != body:
                drift.append(str(target))
        else:
            write_if_changed(target, body)

    index_target = output_dir / "README.md"
    expected_files.add(index_target)
    index_body = _build_index(output_dir)
    if check:
        try:
            current_index = index_target.read_text(encoding="utf-8")
        except FileNotFoundError:
            current_index = ""
        if current_index != index_body:
            drift.append(str(index_target))
    else:
        write_if_changed(index_target, index_body)

    # Detect unknown stale schema files left behind from removed entries.
    for existing in sorted(output_dir.glob("*.schema.json")):
        if existing not in expected_files:
            drift.append(f"{existing} (stale; not in SCHEMAS registry)")

    if drift:
        for path in drift:
            print(f"render cli-output-schemas: out of sync: {path}", file=sys.stderr)
        print(
            "render cli-output-schemas: run: " + control_plane_command("render cli-output-schemas"),
            file=sys.stderr,
        )
        return 1
    if check:
        print(f"render cli-output-schemas: sync OK: {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render JSON Schema artifacts for stable CLI output payloads (#1272).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when published schemas are out of sync with the source models.",
    )
    args = parser.parse_args(argv)
    return render_all(args.output_dir, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CliOutputSchema", "SCHEMAS", "main", "render_all"]
