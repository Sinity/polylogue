"""lane-brief: generate a dispatch brief for a bead lane with live evidence.

Bead prose alone produces bad lanes -- descriptions/design fields drift from
the current tree (files get renamed or deleted, migrations land, generated
surfaces regenerate) and a dispatcher copying bead text into a subagent
prompt has no way to tell current fact from stale claim without a manual
`bd show` + `git log` + `find` round-trip per bead.

This command does that round-trip once, for a whole lane (a batch of bead
ids meant to land on one branch), and emits a markdown brief:

  1. Full bead records (id/priority/type/title/description/design/AC/
     notes tail/dependencies) via `bd show <id> --json`.
  2. Footprint extraction (reusing devtools.bead_cluster's regex/helpers)
     over the combined bead text, then LIVE verification of each extracted
     path: does it exist right now, how many lines, what are its last 3
     commits. A path that no longer exists is flagged loudly -- it means
     the bead prose is stale and must not be trusted uncritically.
  3. Prior art: closed beads (from a fresh `bd export`) whose text mentions
     any of the same file paths, so the dispatcher can see what was already
     tried/decided here.

Sections the tool cannot fill (measured baseline, non-goals) are emitted as
explicit `<!-- DISPATCHER MUST FILL -->` placeholders rather than silently
omitted -- a missing section is a worse failure mode than a visible gap,
because a silently-dropped section reads as "not needed" instead of
"not yet filled in".

Usage:
    devtools workspace lane-brief polylogue-abc polylogue-def
    devtools workspace lane-brief polylogue-abc --out .agent/scratch/lane-abc.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from devtools.bead_cluster import _FILE_PAT as _LANE_FILE_PAT
except ImportError:  # pragma: no cover -- defensive: keep working if bead_cluster moves/breaks
    import re

    _LANE_FILE_PAT = re.compile(
        r"(?:polylogue|tests|\.agent|docs|storage|pipeline|daemon|cli|mcp"
        r"|browser[_-]extension|browser_capture|coordination|archive|insights"
        r"|context|core|hooks|maintenance|artifacts)"
        r"/[\w./\-]+\.(?:py|ts|js|yaml|yml|md|json|sql|html)",
        re.IGNORECASE,
    )

BeadDict = dict[str, Any]

_ANTI_VACUITY_CONTRACT = (
    "Name the production caller that exercises every new surface this lane adds. "
    "A test-local reimplementation, self-authorized registry, or write-only table is "
    "a FAILED lane even if green. State the implementation mutation that would make "
    "your test fail."
)

_HAZARDS = (
    "commit every logical chunk (worktree auto-clean destroys uncommitted work)",
    "never cd to the main checkout",
    "`bd` reimports .beads/issues.jsonl on every invocation -- run "
    "`bd export -o .beads/issues.jsonl` after every bead write, do not commit that file in this lane",
    "adding any module under polylogue/ requires "
    "`devtools render topology-projection && devtools render topology-status`",
    "no `git stash` (refs/stash is shared across worktrees)",
    "TMPDIR=/realm/tmp (system /tmp is a 6GiB tmpfs)",
    "run `python -m devtools ...` from the worktree root so the worktree's code wins over the shared-venv .pth",
)

_VERIFICATION_TIER = (
    "Inner loop: `devtools test <files>` / testmon-affected selection.\n"
    "Pre-PR: `devtools verify`.\n"
    "Do NOT run whole-directory pytest.\n"
    "Per-PR CI skips the heavy test suite, so green checks alone are not test evidence -- "
    "verify locally."
)


@dataclass
class BeadRecord:
    id: str
    found: bool
    priority: int | None = None
    issue_type: str | None = None
    title: str = ""
    description: str = ""
    design: str = ""
    acceptance_criteria: str = ""
    notes_tail: str = ""
    dependencies: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class FootprintEvidence:
    path: str
    exists: bool
    line_count: int | None = None
    recent_commits: list[str] = field(default_factory=list)


@dataclass
class PriorArtHit:
    id: str
    title: str
    close_reason_head: str


def _dep_labels(record: BeadDict) -> list[str]:
    labels: list[str] = []
    for dep in record.get("dependencies") or []:
        if isinstance(dep, dict):
            target = dep.get("depends_on_id") or dep.get("to_id") or dep.get("id") or "?"
            dep_type = dep.get("type") or dep.get("dep_type") or "?"
            labels.append(f"{target}({dep_type})")
        else:
            labels.append(str(dep))
    return labels


def _fetch_bead(bead_id: str) -> BeadRecord:
    result = subprocess.run(["bd", "show", bead_id, "--json"], capture_output=True, text=True)
    if result.returncode != 0:
        return BeadRecord(id=bead_id, found=False, error=result.stderr.strip()[:200] or "bd show failed")
    try:
        payload = json.loads(result.stdout)
        record = payload[0] if isinstance(payload, list) else payload
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        return BeadRecord(id=bead_id, found=False, error=f"unparseable bd show output: {exc}")

    notes = record.get("notes") or ""
    return BeadRecord(
        id=record.get("id", bead_id),
        found=True,
        priority=record.get("priority"),
        issue_type=record.get("issue_type"),
        title=record.get("title", ""),
        description=record.get("description") or "",
        design=record.get("design") or "",
        acceptance_criteria=record.get("acceptance_criteria") or "",
        notes_tail=notes[-500:],
        dependencies=_dep_labels(record),
    )


def _combined_text(records: Sequence[BeadRecord]) -> str:
    parts: list[str] = []
    for r in records:
        parts.extend([r.description, r.design, r.acceptance_criteria, r.notes_tail])
    return " ".join(p for p in parts if p)


def _extract_paths(text: str) -> list[str]:
    return list(dict.fromkeys(_LANE_FILE_PAT.findall(text)))


def _git_log_recent(repo_root: Path, path: str, limit: int = 3) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--oneline", f"-{limit}", "--", path],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _verify_footprint(repo_root: Path, paths: list[str]) -> list[FootprintEvidence]:
    evidence: list[FootprintEvidence] = []
    for path in paths:
        full = repo_root / path
        if full.is_file():
            try:
                line_count = sum(1 for _ in full.open("r", errors="replace"))
            except OSError:
                line_count = None
            evidence.append(
                FootprintEvidence(
                    path=path,
                    exists=True,
                    line_count=line_count,
                    recent_commits=_git_log_recent(repo_root, path),
                )
            )
        else:
            evidence.append(FootprintEvidence(path=path, exists=False))
    return evidence


def _load_bd_export(repo_root: Path, tmpdir: Path) -> list[BeadDict]:
    """Export the full bd backlog and parse it for closed-bead prior art.

    Degrades quietly to an empty list on any subprocess/parse failure --
    prior-art is advisory, and a brief that omits it is still useful.
    """
    export_path = tmpdir / "lane-brief-export.jsonl"
    try:
        result = subprocess.run(
            ["bd", "export", "-o", str(export_path)],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0 or not export_path.exists():
        return []

    records: list[BeadDict] = []
    try:
        with export_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _find_prior_art(
    export_records: list[BeadDict],
    paths: list[str],
    exclude_ids: set[str],
    limit: int = 5,
) -> list[PriorArtHit]:
    if not paths:
        return []
    hits: list[PriorArtHit] = []
    for rec in export_records:
        if rec.get("_type", "issue") != "issue":
            continue
        if rec.get("status") != "closed":
            continue
        bead_id = rec.get("id", "")
        if bead_id in exclude_ids:
            continue
        text = " ".join(
            filter(
                None,
                [rec.get("description", ""), rec.get("design", ""), rec.get("notes", "")],
            )
        )
        if any(p in text for p in paths):
            close_reason = (rec.get("close_reason") or rec.get("notes") or "").strip().replace("\n", " ")
            hits.append(
                PriorArtHit(
                    id=bead_id,
                    title=rec.get("title", ""),
                    close_reason_head=close_reason[:160],
                )
            )
            if len(hits) >= limit:
                break
    return hits


def _render_markdown(
    lane_ids: list[str],
    records: list[BeadRecord],
    footprint: list[FootprintEvidence],
    prior_art: list[PriorArtHit],
) -> str:
    lines: list[str] = []
    lines.append(f"# Lane brief: {', '.join(lane_ids)}")
    lines.append("")

    lines.append("## Scope")
    lines.append("")
    for r in records:
        if not r.found:
            lines.append(f"### {r.id} -- NOT FOUND")
            lines.append(f"`bd show {r.id}` failed: {r.error}")
            lines.append("")
            continue
        lines.append(f"### {r.id} -- P{r.priority} {r.issue_type} -- {r.title}")
        lines.append("")
        lines.append("**Description:**")
        lines.append("")
        lines.append(r.description or "(empty)")
        lines.append("")
        lines.append("**Design:**")
        lines.append("")
        lines.append(r.design or "(empty)")
        lines.append("")
        lines.append("**Acceptance criteria:**")
        lines.append("")
        lines.append(r.acceptance_criteria or "(empty)")
        lines.append("")
        if r.notes_tail:
            lines.append("**Notes (last 500 chars):**")
            lines.append("")
            lines.append(r.notes_tail)
            lines.append("")
        if r.dependencies:
            lines.append(f"**Dependencies:** {', '.join(r.dependencies)}")
            lines.append("")
    lines.append("")

    lines.append("## Footprint (verified)")
    lines.append("")
    if not footprint:
        lines.append("No file paths were extracted from the combined bead text.")
    else:
        for ev in footprint:
            if ev.exists:
                lines.append(f"- `{ev.path}` -- exists, {ev.line_count} lines")
                for commit in ev.recent_commits:
                    lines.append(f"    - {commit}")
                if not ev.recent_commits:
                    lines.append("    - (no commit history found for this path)")
            else:
                lines.append(f"- `{ev.path}` -- **PATH NOT FOUND** -- bead prose may be stale, verify before trusting")
    lines.append("")

    lines.append("## Prior art")
    lines.append("")
    if prior_art:
        for hit in prior_art:
            lines.append(f"- **{hit.id}** -- {hit.title}")
            if hit.close_reason_head:
                lines.append(f"    - close reason: {hit.close_reason_head}")
    else:
        lines.append("No closed beads found mentioning the same file paths.")
    lines.append("")

    lines.append("## Measured baseline")
    lines.append("")
    lines.append("<!-- DISPATCHER MUST FILL -->")
    lines.append(
        "Paste the actual command output that motivates this lane (a failing test, a "
        "measured metric, a reproduction). Bead prose alone is not evidence."
    )
    lines.append("")

    lines.append("## Non-goals")
    lines.append("")
    lines.append("<!-- DISPATCHER MUST FILL -->")
    lines.append("State explicitly what this lane will NOT change, to bound scope creep.")
    lines.append("")

    lines.append("## Anti-vacuity contract")
    lines.append("")
    lines.append(_ANTI_VACUITY_CONTRACT)
    lines.append("")

    lines.append("## Hazards (standing)")
    lines.append("")
    for hazard in _HAZARDS:
        lines.append(f"- {hazard}")
    lines.append("")

    lines.append("## Verification tier")
    lines.append("")
    lines.append(_VERIFICATION_TIER)
    lines.append("")

    return "\n".join(lines)


def _repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return Path.cwd()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bead_ids", nargs="+", help="Bead ids in this lane (e.g. polylogue-abc polylogue-def)")
    parser.add_argument("--out", metavar="FILE", help="Write the brief to this path instead of stdout")
    parser.add_argument(
        "--tmpdir",
        metavar="DIR",
        default="/realm/tmp",
        help="Scratch dir for the bd export used for prior-art scanning (default: /realm/tmp)",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    records = [_fetch_bead(bead_id) for bead_id in args.bead_ids]

    combined_text = _combined_text(records)
    paths = _extract_paths(combined_text)
    footprint = _verify_footprint(repo_root, paths)

    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    export_records = _load_bd_export(repo_root, tmpdir)
    exclude_ids = {r.id for r in records}
    prior_art = _find_prior_art(export_records, paths, exclude_ids)

    brief = _render_markdown(list(args.bead_ids), records, footprint, prior_art)

    if args.out:
        Path(args.out).write_text(brief)
        print(f"Wrote lane brief to {args.out}")
    else:
        print(brief)

    return 0


if __name__ == "__main__":
    sys.exit(main())
