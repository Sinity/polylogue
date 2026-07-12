"""Verify Beads backlog structure invariants (`.beads/issues.jsonl`).

Background
----------

Backlog structure trails filing unless an invariant lint enforces it: the
2026-07-06 session needed a 41-agent sweep to recover from accumulated drift
(missing acceptance criteria, dangling dependency refs, unlabeled beads,
stale "adopted" decisions left open). This is the backlog equivalent of
`devtools lab policy schema-versioning` / `docs-drift` / `timestamp-doctrine`:
a mechanical check that fails a gate instead of drift silently accumulating
until an archaeology session (polylogue-8jg9.1).

Checks over `.beads/issues.jsonl`:

  D1  no dangling dependency refs
  D2  no dependency cycles among blocks-edges
  H1  open tech-tree bead has a horizon label (frontier/mid/vision)
  H2  horizon:vision => priority P3/P4 (keeps `bd ready` clean)
  H3  open horizon:frontier bead has acceptance criteria (field or notes sidecar)
  H4  open horizon:frontier bead has design content (field, notes, or description with file paths)
  P1  open P0/P1 bead has acceptance criteria
  E1  epic has members: id-prefix children, dep edges, or bead ids named in its text
  E2  epic has a non-empty description (WHY + member map)
  T1  no ephemeral-path ground truth: /realm/inbox/ or /tmp/ cited outside provenance context
  X1  duplicate open titles (exact, case-folded)
  X2  bead id named in an open bead's text does not exist
  R1  READY bead (open, all blocks-deps closed) at P1/P2 lacking AC — the fast-execution gap
  A1  open non-epic bead has at least one area:* label
  B1  open decision-type bead whose text declares Status: adopted/decided should be closed

(8jg9.1's design names five conceptual classes: (a) P0/P1 missing AC = P1;
(b) decision-type bead stuck past adopted/decided = B1; (c) no area:* label =
A1; (d) orphan beads with no epic parent — covered by the native `bd orphans`
command, not duplicated here; (e) a blocks-edge pointing at a closed bead —
not representable, since bd computes "blocked" live from dependency status
rather than persisting a blocked flag that could go stale.)

This module supersedes the standalone `.agent/tools/bead-lint.py` script
(same algorithm, ported so the check runs through the standing `devtools
verify --lab` gate instead of requiring a manual invocation). The allowlist
at `.agent/tools/bead-lint-allow.txt` (format: `CHECK<TAB>bead-id<TAB>reason`
per line) is unchanged.

Wired into ``devtools verify --lab`` alongside the other policy checks, since
this is a repo-hygiene boundary check rather than a per-edit gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root

_HORIZONS = {"horizon:frontier", "horizon:mid", "horizon:vision"}
_EPHEMERAL_RE = re.compile(r"(/realm/inbox/|(?<![\w.])/tmp/)")
# Provenance-ish context that legitimizes an ephemeral path mention.
_PROVENANCE_HINTS = ("verbatim spec", "preserved as", "provenance", "escrow", "was in /realm/inbox", "corpus")
_BEAD_REF_RE = re.compile(r"polylogue-[a-z0-9]+(?:\.[0-9]+)?")

_DEFAULT_ISSUES_RELPATH = ".beads/issues.jsonl"
_DEFAULT_ALLOWLIST_RELPATH = ".agent/tools/bead-lint-allow.txt"


@dataclass(frozen=True, slots=True)
class Finding:
    check: str
    bead_id: str
    message: str


def _load(path: Path) -> tuple[dict[str, dict[str, object]], list[tuple[str, str, str]]]:
    issues: dict[str, dict[str, object]] = {}
    deps: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("_type") == "issue":
            issues[d["id"]] = d
            for dep in d.get("dependencies") or []:
                deps.append((d["id"], dep.get("depends_on_id"), dep.get("type", "blocks")))
        elif d.get("_type") == "dependency":
            deps.append((d.get("issue_id"), d.get("depends_on_id"), d.get("type", "blocks")))
    return issues, deps


def _text_of(d: dict[str, object]) -> str:
    return " ".join(str(d.get(k) or "") for k in ("description", "design", "acceptance_criteria", "notes"))


def _has_ac(d: dict[str, object]) -> bool:
    if str(d.get("acceptance_criteria") or "").strip():
        return True
    notes = str(d.get("notes") or "").lower()
    return "acceptance" in notes or "verify:" in notes or "ac:" in notes


def _has_design(d: dict[str, object]) -> bool:
    if str(d.get("design") or "").strip():
        return True
    blob = str(d.get("description") or "") + str(d.get("notes") or "")
    # A description that names concrete code surfaces counts as design-bearing.
    return bool(re.search(r"\w+\.py|\w+/\w+\.|::|polylogue/", blob))


def _labels_of(d: dict[str, object]) -> set[str]:
    raw = d.get("labels") or []
    return {str(lab) for lab in raw} if isinstance(raw, list) else set()


def _priority_of(d: dict[str, object]) -> int:
    prio = d.get("priority", 2)
    return int(prio) if isinstance(prio, int | float | str) and str(prio).lstrip("-").isdigit() else 2


def _load_allowlist(allow_path: Path) -> set[tuple[str, str]]:
    allow: set[tuple[str, str]] = set()
    if allow_path.exists():
        for line in allow_path.read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                allow.add((parts[0], parts[1]))
    return allow


def collect_findings(path: Path | None = None, allow_path: Path | None = None) -> list[Finding]:
    """Run all 12 backlog-hygiene checks against a Beads jsonl export.

    ``path`` defaults to ``.beads/issues.jsonl`` under the repo root;
    ``allow_path`` defaults to ``.agent/tools/bead-lint-allow.txt``.
    """
    root = _get_root()
    if path is None:
        path = root / _DEFAULT_ISSUES_RELPATH
    if allow_path is None:
        allow_path = root / _DEFAULT_ALLOWLIST_RELPATH
    allow = _load_allowlist(allow_path)

    issues, deps = _load(path)
    open_ids = {i for i, d in issues.items() if d.get("status") in ("open", "in_progress")}
    findings: list[Finding] = []

    def add(check: str, bid: str, msg: str) -> None:
        if (check, bid) not in allow:
            findings.append(Finding(check, bid, msg))

    # D1 dangling deps
    for src, dst, typ in deps:
        if dst not in issues:
            add("D1", src, f"dangling dep -> {dst} ({typ})")

    # D2 cycles among blocks deps (open issues only)
    graph: dict[str, set[str]] = defaultdict(set)
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            graph[src].add(dst)
    white, gray, black = 0, 1, 2
    color: dict[str, int] = defaultdict(int)

    def dfs(n: str, stack: list[str]) -> None:
        color[n] = gray
        for m in graph[n]:
            if color[m] == gray:
                cyc = stack[stack.index(m) :] + [m] if m in stack else [n, m]
                add("D2", n, "blocks-cycle: " + " -> ".join(cyc))
            elif color[m] == white:
                dfs(m, stack + [m])
        color[n] = black

    for n in list(graph):
        if color[n] == white:
            dfs(n, [n])

    # Per-issue checks.
    titles: dict[str, list[str]] = defaultdict(list)
    children: dict[str, int] = defaultdict(int)
    for i in issues:
        if "." in i.removeprefix("polylogue-"):
            children[i.rsplit(".", 1)[0]] += 1
    dep_touch: dict[str, int] = defaultdict(int)  # epics may group members via dep edges instead of id-prefix
    for src, dst, _typ in deps:
        dep_touch[src] += 1
        dep_touch[dst] += 1

    for i, d in issues.items():
        if d.get("status") not in ("open", "in_progress"):
            continue
        labels = _labels_of(d)
        prio = _priority_of(d)
        horizon = labels & _HORIZONS
        titles[str(d.get("title", "")).strip().casefold()].append(i)

        if "tech-tree" in labels and not horizon:
            add("H1", i, "tech-tree bead without horizon label")
        if "horizon:vision" in labels and prio < 3:
            add("H2", i, f"vision bead at P{prio} (should be P3/P4)")
        if "horizon:frontier" in labels and not _has_ac(d):
            add("H3", i, "frontier bead without acceptance criteria")
        if "horizon:frontier" in labels and not _has_design(d):
            add("H4", i, "frontier bead without design content")
        if prio <= 1 and d.get("issue_type") != "epic" and not _has_ac(d):
            add("P1", i, f"P{prio} bead without acceptance criteria")
        if d.get("issue_type") != "epic" and not any(lab.startswith("area:") for lab in labels):
            add("A1", i, "open non-epic bead without an area:* label")
        if d.get("issue_type") == "decision" and re.search(r"status:\s*(adopted|decided)", _text_of(d), re.IGNORECASE):
            add("B1", i, "decision bead declares adopted/decided but is still open")
        if d.get("issue_type") == "epic":
            named_members = [r for r in _BEAD_REF_RE.findall(_text_of(d)) if r != i and r in issues]
            if children[i] == 0 and dep_touch[i] == 0 and not named_members:
                add("E1", i, "epic with no members (no children, no dep edges, no named bead ids)")
            if not str(d.get("description") or "").strip():
                add("E2", i, "epic without description")
        blob = _text_of(d)
        for ref in set(_BEAD_REF_RE.findall(blob)):
            token = ref.removeprefix("polylogue-").split(".", 1)[0]
            # id-shaped tokens only: pure-alpha words >=4 chars are English compounds
            # ("polylogue-substrate intake"); pure-numeric are #N-style refs.
            if token.isalpha() and len(token) >= 4:
                continue
            if token.isdigit():
                continue
            # Tolerate .N suffix references to a future child of an existing bead.
            if ref not in issues and ref.rsplit(".", 1)[0] not in issues:
                add("X2", i, f"names nonexistent bead {ref}")
        if _EPHEMERAL_RE.search(blob):
            low = blob.lower()
            if not any(h in low for h in _PROVENANCE_HINTS):
                add("T1", i, "ephemeral path (/realm/inbox or /tmp) cited without provenance framing")

    for _t, ids in titles.items():
        if _t and len(ids) > 1:
            for i in ids:
                add("X1", i, f"duplicate open title with {[x for x in ids if x != i]}")

    # R1 ready-queue executable check.
    blocked: set[str] = set()
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            blocked.add(src)
    for i in sorted(open_ids - blocked):
        d = issues[i]
        if _priority_of(d) <= 2 and d.get("issue_type") not in ("epic",) and not _has_ac(d):
            add("R1", i, f"READY P{_priority_of(d)} bead without AC (cold agent cannot execute fast)")

    return findings


def _format_report(findings: list[Finding], *, issues_scanned: int) -> str:
    if not findings:
        return f"backlog hygiene: zero unhandled findings across {issues_scanned} issues scanned."
    by: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by[f.check].append(f)
    lines: list[str] = []
    for check in sorted(by):
        lines.append(f"[{check}] {len(by[check])} finding(s)")
        for f in by[check]:
            lines.append(f"    {f.bead_id}: {f.message}")
    lines.append(f"\n{len(findings)} finding(s) across {len(by)} check(s); {issues_scanned} issues scanned.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="run `bd export -o <path>` first (bd updates do not immediately re-export the jsonl)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="path to issues.jsonl (default: .beads/issues.jsonl under the repo root)",
    )
    args = parser.parse_args(argv)

    root = _get_root()
    path = Path(args.path) if args.path else root / _DEFAULT_ISSUES_RELPATH

    if args.fresh:
        subprocess.run(["bd", "export", "-o", str(path)], check=True, capture_output=True)

    if not path.exists():
        message = f"backlog hygiene: {path} does not exist (no Beads workspace to check)."
        if args.json:
            print(json.dumps({"ok": True, "findings": [], "issues_scanned": 0, "skipped": message}, indent=2))
        else:
            print(message)
        return 0

    findings = collect_findings(path=path, allow_path=root / _DEFAULT_ALLOWLIST_RELPATH)
    issues_scanned = len(_load(path)[0])

    if args.json:
        payload = {
            "ok": not findings,
            "issues_scanned": issues_scanned,
            "findings": [{"check": f.check, "id": f.bead_id, "msg": f.message} for f in findings],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(findings, issues_scanned=issues_scanned))

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
