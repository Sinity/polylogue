#!/usr/bin/env python3
"""bead-lint: mechanical execution-readiness + coherence audit of .beads/issues.jsonl.

Invariants (tech-tree conventions, operator-approved 2026-07-06):
  D1  no dangling dependency refs
  D2  no dependency cycles among blocks-edges
  H1  open tech-tree bead has a horizon label (frontier/mid/vision)
  H2  horizon:vision => priority P3/P4 (keeps bd ready clean)
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
  (8jg9.1 class (d) orphans is covered by `bd orphans`; class (e) stale-block is
   not representable — bd computes blocked from deps, no persisted blocked status)

Usage: python3 .agent/tools/bead-lint.py [--json] [--fresh] [path-to-issues.jsonl]
--fresh runs `bd export -o .beads/issues.jsonl` first (bd updates do NOT
immediately re-export the jsonl; a stale file yields stale findings).
Exit 1 if any finding, 0 clean. Allowlist: .agent/tools/bead-lint-allow.txt
(lines: CHECK<TAB>bead-id<TAB>reason).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HORIZONS = {"horizon:frontier", "horizon:mid", "horizon:vision"}
EPHEMERAL_RE = re.compile(r"(/realm/inbox/|(?<![\w.])/tmp/)")
# provenance-ish context that legitimizes an ephemeral path mention
PROVENANCE_HINTS = ("verbatim spec", "preserved as", "provenance", "escrow", "was in /realm/inbox", "corpus")


def load(path: Path):
    issues, deps = {}, []
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


def text_of(d) -> str:
    return " ".join(str(d.get(k) or "") for k in ("description", "design", "acceptance_criteria", "notes"))


def has_ac(d) -> bool:
    if (d.get("acceptance_criteria") or "").strip():
        return True
    notes = (d.get("notes") or "").lower()
    return "acceptance" in notes or "verify:" in notes or "ac:" in notes


def has_design(d) -> bool:
    if (d.get("design") or "").strip():
        return True
    blob = (d.get("description") or "") + (d.get("notes") or "")
    # description that names concrete code surfaces counts as design-bearing
    return bool(re.search(r"\w+\.py|\w+/\w+\.|::|polylogue/", blob))


def main() -> int:
    args = [a for a in sys.argv[1:] if a not in ("--json", "--fresh")]
    as_json = "--json" in sys.argv
    path = Path(args[0]) if args else Path(".beads/issues.jsonl")
    if "--fresh" in sys.argv:
        import subprocess

        subprocess.run(["bd", "export", "-o", str(path)], check=True, capture_output=True)
    allow_path = Path(".agent/tools/bead-lint-allow.txt")
    allow = set()
    if allow_path.exists():
        for line in allow_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and not line.startswith("#"):
                allow.add((parts[0], parts[1]))

    issues, deps = load(path)
    open_ids = {i for i, d in issues.items() if d.get("status") in ("open", "in_progress")}
    findings: list[tuple[str, str, str]] = []

    def add(check: str, bid: str, msg: str):
        if (check, bid) not in allow:
            findings.append((check, bid, msg))

    # D1 dangling deps
    for src, dst, typ in deps:
        if dst not in issues:
            add("D1", src, f"dangling dep -> {dst} ({typ})")

    # D2 cycles among blocks deps (open issues only)
    graph = defaultdict(set)
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            graph[src].add(dst)
    white, gray, black = 0, 1, 2
    color = defaultdict(int)

    def dfs(n, stack):
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

    # per-issue checks
    titles = defaultdict(list)
    children = defaultdict(int)
    for i in issues:
        if "." in i.removeprefix("polylogue-"):
            children[i.rsplit(".", 1)[0]] += 1
    dep_touch = defaultdict(int)  # epics may group members via dep edges instead of id-prefix
    for src, dst, _typ in deps:
        dep_touch[src] += 1
        dep_touch[dst] += 1
    bead_ref_re = re.compile(r"polylogue-[a-z0-9]+(?:\.[0-9]+)?")
    for i, d in issues.items():
        if d.get("status") not in ("open", "in_progress"):
            continue
        labels = set(d.get("labels") or [])
        prio = d.get("priority", 2)
        horizon = labels & HORIZONS
        titles[d.get("title", "").strip().casefold()].append(i)

        if "tech-tree" in labels and not horizon:
            add("H1", i, "tech-tree bead without horizon label")
        if "horizon:vision" in labels and prio < 3:
            add("H2", i, f"vision bead at P{prio} (should be P3/P4)")
        if "horizon:frontier" in labels and not has_ac(d):
            add("H3", i, "frontier bead without acceptance criteria")
        if "horizon:frontier" in labels and not has_design(d):
            add("H4", i, "frontier bead without design content")
        if prio <= 1 and d.get("issue_type") != "epic" and not has_ac(d):
            add("P1", i, f"P{prio} bead without acceptance criteria")
        if d.get("issue_type") != "epic" and not any(lab.startswith("area:") for lab in labels):
            add("A1", i, "open non-epic bead without an area:* label")
        if d.get("issue_type") == "decision" and re.search(r"status:\s*(adopted|decided)", text_of(d), re.IGNORECASE):
            add("B1", i, "decision bead declares adopted/decided but is still open")
        if d.get("issue_type") == "epic":
            named_members = [r for r in bead_ref_re.findall(text_of(d)) if r != i and r in issues]
            if children[i] == 0 and dep_touch[i] == 0 and not named_members:
                add("E1", i, "epic with no members (no children, no dep edges, no named bead ids)")
            if not (d.get("description") or "").strip():
                add("E2", i, "epic without description")
        blob = text_of(d)
        for ref in set(bead_ref_re.findall(blob)):
            token = ref.removeprefix("polylogue-").split(".", 1)[0]
            # id-shaped tokens only: pure-alpha words >=4 chars are English compounds
            # ("polylogue-substrate intake"); pure-numeric are #N-style refs
            if token.isalpha() and len(token) >= 4:
                continue
            if token.isdigit():
                continue
            # tolerate .N suffix references to a future child of an existing bead
            if ref not in issues and ref.rsplit(".", 1)[0] not in issues:
                add("X2", i, f"names nonexistent bead {ref}")
        if EPHEMERAL_RE.search(blob):
            low = blob.lower()
            if not any(h in low for h in PROVENANCE_HINTS):
                add("T1", i, "ephemeral path (/realm/inbox or /tmp) cited without provenance framing")

    for t, ids in titles.items():
        if t and len(ids) > 1:
            for i in ids:
                add("X1", i, f"duplicate open title with {[x for x in ids if x != i]}")

    # R1 ready-queue executable check
    blocked = set()
    for src, dst, typ in deps:
        if typ == "blocks" and src in open_ids and dst in open_ids:
            blocked.add(src)
    for i in sorted(open_ids - blocked):
        d = issues[i]
        if d.get("priority", 2) <= 2 and d.get("issue_type") not in ("epic",) and not has_ac(d):
            add("R1", i, f"READY P{d.get('priority')} bead without AC (cold agent cannot execute fast)")

    if as_json:
        print(json.dumps([{"check": c, "id": i, "msg": m} for c, i, m in findings], indent=1))
    else:
        by = defaultdict(list)
        for c, i, m in findings:
            by[c].append((i, m))
        for c in sorted(by):
            print(f"[{c}] {len(by[c])} finding(s)")
            for i, m in by[c]:
                print(f"    {i}: {m}")
        print(f"\n{len(findings)} finding(s) across {len(by)} check(s); {len(issues)} issues scanned.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
