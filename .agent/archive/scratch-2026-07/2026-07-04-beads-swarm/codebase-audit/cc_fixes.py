#!/usr/bin/env python3
"""Direct fixes: test-debt beads + manifest rewrite + attachment reframe + notes memory. --live."""

import re
import subprocess
import sys

LIVE = "--live" in sys.argv


def bd(a):
    if not LIVE:
        print("DRY: bd", *a[:3])
        return ""
    r = subprocess.run(["bd"] + a, capture_output=True, text=True)
    if r.returncode != 0:
        print("  ERR:", (r.stdout + r.stderr)[:120])
    return r.stdout + r.stderr


def create(title, typ, prio, parent, design, acc, labels):
    out = bd(
        [
            "create",
            title,
            "--type",
            typ,
            "--priority",
            str(prio),
            "--parent",
            parent,
            "--design",
            design,
            "--acceptance",
            acc,
            "--labels",
            labels,
        ]
    )
    m = re.search(r"polylogue-[a-z0-9.]+", out)
    nid = m.group(0) if m else "DRY"
    print(f"  CREATE {nid}: {title[:50]}")
    return nid


# ===== 1. TEST-DEBT beads for the 6 genuinely-unowned #590 gaps =====
b_storage = create(
    "Storage-layer correctness scenario family",
    "task",
    2,
    "polylogue-9e5",
    "scenario-coverage.yaml gap 'storage-correctness' orphaned on gh#590. Build a scenario family (devtools lab projections / scenarios) exercising split-tier writes, content-hash idempotency, FTS trigger integrity, blob-lease GC, and lineage composition against seeded archives.",
    "A storage-correctness scenario family exists and runs via devtools lab lanes; it covers idempotent re-ingest, FTS trigger drift, and lineage composition; scenario-coverage.yaml references this bead, not gh#590. Verify: devtools lab projections + lab lanes.",
    "area:audit,area:storage",
)
b_perf = create(
    "Performance/throughput scenario family",
    "task",
    3,
    "polylogue-20d",
    "scenario-coverage.yaml gap 'performance' orphaned on gh#590. A scenario family exercising memory + throughput budgets (ties 20d.14 SLO budgets + the bench campaigns) so perf regressions are caught as scenarios, not ad-hoc.",
    "A performance scenario family runs the memory/throughput budgets from 20d.14 and fails on regression; manifest references this bead. Verify: devtools bench + lab lanes.",
    "area:audit,area:perf",
)
b_rebuild = create(
    "Schema rebuild-safety scenario",
    "task",
    2,
    "polylogue-1xc",
    "scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590. A scenario proving derived-tier rebuild (index/embeddings) from durable source/user evidence is lossless and idempotent, and durable-tier additive migration preserves user.db assertions. Ties 1xc.7 scale-regression lane + z7rv migration framework.",
    "A rebuild-safety scenario resets a derived tier and rebuilds from source, asserting byte/row parity + no user.db loss; a durable additive migration round-trips behind the backup gate. Verify: the scenario under devtools lab lanes.",
    "area:audit,area:storage",
)
b_flaky = create(
    "Flakiness tracking + quarantine lane",
    "task",
    3,
    "polylogue-9e5",
    "test-quality-coverage.yaml gap 'flakiness' orphaned on gh#590. No systematic flakiness tracking/retry/quarantine. Add flaky-test detection (rerun-on-fail classification) + a quarantine registry so intermittent failures are tracked, not silently reruns.",
    "Flaky nodes are detected and recorded; a quarantine list exists; the manifest references this bead. Verify: devtools verify evidence / lab lanes.",
    "area:audit,area:test",
)
b_mock = create(
    "Mock-depth measurement",
    "task",
    3,
    "polylogue-9e5",
    "test-quality-coverage.yaml gap 'mock-depth' orphaned on gh#590. Mock depth is neither measured nor enforced, risking tests that assert against mocks rather than behavior.",
    "A measure reports mock depth per test module and flags over-mocked suites; manifest references this bead. Verify: the mock-depth measure output.",
    "area:audit,area:test",
)
b_permod = create(
    "Per-module coverage tracking (beyond aggregate floor)",
    "task",
    3,
    "polylogue-9e5",
    "test-quality-coverage.yaml gap 'per-module-coverage' orphaned on gh#590. Only aggregate fail_under is tracked; per-module coverage is invisible, hiding under-covered modules behind a green aggregate.",
    "devtools verify coverage reports per-module coverage and flags modules below a floor; manifest references this bead. Verify: devtools verify coverage --per-module.",
    "area:audit,area:test",
)

# gap-id -> owning bead id (new + existing mappings)
GAPMAP = {
    "scenario.storage-correctness": b_storage,
    "scenario.performance": b_perf,
    "scenario.security-privacy": "polylogue-kwsb",
    "scenario.distribution": "polylogue-3tl.7",
    "scenario.schema-rebuild-safety": b_rebuild,
    "test-quality.flakiness": b_flaky,
    "test-quality.mock-depth": b_mock,
    "test-quality.fuzz-ci": "polylogue-9e5.18",
    "test-quality.per-module-coverage": b_permod,
}


# ===== 2. REWRITE manifests: issue: 590 -> bead: <id> per gap =====
def rewrite(path):
    with open(path) as fh:
        lines = fh.read().splitlines(keepends=True)
    cur = None
    out = []
    n = 0
    for ln in lines:
        m = re.match(r"\s*-\s*id:\s*(\S+)", ln)
        if m:
            cur = m.group(1).strip()
        if re.match(r"\s*issue:\s*590\s*$", ln) and cur in GAPMAP:
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}bead: {GAPMAP[cur]}\n")
            n += 1
            continue
        out.append(ln)
    if LIVE:
        with open(path, "w") as fh:
            fh.writelines(out)
    print(f"  REWRITE {path}: {n} gaps -> bead owners")


rewrite("docs/plans/scenario-coverage.yaml")
rewrite("docs/plans/test-quality-coverage.yaml")

# close the umbrella adoption bead (its job is now done)
bd(
    [
        "close",
        "polylogue-9e5.17",
        "--reason",
        "Adoption complete: the 9 manifest coverage-gaps now each own a bead (storage/perf/rebuild/flakiness/mock/per-module as new 9e5 children; security->kwsb, distribution->3tl.7, fuzz-ci->9e5.18) and scenario-coverage.yaml + test-quality-coverage.yaml reference bead owners instead of gh#590. The anonymous debt is retired; implementing each scenario/measure is the tracked residual.",
    ]
)
print("  CLOSE 9e5.17 (adoption complete)")

# ===== 3. ATTACHMENT REFRAME: forward-capture-first =====
bd(
    [
        "update",
        "polylogue-83u",
        "--acceptance",
        "REFRAMED (operator 2026-07-04): the goal is to CAPTURE attachment bytes going forward, not miss-then-account. (1) Forward capture is default at ingest/browser-capture: uploaded + inline bytes land in the blob store at acquisition time (83u.3, 83u.1). (2) Non-inline bytes that STILL EXIST at their source are re-acquired (83u.2) — 'we're not getting some that exist' is a bug, not acceptable loss. (3) A permanent unfetchable floor is NORMAL and expected (source deleted, pre-install history, provider expiry) — the census (83u.6) reports it as honest baseline accounting, never as a failure to fix. Terminal state: no attachment whose bytes were reachable at capture time is lost; the unfetchable floor is measured and explained; no synthetic hashes. Verify: a live-capture session with an upload stores the blob; the census separates reachable-but-missed (bug) from genuinely-unfetchable (normal).",
    ]
)
bd(["update", "polylogue-83u.3", "-p", "1"])  # forward capture in browser = the priority
bd(
    [
        "update",
        "polylogue-83u.2",
        "--append-notes",
        "REFRAME: prioritize the 'bytes still exist at source but we're not getting them' subset — that is a capture bug, distinct from the genuinely-unfetchable floor. Re-acquire what is reachable.",
    ]
)
bd(
    [
        "update",
        "polylogue-83u.6",
        "--append-notes",
        "REFRAME: the census reports the unfetchable floor as NORMAL expected accounting (source-deleted / pre-install / provider-expiry), not a defect backlog. Its job is to separate reachable-but-missed (feeds 83u.2/83u.3 as bugs) from genuinely-gone (baseline).",
    ]
)
print("  ATTACHMENT reframe: 83u epic AC + 83u.3->P1 + 83u.2/83u.6 notes")

# ===== 4. NOTES-SIDECAR memory =====
bd(
    [
        "remember",
        "--key",
        "notes-sidecar-trap",
        "NOTES SIDECAR TRAP (2026-07-04): a Beads issue's `notes` field often carries execution-grade design/acceptance content (operator sidecars) that the `design`/`acceptance_criteria` fields are null for. Before declaring a bead underspecified or filling its AC, ALWAYS read `notes` (bd show <id> --json | .notes) and lift/formalize existing content rather than inventing. The standing hygiene lint (polylogue-8jg9.1) treats notes-with-content as satisfying the AC invariant.",
    ]
)
print("  MEMORY notes-sidecar-trap recorded")
print("\nDONE" if LIVE else "\nDRY DONE")
