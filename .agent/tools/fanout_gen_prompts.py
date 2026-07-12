#!/usr/bin/env python3
"""Generate execution-grade fanout lane prompts from the live beads file.

Permanent tool: lane specs live here; bead titles/status are pulled from
.beads/issues.jsonl at generation time so prompts never embed stale scope.
Regenerate after any beads-adjustment pass:

    python3 .agent/tools/fanout_gen_prompts.py            # writes prompts
    python3 .agent/tools/fanout_gen_prompts.py --check    # roster health only

Output: .agent/scratch/fanout-prompts/<lane>.prompt  (lane name == worktree
dir name under /realm/worktrees/, launched via .agent/tools/fanout-launch.sh).

Roster health rules enforced at generation:
  - a bead assigned to a lane that is CLOSED -> error (roster stale)
  - a bead IN_PROGRESS -> warning (possible stale claim from a killed agent;
    coordinator must confirm ownership before launch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BEADS = REPO / ".beads/issues.jsonl"
OUT = REPO / ".agent/scratch/fanout-prompts"

# wave 2 lanes conflict with a wave-1 lane's footprint; launch them only after
# the named lane's PR merges.
LANES: dict[str, dict] = {
    "polylogue-fable-demo": {
        "beads": ["xiyv", "fnm.1", "212.9.1"],
        "goal": (
            "Fable-delegation demo enablers, in order: (1) xiyv deterministic "
            "cohort/sample manifests; (2) fnm.1 NARROWED first slice — "
            "multi-field group-by + count/proportion aggregates with explicit "
            "denominator, n, and unknown/missing counts (percentiles/time "
            "buckets stay in-bead but later); (3) 212.9.1 the private "
            "descriptive Fable delegation packet, only if 1+2 land cleanly."
        ),
        "own": ["polylogue/archive/query/", "polylogue/insights/", "tests/unit/archive/", "tests/unit/insights/"],
        "avoid": ["polylogue/storage/sqlite/", "polylogue/daemon/", "browser-extension/"],
        "verify": "devtools test tests/unit/archive tests/unit/insights -k 'aggregate or cohort or manifest' (narrow further per change)",
    },
    "polylogue-hermes-wedge": {
        "beads": ["fs1.3", "fs1.11", "fs1.12"],
        "goal": (
            "Hermes integration critical path, strictly in order: fs1.3 "
            "per-source coverage/fidelity declaration; fs1.11 bounded "
            "read-only recall + effective-context audit (exact delivery "
            "manifests); fs1.12 the compact evidence-and-continuity demo "
            "composing fs1.1/fs1.3/fs1.11 — NO parallel importer/manifest/"
            "report machinery (epic note 2026-07-10)."
        ),
        "own": [
            "polylogue/sources/parsers/hermes_state.py",
            "polylogue/mcp/",
            "polylogue/daemon/recall*",
            "tests/unit/sources/",
            "tests/unit/mcp/",
        ],
        "avoid": ["polylogue/storage/sqlite/archive_tiers/", "browser-extension/"],
        "verify": "devtools test tests/unit/sources/test_hermes* tests/unit/mcp -k 'hermes or recall or manifest'",
    },
    "polylogue-extension-redesign": {
        "beads": [],  # ids assigned by the design-pack import in the beads pass
        "goal": (
            "Extension redesign per docs/design/browser-capture-redesign/ and "
            "the handoff pack (coordinator files beads before launch; read "
            "them via bd ready --json | grep redesign). Order: silent-capture "
            "P1 bug (auto-capture trigger never fires — 160 GETs, 0 POSTs) "
            "first; then popup mission-control; operator status vocabulary; "
            "'What Polylogue did here' timeline (doing-nothing must be a "
            "logged visible event). Pixel specs: project/Polylogue "
            "Redesign.dc.html in the design dir."
        ),
        "own": ["browser-extension/"],
        "avoid": ["polylogue/", "webui/"],
        "verify": "cd browser-extension && npm test (vitest suite; 143 tests baseline must not regress)",
    },
    "polylogue-capture-hardening": {
        "beads": ["57rp", "5k5l", "3v1.1"],
        "goal": (
            "Capture-path raw-authority + asset hardening: 57rp reacquire "
            "replaced browser-capture snapshots under typed raw authority; "
            "5k5l asset acquisition (sandbox + file-service bytes) — check "
            "5k5l.1 state first, another agent may have finished it; 3v1.1 "
            "concurrent-instance attribution/dedup/spool safety."
        ),
        "own": [
            "browser-extension/src/background.js",
            "polylogue/daemon/receiver*",
            "polylogue/browser_capture/",
            "tests/unit/daemon/",
        ],
        "avoid": ["browser-extension/src/popup*", "browser-extension/src/content/"],
        "verify": "devtools test tests/unit/daemon -k 'receiver or capture' + extension tests for background.js changes",
    },
    "polylogue-web-cockpit": {
        "beads": ["bby.17", "nhjs"],
        "goal": (
            "Web cockpit depth: bby.17 privacy-safe overview + evidence "
            "aggregates API; nhjs bounded web reader shapes for long sessions "
            "and aggregates (no unbounded payloads)."
        ),
        "own": ["polylogue/daemon/web_shell_*.py", "polylogue/daemon/http.py (web routes only)", "webui/tests/"],
        "avoid": ["polylogue/daemon/convergence*", "polylogue/storage/"],
        "verify": "devtools test tests/unit/daemon -k 'web or shell' + playwright smoke if route shapes change",
    },
    "polylogue-distribution": {
        "beads": ["y8s5"],
        "goal": (
            "Distribution/legibility: everything y8s5 unblocks — PyPI/pipx "
            "publishable package, Homebrew/GHCR smoke lanes, install matrix "
            "doc, extension packaging for store submission. v0.2.0 is tagged; "
            "wire the release artifacts, don't re-cut."
        ),
        "own": [
            "pyproject.toml",
            ".github/workflows/",
            "packaging/",
            "docs/install*",
            "browser-extension/package.json",
        ],
        "avoid": ["polylogue/ (runtime code)"],
        "verify": "wheel/sdist build + install smoke in a clean venv; CI workflow dry runs",
    },
    "polylogue-provider-origin": {
        "beads": ["9e5.8"],
        "model": "gpt-5.6-sol",
        "goal": (
            "Execute the next sequenced phase of the provider->origin "
            "retirement per 9e5.8's plan (read the bead notes FIRST). If the "
            "plan in the bead is incomplete or ambiguous about the phase you "
            "would execute, STOP and report — planning the vocabulary "
            "semantics is coordinator/operator work, not lane work. "
            "GEMINI+DRIVE collapse to "
            "AISTUDIO_DRIVE is non-injective — use the Source-family "
            "disambiguator (#2737, 4rrv). STRICT no-drive-by rule: touch "
            "only the files the phase names."
        ),
        "own": ["polylogue/core/enums.py", "polylogue/core/sources.py", "projection shims named by the plan"],
        "avoid": ["everything else — repo-wide renames are FORBIDDEN in this lane"],
        "verify": "mypy --strict via devtools verify --quick + testmon-affected set; census tool from #2737",
    },
    "polylogue-embeddings-hygiene": {
        "beads": ["1dk1", "egm8"],
        "goal": (
            "Embedding lifecycle honesty: 1dk1 reconcile orphan embedding "
            "rows across index rebuild generations (coordinate with the v35 "
            "transition — the authoritative generation may have just "
            "changed); egm8 make terminal embedding failures inspectable and "
            "resolvable."
        ),
        "own": [
            "polylogue/storage/sqlite/archive_tiers/embeddings.py",
            "polylogue/archive/semantic/",
            "tests/unit/storage/",
        ],
        "avoid": ["polylogue/storage/sqlite/archive_tiers/index.py"],
        "verify": "devtools test tests/unit/storage -k embedding",
    },
    "polylogue-annotation-judgment": {
        "beads": ["mrxt", "37t.12", "p5g", "rxdo.4"],
        "goal": (
            "Close the assertion flywheel loop: mrxt first assertion row via "
            "auto-mark/auto-promote; 37t.12 operator bulk judgment queue; "
            "p5g `polylogue judge` interactive terminal triage; rxdo.4 "
            "AssertionKind.FINDING reusing the candidate->judge lifecycle "
            "verbatim. Substrate (rxdo.7 schemas/batches, kmts joins) is "
            "merged — build on it, do not extend it."
        ),
        "own": [
            "polylogue/storage/sqlite/archive_tiers/user*",
            "polylogue/cli/judge*",
            "polylogue/mcp/ (assertion tools)",
            "tests/unit/user/",
        ],
        "avoid": ["polylogue/archive/query/"],
        "verify": "devtools test -k 'assertion or judge or annotation' + user_audit every-kind invariant",
    },
    "polylogue-live-performance": {
        "beads": ["s7ae.8", "20d.2", "20d.4", "oxz"],
        "goal": (
            "Interactive-latency cluster: s7ae.8 budget+reduce coordination "
            "status latency; 20d.2 defer heavy imports off CLI startup; "
            "20d.4 CLI structured-query routing parity with daemon; oxz "
            "instrumentation doctrine (slow-query log, phase timings). "
            "Measure before/after — perf claims need numbers."
        ),
        "own": ["polylogue/cli/ (startup+routing)", "polylogue/daemon/ (status paths)", "tests/benchmarks/"],
        "avoid": ["polylogue/storage/sqlite/archive_tiers/", "polylogue/archive/query/expression.py"],
        "verify": "devtools test -k 'startup or routing or status_latency' + benchmark deltas quoted in PR",
    },
    "polylogue-consolidation": {
        "beads": ["a7xr.9", "a7xr.15", "a7xr.16", "a7xr.11"],
        "goal": (
            "Mechanical consolidation sweep (one PR): a7xr.9 helper dedup "
            "(scalar coercion quadruplet, _table_exists x40, provenance "
            "vocab x6); a7xr.15 generic from_row for payloads.py's 74 "
            "identical copy lines; a7xr.16 table-drive the hand-aligned "
            "column triplicates in archive_tiers; a7xr.11 prune "
            "zero-consumer protocols. Behavior-preserving; mypy --strict is "
            "the net."
        ),
        "own": ["cross-cutting textual dedup — everything, which is why this lane launches LAST in the merge train"],
        "avoid": ["any semantic change; any file another active lane has open PRs against — rebase late"],
        "verify": "devtools verify --quick + testmon-affected; zero behavior diffs expected",
    },
    "polylogue-origin-interop": {
        "beads": ["2ilz"],
        "goal": (
            "Origin identity hygiene: 2ilz durable capture-mode field "
            "splitting GEMINI export vs live-Drive AISTUDIO_DRIVE; fold in "
            "4rrv completion only if its claim is stale (coordinator "
            "confirms). Classify the schema change per the durability "
            "regime before touching DDL; if it needs an index bump, declare "
            "the delta class for the fast-forward mechanism."
        ),
        "own": ["polylogue/core/sources.py (capture-mode)", "polylogue/schemas/", "tests/unit/core/"],
        "avoid": ["polylogue/core/enums.py (provider-origin lane owns it)"],
        "verify": "devtools test tests/unit/core -k 'source or origin'",
    },
    "polylogue-demo-corpus": {
        "beads": ["212.11"],
        "goal": (
            "Demo construct validity: 212.11 shared deterministic proof "
            "world for flagship demos; PLUS the measured-result experiment — "
            "sample N assistant completion claims from the live archive, "
            "compare against structural tool evidence, produce the honest "
            "headline number (X% unsupported / Y% contradicted-then-"
            "repaired) with full population/sample manifest, packaged so "
            "`demo receipts` and the README can cite it."
        ),
        "own": [".agent/demos/", "polylogue/cli/demo*", "tests/unit/demo/"],
        "avoid": ["polylogue/storage/", "polylogue/archive/query/"],
        "verify": "polylogue demo seed && polylogue demo verify && polylogue demo receipts (deterministic, CI-green)",
    },
    "polylogue-fastforward-mech": {
        "beads": [],  # bead filed in the beads pass; informed by the live v35 run
        "goal": (
            "Productize derived-tier fast-forward plans (bead filed by "
            "coordinator; read it + the fanout-prep doc section 'v35 "
            "rebuild'). Each index bump declares a delta class (constraint/"
            "view/index-only vs semantic-reparse); non-semantic deltas get a "
            "generated SQL fast-forward validated by equivalence sampling "
            "on a reflink clone. Codify what the live v32->v35 fast-forward "
            "just did manually; add the policy lint so bumps without a "
            "declared class fail devtools lab policy schema-versioning."
        ),
        "own": ["polylogue/storage/sqlite/lifecycle*", "devtools/ (policy)", "tests/unit/storage/"],
        "avoid": ["archive_tiers DDL semantics"],
        "verify": "devtools lab policy schema-versioning + focused lifecycle tests",
    },
    # ---- wave 2: launch only after the named lane's PR merges ----
    "polylogue-query-polish": {
        "wave2_after": "polylogue-fable-demo",
        "beads": ["9srm", "fnm.6"],
        "goal": (
            "DSL polish (WAVE 2 — launch after polylogue-fable-demo merges; "
            "shared footprint on the query executor): 9srm terminal `| "
            "count` stage consistency across unit types (confirm claim "
            "state; may be stale from a killed agent); fnm.6 wire the "
            "terminal stage to projections (| read / | context-image)."
        ),
        "own": ["polylogue/archive/query/", "tests/unit/archive/"],
        "avoid": ["polylogue/insights/"],
        "verify": "devtools test tests/unit/archive -k 'pipeline or stage or count'",
    },
    "polylogue-webui-2": {
        "wave2_after": "polylogue-web-cockpit",
        "beads": ["t67b", "9l5.5"],
        "goal": (
            "Web workbench second slice (WAVE 2 — after polylogue-web-cockpit "
            "merges): t67b compile browser proof obligations from workflow "
            "claims; 9l5.5 ship opinionated saved views as product defaults."
        ),
        "own": ["polylogue/daemon/web_shell_*.py", "polylogue/product/workflows.py", "webui/tests/"],
        "avoid": ["polylogue/storage/"],
        "verify": "devtools test -k 'workflow or saved_view or proof'",
    },
}

PROMPT_TEMPLATE = """You are lane {lane} of a parallel fanout on the Polylogue repo.
Worktree = your root: /realm/worktrees/{lane}. NEVER cd to /realm/project/polylogue
(a hook will abort commits; the canonical checkout belongs to the coordinator).
Branch: you are already on a dedicated feature branch; confirm with
`git branch --show-current` before the first commit.

MISSION
{goal}

BEADS (read each with `bd show polylogue-<id>` INCLUDING notes before coding;
prework packets under .agent/scratch/corpus-gpt-pro-2026-07-07/ are
accelerators, not authority — anchors are from 2026-07-06, re-verify against
your checkout):
{bead_lines}

FILE OWNERSHIP
OWN (you may modify): {own}
AVOID (another lane owns these; do not touch): {avoid}
Generated surfaces (docs/cli-reference.md, docs/plans/topology-target.yaml,
openapi/cli-output schemas, EXPECTED_TOOL_NAMES): regenerate ONLY if your
change requires it, in a separate commit, and expect the coordinator to
re-run regeneration at merge time.

VERIFICATION (worker-owned — coordinator runs broad gates, you run these):
{verify}
Anti-vacuity requirement: for every test you add, state in the PR body which
production dependency it exercises and which implementation mutation or
removal would make it fail. Toy replicas, test-only validators, and mocks
that only prove their own wrapping will be rejected.

DISCIPLINE
- Commit every logical chunk immediately (worktree cleanup discards
  uncommitted work). Push after each commit.
- Do NOT run `bd close`/`bd update --claim`; report per-bead AC status
  (satisfied / deferred / misframed) in your final message AND the PR body.
- Do NOT merge your PR; open it with Summary/Problem/Solution/Verification
  sections and `Ref polylogue-<id>` lines, then stop.
- If a bead turns out to be already-done, stale-claimed, or blocked by
  missing substrate, say so explicitly and move to the next bead rather
  than improvising scope.
- Archive safety: never export POLYLOGUE_ARCHIVE_ROOT to the live archive
  (/home/sinity/.local/share/polylogue); tests use isolated roots.
"""


def load_beads() -> dict[str, dict]:
    out = {}
    for line in BEADS.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d["id"].removeprefix("polylogue-")] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="roster health only")
    args = ap.parse_args()

    beads = load_beads()
    errors, warnings = [], []
    OUT.mkdir(parents=True, exist_ok=True)

    for lane, spec in LANES.items():
        bead_lines = []
        for bid in spec["beads"]:
            b = beads.get(bid)
            if b is None:
                errors.append(f"{lane}: bead {bid} not found")
                continue
            status = b.get("status")
            title = (b.get("title") or "").strip()
            if status == "closed":
                errors.append(f"{lane}: bead {bid} is CLOSED — roster stale")
            elif status == "in_progress":
                warnings.append(
                    f"{lane}: bead {bid} is in_progress — confirm the claim is not stale (killed agent) before launch"
                )
            bead_lines.append(f"- polylogue-{bid} [{status}]: {title}")
        if not spec["beads"]:
            bead_lines.append(
                "- (bead ids assigned by the coordinator's beads pass — ask for them / check bd ready before starting)"
            )
        if args.check:
            continue
        prompt = PROMPT_TEMPLATE.format(
            lane=lane,
            goal=spec["goal"],
            bead_lines="\n".join(bead_lines),
            own="; ".join(spec["own"]),
            avoid="; ".join(spec["avoid"]),
            verify=spec["verify"],
        )
        if spec.get("wave2_after"):
            prompt = (
                f"## WAVE 2 LANE — do not launch before {spec['wave2_after']} "
                "has merged (shared footprint).\n\n" + prompt
            )
        (OUT / f"{lane}.prompt").write_text(prompt)

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if not args.check:
        print(f"\nwrote {len(LANES)} prompts to {OUT}")
        sol = [name for name, s in LANES.items() if s.get("model") == "gpt-5.6-sol"]
        if sol:
            print(f"SOL lanes (launch separately with FANOUT_MODEL=gpt-5.6-sol): {', '.join(sol)}")
        wave2 = [name for name, s in LANES.items() if s.get("wave2_after")]
        if wave2:
            print(f"WAVE-2 lanes (hold): {', '.join(wave2)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
