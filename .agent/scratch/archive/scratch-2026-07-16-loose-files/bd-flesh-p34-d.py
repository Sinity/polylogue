#!/usr/bin/env python3
"""Flesh out thin P3/P4 beads, batch D: 9e5 audit cluster + test infra. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:110] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


def label(bid, *labs):
    for lab in labs:
        subprocess.run(["bd", "label", "add", bid, lab], capture_output=True, text=True)


bd(
    "update",
    "polylogue-9e5.8",
    "--design",
    "The sequenced retirement plan for provider vocabulary (tracked context: 9e5 epic; jnj.7 owns "
    "the CLI-help slice; 2qx owns OriginSpec). Constraint that makes naive rename unsafe: "
    "provider->origin is NON-INJECTIVE (GEMINI and DRIVE both collapse to AISTUDIO_DRIVE). Method: "
    "inventory every provider-token surface (rg Provider/provider_from_origin/"
    "project_origin_payload call sites) and classify: wire-boundary-legitimate (raw export "
    "parsing, schemas/providers/), transitional-shim (project_origin_payload consumers), leak "
    "(public filters/payloads/help). Output: ordered PR sequence where each step keeps "
    "byte-compat via the shim until its consumers flip, ending with Provider enum gated to "
    "sources/ + schemas/ only. Anchors: core/enums.py, core/sources.py, insights/registry.py "
    "project_origin_payload.",
    "--acceptance",
    "Committed sequence doc lists every provider-token surface with class + flip step; no step "
    "breaks public output byte-compat (goldens); final state = Provider importable only from "
    "wire modules (layering lint enforces). Verify: rg census + goldens per step.",
)
label("polylogue-9e5.8", "horizon:mid")

bd(
    "update",
    "polylogue-9e5.10",
    "--design",
    "Observational (no A/B): join resume-shaped evidence the archive already has — "
    "get_resume_brief/compose_context_preamble usage (hook events, MCP call logs in ops.db), "
    "session_links resume chains, and outcome proxies (time-to-first-edit, early-tool-error "
    "rate, repeated-orientation queries) — comparing resumed-with-context vs resumed-bare "
    "sessions on the same repo. Confounders stated, not modeled away (self-selection: harder "
    "tasks may attract context use). Relation: cfk owns the CONTROLLED two-arm test; this is "
    "the cheap corpus-wide observational complement.",
    "--acceptance",
    "A committed analysis artifact over the live archive: n per arm, the 3-4 outcome proxies "
    "with uncertainty, confounders section, and a verdict on whether the controlled cfk result "
    "generalizes. Verify: re-runnable script + insight_rigor_audit passes on any derived "
    "numbers.",
)
label("polylogue-9e5.10", "horizon:mid")

bd(
    "update",
    "polylogue-9e5.11",
    "--design",
    "Map where tests earn their runtime: per-module (a) coverage percent, (b) historical "
    "fix-density (git log --grep fix -- <module> commit counts), (c) test wall-time share "
    "(.cache/verify pytest artifacts have per-test durations), (d) testmon selection frequency. "
    "Quadrants: high-fix low-coverage = write tests; low-fix high-cost = candidates for "
    "slow-marking or property consolidation. Output: one committed table + the top-5 actions. "
    "This is measurement for the TESTING.md doctrine, not a coverage-chasing exercise — the "
    "90% floor stays.",
    "--acceptance",
    "Committed matrix for every polylogue/ package; five concrete actions each with expected "
    "effect (minutes saved or risk covered); actions filed as beads or done inline. Verify: "
    "re-runnable script, numbers reproducible.",
)
label("polylogue-9e5.11", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.12",
    "--design",
    "Question: does the schemas/ package (Pydantic provider-record validation driving "
    "detect_provider tightness) earn its maintenance cost? Measure: (a) which detectors "
    "actually gate on validated records vs dict-key checks (sources/dispatch.py census), "
    "(b) parse-failure telemetry — how often validation rejects real-world records that the "
    "loose path would have accepted (ops.db attempts/errors), (c) maintenance cost proxy = "
    "commits touching schemas/providers/ per provider format change. Verdict per provider: "
    "load-bearing (keep), ceremonial (simplify to dict-key), or missing (loose check should "
    "be tightened).",
    "--acceptance",
    "Per-provider verdict table committed with the three measurements; at least one "
    "simplify/tighten action executed or beaded; detector-order tests still green. Verify: "
    "devtools test -k detect + the census script.",
)
label("polylogue-9e5.12", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.13",
    "--design",
    "One diff pass, one PR: for each doc in the Reference-docs table (CLAUDE.md lists them), "
    "extract checkable claims (file paths, command names, schema versions, table names, tool "
    "counts) and verify against live source mechanically where possible (paths exist, commands "
    "in --help, versions match constants). Known confirmed drift to include: internals.md "
    "describes external-content FTS while the build is contentless (3tl.14 owns that fix — "
    "coordinate, do not duplicate; this bead sweeps the REST). Output: one docs-correction PR "
    "+ a claims-extraction script rerunnable as a lane.",
    "--acceptance",
    "Every reference doc swept; each stale claim fixed in the PR or beaded with reason; the "
    "extraction script committed so the sweep is repeatable. Verify: render all --check + "
    "script re-run reports zero unhandled drift.",
)
label("polylogue-9e5.13", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.18",
    "--design",
    "tests/fuzz exists but runs nowhere. Wire: a scheduled CI job (nightly/weekly, not per-PR — "
    "per-PR CI already skips heavy suites) running each atheris target for a bounded corpus "
    "time, uploading crashes as artifacts + opening/annotating an issue on new findings. Local "
    "entry: devtools test --fuzz or a lab command. Targets to confirm still import-clean after "
    "the split-file refactor. Seed corpora from real (sanitized/synthetic) provider fixtures — "
    "fuzzing parsers with structureless bytes wastes cycles; mutate from valid records.",
    "--acceptance",
    "Scheduled workflow green on a first run; a seeded crash (assert False target) produces an "
    "artifact + notification path; README-of-fuzz documents adding a target. Verify: workflow "
    "run link + local bounded run.",
)
label("polylogue-9e5.18", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.20",
    "--design",
    "Flakiness is currently folklore (the 3.11 test_concurrent_reads_during_writes memory). "
    "Make it data: parse .cache/verify run artifacts + CI logs into a per-test outcome history; "
    "flaky = pass-and-fail on identical commit. Quarantine lane: a marker that keeps the test "
    "running-but-nonblocking with an owning bead required (no silent skip — quarantine without "
    "an owner is deletion in slow motion). Auto-expire: quarantined test green N consecutive "
    "runs -> proposed for unquarantine.",
    "--acceptance",
    "Flakiness ledger generated from existing artifacts; the known 3.11 flake appears in it; "
    "quarantine marker exists with lint requiring owner-bead ref; CI treats quarantined "
    "failures as warnings. Verify: seed a random-fail test, watch it get ledgered.",
)
label("polylogue-9e5.20", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.21",
    "--design",
    "Measure where tests mock so deep they test the mocks: AST scan of tests/ counting patch "
    "targets per test, patch depth class (own-module boundary vs foreign-internal vs stdlib), "
    "and assert-on-mock ratio (asserts against Mock attrs vs real outputs). The workspace_env/"
    "SessionBuilder infra means most tests CAN run real — high foreign-internal patch counts "
    "flag conversion candidates. Output: ranked worst-20 list + convert 3 as proof.",
    "--acceptance",
    "Committed mock-depth report over tests/unit; three worst offenders converted to "
    "infra-backed tests with equal-or-better assertions; scan script re-runnable. Verify: "
    "devtools test on the three converted files.",
)
label("polylogue-9e5.21", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.22",
    "--design",
    "The 90% aggregate floor hides per-module holes (a 60% storage module offset by 99% "
    "rendering). Add per-package floors: coverage config gains per-package minimums set at "
    "current-actual minus small slack (ratchet, no aspirational jumps); the verify pipeline "
    "reports the three worst modules each run. Anchor: pyproject.toml coverage config + "
    ".cache/verify summary emitters. Feeds 9e5.11's economics map (same data, different "
    "consumer).",
    "--acceptance",
    "Per-package floors active in CI; lowering a module below its floor fails; worst-3 report "
    "visible in verify output; floors documented as ratchet policy. Verify: deliberately "
    "un-cover one module locally, watch it fail.",
)
label("polylogue-9e5.22", "horizon:frontier")

bd(
    "update",
    "polylogue-20d.16",
    "--design",
    "Scenario family for perf/throughput regression: seed archives at three scales (demo-size, "
    "10%-of-live sample shape, live-shape synthetic) via scenarios/ + corpus_seeded_db infra; "
    "measured flows = ingest batch, rebuild-index, hot find query set, read --all of largest "
    "session, convergence catch-up. Emit per-flow wall/RSS to a committed baseline file; "
    "regression = >X% over baseline on same machine class. Ties: 20d.8 (claim-vs-evidence 43s "
    "regen) and 20d.11 (mmap tuning) become measured flows instead of anecdotes.",
    "--acceptance",
    "polylogue lab perf (or devtools equivalent) runs the family and diffs against baseline; "
    "one seeded regression (sleep injection) is caught; baselines refreshed with rationale in "
    "the same PR that changes them. Verify: two consecutive runs stable within noise band.",
)
label("polylogue-20d.16", "horizon:mid")

bd(
    "update",
    "polylogue-t46.1",
    "--design",
    "Inventory what 'showcase QA' still is (rg showcase + the demo/QA command surfaces; the old "
    "qa CLI became demo per the stale-notes memory): keep = polylogue demo seed/verify "
    "(synthetic, private-data-free), real CLI subprocess checks (tests/ integration-style), "
    "visual tests (visual-tapes recordings per 3tl.5). Remove = any bespoke QA harness "
    "layer that duplicates what devtools test + demo verify already prove. Migration: each "
    "removed check either maps to an existing test/demo (name it) or is dead (delete with the "
    "removal PR listing the mapping).",
    "--acceptance",
    "Zero bespoke QA-harness code remains; the removal PR contains the check->replacement "
    "mapping table; demo verify + devtools test cover every behavior the old layer claimed. "
    "Verify: devtools test -k demo + rg showcase returns only historical docs.",
)
label("polylogue-t46.1", "horizon:frontier")

print("--- batch D done")
