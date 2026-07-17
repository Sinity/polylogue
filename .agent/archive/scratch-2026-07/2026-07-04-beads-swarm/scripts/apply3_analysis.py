#!/usr/bin/env python3
"""Apply pass — Script 3: analysis-driven upgrades. --live to execute."""

import re
import subprocess
import sys

LIVE = "--live" in sys.argv
ledger = []
ids = {}


def run(args):
    if not LIVE:
        ledger.append("DRY: " + " ".join(a if len(a) < 48 else a[:45] + "..." for a in args))
        return ""
    r = subprocess.run(["bd"] + args, capture_output=True, text=True)
    ledger.append("bd " + args[0] + " " + (args[1] if len(args) > 1 else "") + f" rc={r.returncode}")
    if r.returncode != 0:
        ledger.append("   ERR: " + (r.stdout + r.stderr)[:150])
    return r.stdout + r.stderr


def create(title, typ, prio, parent=None, design=None, acc=None, labels=None, extref=None):
    a = ["create", title, "--type", typ, "--priority", str(prio)]
    if parent:
        a += ["--parent", parent]
    if design:
        a += ["--design", design]
    if acc:
        a += ["--acceptance", acc]
    if labels:
        a += ["--labels", labels]
    if extref:
        a += ["--external-ref", extref]
    out = run(a)
    m = re.search(r"polylogue-[a-z0-9.]+", out)
    nid = m.group(0) if m else ("DRY-" + title[:6])
    ids[title] = nid
    ledger.append(f"  -> {nid} : {title[:60]}")
    return nid


def update(bid, *flags):
    run(["update", bid, *flags])


def close(bid, reason):
    run(["close", bid, "--reason", reason])


def comment(bid, text):
    run(["comment", bid, text])


# ========== NEW EPICS (strong product identity) ==========
sec = create(
    "Security & privacy: the archive can forget on purpose and never leaks secrets",
    "epic",
    2,
    design="No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned. Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing. Also owns the security-privacy-coverage.yaml manifest gaps. NON-GOAL: do not resurrect the paused sanitize/redaction cluster (chatlog != spec).",
    acc="Excision (right-to-forget + secret redaction + blob excision) is execution-grade and shares one mutation-audit/dry-run/--yes contract with reset (jnj.5); the security-privacy-coverage.yaml gaps each have an owning bead or test; the MCP write/admin destructive path shares the same audit-row contract. Verify: devtools verify + the reset/excision dry-run tests.",
    labels="area:security,spine",
)
update("polylogue-jnj.5", "--parent", sec)  # lift the reset security bug out of the P3 hygiene epic
for a in ["polylogue-27m", "polylogue-jsy"]:
    update(a, "--parent", sec)

res = create(
    "Operational resilience: recoverable, restorable, survives daemon death and deploy",
    "epic",
    2,
    design="Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home. This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).",
    acc="A quarterly restore drill proves backups restore (4be); daemon crash mid-convergence recovers without stranding debt (peo, ties 1xc.3/1xc.4); deployed state is provable via deployment-smoke when prod is re-activated (s8q). Verify: devtools workspace deployment-smoke --json + a restore-drill artifact.",
    labels="area:ops,spine",
)
for a in ["polylogue-4be", "polylogue-peo", "polylogue-s8q"]:
    update(a, "--parent", res)

# ========== cpf -> epic + 3 enforcement-hook beads ==========
update("polylogue-cpf", "--type", "epic")
create(
    "Doctrine lint: reject TEXT timestamps in new durable DDL",
    "task",
    3,
    parent="polylogue-cpf",
    design="Time doctrine: UTC epoch-ms canon. A schema-audit check should reject new durable-tier columns storing timestamps as TEXT (should be INTEGER epoch-ms). Extend devtools lab schema audit or add a policy lint.",
    acc="A test DDL adding a TEXT timestamp column fails the lint; existing INTEGER epoch-ms columns pass. Verify: devtools lab policy (new check) on a fixture.",
    labels="area:substrate",
)
create(
    "Doctrine: writer-class docstring convention + layering check",
    "task",
    3,
    parent="polylogue-cpf",
    design="Writer-class doctrine: one writer-class per file, cross-tier interruption validity. Add a docstring convention + a layering check that flags files mixing writer classes.",
    acc="A file declaring two writer classes fails the check; single-class files pass. Verify: devtools verify layering (extended).",
    labels="area:substrate",
)
create(
    "Doctrine: injected-context trust deny-lexicon tripwire fixture",
    "task",
    3,
    parent="polylogue-cpf",
    design="Injected-context trust classes (OPERATOR/SYSTEM/QUOTED). 37t.11 carries the ContextSource typing; this bead lands the deny-lexicon tripwire test fixture so QUOTED content can never emit OPERATOR-class directives.",
    acc="A fixture where QUOTED content contains an OPERATOR-style directive is caught by the tripwire test. Verify: the trust-class pytest fixture.",
    labels="area:context",
)
comment(
    "polylogue-cpf",
    "Promoted to epic (E4 audit): bundled 6 doctrine texts + 6 hooks was too large for one task. Hook beads for the 3 cheap lints now filed as children. Doctrine texts still land under docs/doctrine/. Note: cpf's finding-provenance hook mis-pointed at 3tl.4 (a docs-publishing lane) — provenance-stanza gate belongs in the findings lane once it exists, not as 3tl.4's identity.",
)

# ========== NEW TRACKED BEADS (analysis-surfaced untracked work) ==========
create(
    "Standing backlog-hygiene invariant lint (bd devloop gate)",
    "task",
    2,
    parent=res,
    design="This session needed a 41-agent sweep because structure trails filing. Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead (false-block). The backlog equivalent of automagic-invariants.",
    acc="The lint runs in the devloop and fails on a seeded violation of each of the 5 classes; a clean backlog passes; wired into devtools verify or a bd hook. Verify: seed one violation per class, assert non-zero exit.",
    labels="area:ops",
)
create(
    "Durable-tier additive migration framework: backup-gate + numbered runner contract",
    "task",
    1,
    design="schema-evolution-v2 SHIPPED on this branch (migrations/source/002_raw_capture_multimap.sql, migrations/user/004_user_settings.sql, migration_runner.py with backup-manifest gate at ~:73-106) with NO owning bead. Own the contract: durable tiers (source/user) advance PRAGMA user_version one step at a time behind a verified backup manifest; derived tiers still rebuild. Reconcile the docs that still say 'no in-place upgrade chains'.",
    acc="The migration runner's backup-gate + one-step-advance is covered by a test; docs/architecture-spine.md and internals.md schema-versioning sections match the shipped two-regime model; devtools lab policy schema-versioning still passes. Verify: pytest on the runner + render all --check.",
    labels="area:storage",
)
create(
    "Adopt manifest-declared coverage gaps as tracked beads (retire gh#590 umbrella)",
    "task",
    2,
    parent="polylogue-9e5",
    design="scenario-coverage.yaml + test-quality-coverage.yaml declare 9 coverage_gaps all owned by external gh#590 with 0 beads — the anonymous-debt anti-pattern the doctrine forbids. Split into tracked beads (storage-correctness, performance, security-privacy, distribution, schema-rebuild-safety, flakiness, mock-depth, fuzz-ci, per-module-coverage) and rewrite the manifest owner strings to the bead ids.",
    acc="Each of the 9 gaps maps to a bead id in the manifest; no manifest gap cites only a GH issue. Verify: devtools verify manifests + grep the yaml for issue:590.",
    labels="area:audit",
)
create(
    "Blob-GC lease/orphan concurrency test (the acquire->commit race)",
    "task",
    2,
    parent=res,
    design="internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test. #818 has real orphan-detection bugs. Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.",
    acc="A test acquires a lease, starts GC, and asserts the leased blob survives; a released-lease orphan is reclaimed; sweep_orphaned_blob_leases clears a SIGKILLed writer's lease past ORPHAN_LEASE_MAX_AGE_S. Verify: the new pytest under tests/unit/storage.",
    labels="area:storage",
)
create(
    "Wire atheris fuzz targets into CI",
    "task",
    3,
    parent="polylogue-9e5",
    design="tests/fuzz/ ships 4 real atheris targets with no CI execution gate and no bead. Add a bounded fuzz job (short duration) to a workflow so regressions in parser crashlessness are caught.",
    acc="A CI job runs each fuzz target for a bounded budget; a seeded crash is caught. Verify: grep atheris .github/workflows + a green run.",
    labels="area:ci",
)
create(
    "Reconcile schema-versioning docs + retire superseded execution-plan.md",
    "task",
    2,
    parent="polylogue-3tl",
    design="architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent). docs/execution-plan.md is fully superseded (dropped #1807 umbrella; every issue re-encoded as a bead) yet README.md:14 still calls it 'current sequencing plan'. Fix the spine section, reconcile internals.md, retire execution-plan.md with a pointer to Beads, and repoint README:14.",
    acc="architecture-spine + internals schema-versioning sections describe the two-regime model consistently; execution-plan.md is archived/removed and no doc calls it current; README points at Beads. Verify: render docs-surface --check + grep 'execution-plan' docs README.",
    labels="area:legibility",
)

# ========== HYGIENE RETYPES ==========
update("polylogue-f94", "--type", "task")  # decided KILL, execution remains
update("polylogue-gjg", "--type", "epic")  # epic-sized compaction-lifecycle
update("polylogue-20d.4", "--type", "bug")  # FTS-gate parity defect, not perf
update("polylogue-7aw", "--type", "epic")  # 5 multi-PR outcomes

# ========== CLOSE RESOLVED DECISIONS (residuals already created) ==========
close(
    "polylogue-jgp",
    "Adopted doctrine (ambient/default-on criterion). No residual task; recorded as decision. Re-open only if the criterion is contested.",
)
close(
    "polylogue-6mv",
    "Adopted: Polylogue<->Sinex evidence boundary. Residual Sinex-side emitter tracked as fs1.9. Sinex receiving half is sinex-4j2/zi6.",
)
close(
    "polylogue-lnd",
    "Working doctrine: beads vs assertions boundary. Revisit tracked as 37t.13 (gated on beads-history ingestion 7fj).",
)

# ========== FLAGS / CROSS-REPO REFS ==========
comment(
    "polylogue-fs1.7",
    "EXTERNAL DEPENDENCY (N2): this is an upstream PR to the open-source Hermes repo, tracked in no local system. It can silently block the fs1 importer children — treat as external-blocked, do not count toward fs1 terminal state until merged upstream.",
)
comment(
    "polylogue-k8k",
    "CROSS-REPO (N2): root cause + fix live in /realm/project/sinnix/scripts/sinnix-direnvrc; only the before/after timing is Polylogue-local. Sinnix has no bead tracker — remediation is a Sinnix-repo change, this bead owns the measurement half only.",
)
comment(
    "polylogue-lio",
    "CROSS-REPO (N2): the devloop-checkpoint --queue tooling is Sinex-owned; Sinex tracks the receiving half as sinex-hlv. This bead's contract update must land in both repos.",
)
comment(
    "polylogue-37t.4",
    "CROSS-REPO (N2): the SessionStart rollout's sinnix half edits the Sinnix repo (global CLAUDE.md section -> injected preamble). 3gd owns the CLAUDE.md-section migration; 37t.4 depends on it. Sinnix edit is untracked in Beads.",
)

print("\n".join(ledger))
print(f"\n=== SUMMARY ({'LIVE' if LIVE else 'DRY'}) created={len(list(ids.values()))} ===")
