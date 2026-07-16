#!/usr/bin/env python3
"""Design fields for the 14 design-less 400-950-band beads. All design fields verified empty. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:110] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


bd(
    "update",
    "polylogue-20d.8",
    "--design",
    "Hinge: the pairing cost is Python-side row inspection; 1vpm action-unit outcome fields move it "
    "into SQL. Sequence: (1) after the action-unit fields land, re-measure with the staged timings "
    "the devloop memory prescribes (per-origin counts, unpaired counts, per-origin sampling — no "
    "whole-regen reruns while diagnosing); (2) if still >10s, the residual is the failure-predicate "
    "legs — apply the indexed disjoint-leg pattern that fixed the earlier OR/COALESCE scan. Budget: "
    "full live regen <10s or the demo documents why not.",
)

bd(
    "update",
    "polylogue-f94",
    "--design",
    "Execution list (decision already made — KILL): delete polylogue/ui/tui/ (rg first for the "
    "actual module path), its command registration in cli/ (command_inventory + click registration), "
    "its tests, and the textual dependency from pyproject if nothing else imports it. Then: devtools "
    "render topology-projection + topology-status (module removal changes the projection), render "
    "all --check, and the command-inventory tests. One PR, surgical-renewal shaped.",
)

bd(
    "update",
    "polylogue-9l5.16",
    "--design",
    "Component subscores (each an existing/planned measure, coverage-gated): outcome_sub (structural "
    "success per 9l5.1), efficiency_sub (from the 9l5.14 scorecard vector), error_sub (tool-error + "
    "unacknowledged-failure rates), fragmentation_sub (phase-churn heuristic — LOWEST weight, "
    "labeled heuristic-tier), correction_sub (operator-correction density inverted). Composite = "
    "weighted mean over AVAILABLE components with per-component values + weights + coverage in the "
    "payload; NULL propagates (a session missing cost evidence has no efficiency_sub, and the "
    "composite says so). Registered as MeasureSpec with the Goodhart caveat in the spec text. "
    "Consumer: fs1.5 eval export (reward shaping lane) — never a default-on dashboard number.",
)

bd(
    "update",
    "polylogue-37t.9",
    "--design",
    "Rail steps: (1) WRITER: an MCP/CLI surface that records a PROMPT_EVAL assertion "
    "(AssertionKind.PROMPT_EVAL exists at enums.py:426 with no writer) — payload: task ref, "
    "context-spec variant ids, outcome observations, evidence refs; lands CANDIDATE via the 37t.15 "
    "chokepoint like every agent write. (2) VARIATION: compile_context already accepts specs — a "
    "harness recipe runs the same task N times with varied specs via subagents (runnable today, no "
    "new machinery). (3) DREAMING: a bounded background pass (daemon idle lane or explicit command) "
    "prompts over a session cohort and writes candidate observations — same writer, same gate. "
    "Sequencing: writer first (small), variation harness second, dreaming last (needs the "
    "convergence idle-lane slot).",
)

bd(
    "update",
    "polylogue-l4kf.3",
    "--design",
    "Anchors: a new cli/commands/cite.py (cite commit <sha> / cite pr) reading finding/query/session "
    "refs from the archive and writing (a) git notes under refs/notes/polylogue via subprocess git "
    "notes --ref=polylogue add (never touches working tree; push requires explicit --push with "
    "refspec), (b) a PR-footer text block to stdout for manual paste — NEVER auto-mutates GitHub. "
    "SARIF lane: accepted pathology findings render as SARIF runs (rule id = pathology kind, "
    "level from severity, evidence refs in relatedLocations) — a render profile, not a new "
    "subsystem. Candidates stay out of SARIF (judged-only).",
)

bd(
    "update",
    "polylogue-9yz",
    "--design",
    "Package the existing bounded-dialogue projection as a named read-view/render profile "
    "(list_read_view_profiles surface already exists — add profile id operator-dialogue): bounded "
    "window with explicit elision markers ([... N messages elided, M tokens]), operator-readable "
    "role labeling, token cap as a profile PARAMETER not a hardcoded max_tokens in the workspace "
    "command. Consumers: chatlog export switches to the profile; the mass-grab named-workflow ask "
    "becomes profile x cohort loop. Anchor: wherever the current first-window projection lives in "
    "the chatlog-export path (rg bounded/first-window in workspace/export code).",
)

bd(
    "update",
    "polylogue-mhx.7",
    "--design",
    "Canonical site: storage/sqlite/archive_tiers/embeddings.py (the tier owner per architecture; "
    "search_providers/sqlite_vec_runtime.py becomes a consumer importing the DDL constant). "
    "Metadata naming: +origin (the retirement direction), migrated per the DERIVED-tier regime — "
    "bump embeddings schema version + rebuild, no in-place migration. Drift-lock: a test that "
    "extracts CREATE VIRTUAL TABLE statements from both modules (or asserts the runtime imports "
    "the tier constant) and fails on any second definition appearing anywhere (rg-based).",
)

bd(
    "update",
    "polylogue-9l5.17",
    "--design",
    "Changepoint mechanics (classical, no ML): per (model_family, task-shape cohort) monthly series "
    "of median cost/turns/error-rate; candidate changepoint = rolling two-window median shift "
    "exceeding a MAD-scaled threshold, confirmed by permutation test (label-shuffle p<0.05); "
    "n_min per window enforced by the MeasureSpec coverage gate (REFUSE below floor, never "
    "extrapolate). Anchor: rides 9l5.8 temporal-analytics substrate; cohort anchors are explicit "
    "assertions (desc). Output: candidate changepoints as CANDIDATE findings (rxdo.4 lifecycle) — "
    "an operator judges 'model X got worse at Y'; the observatory never asserts causality.",
)

bd(
    "update",
    "polylogue-jnj.8",
    "--design",
    "Three surfaces to converge: root bare invocation (jnj.13 owns the triage screen), polylogue "
    "init/tutorial (currently prints five status lines on an already-configured install — make it "
    "state-aware: configured -> point at triage/cookbook; fresh -> guided init), and the reader "
    "launcher (web reader open command). Rule: every onboarding path ends by TEACHING the next "
    "command, not by dumping status. Fresh-machine flow (notes): bare polylogue on absent archive "
    "offers guided init. Anchors: cli/commands/init.py + the tutorial command module + jnj.13's "
    "triage entry.",
)

bd(
    "update",
    "polylogue-37t.18",
    "--design",
    "Storage (derived, index-tier — rebuild regime): entities(entity_id, kind, canonical_name), "
    "entity_mentions(entity_id, block_id, mention_kind: structural|candidate, extractor_version, "
    "confidence), entity_topics join, entity_backlinks VIEW over mentions. Extractors: STRUCTURAL "
    "= deterministic (bare #N with repo scope, explicit bead/session/file refs from 37t.2 "
    "notation, URLs, git SHAs) — trusted, no gate; CANDIDATE = prose-mined names/concepts — "
    "enters via the 37t.15 chokepoint as candidate assertions, promoted only by judgment (the "
    "recovery-digest fabrication incident is the standing regression fixture). Topic clustering "
    "rides mhx.5 (semantic analytics), not its own pipeline.",
)

bd(
    "update",
    "polylogue-37t.21",
    "--design",
    "Pipeline: (1) COHORT: select high-value sessions by structural outcome (verify-success, "
    "low-correction, high-reuse) via the DSL — the selection query is part of each template's "
    "provenance; (2) INDUCE: an external-model pass (find|compact pack -> model -> annotation "
    "import per rxdo.7) proposes parametrized meta-prompts (params: repo, task-type, risk-tier); "
    "(3) LAND: PROMPT_TEMPLATE candidates in git-YAML (code-review lane, NOT user.db — recipes "
    "are code); (4) EVALUATE: A/B via the 37t.9 variation harness; the evaluator REFUSES verdicts "
    "below the evidence floor (INSUFFICIENT_EVIDENCE is a valid, expected outcome). Each template "
    "cites its source sessions.",
)

bd(
    "update",
    "polylogue-8jg9.3",
    "--design",
    "Anchors: daemon_events + cursor-lag tables/samplers already exist (ops.db; "
    "daemon/cursor_lag_*.py modules) — this ADDS slo_samples (closed-set label enum, retention "
    "GC) + reducers (level/quantile/slope/ETA/burn-rate) as pure functions over those tables, "
    "and the idle-vs-stalled verdict: stalled = offered_work > 0 AND drain_rate == 0 over the "
    "window; idle = backlog with no offered work. Surface: readiness/status verdict field + one "
    "MCP status payload. Bulk-import suppression: ingest SLO pauses while a bulk-import marker "
    "event is open.",
)

bd(
    "update",
    "polylogue-3tl.3",
    "--design",
    "Reuse the claim-vs-evidence harness (campaign artifacts under .agent/demos/claim-vs-evidence) "
    "— the variant axis is MODEL: silent-proceed / unsupported-claim rates per model family over "
    "identical task classes, with cost + cache columns from f2qv-honest accounting. Open-model "
    "rows (DeepSeek, Hermes, local via LiteLLM) are the headline. Precondition: enough non-Claude "
    "sessions in the archive per task class (coverage gate REFUSES cells below n_min rather than "
    "publishing thin comparisons). Output: a Demo Finding Packet (212.7 shape) + leaderboard "
    "table render.",
)

bd(
    "update",
    "polylogue-rii.3",
    "--design",
    "Split by tier regime: parser_fingerprint + decode_failure_class are DURABLE source-tier "
    "columns -> numbered source v3 migration, batched in the 60i5 window; raw_fidelity records "
    "(byte-ratio bands, unparsed-key census, round-trip equality) are DERIVED -> index-tier "
    "rebuild regime. Round-trip check: parse -> re-serialize -> structural compare (field "
    "multiset, not byte) per origin; unparsed-key census = recursive key-walk of raw payload "
    "minus keys the parser consumed (instrument LoweredPayloadSpec consumption). Fingerprint "
    "change -> convergence enqueues reprocess for affected raw rows (existing debt machinery).",
)

print("--- band batch done")
