# Fanout prep pack — 2026-07-12

Prepared by the Fable session so the post-Codex beads pass + launch is rapid.
Companion: `.agent/tools/fanout-launch.sh` (kitty os-window launcher, workspace 6,
persistent windows, exit markers, attested job-ids).

## Launch procedure (when ready)

1. Wait for the current Codex session to finish: kmts merge/close + staged
   Claude.ai/AI-Studio bundle imports. v35 rebuild stays DEFERRED (below).
2. Run the beads-adjustment pass (JSONL drafts below), regenerate prompts if
   scopes shifted.
3. Create worktrees: for each lane `git -C /realm/project/polylogue worktree add
   -b feature/fanout/<lane> /realm/worktrees/polylogue-<lane> origin/master
   (lane name passed to the launcher is the full dir name, `polylogue-<lane>`)`.
4. Pre-seed the shared pytest cache once (one `devtools test` run touching
   corpus_seeded_db) BEFORE launching, or give lanes disjoint tmp overrides.
5. Write prompts to `.agent/scratch/fanout-prompts/<lane>.prompt` where lane
   name = worktree dir name (e.g. `polylogue-hermes-wedge`). Launch:
   `.agent/tools/fanout-launch.sh polylogue-hermes-wedge polylogue-web-cockpit ...`
   (thin wrapper; canonical implementation upstreamed to sinnix commit 1a26aff
   in launch_agent_tabs.sh: --persist-windows/--per-task-workdir-base/
   --job-prefix/--status). Terra/high default; Sol lane:
   `FANOUT_MODEL=gpt-5.6-sol .agent/tools/fanout-launch.sh polylogue-provider-origin-p2`
6. Monitor: `.agent/tools/fanout-launch.sh --status`; interrupt via
   agent_job_control.sh by attested id `fanout-<lane>`. Windows persist after
   completion (interactive shell in the lane worktree).
7. Merge train (coordinator): merge one lane; others rebase + re-run
   `devtools render all`; bd bookkeeping in coordinator batches, never in lanes.

## Prompt skeleton (every lane)

```
You are one lane of a parallel fanout on the Polylogue repo. Your worktree is
your root; NEVER cd to /realm/project/polylogue. Branch: feature/fanout/<lane>.

SCOPE: <bead ids + one-line goals>. Read each bead fully (bd show <id>) INCLUDING
notes; prework packets under .agent/scratch/corpus-gpt-pro-2026-07-07/ are
accelerators, not authority — re-verify anchors against your checkout.

OWN (may modify): <file/dir list>
AVOID (do not touch; another lane owns them): <list + generated surfaces:
docs/cli-reference.md, docs/plans/topology-target.yaml, openapi/cli-output
schemas — regenerate only if your change requires it, commit regen separately>

VERIFICATION (worker-owned): focused devtools test for changed behavior + exact
-path static checks + affected-area check if crossing modules. Anti-vacuity: for
each test, state the production dependency it exercises and the implementation
mutation that would make it fail. Broad gates are the coordinator's.

COMMIT every logical chunk (worktree cleanup discards uncommitted work). Do NOT
close/claim beads; report per-bead AC status in your final message + PR body.
Open a PR when your cluster is coherent; do not merge it.
```

## Lane definitions — SOURCE OF TRUTH: .agent/tools/fanout_gen_prompts.py

The roster, footprints, and goals live in the generator; regenerate prompts
after any beads change: `python3 .agent/tools/fanout_gen_prompts.py`
(--check for roster health only; it errors on closed beads and warns on
possibly-stale in_progress claims).

Final 16 roster (14 wave-1 + 2 wave-2):
  wave 1: fable-demo, hermes-wedge, extension-redesign, capture-hardening,
          web-cockpit, distribution, provider-origin (SOL), embeddings-hygiene,
          annotation-judgment, live-performance, consolidation (merge LAST),
          origin-interop, demo-corpus, fastforward-mech
  wave 2: query-polish (after fable-demo merges), webui-2 (after web-cockpit)

Launch: `.agent/tools/fanout-launch.sh <lane...>`; Sol lane separately:
`FANOUT_MODEL=gpt-5.6-sol .agent/tools/fanout-launch.sh polylogue-provider-origin`

Worktree creation one-liner (run at launch time, from fresh origin/master):
```
for l in $(ls .agent/scratch/fanout-prompts | sed 's/.prompt//'); do
  git -C /realm/project/polylogue worktree add -b feature/fanout/${l#polylogue-} \
    /realm/worktrees/$l origin/master
done
```

Scaling notes: beyond ~12 concurrent the coordinator merge train (render-regen
per merge + rebase queue) is the bottleneck — batch merges 2-3x/day.
consolidation lane rebases/merges strictly last. Quota: all-terra 14-way is
roughly 5-7 sols-equivalent.

## v35 rebuild — rapid-rebuild findings + plan

Facts (verified 2026-07-12): live index = v32; code INDEX_SCHEMA_VERSION = 34
(polylogue/storage/sqlite/archive_tiers/index.py:36). Deltas:
- v32->v33: CHECK-constraint widening (new writes would violate old CHECK; all
  existing rows conform to the wider constraint).
- v33->v34: delegations-view rebuild (#2739) + message FK backreference index
  (#2738; that PR also diagnosed rebuild tail latency).
Both deltas are REPRESENTATIONAL (constraints/views/indexes), not
semantic-reparse. A full rebuild re-parses ~21k raw artifacts / 4.5M messages
through Python parsers (hours); the schema delta itself is minutes of SQL.

Plan adopted:
1. DEFER the rebuild until the fanout wave lands its index-tier bumps +
   5q2u (lineage-ordered replay) — batch to a single rebuild (schema-bumps
   doctrine).
2. File bead: **derived-tier fast-forward plans** — each index bump declares a
   delta class (constraint-only / view-only / index-only / semantic-reparse).
   Non-semantic deltas get a generated SQL fast-forward (new-table copy for
   CHECK changes, CREATE VIEW/INDEX for the rest) validated by equivalence
   sampling: rebuild N sessions on a reflink clone, compare content hashes vs
   fast-forwarded tables. Semantic deltas -> full rebuild or targeted
   `reprocess` of affected sessions only. This is the missing execution arm of
   the existing "classify before editing schema" doctrine; it does NOT create
   a migration chain (each plan is version-pair-specific and disposable).
   CAVEAT the bead must state: content drift from parser changes since the
   live index was built (e.g. #2730 terminal_state, #2737 origin dedupe) is
   NOT covered by schema fast-forward; those need convergence/reprocess, and
   the equivalence sample will surface them honestly.
3. When rebuild does run: offline generation + promotion (#2685 machinery) on
   a btrfs reflink clone under the nix-build slice, overnight.

## Beads-adjustment JSONL drafts (run AFTER Codex finishes)

Extension design-pack (reconcile under 90y/3v1; source: /realm/inbox/download/
"Claude design pack-handoff.zip" -> copy into docs/design/browser-capture-redesign/):

```jsonl
{"title":"Extension redesign epic: ambient two-way surface","type":"epic","priority":1,"description":"IA change per Claude Design handoff pack (docs/design/browser-capture-redesign/). Supersedes yajm/x5k3 incremental framing. Two-layer rule (from 1nb2, verbatim on 90y): per-message state blends in; cross-conversation intelligence floats."}
{"title":"Popup mission-control (multi-tab list + active-conversation card)","type":"feature","priority":1}
{"title":"'What Polylogue did here' per-conversation timeline; doing-nothing is a logged event","type":"feature","priority":1}
{"title":"Operator status vocabulary (Safe/Catching up/Needs attention/Failed/Not saved + Partial fidelity)","type":"feature","priority":2}
{"title":"In-page Layer 1: blended per-message capture dot + save action","type":"feature","priority":1,"parent":"polylogue-90y"}
{"title":"In-page Layer 2: corner chip + slide-over (cost, recall, assertions)","type":"feature","priority":1,"parent":"polylogue-90y"}
{"title":"Selection -> assertion write flow (evidence ref to exact message)","type":"feature","priority":1,"parent":"polylogue-90y"}
{"title":"Multi-tab aggregate + calm offline spool model","type":"feature","priority":2}
{"title":"Agent-control (reverse channel) popup home","type":"feature","priority":2,"related":"polylogue-ptx"}
{"title":"[bug] Auto-capture trigger never fires: 160 archive-state GETs, zero capture POSTs","type":"bug","priority":1,"description":"Diagnosed live in design pass: conversation detected missing twice, no POST until manual capture. Trust bug + data loss. Fix trigger AND make 'saw it, did nothing' a logged visible event (timeline bead)."}
```

Other mutations:
- y8s5: priority 3 -> 1 (distribution is the cheapest legibility multiplier).
- xiyv: priority 2 -> 1 (blocks P1 212.9.1 — priority inversion).
- fnm.1: add-note confirming the narrowed campaign slice is the first
  deliverable (note exists 2026-07-10; reaffirm scope).
- New bead: derived-tier fast-forward plans (text above).
- New bead: demo receipts "measured result" experiment — one falsifiable
  headline number (N sampled completion claims, X% lacking structural
  evidence) packaged into demo/README.
- Stamp v35-batching note on open index-tier beads (5q2u, ma2-class).
- Label hygiene: 107 open beads carry no delivery:* label; label
  D-agent-coordination is outside the gate registry (rename to
  D-agent-context-coordination).
- fs1.12 note: record critical path fs1.3 -> fs1.11 -> fs1.12; it is the Nous
  artifact; enablers are lane hermes-wedge.
- STALE-CLAIM AUDIT (new): wave-3 agents died at session quota holding
  claims; for every in_progress bead without an open PR or live worktree,
  `bd update <id> --status open` (candidates seen 2026-07-12: 013x, 4rrv,
  9srm, 8jg9.1, 6rvt, 9e5.8 — verify each against open PRs first).
- Extension design-pack import runs FIRST in the beads pass (extension-redesign
  lane prompts reference the new ids); copy the handoff zip into
  docs/design/browser-capture-redesign/ in the same change.
- Blanket note where packets are stale: packets generated from master@8a975a40
  (2026-07-06); ~1300 commits landed since — anchors need re-verification.

## Holistic backlog observations (beyond discussed clusters)

- **Priority inversions**: P1 leaf work gated on P2 enablers in at least two
  places (212.9.1 <- xiyv/fnm.1; bby.17 P2 while H-gate is the flagship UI).
  Sweep: any P1 bead's open dependency should be >= its priority.
- **Gate I (analytics) and J (embeddings) hold 31 ready beads that should not
  attract lanes** until L ships — consistent with stop-doing list. The tech
  tree's gravity pulls toward I/J because those beads are best-specified;
  resist by labeling the fanout lanes explicitly.
- **M-substrate-consolidation (30 ready)** is almost all mechanical-sweep
  batch-affinity work — ideal single "consolidation-sweep" lane periodically,
  not 30 separate claims.
- **Duplicate-ish surface**: several capture/extension beads predate the
  design pack (3v1 family vs pack beads) — the reconcile pass must merge or
  parent them, not add 9 parallel siblings.
- **Horizon labels**: vision beads (P3/P4 horizon:vision) are well-behaved;
  no cleanup needed there.
- **The backlog's real gap**: nothing tracks "first external user" as a
  deliverable — distribution (y8s5), a measured-result demo, and a
  SHOW-someone milestone exist as scattered items but no epic owns "outside
  adoption v1". Consider a small epic tying them.
