---
created: 2026-07-04
purpose: Provenance + audit trail for the 41-agent Beads backlog restructuring
status: complete (all drafts APPLIED to Beads; see ledgers)
project: polylogue
---

# Beads Swarm — 2026-07-04

A 41-subagent audit + single-writer apply pass that restructured the Polylogue
Beads backlog (333 -> 370 issues). All changes are live in Beads and committed
to git as `chore(beads): backlog restructuring` (06fe9a621).

## What's here

### findings/ (30 files) — raw audit outputs (READ-ONLY agents)
The first-order swarm. Each agent audited one slice and returned structured JSON
(underspecified / gaps / inconsistencies / dep_priority / reorg).
- `findings_D01..D16` — 14 epic-domain audits + D15 graph/orphans + D16 priority/hygiene
- `findings_E1..E6` — cross-set: E1 dedup, E2 closed-bead verify, E3 beads-vs-code
  reality, E4 doctrine coverage, E5 GitHub reconciliation, E6 AC-testability/sizing
- `findings_N1..N8` — N1 labels, N2 cross-repo, N3 rawlog-freshness, N4 security,
  N5 test-coverage, N6 docs, N7 critical-path leverage, N8 wave sequencing

### drafts/ (12 files) — apply-ready artifacts (second-order "S" agents)
These consumed the findings and produced exact bead bodies / command sets.
ALL HAVE BEEN APPLIED (via scripts/apply1-4). They are now the "why" record.
- `draft_S1_1xc.json` — 7 execution-grade children for the empty 1xc scale epic
- `draft_S2_decisions.json` — dispositions for the 4 resolved-but-open decision beads
- `draft_S3_ac_A.json` / `draft_S4_ac_B.json` — finalized acceptance/design for
  ~33 underspecified beads (context/query/perf ; storage/analytics/embeddings/audit)
- `draft_S5_deps.json` — 48 de-conflicted dependency edges (alias fixes, cycle-checked)
- `draft_S6_reorg.json` — full orphan-reparent + new-epic reconciliation (8 proposed)
- `draft_S7_s7ae.json` — s7ae.1 honest-narrowing + archive-composition + two-agent-proof
- `draft_S8_variants.json` — 4smp read-algebra sequencing + 4p1 decision content
- `draft_S9_launch.json` — cfk/3tl launch chain + 3tl.4 spec + 212/cpf fixes
- `draft_S10_usage.json` — provider usage/cost-honesty epic + 4 children
- `draft_S11_judgment.json` — the missing 37t judgment-queue bead
- `draft_S12_critic.json` — ADVERSARIAL CRITIC: contradictions/over-reach/checkpoints
  that gated the apply pass (the safety layer)

### ledgers/ — exactly what was applied
- `ledger_apply1.json` (field-fills/reprios/reparents/deps), `ledger_apply2.json` (creates)
- `apply{1,2,3,4}_live.log` — per-script execution logs (all rc=0)

### scripts/ — the applier code (python -> bd subprocess, single writer)
- `apply1_updates.py` field-fills + reprios + reparents-into-existing + dep edges
- `apply2_creates.py` new beads + adopts + new-bead dep wiring
- `apply3_analysis.py` analysis-driven epics/beads/retypes/closures
- `apply4_final.py` completion-state (s7ae closes) + remaining epics
- `aggregate.py` findings aggregator

### Root
- `SHARED_BRIEF.md` — the rules every audit agent operated under
- `beads_all.json` — pre-restructure snapshot (333 issues)
- `AGG.json` — aggregated findings; `partition_manifest.json` — the 16-way domain split
