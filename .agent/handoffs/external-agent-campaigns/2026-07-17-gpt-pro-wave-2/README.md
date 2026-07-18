# GPT Pro wave 2 — 2026-07-17 (war-room push)

32 browser-hosted ChatGPT Pro jobs designed to run WHILE local coding lanes
(A: demo credibility, B: hermes forensics, C: MCP six-tool cutover, D: raw
authority closure) work the serial-hard immediate front. Wave jobs are
deliberately NON-immediate: implementation drafts, designs, runbooks, and
research we adjudicate and integrate after the lanes land. De-overlap is by
construction; where a wave job borders a lane (mcp-* vs Lane C), the job
consumes the lane's contract as given and produces surrounding material, and
its result is adjudicated against whatever the lane actually landed.

## Launch flow

1. Project-state archive: DONE — `polylogue-all.tar.gz` in this directory
   (generated 2026-07-17 evening from the operator's Chisel run). Attach it
   to every job (same snapshot for all; regenerate via
   `cd /realm/project/sinity-lynchpin && just chisel polylogue <out>` only
   if master moves by >~30 commits mid-wave).
2. Build the final prompts: `./make-prompts.sh` (concatenates each mission with
   the shared contract from `../contracts/`). Final prompts land in `prompts/`.
3. Paste one prompt per fresh ChatGPT Pro chat (GPT-5.6 Pro, extended
   reasoning), attach the Chisel archive, launch. Suggested order: clusters in
   manifest order; within a cluster, jobs are independent unless `depends_on`
   says otherwise.
4. Results: download each Result ZIP into `results/<job-id>/rNN/raw/` and let
   the intake lane (or Fable) extract/triage — same flow as wave 1.

## Clusters

- `webui` (8): WebUI v2 verticals per the ratified TS+Preact+Vite design.
- `mcp` (4): agent-surface material around the six-tool cutover.
- `ann` (5): the annotation/judgment program (rxdo, 37t.12, dve1) — ontology,
  calibration, mass-annotation runbook, judgment transaction, claims view.
- `perf` (4): hot/thick daemon, status snapshots, CLI snappiness, resume quality.
- `lin` (4): lineage law, continuity scenarios, subagent compaction, codex
  delegation lowering.
- `res` (4): deep research + outreach/application drafts (use the
  deep-research contract for res-01/res-02).
- `misc` (3): config closure, embedding freshness, FTS identity ledger.
- `mandate` (3): Claude artifact admission, observed-work effects, and the
  terminal replay gate; run in dependency order.
- `lane-support` (4): independent, implementation-grade support lanes for the
  four local war-room lanes. They own distinct reusable substrate and synthetic
  proofs; the paired local lane remains owner of private/live validation.

Contracts: `../contracts/chatgpt-pro-implementation.md` (default),
`../contracts/chatgpt-pro-analysis.md` (ann-03, res-04),
`../contracts/chatgpt-deep-research.md` (res-01, res-02).
