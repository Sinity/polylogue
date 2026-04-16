---
created: "2026-04-12T19:04:00+02:00"
purpose: "Detailed working plan for the remaining runtime vetting, robustness, performance, and architectural cleanup on feature/chore/repo-cleanup-governance"
status: "active"
project: "polylogue"
---

# Remaining Branch Plan

This note is the working map for the rest of the branch after the repo/governance/docs cleanup phase.

The documentation-surface optimization pass is being handled in parallel elsewhere.
This plan covers the remaining scope that still belongs on this branch:

- runtime/product vetting
- live-archive rebuild validation
- parser robustness
- memory and throughput investigation
- command-surface UX defects found under real use
- architectural drift that directly interferes with product reliability or maintainability

This note should stay concrete. If something is fixed, move it to the “done” section or tighten the open work. Do not let it devolve into vague roadmap prose.

## Current State

### Branch state

- branch: `feature/chore/repo-cleanup-governance`
- current latest runtime-vetting commit at note creation:
  - `28b512bd` `fix: emit acquisition heartbeats for slow files`

### Cleanup phase status

The repo/process/tooling cleanup phase is mostly complete.

Already substantially handled on this branch:

- license/public metadata cleanup
- GitHub workflow/template/governance cleanup
- docs generation and deduplication pass
- `CLAUDE.md` / `AGENTS.md` cleanup and transclusion tightening
- `devtools` consolidation as the control plane
- stale config/test debris cleanup
- multiple CLI consistency fixes
- zsh/archive-backed completions

### Runtime vetting phase status

Still actively in progress.

We have enough evidence now to say:

- the product works broadly enough to ingest and serve real archives
- the command surface still has real paper-cut inconsistencies, which are being found and fixed
- long-running rebuilds are still too heavy and not yet sufficiently observable
- parser robustness on malformed-but-salvageable JSONL is not finished

## Division Of Labor

### Parallel docs worker

Handled elsewhere:

- `README.md` / `docs/README.md` split and duplication
- final pass over `CONTRIBUTING.md`, `CLAUDE.md`, `AGENTS.md`
- natural-language/doc-surface cleanup
- transclusion/dynamic-generation optimization for docs and agent memory

### My lane

Remain responsible for:

- live product vetting
- runtime UX defects
- parser and ingestion robustness
- heavy-path performance and memory behavior
- architectural cleanup that materially affects runtime correctness or operability

## Workstreams

## 1. Live Rebuild Completion

### Goal

Complete a full `polylogue --plain run all` against the canonical default archive root after the reset and treat that run as the primary source of truth for:

- end-to-end wall time
- peak RSS
- stage-by-stage throughput
- malformed artifact behavior
- progress/observability quality

### Current evidence

- default archive root was reset in place under:
  - `~/.local/share/polylogue/reset-backup-20260412T183637+0200`
- fresh rebuild log:
  - `.local/logs/run-all-default-20260412T183655+0200.log`
- acquisition heartbeat fix already landed because the live run exposed multi-minute silent gaps

### Open questions

- total wall time after full completion
- true peak RSS for the full run
- whether later stages are dominated by a few giant artifacts or by broad baseline inefficiency
- whether final product rebuild/index phases behave acceptably after parse

### Exit criteria

- rebuild finishes successfully
- timings and RSS captured in the vetting log
- any concrete user-visible defect discovered during the run is either:
  - fixed and committed, or
  - explicitly logged with rationale for deferral

## 2. JSONL Robustness

### Goal

Make JSONL sampling/validation behavior consistent with the actual stream parser for real-world session files.

### Problem statement

Current bug:

- `sample_jsonl_payload()` used `orjson.loads()` only
- the streaming JSONL decoder used stdlib `json.loads()`
- some Claude Code lines contain escaped lone surrogates like `\\udce2`
- stdlib accepts them
- `orjson` rejects them with:
  - `invalid low surrogate in string`
- strict validation then marks the whole raw record as malformed even though the actual stream parser can process it

### Evidence already gathered

- real failing file:
  - `/home/sinity/.claude/projects/-realm-project-sinex.pre-enrich/b8c8d990-f5c4-4d01-881a-f4af42ceb7f2.jsonl`
- observed malformed lines:
  - line `2409`
  - line `2410`
- direct comparison:
  - `json.loads()` accepts the line
  - `orjson.loads()` rejects it
- a second live raw record with the same class of failure already surfaced during the fresh rebuild:
  - `raw_id = 57399b8676d88e84698827874e6f7cee6700f8138841ca058205af05cc73fd7e`

### Desired outcome

- keep `orjson` as the fast path
- add stdlib fallback only for line-level JSONL decode failures where `orjson` is stricter
- align the sampling/validation path with the stream parser’s tolerance
- verify on:
  - synthetic regression case
  - the real offending file(s)
  - a focused unit-test set

### Exit criteria

- fix committed atomically
- regression tests added
- real offending files no longer report malformed-line counts from this mismatch alone

## 3. Heavy Raw Artifact Investigation

### Goal

Understand and reduce the cost of giant raw session files.

### Evidence already gathered

From the live rebuild:

- current RSS has exceeded `1.0 GiB` during ingest
- many slow batches correspond to very large single-record raws
- examples from the log:
  - `blob_mb=244.5` with one Codex conversation / `96` messages
  - `blob_mb=277.5` with one Codex conversation / `277` messages
  - `blob_mb=515.6` with one Codex conversation / `274` messages
- provider-level blob-size distribution snapshot:
  - `claude-code`: average `1.3 MB`, max `1535.0 MB`
  - `codex`: average `9.1 MB`, max `515.6 MB`
  - `gemini`: average `0.7 MB`, max `25.9 MB`
- top concrete offenders are raw session files under:
  - `/home/sinity/.codex/sessions/...`

### Questions to answer

- are those huge files expected one-session transcripts or evidence of bad capture/export shape?
- where is the dominant time spent for giant records:
  - JSON decode
  - validation
  - parsing
  - transformation
  - SQL write
  - serialized IPC between process worker and parent
- can the ingest path be made less memory-heavy or less bursty for giant raws without destabilizing normal cases?

### Likely investigation steps

- correlate slow batches with concrete source files and conversation IDs
- inspect Codex parser/worker behavior for giant JSONL files
- inspect serialized worker-result size where relevant
- examine whether validation sampling or full parse repeatedly materializes too much structure
- measure whether worker count heuristics are still too optimistic for giant records

### Exit criteria

- at least one concrete performance improvement lands, or
- the main bottleneck is narrowed to a specific subsystem with evidence strong enough for a follow-up issue/branch

## 4. Command-Surface Vetting

### Goal

Exercise Polylogue as a real user would on the rebuilt archive and continue fixing obvious UX defects immediately.

### Already fixed during this phase

- `open --print-path <conversation-id>` exact-ID UX
- post-verb `stats --limit`
- `tags --format json`
- archive-backed zsh completions
- acquisition heartbeats during slow file reads

### Commands to keep exercising

- root query/list/stats/open/tags flows
- `products` family
- `doctor`
- `audit`
- `run` subcommands
- packaged binary via `.local/result/bin/polylogue`

### What to look for

- inconsistent option naming
- root-option vs verb-option confusion
- poor JSON envelope consistency
- bad error messages
- commands that work only in one mode (`--json` vs `--format json`)
- commands that force the user to know arbitrary internal IDs when natural discovery is available
- progress output that looks dead or misleading

### Exit criteria

- no easy command-surface inconsistencies remain unfixed
- remaining known CLI defects are either substantial work or explicitly logged

## 5. Architectural Drift With Runtime Impact

### Goal

Tackle only the architectural drift that meaningfully affects product reliability or maintainability on this branch.

### Known drift already documented elsewhere

See:

- `.claude/scratch/active/009-operator-brief.md`

### Highest-priority runtime-relevant drift

- sync/async storage split leaking into real production paths
- duplicated schema/version enforcement
- config/env surfaces that are broader or more hidden than justified
- giant-raw handling and stage heuristics not being fully shaped by real memory/runtime cost

### Rule for this branch

Do not try to “solve architecture” in the abstract.

Only act when:

- a drift item is directly causing a runtime problem seen in live vetting, or
- the cleanup is small enough and clear enough to finish safely within this branch

## 6. Evidence Capture

### Primary records

- `.claude/scratch/active/005-thorough-vetting-log.md`
  - append-only operational log
- `.claude/scratch/active/009-operator-brief.md`
  - current operator state and unresolved structural/runtime findings
- this file
  - forward-looking execution plan and scope map

### Logging discipline

Whenever a real finding appears:

- if quick to fix:
  - fix immediately
  - verify
  - commit atomically
  - log the finding and resolution
- if not quick:
  - capture concrete evidence
  - do not leave it as vague prose
  - decide whether it belongs in:
    - this branch later
    - a GitHub issue
    - a future branch

## Near-Term Sequence

1. Continue reducing default `doctor` latency on the live archive; exact maintenance counts should stay available, but the ordinary health path should behave like a probe, not an audit.
2. Re-run real-user smoke tests against the rebuilt archive after each runtime-facing fix.
3. Resolve the remaining true malformed JSONL case (`da69faf4...`, line 819) with an explicit policy: repair, quarantine, or improved diagnostics.
4. Investigate giant raw artifact performance with concrete measurements and narrow fixes.
5. Keep converting real user-visible defects into small atomic commits instead of stacking theory.

## “Done” Threshold For This Branch

The branch is ready only when all of the following are true:

- repo/governance/docs cleanup is in a stable state
- the default archive rebuild completes successfully on the cleaned branch
- obvious CLI/UX inconsistencies found during live use are fixed
- the JSONL robustness inconsistency is fixed
- the main remaining performance problems are either:
  - improved materially, or
  - narrowed to explicit evidence-backed follow-up work

If the repo is pristine but the product still feels operationally flimsy under real use, the branch is not done.
