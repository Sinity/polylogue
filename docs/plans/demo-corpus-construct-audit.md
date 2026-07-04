# Demo Corpus Construct Audit

This is the current requirements-vs-corpus audit for `polylogue-uhl`.
It exists to keep the deterministic demo archive construct-valid: demos and
seeded acceptance tests should not pass because the fixture world is too small
to exercise the thing being claimed.

## Evidence Snapshot

- Beads input: `bd list --status open --limit 0 --json` plus
  `bd list --status in_progress --limit 0 --json`, filtered for
  demo/corpus/scenario/archive language.
- Matching issues: `158` open or in-progress beads.
- P1 matching issues: `17`.
- Current seed command:
  `polylogue demo seed --root /realm/tmp/polylogue-uhl-demo-current/archive --force --with-overlays --format json`.
- Current seed wall time on this workstation: `4.602s`.
- Current verifier:
  `polylogue demo verify --root /realm/tmp/polylogue-uhl-demo-current/archive --require-overlays --format json`.

## Current Demo Archive Coverage

The current deterministic archive is intentionally tiny:

| Fact | Current |
| --- | ---: |
| Sessions | 8 |
| Messages | 35 indexed / 37 processed before prefix dedup |
| Blocks | 79 |
| Session profiles | 8 |
| Origins | chatgpt-export, claude-ai-export, claude-code-session, codex-session, aistudio-drive |
| Tool-use blocks | 19 |
| Tool-result blocks | 21 |
| Failed tool results | 4 |
| Messages with provider usage lanes | 7 |
| Attachment rows | 1 |
| Acquired attachment rows | 1 (`byte_count=53`, true blob hash present) |
| Temporary sessions | 1 |
| Session-link rows | 2 |
| Prefix-sharing lineage links | 1 |
| Subagent links | 1 |
| Run rows | 8 |
| Observed-event rows | 32 |
| Context-snapshot rows | 9 |
| Subagent-start context snapshots | 1 |

Those constructs are now declared in `polylogue.demo.constructs` and checked by
`polylogue demo verify`. If a declared construct produces zero or too few rows,
the verifier fails.

The demo source families are also declared in `polylogue.scenarios.corpus` as
`DEMO_CORPUS_FAMILIES`, so each source family names the construct IDs it exists
to exercise. The temporary-session family writes a Claude.ai export payload
(`claude-ai/temporary-demo.json`) with source-level `is_temporary: true`; the
lineage/subagent family writes explicit Codex source files
(`codex/lineage-parent.jsonl`, `codex/lineage-fork.jsonl`,
`codex/lineage-subagent.jsonl`). Both go through the normal parser/storage path;
they are not DB-side row patches.

## Requirements Buckets From Beads

The current Beads text clusters into these construct requirements:

| Construct bucket | Matching beads |
| --- | ---: |
| Web/read/demo UX | 137 |
| Tool/action outcomes | 122 |
| Provider usage/cost | 86 |
| Query DSL/analytics | 77 |
| Temporary/abandoned sessions | 40 |
| Browser capture/import convergence | 39 |
| Lineage/fork/resume/compaction | 37 |
| Attachments/blobs | 29 |
| Subagent/sidechain | 29 |

## Gaps

These P1-relevant constructs are not yet exercised by the deterministic demo
archive and must become explicit generator families before demos can safely
claim them:

| Gap | Current evidence | Driver beads |
| --- | --- | --- |
| Richer lineage matrix | one prefix-sharing fork only; no resume/compaction/sidechain variants | `polylogue-4ts`, `polylogue-4ts.1`, `polylogue-37t.11` |
| Abandoned / censored sessions | temporary-session coverage exists; no abandoned/censored variants yet | `polylogue-cfk`, temporal-analysis beads |
| Browser-capture gap and convergence cases | no capture-gap scenario in the demo seed | `polylogue-b5l`, capture-completeness work |
| Embedding-lane prose | no synthetic embedding coverage in the demo seed | embedding/status beads |
| Subagent run projection collision | parent subagent run currently collides with child main `run_ref`; tracked as `polylogue-85z0` | `polylogue-37t.11`, `polylogue-4ts.1` |

## Next Implementation Order

1. Add resume/compaction/sidechain variants to the lineage family.
2. Add abandoned/censored and capture-gap families.
3. Add embedding-lane prose or synthetic local embedding coverage.
4. Render this audit into a generated datasheet from the family registry,
   so the docs stop being hand-maintained.
