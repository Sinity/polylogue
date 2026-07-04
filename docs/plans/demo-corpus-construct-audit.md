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
- Current seed wall time on this workstation: `39.339s`.
- Current verifier:
  `polylogue demo verify --root /realm/tmp/polylogue-uhl-demo-current/archive --require-overlays --format json`.

## Current Demo Archive Coverage

The current deterministic archive is intentionally tiny:

| Fact | Current |
| --- | ---: |
| Sessions | 3 |
| Messages | 23 |
| Blocks | 63 |
| Session profiles | 3 |
| Origins | chatgpt-export, claude-code-session, codex-session |
| Tool-use blocks | 18 |
| Tool-result blocks | 20 |
| Failed tool results | 4 |
| Messages with provider usage lanes | 7 |
| Run rows | 3 |
| Observed-event rows | 25 |
| Context-snapshot rows | 3 |

Those constructs are now declared in `polylogue.demo.constructs` and checked by
`polylogue demo verify`. If a declared construct produces zero or too few rows,
the verifier fails.

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
| Lineage/fork/resume/compaction | `session_links = 0` | `polylogue-4ts`, `polylogue-4ts.1`, `polylogue-37t.11` |
| Subagent/sidechain trees | no subagent/sidechain rows in the demo seed | `polylogue-37t.11`, `polylogue-4ts.1` |
| Attachments with acquired bytes | `attachments = 0`, `acquired_attachments = 0` | `polylogue-83u`, `polylogue-83u.1`, `polylogue-83u.4` |
| Temporary / abandoned / censored sessions | `temporary_sessions = 0` | `polylogue-cfk`, temporal-analysis beads |
| Browser-capture gap and convergence cases | no capture-gap scenario in the demo seed | `polylogue-b5l`, capture-completeness work |
| Embedding-lane prose | no synthetic embedding coverage in the demo seed | embedding/status beads |

## Next Implementation Order

1. Add a declared corpus-family layer so each demo family names the construct it
   exists to exercise.
2. Add the first missing family: attachment bytes, because `polylogue-83u.1` is
   a concrete P1 bug with a narrow expected shape.
3. Add lineage/fork/resume and subagent families together; they share topology
   and composed-read assertions.
4. Add temporary/abandoned and capture-gap families.
5. Render this audit into a generated datasheet once the family registry exists,
   so the docs stop being hand-maintained.
