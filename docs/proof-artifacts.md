# Polylogue Proof Artifacts

This page maps public-facing claims to inspectable artifacts. It is deliberately
stricter than a feature list: if a claim has no artifact, phrase it as a
capability under construction instead of as a proven result.

## Current Claim Map

| Claim | Status | Artifact | What It Proves | Caveat |
| --- | --- | --- | --- | --- |
| A deterministic private-data-free corpus exercises the advertised demo constructs. | Fact | [Demo Corpus Construct Audit](plans/demo-corpus-construct-audit.md) | A fresh no-daemon seed/verify run produces non-empty coverage for declared source families: five origins, acquired attachments, browser-capture coalescing, lineage, subagent runs, terminal states, synthetic embeddings, and user overlays. | The corpus proves product behavior and construct coverage, not private archive scale. |
| Polylogue reads tool outcomes from structure, not assistant prose. | Fact | [Claim-vs-Evidence summary](#claim-vs-evidence-failure-follow-up) and `devtools workspace claim-vs-evidence` | The failure predicate is normalized tool-result evidence: `is_error=1` or non-zero `exit_code`. Assistant prose is only a follow-up acknowledgment signal. | The live aggregate is private-archive evidence; reproduce the method on the demo archive. |
| Codex token accounting bills disjoint lanes instead of double-counting cached input. | Fact | [Cost accounting example](examples/cost-accounting-demo.txt) and [Cost Model](cost-model.md#codex-disjoint-billing-lanes) | The synthetic run uses the real writer and pricing logic to show `fresh_input + cache_read = provider input`, avoiding the old cached-token double bill. | Provider prices and subscription equivalence are curated estimates, not billing authority. |
| Polylogue can analyze agent affordance usage across MCP and tool-call spellings. | Fact | [Agent affordance usage summary](#agent-affordance-usage) and `devtools workspace affordance-usage` | Family-normalized counts group tool aliases such as Serena, Lynchpin, Polylogue MCP, Context7, and codebase tools across captured action evidence. | Counts measure captured usage and failure signals, not independent utility or user benefit. |
| Agents with reviewed memory should recover better. | Promise | Pending uplift experiment (`polylogue-cfk`) | The substrate can compile context bundles, judged notes, and resume briefs. | Do not claim measured performance uplift until the paired experiment is complete. |

## Demo Corpus

Regenerate the construct audit:

```bash
devtools render demo-corpus-datasheet
devtools render all --check
```

The current tracked datasheet reports:

- 11 sessions
- 43 indexed messages
- 87 blocks
- 11 session profiles
- 12 run rows
- 37 observed-event rows
- 12 context-snapshot rows
- five origins: `aistudio-drive`, `chatgpt-export`, `claude-ai-export`,
  `claude-code-session`, `codex-session`
- no residual declared-construct gaps

## Claim-vs-Evidence Failure Follow-Up

The current private-archive demo shelf carries a public-safe aggregate summary
generated on 2026-07-04 against archive schema v24. It inspected a bounded,
origin-stratified sample of 5,000 structured failures from a frame of 42,033
structured failures:

- acknowledged next turn: 420
- silent-proceed next turn: 1,205
- ambiguous next turn: 3,375
- silent lower bound: 24.1%
- acknowledged within the next three assistant turns: 722
- marker calibration rows: 50 labeled
- acknowledged-marker precision: 100.0%
- acknowledged-marker recall: 84.2%

Regenerate the local packet:

```bash
devtools workspace claim-vs-evidence \
  --limit 5000 \
  --out-dir .agent/demos/claim-vs-evidence \
  --json
devtools workspace demo-shelf
```

Reproduce the method without private data:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/polylogue-claim-vs-evidence-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
devtools workspace claim-vs-evidence \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" \
  --limit 5000 \
  --out-dir /realm/tmp/polylogue-claim-vs-evidence-repro \
  --json
```

## Agent Affordance Usage

The current demo shelf includes a recent-session-window affordance report. Top
families in that window:

- `lynchpin`: 32 actions across 9 raw tool names; 3 distinct sessions; errors 0.
- `polylogue`: 21 actions across 10 raw tool names; 6 distinct sessions; errors 3.
- `serena`: 15 actions across 3 raw tool names; 4 distinct sessions; errors 0.

Regenerate:

```bash
devtools workspace affordance-usage \
  --out-dir .agent/demos/agent-affordance-usage \
  --json
devtools workspace demo-shelf
```

Use this as product-development evidence, not as a leaderboard. It answers
"which affordances are actually being used, with what structured failure
signals?" A utility judgment still requires reading the session context around
the samples.

## Release Readiness

The release gate is tracked separately at
[Release Readiness Gate](plans/release-readiness-gate.md). Before a public
release, the README claims, generated docs, demo corpus, visual evidence, and
packaging checks must all agree with that gate.
