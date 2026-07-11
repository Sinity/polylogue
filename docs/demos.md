# Demos and Proofs

Polylogue demos are evidence packets, not feature montages. A public demo should declare one primary claim, the structural oracle that decides it, at least one negative or missing-evidence control, exact evidence refs, and the boundary of what it does not prove.

## Run the current private-data-free tour

```bash
nix run github:Sinity/polylogue -- demo tour
```

From a checkout:

```bash
git clone https://github.com/Sinity/polylogue.git
cd polylogue
nix develop -c polylogue demo tour
```

The tour creates a throwaway archive, imports synthetic provider-shaped artifacts through the normal parsers, verifies declared constructs, runs canonical query/read/analysis paths, and writes a report, transcript, command outputs, and recording source. It requires no private transcript or provider account.

The human transcript is deliberately evidence-first rather than audit-first:

1. compare a false success claim with the structural receipt that existed at claim time;
2. distinguish the later verified repair from the earlier contradiction;
3. prove that prose containing the word `error` contributes zero failed actions;
4. aggregate failed actions from structural fields rather than prose;
5. read a fork as one composed chronicle while preserving parent refs;
6. only then zoom out to archive facets.

The full fixture audit remains machine-readable in `report.json`, so the public story stays compact without weakening verification. The current fixture world covers five origins, structured tool outcomes, attachment bytes, browser-capture coalescing, lineage, a subagent, a compaction boundary, context snapshots, user overlays, and deterministic synthetic embeddings. See the [construct audit](plans/demo-corpus-construct-audit.md) and [proof map](proof-artifacts.md).

## Current public proofs

### Deterministic tour

**Claim:** normal product paths can ingest and query the constructs declared by the demo corpus, compare a false assistant success claim with the structural failure that existed at the claim boundary, distinguish a later verified repair, reject a prose-only anti-grep control, and compose copied lineage with original refs intact.

**Oracle:** the independent construct verifier, exact expected claim text, message order, structural `exit_code`/`is_error` fields, a prose-only negative control, source-material hash, and stable refs in the composed chronicle.

**Does not prove:** private-archive scale, provider completeness, real-world prevalence, memory uplift, or the planned Sinex-backed storage architecture.

### Cost-accounting example

**Claim:** the demonstrated Codex input and cache lanes are normalized into disjoint pricing inputs rather than additively double-billing an inclusive provider input field.

**Oracle:** crafted provider payload, real writer/pricing code, and independent arithmetic fixture.

**Does not prove:** that a catalog estimate is a provider invoice or that subscription credits have a defensible currency conversion.

### Honest refusal

**Claim:** when required modalities are absent, a demo packet can return `not_supported` and name the missing evidence instead of fabricating a reconstruction.

**Oracle:** the demo corpus coverage manifest.

**Does not prove:** that every possible source outage is currently detected.

## Field finding: claim versus structural failure

A bounded private-archive study sampled 5,000 structured failures from a frame of 42,033. It classified 1,205 cases as silent proceed on the next assistant turn, a 24.1% lower bound, while 3,375 cases remained ambiguous. This is a field observation from one archive and one method—not a population estimate.

Read the full [finding, method, calibration, and caveats](findings/claim-vs-evidence.md).

## Flagship demonstrations under construction

These are roadmap items, not present capabilities unless their packets are linked above.

### The Receipts: deterministic contract proof

The current `polylogue demo receipts` command is a deterministic product-contract proof. It compares a false success claim with the structural failed test result that existed at the claim boundary, then distinguishes a later successful rerun. A separate prose-only session contains the word `error` while contributing zero structurally failed actions.

This is not the real-PR field proof owned by `polylogue-212.2`. That Bead still requires a merged pull request whose claim, tool evidence, code change, and verification receipts are independently reconstructable.

### Count It Once

A lineage proof that preserves copied provider artifacts while separating physical transcript volume from logical unique work. A fresh subagent and a compaction summary serve as non-deduplication controls. Planned under the `polylogue-212` demo portfolio and lineage program.

### Context Autopsy

A view of exactly which evidence and reviewed assertions an agent received, what was omitted under budget, and which stale candidates were excluded. This is capability and experiment work under the judged-context program.

### Resume Under Oath

A preregistered paired comparison between strong raw-reference access and a generated resume packet. A workflow demonstration is insufficient; any benefit claim requires fixed sampling, equal budgets, preserved outputs, and independent scoring.

## Demo Packet v2 contract

The Demo Finding Packet contains:

- an executable prompt;
- a provenance stanza;
- a fixed-section report;
- evidence and query rows;
- checks, unsupported claims, and coverage notes;
- the raw run transcript.

The versioned machine contract is
[`schemas/demo-packet-v2.schema.json`](schemas/demo-packet-v2.schema.json). It
also requires exactly one primary construct, a receipt-backed claim declared
before execution, an independent oracle, a comparative baseline, negative and
missing-evidence controls, an explicit falsifier, bounded non-claims, and local
SHA-256-bound receipt artifacts. `devtools lab policy demo-packet-registry`
validates every registered packet and rejects unregistered v2 packets.

A deterministic public packet proves a product contract on the approved seed
1843 fixture world. It does not establish field prevalence, production scale,
or model behavior in the wild. Private-archive benchmarks remain a separate
local-only lane with their own sampling and privacy obligations.

## Why the anti-demo belongs beside the successes

Polylogue’s category claim is not merely “it can answer difficult questions.” It is also “it can say when the archive does not support an answer.” A public portfolio that publishes only successful reconstructions would hide the project’s most important epistemic behavior.
