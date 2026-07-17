# Polylogue + Sinex external-legibility kit

This package turns the static repository audits into an executable public-presentation program. It contains mergeable proposal patches, a redesigned proof portfolio, a Beads-aligned Polylogue launch cut, a single-machine agent-swarm runbook, two sets of parallel fork prompts, machine-readable claim/demo contracts, and visual mockups.

## Fastest path through the package

Read these five artifacts first:

1. [Diagnostic and positioning](01-diagnostic-and-positioning.md) — what currently obscures each project, the category language to standardize, and the joint story.
2. [Expanded demo redesign](02b-demo-portfolio-expanded.md) — complete reconsideration of the portfolio, including the shared Incident 14:32 corpus, oracles, controls, rejected demos, and launch sequence.
3. [Frontier-agent execution plan](03b-polylogue-agent-execution-expanded.md) — the rapid Polylogue presentability cut, lane ownership, Beads, dependencies, merge gates, commands, and resource scheduling.
4. [Single-machine swarm runbook](05-single-machine-swarm-runbook.md) — worktrees, ownership locks, handoffs, validation tiers, integration strategy, and failure policy.
5. [Primary 16 fork prompts](08-fork-prompts.md) — the copy-paste prompts designed to run as parallel forks of this conversation.

For a directory-style index, use the [artifact map](16-artifact-map.md).

## Concrete repository patches

The patch bundle is under [`patches/`](patches/README.md):

- [`polylogue-external-legibility.patch`](patches/polylogue-external-legibility.patch), based on `f6c1da99`;
- [`sinex-external-legibility.patch`](patches/sinex-external-legibility.patch), based on `b70a08d9`.

The Polylogue patch improves its README, generated-site source, navigation, demos/proof/findings surfaces, public claim ledger, and maximal Sinex direction. The Sinex patch rewrites its public entry point, adds a skim-oriented docs map, concepts/product page, demo portfolio, proof index, claim ledger, and current-versus-target Polylogue backend contract.

Apply from a clean checkout at the matching base commit:

```bash
git apply --index /path/to/polylogue-external-legibility.patch
# or
git apply --index /path/to/sinex-external-legibility.patch
```

See [the validation report](15-validation-report.md) for exact checks, patch-application proof, verified claims, and explicit non-verification.

## Public story

**Polylogue is the local flight recorder and system of record for AI work.** It makes provider-native transcripts, tool outcomes, lineage, usage, reviewed memory, and context delivery inspectable as one evidence-backed body of work.

**Sinex is the local, replayable evidence substrate for digital life and agent work.** It preserves source material, records interpretations without erasing prior readings, represents source gaps explicitly, and joins heterogeneous activity under replayable provenance.

**Together:** Polylogue explains AI work; Sinex preserves the wider evidentiary world in which that work happened. In the maximal architecture, Sinex stores provider-native and normalized transcripts plus durable Polylogue-domain history. Polylogue remains the AI-work ontology, parser, query, memory, and user-experience layer. SQLite remains Polylogue's standalone store and local/offline projection.

Read [the one-page architecture](08-joint-architecture-one-pager.md) and [the fuller joint story](08a-joint-public-story.md).

## Demo portfolio

The redesigned portfolio is not a generic feature tour. Every demonstration has one primary construct, a declared oracle, a falsifier, controls, evidence refs, and a scope statement.

The shared deterministic proof world, **Incident 14:32**, contains:

- a structural test failure followed by an assistant success claim;
- a later verified repair;
- copied lineage and a fresh-subagent control;
- a compaction summary that omits the failed attempt;
- disjoint usage/cache/reasoning lanes;
- an attachment with retained bytes;
- candidate, accepted, rejected, and stale assertions;
- a context image and delivery record;
- terminal, Git, filesystem, browser, desktop, and Beads evidence;
- one deliberate source outage;
- parser semantics v1/v2;
- one ambiguous cross-material duplicate.

Recommended first-contact sequence:

1. **Polylogue: The Receipts** — compare assistant claim with structured tool evidence.
2. **Polylogue: Count It Once** — preserve physical sessions while accounting for copied lineage once.
3. **Sinex: Missing Source** — distinguish evidence-backed absence from a capture gap.
4. **Sinex: Changes Mind Honestly** — replay one occurrence under corrected semantics without erasing the old interpretation.
5. **Joint: World Around the Claim** — combine AI-work evidence with terminal, Git, task, and source-coverage evidence.

Start with [the executive portfolio](02-demo-portfolio-redesign.md), then read [the expanded portfolio](02b-demo-portfolio-expanded.md). Machine-readable forms live in [`11-demo-portfolio.yaml`](11-demo-portfolio.yaml), [`10-demo-packet-v2.schema.json`](10-demo-packet-v2.schema.json), and [`10-demo-packet-v2-example.yaml`](10-demo-packet-v2-example.yaml).

## Polylogue rapid launch cut

The launch cut intentionally defers broad autonomy, a full evidence cockpit, cost-by-outcome, and general memory-uplift claims. It prioritizes:

1. truthful degraded/readiness states;
2. one consistent category and claims ledger;
3. a narrow provider-neutral semantic transcript renderer;
4. The Receipts and Count It Once packets;
5. findings/proof pages;
6. clean install and recording receipts;
7. cold-reader and adversarial claim gates.

Use [the compact plan](03-polylogue-presentability-plan.md), [the expanded execution program](03b-polylogue-agent-execution-expanded.md), [`12-beads-launch-cut.csv`](12-beads-launch-cut.csv), and [`13-worktree-lanes.csv`](13-worktree-lanes.csv).

## Parallel fork prompts

The **primary suite** is [`08-fork-prompts.md`](08-fork-prompts.md), with individual files under [`fork-prompts/`](fork-prompts/). It contains 16 substantial chat-fork tasks spanning fixture construction, flagship demos, renderer/readiness work, public surfaces, Sinex demos, maximal interop, red-team review, and integration.

An **optional alternate suite** is [`14-alternate-worktree-prompts.md`](14-alternate-worktree-prompts.md), with individual files under [`alternate-prompts/`](alternate-prompts/). These divide the work into narrower repository-owned lanes and are useful for a second wave or when strict file ownership matters more than shared artifact synthesis.

## Concrete proof and visual artifacts

The validated Polylogue tour packet is under [`polylogue-demo-tour/`](polylogue-demo-tour/report.md). It includes the human evidence story, complete machine report, raw command outputs, a VHS recipe, a GIF, and its deterministic archive. The patched tour starts with a structural failed-tool receipt, aggregates failures from typed fields, demonstrates composed lineage, and only then expands to archive scope.

The `mockups/` directory contains source HTML and screenshots for:

- a [Polylogue landing page](mockups/polylogue-home.png) centered on evidence, lineage, and reviewed memory;
- a [Sinex landing page](mockups/sinex-home.png) centered on material, replay, coverage gaps, and authority;
- a joint [Resume This Bead](mockups/resume-this-bead.png) Agent Work Packet surface.

The current Polylogue generated-site preview is under `previews/polylogue-site/`. A 13-slide executive deck is available as [PowerPoint](Polylogue-Sinex-external-legibility.pptx) and [PDF](Polylogue-Sinex-external-legibility.pdf).

## Machine-readable coordination and claims

- [`09-public-claims-ledger.yaml`](09-public-claims-ledger.yaml) — joint public claim inventory.
- [`10-public-claims-ledger.yaml`](10-public-claims-ledger.yaml) — compact claims companion.
- [`11-demo-portfolio.yaml`](11-demo-portfolio.yaml) — demo portfolio as data.
- [`12-beads-launch-cut.csv`](12-beads-launch-cut.csv) — Beads subset and launch treatment.
- [`13-worktree-lanes.csv`](13-worktree-lanes.csv) — worktree ownership and dependencies.
- [`scripts/bootstrap-worktrees.sh`](scripts/bootstrap-worktrees.sh) — create an integration-first single-machine worktree swarm.
- [`scripts/check-public-artifacts.sh`](scripts/check-public-artifacts.sh) — basic path/credential leak scan.

## Deliberate non-claims

The static patches do not implement the full semantic transcript renderer, Sinex-backed Polylogue data plane, full physical excision, general agent-memory uplift, or the proposed flagship multi-source demonstrations. They label those targets and turn them into bounded work packages.

The Polylogue patch passed focused Ruff, format, strict MyPy, documentation generation, command verification, site-link checks, and 41 targeted tests; its deterministic tour completed with 30/30 declared constructs and no path leaks. Both binary patches apply cleanly to their exact base commits. The Sinex patch passed static documentation, claims-ledger, link, roadmap-policy, and diff checks. The Sinex runtime was not compiled or deployed because Rust, Cargo, Nix, PostgreSQL, and NATS were not available in the execution environment. Exact scope is recorded in [the validation report](15-validation-report.md).

## Integrity manifest

[`artifact-manifest.json`](artifact-manifest.json) records sizes and SHA-256 digests for the primary deliverables. [`MANIFEST.sha256`](MANIFEST.sha256) covers the complete package contents except the manifest file itself.
