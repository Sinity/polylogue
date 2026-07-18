# Framing and structural decisions

## D-01: Name the category as “local-first archive and forensics layer”

**Decision.** Open with “Polylogue is a local-first archive and forensics layer for AI and agent sessions.”

**Why.** “Archive” is directly supported by the five-tier storage model and durable raw evidence. “Forensics” is supported by typed tool outcomes, claim-vs-evidence artifacts, lineage, usage provenance, and explicit refusals. “System of record” implies organizational authority, multi-user governance, and completeness that the trusted single-user host model does not establish. “Memory system” invites benchmark and outcome-improvement claims that remain unproven.

**Reversal trigger.** A future multi-user governance model, completeness SLA, or validated memory-outcome study could justify a broader category. Until then, archive plus forensics is exact.

## D-02: Put a runnable receipt before architecture

**Decision.** The first operational path is import, find, exact read, and structured action aggregation over a private-data-free corpus.

**Why.** The target audience can inspect the evidence chain immediately. Architecture then explains why the result is trustworthy instead of asking readers to infer product value from components.

**Rejected alternative.** Lead with the four-ring diagram or a feature inventory. That repeats the current underselling problem and invites “just use grep” categorization.

## D-03: Merge-gate the requested daemon quickstart

**Decision.** Include the exact intended `polylogue import --demo --wait` path but mark it non-publishable until daemon/direct parity passes.

**Why.** Omitting the command would fail the mission. Presenting it as working would violate the evidence contract. The snapshot failure is semantic and reproducible, not a cosmetic count drift.

**Acceptance trigger.** The same canonical verifier passes after daemon import and direct seed, with generated expectations and idempotent repeat import.

## D-04: Publish no fixed demo corpus count yet

**Decision.** Use qualitative expected observations in the quickstart and leave `[demo numbers]` for a generated source.

**Why.** Four incompatible states exist across current source, tests, docs, and execution. A number copied into README would become another owner of drift.

**Reversal trigger.** One generated datasheet feeds seeder, verifier, tests, CLI output, docs, and demo shelf.

## D-05: Separate origin, provider wire, and capture mode

**Decision.** The provider table is keyed by public `origin`; capture mode and internal provider-wire identity appear only where they explain fidelity.

**Why.** The repository doctrine explicitly prevents one source-shaped value from owning parser family, public query token, material root, capture mode, and session identity. Browser capture is therefore not listed as a provider origin.

**Rejected alternative.** A vendor-logo matrix that collapses ChatGPT Takeout, browser capture, API provider, and parser shape into one “provider supported” checkmark.

## D-06: Treat package completeness and losslessness as different claims

**Decision.** Explain that an accepted/complete importer row covers detector, parser, fixtures, schema, read surfaces, ImportExplain, privacy, and docs. It does not promise lossless provider parity.

**Why.** Hermes ATIF has an explicit exact/inferred/absent matrix. Chat exports omit attachment bytes. Usage coverage varies. The table needs to reward the project’s actual rigor rather than flattening it into misleading checkmarks.

## D-07: Use four proof cards with one headline number each

**Decision.** Lead the demo shelf with deterministic receipts, bounded claim-vs-evidence, longitudinal usage forensics, and a structural refusal.

**Why.** Together they show contract correctness, bounded field analysis, scale on an operator archive, and disciplined non-claim behavior. One number per card keeps provenance legible.

**Rejected alternatives.** Use every demo on the shelf, or use the uplift pilot as a headline. The former becomes inventory prose; the latter overstates an explicitly non-publishable experiment.

## D-08: Keep private field findings bounded and aggregate-only

**Decision.** The claim-vs-evidence card publishes only its inspected sample size in the README. Rates, calibration, and denominator details stay in the linked packet and evidence ledger.

**Why.** The private archive is unavailable for independent reproduction. The public deterministic path reproduces method and artifact shape, not field rates.

**Reversal trigger.** A public or shareable corpus with the same sampling and calibration contract could support a fully reproducible field card.

## D-09: Keep physical, logical, and pricing grains separate

**Decision.** The forensics card says the command separates physical-session totals, logical-session high-water totals, and pricing provenance. It does not lead with a dollar headline.

**Why.** The cost model explicitly rejects billing equivalence. A dollar-first card would obscure missing models, origin-reported lanes, cache semantics, and logical replay.

## D-10: Include an anti-demo in the main README

**Decision.** Show one refusal based on absent ambient source tables.

**Why.** “Look how rigorously this is audited” is stronger when the README demonstrates a claim the product refuses. It preempts the common assumption that nearby transcript and git evidence can be fused into a minute-level activity timeline.

## D-11: Adapt the four-ring architecture, then show five tiers

**Decision.** Use one compact Mermaid diagram followed by the commands that expose paths, health, and rebuild behavior.

**Why.** The rings explain responsibility flow. The tier names explain durability. The command references keep the architecture from becoming decorative prose.

## D-12: Preserve a six-tool integration slot without inventing tools

**Decision.** Insert `[six-tool table]` with required columns and a current 104-tool fallback.

**Why.** The user states that a six-tool replacement is landing in parallel, but no names or contracts exist in the snapshot. Inventing them would create a parallel product model. Omitting current fallback would make the draft incomplete before integration.

**Removal rule.** The fallback and six-tool table must never be published together.

## D-13: Use runtime source over stale generated prose for MCP roles

**Decision.** Current fallback names `read`, `write`, `review`, and `admin`.

**Why.** `polylogue-mcp --help` and source register four roles. `docs/mcp-reference.md` still describes three. Runtime is the stronger authority, while the generated tool count remains usable.

## D-14: Put search omissions next to search capability

**Decision.** State default lexical behavior, explicit semantic activation, and the Write/Edit-body FTS exclusion in one section.

**Why.** “Search everything” is false. The repository already made a deliberate option-B decision to document and provide an unindexed action-input workaround rather than silently expanding FTS.

## D-15: Put trust and rebuild limits in the README, not only security docs

**Decision.** Name trusted single-user host, single writer, OS encryption, rebuild cost, uneven provider fidelity, cost non-billing, absent ambient sources, proposed browser capture, Grok absence, and pre-1.0 status.

**Why.** These boundaries materially change how an agentic-coding or eval reader should classify the product. Hiding them would weaken the evidence-led position.

## D-16: Exclude outcome-improvement positioning

**Decision.** “What it is not” says Polylogue is not a memory-retrieval benchmark player. The main description claims context delivery capability, not improved task performance.

**Why.** The current uplift pilot is small, uses hand-written summaries rather than the production pipeline, includes a compromised blind, and contains a false-fact counterexample.

**Reversal trigger.** Preregistered independent production-pack experiment at the protocol’s publishable sample tier, with counterexamples and selection bias stated.

## D-17: Keep installation claims channel-specific

**Decision.** The quickstart uses the documented pipx route, but the integration checklist requires release-channel smoke before publication.

**Why.** Source metadata proves entry points, not that every current registry artifact is present and version-aligned.

## D-18: Do not preserve the current README’s persuasive inventory cadence

**Decision.** Remove broad rhetorical questions, stacked feature claims, superlative implication, and architecture-first exposition. Use claim, command, evidence, caveat.

**Why.** The target audience rewards artifacts and falsifiable boundaries. The repository’s open `polylogue-3tl.12` acceptance criteria already require reproducible capability claims and a de-persuasion pass.

## D-19: Treat another iteration as integration, not another conceptual rewrite

**Decision.** Preserve this structure unless new evidence falsifies it. The next revision should fill the two slots, rerun commands, and update generated numbers.

**Why.** The high-value uncertainty is executable parity and landed MCP shape, not category wording. Rewriting again before those inputs exist would create churn without stronger evidence.

## D-20: Use the explicit maintenance rebuild path over stale daemon-only doctrine

**Decision.** The README demonstrates `polylogue ops maintenance rebuild-index --plan --output-format json` and names the same maintenance command as the post-reset replay path.

**Why.** Current repository doctrine still says `polylogue ops reset --index && polylogued run`, but the executable reset command prints a different next action. In a throwaway archive, daemon startup alone left search stale; the explicit authority-safe maintenance rebuild completed and reported ready surfaces.

**Reversal trigger.** If daemon startup again becomes the tested canonical replay owner, update the reset output, diagnostics, repository doctrine, operator docs, and integration tests together before changing the README.
