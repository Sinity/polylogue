# Sixteen parallel fork prompts
Fork this chat and send one prompt per fork. Each prompt is self-contained but assumes the uploaded repository snapshots and prior analysis remain available.

---

# Fork prompt 01 — Build the Incident 14:32 evidence corpus

Use the Polylogue and Sinex repositories uploaded in this chat and the analysis already established here. Work independently and deeply. Do not stop at a design memo: produce an implementation-ready, patch-backed shared deterministic evidence corpus named **Incident 14:32**.

The corpus must support both projects without collapsing their ontologies. It should model a small AI-assisted coding incident containing, at minimum:

- a browser/design conversation;
- two coding-agent sessions;
- a copied-prefix fork or continuation;
- a fresh subagent control;
- a compaction boundary and summary;
- one nonzero structured test result followed by an assistant success claim;
- one later successful verification;
- provider usage with cache and reasoning lanes;
- one attachment with actual retained bytes;
- a candidate assertion, accepted assertion, and stale/rejected assertion;
- one context image and delivery snapshot;
- shell, Git, filesystem, browser, and desktop evidence for the same interval;
- one deliberate source outage;
- parser semantics v1 and v2 for one record;
- one ambiguous cross-source duplicate;
- one Beads task with acceptance criteria and a dependency transition.

The critical rigor requirement is that fixture generation and expected-result calculation must not use the same reducer. Create a declarative scenario manifest and an independent oracle manifest covering physical sessions, logical composition, tool outcomes, usage lanes, coverage intervals, parser semantic diff, material hashes, assertion state, and context delivery.

Inspect the existing Polylogue demo machinery under `polylogue/demo`, its construct audit, and Sinex source/test fixture conventions. Reuse real provider-shaped parsers and product paths wherever practical; do not create a demo-only parallel ontology. Preserve public safety: no private text, real secrets, hostnames, or absolute user paths.

Produce:

1. a detailed scenario specification;
2. a fixture and oracle data model;
3. patch-ready changes for Polylogue that seed and verify the corpus;
4. a Sinex fixture bundle or exact follow-on patch plan where direct implementation is too broad;
5. tests proving every declared construct is nonempty and every control behaves as expected;
6. a content manifest and privacy/licensing statement;
7. one patch per repository, generated against the exact snapshot commit;
8. a handoff listing files, commands, test results, unrun checks, and dependent Beads.

Use Beads, not GitHub Issues, as roadmap authority. Propose narrowly scoped new child Beads only where existing ones cannot own the work. Do not claim the full Sinex-backed integration exists. Treat Sinex as the ultimate durable backend target and Polylogue as the AI-work domain kernel.

Store all output under `/mnt/data/incident-1432-corpus/`, including `.patch` files and a top-level `README.md`. Return links to the artifact bundle and the most important files.

---

# Fork prompt 02 — Implement Polylogue “The Receipts”

Use the uploaded Polylogue repository and the complete prior analysis in this chat. Implement the strongest first-contact Polylogue demo: **The Receipts**, aligned with `polylogue-212.2` and especially the ready follow-up `polylogue-xyel`.

Do not merely make a report script. Use existing product primitives and the current Demo Finding Packet contract. Extend that contract only where necessary to carry a primary construct, oracle, falsifier, positive/negative/missing-evidence controls, exact material/evidence refs, scope, and recording parity.

The demo must show a two-column claim-versus-observed view:

- assistant claim: “All tests pass; the fix is complete” or equivalent synthetic text;
- observed structural tool outcome: paired test invocation, nonzero exit status, duration, and result ref.

Required controls:

1. nonzero exit is classified as failure;
2. zero exit with the word “failed” somewhere in text remains success;
3. claim with no paired result becomes `insufficient_evidence` or the project’s equivalent—not failure or success;
4. malformed or ambiguous pairing produces a loud caveat;
5. the verdict derives from structured outcome fields, never prose regex.

The demo must resolve every visible claim, outcome, and follow-up classification to stable refs and source evidence. It must emit a validated packet, report, queries, evidence rows, checks, and raw run transcript. The recording must be generated from the same canonical command.

Inspect `devtools/demo_packet.py`, the existing packet stub, anti-demo packet, claim-vs-evidence tooling, `polylogue/demo`, CLI demo commands, renderer code, and relevant tests. Reuse the deterministic corpus; if Incident 14:32 is not yet implemented in this fork, add the smallest self-contained provider-shaped fixture required and make the patch easy to rebase onto that corpus later.

Produce a patch-ready worktree, tests, generated public-safe packet, and recording source. Run targeted tests and the demo from a clean temporary archive. Report exact commands and exit codes. Do not overclaim prevalence or agent intent.

Keep public wording aligned with: “Polylogue is the local flight recorder and system of record for AI work.” Do not use “Your AI memory” as the primary category.

Store outputs under `/mnt/data/polylogue-receipts-demo/` and return links to the patch, packet, report, and verification receipt.

---

# Fork prompt 03 — Implement Polylogue “Count It Once”

Use the uploaded Polylogue repository and prior analysis. Design and implement a new construct-valid demo under the `polylogue-212` portfolio: **Count It Once**.

Primary claim: Polylogue preserves physical provider artifacts while distinguishing copied transcript replay from logical unique work.

Build a deterministic scenario with:

- a parent session containing a known prefix;
- a fork or continuation that physically repeats that prefix and adds a unique tail;
- a fresh subagent containing superficially similar text but no copied-prefix relationship;
- a compaction summary that is real new material and must not be deduplicated;
- independently declared expected physical and logical counts;
- usage/token lanes sufficient to show physical total, logical high-water total, and replay delta.

The demo should visibly transform a naïve physical total into a lineage-aware logical view while keeping raw evidence accessible. It must explain that logical accounting is not provider billing and that physical and logical views answer different questions.

Required controls:

1. copied prefix is charged once in logical unique work;
2. fresh subagent remains separate;
3. compaction summary remains real new content;
4. near-match text without an identity/topology edge is not fuzzy-deduplicated;
5. every headline total resolves to the exact session/message/edge ranges that contribute to it.

Use existing topology, session-profile, logical-root, usage, query, and demo-packet machinery. Do not hide a core-lineage defect in demo-specific arithmetic. If the current product lacks a required read, implement a reusable view or create a precise blocker with a failing test and deliver the strongest honest partial demo.

Produce:

- a proposed Bead specification for the new child under `polylogue-212`;
- implementation patch and tests;
- validated demo packet and report;
- a compact visual storyboard or generated terminal tape;
- exact reproduction commands;
- a claim/scope statement suitable for the public claims ledger.

Store outputs under `/mnt/data/polylogue-count-it-once/`. Use Beads, not GitHub Issues. Return links to the patch, packet, and handoff.

---

# Fork prompt 04 — Land the first semantic transcript renderer slice

Use the uploaded Polylogue repository and prior analysis. Implement the highest-value vertical slice of `polylogue-ap7`: a provider-agnostic semantic transcript renderer shared by CLI and web.

The first slice must cover:

1. shell/test command cards;
2. structural result cards with success/failure, exit status, duration, and folded output;
3. file edit cards rendered as diffs where evidence supports it;
4. a generic unknown-tool fallback that preserves all data;
5. evidence, derived, candidate/reviewed, missing, degraded, and not-supported badges where the current model exposes those states.

The core rule is that CLI and web consume a shared surface-neutral descriptor. Neither surface may independently decide whether a tool failed. Normalize provider spellings into semantic families while preserving raw tool identity and refs.

Inspect `polylogue/rendering`, CLI read views, web/site reader paths, surface payloads, and current snapshots. Design a renderer registry with explicit fallback and bounded output behavior. Add snapshots and behavioral tests, including huge output folding, invalid/missing fields, unknown tools, and structural failure that contradicts prose.

Make the change visually forceful enough to support The Receipts demo. Produce before/after screenshots or deterministic HTML/terminal fixtures using only public-safe data. Avoid attempting every tool family; document the extension contract and finish the shell/test/edit slice completely.

Run targeted rendering, CLI, and web/site tests. Do not edit README or marketing copy except a narrow developer note if required. Do not modify active token-accounting work.

Store outputs under `/mnt/data/polylogue-semantic-renderer/`, including the patch, design note, screenshots, snapshots, and verification receipt. Return links and call out any files likely to conflict with demo branches.

---

# Fork prompt 05 — Make Polylogue degrade loudly

Use the uploaded Polylogue repository and prior analysis. Implement the smallest coherent, reusable slice of `polylogue-avg` and `polylogue-cpf.4` needed for public demos and ordinary reads to distinguish:

- complete;
- degraded;
- not supported;
- timed out;
- truncated;
- stale/not current;
- incomplete projection or missing modality.

Do not attempt a global health-system rewrite. Start from actual result/view envelope contracts and the existing honesty anti-demo. Ensure the demo/read paths used by The Receipts, Count It Once, findings, and context can carry one bounded machine-readable and human-visible signal instead of returning a normal-looking empty or partial answer.

Required adversarial cases:

1. missing paired tool evidence;
2. stale FTS or derived projection;
3. query timeout;
4. result truncation;
5. unsupported cross-source reconstruction;
6. missing provider modality;
7. incomplete import/revision.

The implementation must be fail-closed where a false complete result would be misleading, but avoid turning every warning into a hard error. Define precedence when several states apply and include evidence/coverage details.

Use existing readiness, derived-status, surface-envelope, anti-demo, and daemon/query contracts. Add targeted tests and one deterministic anti-demo packet proving that refusal is a successful product behavior.

Do not close broad Beads unless their full acceptance criteria are met. Produce a patch, a state vocabulary/design record, tests, a packet, and a verification receipt under `/mnt/data/polylogue-readiness-refusal/`.

---

# Fork prompt 06 — Build the public claims and findings lane

Use the uploaded Polylogue repository and the prior analysis. Build a static but rigorous first public slice toward `polylogue-3tl.16` and `polylogue-3tl.4` without pretending to complete their broader blocked architecture.

Deliver:

1. `docs/public-claims.yaml` with statuses `proven`, `capability`, `aspirational`, and `retired`, plus an orthogonal evidence class;
2. a linter that scans declared public surfaces and rejects unknown claim IDs, retired primary wording, missing evidence refs, and present-tense publication of aspirational claims where practical;
3. a generated or checked public finding page for the claim-vs-evidence field observation;
4. a deterministic synthetic reproduction lane separated from the private field observation;
5. a route or page manifest suitable for the docs site;
6. tests for stale artifacts, missing provenance fields, unsupported quantitative claims, and malformed sample/caveat metadata.

The field finding must preserve exact scope: a bounded origin-stratified sample of 5,000 structured failures from a frame of 42,033, with 1,205 silent-proceed next-turn cases, 3,375 ambiguous cases, a 24.1% lower bound, and the existing marker calibration caveats. Do not generalize it to all agents or models.

Inspect `docs/proof-artifacts.md`, `devtools/demo_packet.py`, `.agent/demos`, page generation, docs drift checks, and Beads. Prefer a simple auditable YAML + generator over a large premature finding database. Make future migration to first-class finding objects explicit.

Produce a patch, generated page, claims inventory, and verification receipt under `/mnt/data/polylogue-claims-findings/`. Return links and list claims that remain unsupported.

---

# Fork prompt 07 — Rewrite Polylogue’s public surface

Use the uploaded Polylogue repository and all prior analysis. Produce a mergeable public-surface patch covering the repository README, generated docs site, docs information architecture, demo/proof pages, and Sinex interop page.

Stable category:

> Polylogue is the local flight recorder and system of record for AI work.

Remove “Your AI memory” as the primary site title/tagline. Memory remains one capability, not the category.

The first screen must communicate:

- what Polylogue is;
- why it is more than grep or a chat viewer;
- one private-data-free command that currently works;
- a visible evidence chain;
- current status and limitations.

Add or revise:

- `README.md`;
- Demos page centered on The Receipts, Count It Once, and Honest Refusal;
- Proof/Findings page;
- maximal Sinex interop direction, explicitly labeled not fully implemented;
- docs map and navigation;
- homepage hero and cards;
- GitHub description/topics/social-card checklist;
- a cold-reader skim ladder.

Use current repository facts only. Supported current install lanes should be source checkout and Nix unless you actually verify more. Roadmap authority is Beads, not GitHub Issues. Distinguish facts, capabilities, field observations, and aspirations. Do not claim memory uplift, cost-by-outcome, complete Sinex backend, or general release readiness.

Inspect and update generated-source files, not only generated output. Run page generation, docs surface checks, command-reference/link/drift tests available in the repo. Generate a patch and rendered-site preview under `/mnt/data/polylogue-public-surface/`.

Return links to the patch, proposed README, rendered homepage screenshot/HTML, and verification receipt.

---

# Fork prompt 08 — Produce the Polylogue install, visual, and launch kit

Use the uploaded Polylogue repository and prior analysis. Work as a release/proof engineer, not a copywriter.

Build the artifacts needed so publication becomes a decision rather than a scramble:

1. prove the current source-checkout lane from a clean checkout;
2. prove the Nix one-shot lane from a clean state;
3. inspect PyPI, Homebrew, OCI, NixOS, browser-extension, and other declared lanes, but mark each `proven`, `wired-not-proven`, or `not-currently-supported` based on actual execution evidence;
4. generate deterministic recordings for the current tour and any available flagship demo from canonical commands;
5. ensure recordings and reports share the same packet/run ID;
6. create a checksum manifest;
7. create launch assets: repository description, topics, social-card copy, short announcement, long announcement, demo captions, FAQ, limitations block, and press/analyst paragraph;
8. add a leak scan for private paths, hostnames, tokens, emails, and transcript text in generated artifacts.

Do not invent package availability. Do not make broad performance claims. Preserve failed install attempts as receipts rather than deleting them.

Inspect release workflows, `docs/installation.md`, release readiness docs, visual tape tooling, Pages generation, and existing demo-tour assets. Run the narrowest appropriate verification, and explain what could not be executed in the environment.

Store the complete kit under `/mnt/data/polylogue-launch-kit/`, including patches for repository-owned scripts/docs. Return the launch bundle link and a one-page go/no-go matrix.

---

# Fork prompt 09 — Make Sinex externally legible

Use the uploaded Sinex repository and prior analysis. Produce a mergeable external-legibility patch. Sinex’s first screen must no longer read primarily as a Rust/PostgreSQL/NATS architecture document.

Stable category:

> Sinex is the local evidence substrate for digital life and agent work.

Lead with the project’s genuinely distinctive concepts:

- source material versus interpretation;
- occurrence, coining, and persistence time;
- replay that changes interpretation without rewriting source history;
- explicit coverage gaps;
- confidence versus authority;
- current projections versus canonical events;
- agent-facing evidence access;
- Polylogue as the AI-work domain product on the maximal backend path.

Produce:

- rewritten `README.md`;
- top-level docs map with a 30-second/3-minute/30-minute skim ladder;
- product/concepts page;
- demos page that separates smoke verification, capability demos, system safety proofs, and experiments;
- proof-artifacts/claims page;
- maximal Polylogue integration page, present state versus target clearly separated;
- removal or replacement of retired GitHub Issue roadmap links with Beads references or stable docs;
- repository description/topics/social-card checklist.

The deterministic `sinexctl ops verify --demo` walkthrough must be described honestly as an operational smoke proof, not the full thesis demo. The private large-deployment observations must be labeled bounded field evidence, including recovery caveats.

Run documentation checks and any generated-doc commands. Do not modify the core runtime except to fix a directly exposed documentation contract. Store outputs under `/mnt/data/sinex-public-surface/`, with patch, rendered docs preview, and verification receipt.

---

# Fork prompt 10 — Implement Sinex “The Missing Source”

Use the uploaded Sinex repository and prior analysis. Implement or drive as far as statically possible the first flagship Sinex demo, strengthening `sinex-cem.2` with `sinex-jdp` and `sinex-60r`.

Primary claim: Sinex distinguishes a true quiet interval from an interval for which a source could not provide trustworthy evidence.

Build a deterministic fault-injection scenario with:

1. a healthy source and a genuinely empty interval;
2. a source process that is down or checkpoint-stalled;
3. a source process that is alive but emits semantically unusable empty records;
4. material acquired but parser/projection behind;
5. another source producing an event in the same interval, proving the overall query is not empty.

The result must carry source coverage, last confirmed occurrence/frontier, reason, evidence strength, and overall completeness. A normal-looking empty timeline is a failure.

Use the real source contracts, staged material, checkpoint/coverage, API/view-envelope, and demo verification paths. Avoid a shell script that bypasses the product. Create an independent expected source-sequence manifest and inject faults through supported test/sandbox mechanisms.

Produce:

- implementation patch and tests;
- demo packet with oracle, controls, and falsifier;
- human report and deterministic terminal/HTML recording source;
- exact commands and reset behavior;
- public claim wording and scope;
- Beads handoff with any uncovered substrate blockers.

Run targeted Rust/xtask tests. Respect checkout-local Postgres/NATS isolation and report actual test commands. Store outputs under `/mnt/data/sinex-missing-source-demo/`.

---

# Fork prompt 11 — Implement Sinex “Import It Twice” and replay contrast

Use the uploaded Sinex repository and prior analysis. Implement the strongest bounded slice of `sinex-cem.13`, coordinated conceptually with `sinex-908` and the later honest-revision demos.

Primary claim: importing the same source occurrences twice creates zero duplicate current occurrences, while replay under a changed semantics version intentionally creates new interpretations over the same stable occurrences.

Required scenario:

- first import of a deterministic export admits N occurrences;
- second import of identical bytes admits zero new current occurrences and emits a receipt for suppressed duplicates;
- a grown export adds exactly one new occurrence and supersedes only where declared;
- the same content from another source with ambiguous identity becomes an adjudication candidate rather than automatic fusion;
- replay under semantics v2 creates new interpretation identities while preserving stable occurrence/domain identities;
- a semantic diff reports which projections changed.

The demo must teach the distinction among occurrence identity, interpretation event identity, content hash, and stable domain object identity. Do not use event UUID as the stable public object ref.

Inspect equivalence/occurrence policy, material registry, parser replay, audit archive, domain reducers, operations, and public refs. Create independent manifests and tests. If stable identity infrastructure is insufficient, produce a failing acceptance test and a precise minimal patch/design rather than faking the result.

Produce patch, tests, packet, report, and verification receipt under `/mnt/data/sinex-import-twice/`. Include a follow-on design note showing how the same substrate supports `sinex-cem.14` and `.3` without conflating them.

---

# Fork prompt 12 — Specify and patch the maximal Sinex–Polylogue contract

Use both uploaded repositories and the full prior analysis. Treat the following premise as settled for this task:

> In Sinex-backed mode, Sinex ultimately stores provider-native transcript material, normalized transcript material, durable Polylogue-domain history, judgments, lifecycle, context deliveries, and model effects. Polylogue remains the AI-work ontology, parser/domain kernel, query/composition layer, and product. SQLite remains standalone authority or a local/offline projection and outbox.

Do not preserve the metadata-only doctrine as the ultimate boundary. Identify and explicitly supersede contradictory Beads or docs without rewriting history.

Produce a concrete cross-project contract covering:

- authority matrix by data class;
- stable domain object IDs, revision IDs, Sinex interpretation event IDs, and material anchors;
- provider-native and normalized immutable material segments;
- multi-digest content descriptors;
- session revision manifests;
- admitted observation vocabulary;
- batch settlement and complete-revision visibility;
- Polylogue domain relationships versus Sinex derivation provenance;
- physical versus logical transcript accounting;
- PostgreSQL domain projections and SQLite edge replica;
- offline writes and conflict handling;
- assertion/judgment/context lifecycle;
- shared model-effect recipe identity;
- public refs and cross-resolution;
- replay, supersession, deletion, shared-prefix safety, and cache/vector invalidation;
- source coverage and projection frontiers;
- security/capability boundaries for raw transcript text;
- a full `polylogue rebuild --from-sinex` acceptance proof.

Inspect `sinex-4j2` and children, current source contract, integration-authority docs, Polylogue `polylogue-6mv`, `polylogue-fs1.9`, storage tiers, assertions, context, refs, and topology. Produce:

1. versioned JSON Schemas and example payloads;
2. an architecture decision record for each repo;
3. exact Beads amendments/new children with dependencies and acceptance criteria;
4. patch-ready documentation changes;
5. the smallest safe code patch that deepens the current bridge without pretending the full architecture is complete;
6. an end-to-end phased implementation/verification plan.

Store outputs under `/mnt/data/maximal-sinex-polylogue-contract/`, with separate patches for both repos. Return links to schemas, ADRs, patches, and Beads plan.

---

# Fork prompt 13 — Prototype stable identity and Sinex-backed rebuild

Use both uploaded repositories and prior analysis. Focus narrowly on the hardest impedance mismatch: Sinex event IDs identify replay-specific interpretations, while Polylogue refs require stable session/message/block/assertion/context identities across replay and resegmentation.

Design and implement a reference prototype for:

- stable Polylogue domain object identity;
- domain revision identity;
- alias/reconciliation records;
- mapping to one or more Sinex interpretation event IDs;
- mapping to exact provider-native and normalized material anchors;
- stable refs that survive a parser v1→v2 replay;
- a projection watermark and local SQLite rebuild cursor;
- a minimal `rebuild from Sinex` path for a deterministic subset: sessions, messages, blocks, one topology edge, one assertion, and one context delivery.

Do not use content hash, mutable byte offset, or Sinex event UUID alone as stable identity. Use immutable normalized segments or record identities so reserialization does not invalidate refs.

The prototype can be schema-first if full runtime integration is too broad, but it must include executable tests or a small harness proving:

1. identical occurrence import resolves to the same stable object;
2. changed parser semantics creates a new interpretation/revision as appropriate;
3. stable public ref still resolves;
4. deleting rebuildable SQLite state and replaying the deterministic Sinex fixture reconstructs equivalent domain objects;
5. intentionally local UI state is excluded from parity;
6. ambiguity routes to reconciliation rather than silent merge.

Produce patches or prototype code in both repos, schema/DDL, tests, parity report format, and a Beads-ready handoff under `/mnt/data/interop-identity-rebuild/`.

---

# Fork prompt 14 — Build the joint “World Around the Claim” demo

Use both uploaded repositories and the prior analysis. Design and implement as much as possible of the combined flagship demonstration: **The World Around the Claim**.

The scene starts from one assistant claim that tests pass. The final evidence stack must join independent refs without flattening them:

- Beads intent and acceptance criteria;
- exact context delivered to the agent;
- assistant claim;
- Polylogue tool call/result and logical lineage;
- Sinex terminal, Git, filesystem, browser, and desktop evidence around the interval;
- successful or failed outcome;
- accepted and stale memory assertions;
- source coverage and unavailable modalities.

Every line must resolve to a native domain ref or material anchor. Missing evidence is a first-class gap. A domain relationship such as a session fork must not be encoded as Sinex derivation provenance.

Produce an Agent Work Packet v1 schema, example packet over Incident 14:32, query/orchestration plan, rendering storyboard, and one runnable prototype path using current APIs or fixture-level adapters. The packet should remain useful when one or more legs are absent.

Include controls:

- claim contradicted by transcript outcome;
- ambient terminal evidence agrees or disagrees;
- source outage prevents a broader conclusion;
- Bead status at the time differs from current status;
- stale assertion is excluded or visibly labeled;
- copied lineage is not counted as new work.

Do not claim the full Sinex backend is already implemented. Clearly distinguish a fixture-level joint proof, a current metadata bridge, and the target architecture.

Store outputs under `/mnt/data/world-around-the-claim/`, including schemas, patch-ready code/docs for each repo, demo packet, and verification/handoff report.

---

# Fork prompt 15 — Red-team every public claim and demo

Use both uploaded repositories, the prior analysis, and any legibility artifacts already produced in this fork. Act as a hostile but technically fair external reviewer. Do not implement the preferred story. Try to falsify it.

Audit:

- READMEs and site taglines;
- proof-artifact pages;
- claims ledger;
- all demo packets and recordings;
- install commands;
- private field findings;
- Sinex–Polylogue present-state versus target wording;
- security/privacy statements;
- Beads-backed roadmap claims;
- logical versus physical cost/accounting language;
- context/memory uplift language;
- source coverage and no-loss language.

Search for:

1. circular oracles;
2. deterministic fixture results presented as prevalence;
3. private archive observations presented as benchmarks;
4. silent denominator changes;
5. present-tense aspirations;
6. stale generated assets;
7. claims that cannot resolve to evidence;
8. missing negative or missing-evidence controls;
9. inaccessible install paths;
10. private paths, hostnames, secrets, emails, or transcript text;
11. GitHub Issues used as current roadmap authority;
12. “privacy” wording that actually means redacted view while originals remain;
13. universal no-loss claims contradicted by recovery history;
14. full-backend claims contradicted by the metadata-only current bridge;
15. UI outcomes that differ from CLI semantics.

Run static scans and available validators. Build a claim-by-claim table with verdicts: supported, overbroad, stale, ambiguous, or false. For every problem, propose exact replacement wording or a falsifying test. Rank launch blockers separately from polish.

Also perform a cold-reader exercise: explain both projects and the joint architecture using only public files, then list where you had to infer missing context.

Produce `/mnt/data/adversarial-legibility-audit/` with a detailed report, machine-readable findings, secret/PII scan receipt, suggested patches, and a go/no-go verdict. Do not soften the conclusion to preserve momentum.

---

# Fork prompt 16 — Integrate the legibility swarm

Use both uploaded repositories, the complete prior chat, and all artifact bundles returned by the other parallel forks that are available in this conversation or uploaded alongside this prompt. Act as the integration coordinator.

Your task is not to produce another high-level plan. Build the best coherent candidate you can from the returned work.

Steps:

1. inventory every incoming artifact, patch, schema, report, and verification receipt;
2. record exact base commits and reject patches against incompatible or unknown revisions unless safely portable;
3. detect file overlap and semantic disagreement;
4. choose one canonical contract for Incident 14:32, demo packet v2, claims vocabulary, semantic renderer, and maximal interop;
5. create clean integration worktrees for Polylogue and Sinex;
6. apply or manually port accepted patches in dependency order;
7. preserve source branch/artifact provenance even when importing files instead of commits;
8. regenerate all generated surfaces from source;
9. run targeted then integration verification;
10. run deterministic demos from clean state;
11. run static privacy/secret scans on public artifacts;
12. conduct a cold-reader test using a fresh agent prompt with no hidden context;
13. update Beads comments/status only where acceptance evidence exists; never use GitHub Issues;
14. produce final patches, rendered previews, demo packets, recordings, claims ledger, launch kit, and a rejected/deferred-work log.

Priority order:

1. truth and reproducibility;
2. semantic transcript rendering;
3. The Receipts;
4. Count It Once;
5. loud refusal/degradation;
6. public README/site/docs;
7. install and visual proofs;
8. Sinex legibility;
9. joint interop artifacts.

Do not let ambitious target architecture appear as current implementation. Do not preserve “Your AI memory” as the primary Polylogue category. Do not preserve the metadata-only Sinex bridge as the ultimate architecture.

When conflict cost is high, sacrifice Git elegance: import nonoverlapping file sets, make integration commits, and record provenance in commit messages. Never sacrifice test/evidence receipts.

Store the integrated candidate under `/mnt/data/legibility-integrated/`, including separate repository patches, a `MERGE_ORDER.md`, `VERIFICATION.md`, `CLAIMS_AUDIT.md`, `DEFERRED.md`, and a single downloadable archive. Return links to all primary artifacts and state exactly which checks did not run.
