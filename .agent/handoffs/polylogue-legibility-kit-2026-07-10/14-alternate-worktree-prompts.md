# Sixteen parallel fork prompts

Each section below is also available as an individual file under `alternate-prompts/`. Run them as independent forks of the repository-analysis conversation. Their file ownership is intentionally separated so a conductor can absorb patches quickly.

---

# Fork 01 — Polylogue landing narrative and public category

Work directly on the supplied Polylogue repository. Treat Beads—not GitHub Issues or speculative Markdown plans—as roadmap authority. Your mission is to make a technically sophisticated stranger understand and care about Polylogue within one screen, one minute, and ten minutes.

## Product thesis

Use this category consistently:

> Polylogue is the local flight recorder and system of record for AI work.

Its public promise is: search heterogeneous agent histories, read tool activity as work rather than chat, audit claims against structural outcomes, understand lineage and accounting, and resume from reviewed evidence.

Do not collapse the project into “AI memory,” “chat archive,” or “LLM observability.” Those are subordinate capabilities.

## Owned scope

You have exclusive ownership of:

- `README.md`;
- package metadata descriptions in `pyproject.toml`, `flake.nix`, and equivalent top-level metadata;
- landing-page copy and layout templates under `devtools/pages_*`;
- top-level documentation navigation/registry files;
- new public orientation docs, but not product runtime code.

Do not change demo fixture behavior, semantic renderers, daemon behavior, query semantics, or database code.

## Required work

1. Audit every top-level category phrase and remove contradictions such as “Your AI memory” versus “system of record.”
2. Rewrite the README around visible payoff rather than repository process. Use this order:
   - category and promise;
   - a public-safe visual or demo command;
   - five concrete questions the product answers;
   - one structural-evidence chain;
   - the trust model;
   - architecture;
   - current status and honest limitations;
   - installation and documentation.
3. Remove stale meta-persuasion and “how to read this README” copy. Show the product rather than defending it.
4. Ensure the landing page does not render zero-valued archive statistics in CI or a fresh checkout. Prefer capability cards and deterministic demo facts.
5. Add a short “Why Polylogue” page comparing it with grep, transcript viewers, vector memory, agent tracing, and Git without constructing a straw man.
6. Make Beads authority explicit and remove public navigation that implies GitHub Issues are the roadmap.
7. Preserve all required generated-doc markers and repository-specific generation contracts.

## Construct-validity constraints

Every public claim must be one of:

- a deterministic fact from the public demo;
- a bounded private-archive field observation with corpus and caveats;
- a current capability with a direct code/test/proof path;
- an explicitly labeled plan.

Do not claim general reliability uplift, universal provider completeness, provider invoice accuracy, or real-scale latency from the small deterministic corpus.

## Validation

Run the repository’s canonical documentation and page-generation checks. At minimum:

- render the generated docs surface;
- build the static site;
- run doc-command verification;
- run generated-site link checks;
- run `git diff --check`;
- inspect the rendered home page rather than trusting source templates alone.

## Deliverables

Produce:

1. a concrete patch;
2. a one-page before/after explanation;
3. a list of every public claim added or removed;
4. validation output;
5. unresolved legibility defects that require runtime or visual work.

Commit hygiene is secondary to speed. Keep edits inside the owned scope so the conductor can merge the branch wholesale.

---

# Fork 02 — Polylogue semantic transcript launch slice

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, particularly `polylogue-ap7`. Implement the smallest high-impact semantic transcript slice that makes Polylogue visibly different from a chat viewer.

## Goal

A reader should see agent work, not serialized provider payloads. The launch slice must render at least these constructs through one shared intermediate contract:

1. shell command and structural result;
2. file edit or patch;
3. lineage/subagent relationship;
4. generic unknown-tool fallback.

Terminal and web surfaces must consume the same semantic card model rather than independently guessing from provider JSON.

## Owned scope

Own the semantic rendering contract, renderer registry/intermediate model, terminal renderer, focused web renderer integration, and their snapshots/tests. Avoid broad changes to ingestion, provider parsers, query grammar, storage, or demo fixture definitions unless a tiny fixture addition is indispensable and documented.

Coordinate with the demo-fixture lane by publishing the exact card inputs you need. Do not edit the same fixture files unless explicitly necessary.

## Design requirements

- Derive cards from normalized content blocks/actions, not prose regexes.
- Preserve exact evidence refs and source-order information.
- Render failure from structural fields such as nonzero exit status or provider `is_error`.
- Distinguish call, result, duration, status, and omitted output.
- Fold large output without hiding the existence or size of omitted material.
- Show file paths under the repository’s privacy/rendering policy.
- A lineage card must link parent/fork/subagent refs without treating domain topology as derivation provenance.
- Unknown tools must remain fully inspectable through a generic typed fallback.
- Terminal and HTML output must be deterministic for snapshots.
- The semantic model must be serializable so future MCP/web clients can consume it.

## Launch acceptance story

The deterministic “Receipts” session should render approximately as:

```text
$ pytest tests/missing_test.py
FAILED · exit 4 · 0.8s
ERROR: file or directory not found: tests/missing_test.py

Assistant response
I hit an error and need the missing path corrected before continuing.
```

The key claim is not styling. It is that the result is grounded in a typed tool-result block and retains a resolvable ref.

## Validation

- focused contract and snapshot tests;
- one cross-surface parity test over the same semantic model;
- unknown-tool fallback test;
- structural-failure test where prose does not contain the word “error”;
- output-folding boundary test;
- relevant MyPy/Ruff checks;
- one generated public-safe screenshot or text artifact.

## Deliverables

Produce a patch, semantic-card schema/example packet, before/after screenshots or terminal captures, test output, and a short note describing the next card types that remain outside the launch slice.

Favor a coherent vertical slice over broad partial support.

---

# Fork 03 — Polylogue flagship “The Receipts” deterministic demo

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-212.2`, the demo-corpus contracts, and related external-legibility Beads.

## Mission

Build a private-data-free flagship demonstration that answers one memorable question:

> The agent said the work succeeded. What actually happened?

The demo must prove structural evidence, not merely search for failure vocabulary.

## Owned scope

Own:

- deterministic scenario/fixture definitions needed for this story;
- demo seed and semantic verification logic;
- the flagship tour command or packet generator;
- focused tests and public-safe generated artifacts.

Do not own landing-page prose, generic semantic-renderer architecture, daemon behavior, or claims-site infrastructure. Coordinate through stable fixture IDs and a small output contract.

## Story

Construct one compact incident with these stages:

1. user asks an agent to make a change and verify it;
2. agent invokes a test command;
3. the structural result fails with a nonzero exit or typed provider error;
4. the assistant either acknowledges it or makes a deliberately bounded claim;
5. a later action or session supplies the repaired verification outcome;
6. a fork or continuation reuses prior context so physical and logical history differ;
7. the final report identifies the evidence chain and safe resume point.

Use provider-shaped fixtures and the normal parser/storage path. Do not insert normalized rows directly merely to make the demo easy.

## Required proof packet

The generated packet must include:

- question and bounded claim;
- exact fixture manifest;
- source/origin inventory;
- stable session/message/block/action refs;
- rendered claim and structural result adjacency;
- a structured action aggregation that grep cannot reproduce semantically;
- lineage view;
- independent fixture oracle;
- negative controls;
- explicit “does not prove” section;
- machine-readable JSON/YAML result;
- human-readable report;
- complete command transcript;
- regeneration command and artifact digests.

## Negative controls

At least:

1. include prose containing “error” with no failed structural result and prove it is not counted;
2. include a failed result whose output avoids the word “error” and prove it is counted;
3. remove the relevant tool-result block and require `not_supported` or a caveat rather than an inferred failure;
4. copy a lineage prefix and prove logical counting does not charge it as unique work.

## Timing and presentation

The first useful receipt should appear quickly. Do not open with a full construct-audit JSON dump. The full audit belongs in the machine-readable packet.

## Validation

Run focused fixture, seed, verify, CLI, snapshot, and public-artifact leak tests. Verify determinism from two fresh output roots. Record all observed timings without converting the fixture result into a real-scale performance claim.

## Deliverables

Produce the patch, generated demo packet, a concise tour transcript suitable for a GIF, test output, and a one-page construct-validity assessment.

---

# Fork 04 — Polylogue anti-grep proof card

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, particularly `polylogue-3tl.15` and existing demo/claims contracts.

## Mission

Create the smallest rigorous public proof that Polylogue performs a domain operation that grep or ordinary transcript search cannot perform.

Do not claim that grep is useless. Demonstrate a precise semantic distinction.

## Preferred construct

Use structural tool failure:

- a transcript with prose containing “error” but no failed tool outcome;
- a transcript with a structurally failed tool result whose output does not contain “error”;
- provider-normalized fields such as exit status or `is_error`;
- a query that returns the latter and excludes the former;
- exact evidence refs and source material drill-down.

A second optional card may show copied lineage and physical-versus-logical counts.

## Owned scope

Own a small deterministic fixture or reuse existing fixture IDs, one proof-card generator/page, focused tests, and generated public artifacts. Avoid changing global README copy, renderer architecture, or query grammar unless a true product defect blocks the proof.

## Independent oracle

The fixture manifest must declare expected structural outcomes independently of the query result. The proof fails if expected labels are derived from the same implementation path being tested.

## Output form

Produce a compact side-by-side artifact:

```text
Text search for “error”       → matches prose, misses semantic distinction
Polylogue structural query    → failed action(s), typed status, exact refs
```

Then explain the exact construct in no more than 200 words.

## Falsifiers

The card must fail when:

- the structural query includes the prose-only control;
- it excludes the failure-without-keyword control;
- evidence refs do not resolve;
- provider-specific fields leak into the public query contract;
- the artifact is regenerated from a stale or different fixture world.

## Validation and deliverables

Run focused parser/query/ref tests and determinism checks. Deliver a patch, HTML/Markdown card, machine-readable packet, test transcript, and the exact public claim wording suitable for the claims ledger.

---

# Fork 05 — Polylogue public claims ledger and findings shelf

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-3tl.4`, `polylogue-3tl.16`, proof-artifact documentation, and existing tracked field packets.

## Mission

Turn scattered proof artifacts into a public evidence shelf where every claim is bounded, classed, caveated, and resolvable.

## Owned scope

Own:

- machine-readable public claims schema/ledger;
- human-readable claims and findings pages;
- generators and drift checks;
- links from existing proof-artifact documentation;
- no changes to the underlying experiments or runtime behavior.

## Claim classes

At minimum distinguish:

- deterministic public proof;
- private-archive field observation;
- current product capability;
- performance observation;
- negative result;
- planned capability.

Every claim entry should carry:

- stable claim ID;
- exact wording;
- class;
- date and revision;
- corpus or fixture;
- product boundary;
- command/query;
- evidence artifacts and refs;
- result;
- caveats;
- unsupported interpretations;
- regeneration or verification path;
- Beads ownership where ongoing.

## Required findings

Index at least:

1. deterministic demo facts;
2. the claim-versus-evidence field finding with sampling/calibration caveats;
3. physical-versus-logical usage gap from the tracked live archive;
4. context/handoff pilot, including its stale/ahead negative case;
5. real-archive performance timeout or other negative operational finding;
6. honesty anti-demo returning unsupported/unavailable rather than guessing.

Do not convert private field observations into universal claims.

## Drift gate

Add a repository check that catches at least:

- missing artifact paths;
- duplicate claim IDs;
- claims with no caveat field where the class requires one;
- GitHub Issue roadmap links where Beads are authoritative;
- generated pages out of date with the ledger.

## Deliverables

Produce the patch, generated shelf, schema/ledger, validation output, and a short editorial note recommending which three findings deserve landing-page links.

---

# Fork 06 — Polylogue install matrix, release channel, and public media

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-3tl.7`, `.5`, `.9`, and `.10`.

## Mission

Make the first-run and first-impression path reproducible from a clean machine. Do not advertise installation channels that are not actually verified.

## Owned scope

Own packaging/install documentation, clean-environment verification scripts or CI matrix, VHS/media tapes and generated public-safe recordings, launch packet assembly, and media drift checks. Avoid changing product semantics or demo fixture definitions.

## Required install matrix

Determine and verify the truth of at least:

- Nix `nix run` path;
- source checkout plus `nix develop` path;
- `uv`/Python source-install path if supported;
- wheel/sdist build and install in a clean virtual environment;
- unsupported or future channels such as PyPI, Homebrew, OCI, or browser stores.

The README must label each channel as supported, experimental, planned, or unavailable. One clean path is better than five aspirational commands.

## Media

Regenerate a slow, comprehensible public recording from deterministic fixtures. It should show:

1. one-command tour;
2. structural failure receipt;
3. semantic aggregation;
4. lineage view;
5. bounded report with “does not prove.”

Avoid tiny text, rapid cuts, full-screen JSON dumps, and private paths. Store the source tape and regeneration command.

## Verification

- run the install paths in isolated roots or containers where available;
- verify the generated artifacts contain no absolute paths, usernames, secrets, private repositories, or volatile timestamps;
- compare generated media/artifact hashes or enforce a drift check;
- record exact environment and duration;
- make failure messages actionable.

## Deliverables

Produce a patch, install matrix report, launch media, release-channel status table, validation transcript, and a concise list of remaining blockers to a first tagged release.

---

# Fork 07 — Polylogue web reliability and truthful degraded states

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-0hqs` and `polylogue-bby.1`. This is the launch-blocking reliability lane.

## Mission

Prevent the web reader from hanging or presenting partial/stale data as an ordinary successful response. The product must return a bounded truthful result under slow archive queries.

## Owned scope

Own daemon HTTP query execution, request concurrency/timeout controls, cancellation and cleanup, response envelopes for degraded/timeout states, and the corresponding web UX. Avoid README/demo-fixture edits.

## Requirements

- bound concurrent archive-query execution;
- impose per-request timeouts with a clear server-side ceiling;
- ensure timed-out work does not continue consuming the shared executor indefinitely;
- return a typed degraded response carrying reason, elapsed time, partiality, and relevant readiness/frontier state;
- distinguish timeout, unavailable projection, stale index, invalid query, and internal failure;
- render slow/loading, partial, unavailable, and retry states visibly in the web reader;
- preserve request IDs and audit/ref information;
- do not silently retry a non-idempotent operation;
- prevent one pathological query from starving other requests.

## Test design

Use deterministic fault injection rather than wall-clock sleeps where possible. Cover:

- saturation of the worker bound;
- one timeout while a second cheap query succeeds;
- cancellation/cleanup;
- late worker completion not mutating an already-final response;
- browser rendering of each degraded state;
- no raw exception or private path leakage;
- metrics or reflection evidence for the timeout.

## Launch gate

Define a narrow public demo-route SLO and test it on the deterministic archive. Keep real-archive timings as field observations, not universal promises.

## Deliverables

Produce the patch, focused load/fault tests, a response-envelope example, screenshots of degraded UX, measured deterministic timings, and any residual risk requiring architectural follow-up.

---

# Fork 08 — Polylogue reviewed-resumption demonstration

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially the judged-context/memory program (`polylogue-37t` family) and relevant demo Beads.

## Mission

Build a deterministic demonstration of the difference between raw historical access, an unreviewed candidate, and reviewed evidence-backed memory.

The public claim must be modest:

> Polylogue can preserve candidate and reviewed assertions separately, compile a bounded context image with omissions and caveats, and record what was delivered to the next agent.

Do not claim generalized task uplift from one fixture.

## Owned scope

Own deterministic assertion/context fixtures, a resume/context demo command or packet, context-selection tests, and generated artifacts. Avoid changing the generic semantic renderer or landing page.

## Scenario

Create a project checkpoint containing:

- source evidence for a failed approach;
- a candidate lesson generated by an assistant;
- an operator-accepted correction or lesson;
- a stale or superseded assertion;
- one relevant evidence item omitted under a low token budget;
- a later agent context delivery.

The demo should show:

1. candidates are not injected by default;
2. accepted assertions can be selected when in scope and fresh;
3. stale/superseded material is excluded or caveated;
4. exact evidence refs and assertion refs are preserved;
5. the context image states omissions and lossiness;
6. a delivery snapshot records the final payload.

## Negative controls

- mark an unsupported assistant claim as high confidence; it must remain non-injectable;
- reduce the budget and require explicit dropped-segment reasons;
- supersede an accepted assertion and prove the old one no longer appears as current;
- remove the evidence ref and require a safety failure or caveat.

## Deliverables

Produce a patch, human-readable context packet, machine-readable context image/snapshot, focused tests, and a proposal for the later blinded resumption experiment. Clearly separate shipped capability from experimental outcome.

---

# Fork 09 — Polylogue physical-versus-logical lineage proof

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-4ts`, `polylogue-gjg`, and lineage demo work.

## Mission

Create a deterministic proof that physical transcript artifacts and logical work are different accounting units.

## Scenario

Build a small lineage graph containing:

- a parent session;
- a fork with a copied prefix and unique tail;
- a continuation;
- a fresh-context subagent;
- a compaction summary or explicit context boundary.

Assign simple token/usage numbers that permit an independently computed oracle.

## Claims to prove

- physical artifacts remain inspectable;
- inherited copied messages are composed into logical reads without being mistaken for new unique work;
- fresh subagent work remains distinct;
- compaction summary is a real context-boundary message, not a replacement for source evidence;
- physical totals, logical high-water totals, and provider/accounting views are named separately;
- every composed message resolves to its physical origin.

## Owned scope

Own lineage fixtures, the proof packet, focused composition/accounting tests, and optionally one visualization. Do not change broad pricing logic or unrelated provider parsers.

## Independent oracle

Store the expected graph, unique-tail sets, and arithmetic in a fixture manifest that is not generated by the production composition algorithm.

## Negative controls

- naïvely sum every physical transcript and show the overcount;
- treat the subagent as copied history and prove the oracle rejects it;
- remove an edge and require an incomplete/unknown caveat rather than an invented relation;
- introduce a cycle and require validation failure.

## Deliverables

Produce a patch, graph diagram, machine-readable receipt, arithmetic table, test output, and concise public explanation connecting the small proof with—but not generalizing from—the tracked live-archive replay gap.

---

# Fork 10 — Sinex public narrative, concepts, and agent relevance

Work directly on the supplied Sinex repository. Treat Beads—not retired GitHub Issues—as roadmap authority. Your mission is to expose the project’s distinctive evidence model before its service topology.

## Category

Use this consistently:

> Sinex is the local evidence substrate for digital life and agent work.

The project is not merely an activity logger, event bus, observability stack, data lake, knowledge graph, or vector memory.

## Owned scope

Own `README.md`, top-level public concepts/agent/demo documentation, public claims index, and documentation navigation. Do not change runtime behavior, database schema, source parsers, or demo seed code.

## Required narrative

Explain early:

- source material versus interpretation;
- `ts_orig`, `ts_coided`, and `ts_persisted`;
- replay as explicit reinterpretation rather than overwrite;
- occurrence identity versus interpretation identity;
- material provenance versus derivation provenance;
- projections versus authority;
- confidence versus judgment;
- coverage gaps as first-class results;
- activity evidence versus system reflection.

Then show the deployed architecture.

## Current demo honesty

Inspect the actual deterministic demo path. If the seed writes directly to PostgreSQL, say so. Distinguish the database/API smoke proof from source acquisition, NATS, event-engine, replay, and capture-completeness proofs.

## Polylogue direction

State the maximal target clearly:

- Sinex stores provider-native and normalized transcript material plus durable Polylogue-domain history;
- Polylogue owns AI-work semantics and product behavior;
- SQLite remains standalone and edge projection;
- generic NATS payloads and generic MCP remain privacy-bounded.

Do not repeat stale metadata-only doctrine as the ultimate architecture.

## Validation and deliverables

Run link checks, formatting/static checks available in the repository, and a retired-GitHub-issue-link scan. Deliver a patch, before/after explanation, claim inventory, validation output, and remaining public-story gaps.

---

# Fork 11 — Sinex flagship multi-source moment demo

Work directly on the supplied Sinex repository. Use Beads as roadmap authority, especially `sinex-cem.8`, the cross-source composite program, and coverage-honesty work.

## Mission

Build the flagship deterministic story:

> Reconstruct what happened around a failed build—including the source that went missing.

## Owned scope

Own a private-data-free multi-source fixture world, staging/admission harness, bounded moment-query product path, proof packet, and focused tests. Do not own the general README or unrelated source families.

## Fixture

Create one bounded interval with independently manifested occurrences from at least:

- terminal command and structural exit result;
- Git or repository change;
- browser research;
- active-window/focus or filesystem evidence;
- one deliberately interrupted or stale source.

All source material must enter through real source/admission paths rather than direct SQL insertion. If a source adapter is too expensive, implement the smallest reusable staged-material adapter rather than a demo-only database shortcut.

## Claim

Sinex joins the interval across heterogeneous sources, preserves native refs/material anchors, and reports the missing source segment as a coverage caveat rather than interpreting silence as inactivity.

## Independent oracle

A fixture manifest declares occurrence IDs, source-domain times, expected joins, and the deliberately missing segment. The query implementation must not generate its own labels.

## Negative controls

- remove browser material and require a coverage caveat;
- shift the interval and require the browser leg to disappear;
- disable a source and distinguish disabled from empty;
- delay one source and prove temporal-quality/late-arrival behavior;
- corrupt one material record and require bounded failure rather than partial success presented as complete.

## Artifacts

Produce a concise timeline, a source coverage panel, exact event/material refs, machine-readable result, complete command transcript, regeneration command, and “does not prove” section.

## Validation

Run from a fresh sandbox twice, compare semantic outputs, and preserve timing observations without claiming a general SLO. Deliver patch, packet, tests, and a cold-reader-ready narrative.

---

# Fork 12 — Sinex interpretation-revision and replay demo

Work directly on the supplied Sinex repository. Use Beads as roadmap authority, especially `sinex-cem.3`, `sinex-cem.14`, `sinex-908`, and derivation-control work.

## Mission

Build a deterministic proof that Sinex can change its interpretation without rewriting the observed occurrence or erasing prior history.

## Scenario

Use one small source record whose parser-v1 interpretation is intentionally wrong or incomplete. Parser/semantics v2 corrects it.

The demo must show:

- source material identity and exact anchor remain stable;
- `ts_orig` remains the source-domain occurrence time;
- the new interpretation receives a new event identity and coining time;
- old interpretation lifecycle is explicit;
- current projection moves to the corrected state;
- a semantic diff explains what changed;
- replay does not create another live occurrence;
- public refs can resolve current and historical interpretations.

## Owned scope

Own the fixture source/parser pair, replay invocation path, semantic-diff packet, focused lifecycle/projection tests, and generated artifacts. Avoid changes to unrelated source runtimes.

## Independent oracle

A hand-authored manifest specifies the source occurrence, expected v1 interpretation, expected v2 interpretation, and fields that must remain invariant.

## Negative controls

- overwrite the v1 row in place: test must fail;
- alter occurrence time on replay: test must fail;
- emit two live occurrences: test must fail;
- reuse event ID across replay: test must fail;
- lose material reachability: test must fail.

## Deliverables

Produce patch, before/after projection, interpretation history, semantic diff, source-material drill-down, machine-readable packet, command transcript, and focused test output.

---

# Fork 13 — Sinex self-diagnosing capture-outage demo

Work directly on the supplied Sinex repository. Use Beads as roadmap authority, especially `sinex-cem.2`, `sinex-jdp`, and `sinex-r6d`.

## Mission

Prove that Sinex can recognize when its own evidence is incomplete and propagate that fact into user-facing claims.

## Fault model

Choose one deterministic, meaningful capture fault such as:

- source cursor advanced before durable admission;
- sequence gap in a stream;
- stale runtime binding;
- interrupted staged material;
- inotify overflow or dropped segment;
- projection frontier behind admitted events.

Inject the fault through a supported test/fault boundary rather than editing database rows after the fact.

## Claim

The source-health and coverage surfaces identify the fault, quantify only what can actually be measured, and cause dependent queries to carry an explicit caveat or unavailable state.

Do not turn “sources with at least one error” into an event-loss percentage.

## Owned scope

Own the fault fixture/harness, coverage result, propagation into one query/read surface, focused tests, and proof packet. Avoid broad source cleanup unrelated to the selected fault.

## Independent oracle

Use known source sequence numbers, material manifest counts, or checkpoint barriers. The expected gap must be independent of the production coverage query.

## Negative controls

- healthy source produces no false outage;
- source disabled by policy is not labeled failed;
- an empty but healthy interval remains distinguishable from missing coverage;
- recovery does not erase the historical gap;
- query cannot silently downgrade the caveat.

## Deliverables

Produce patch, source-health view, affected-query view, machine-readable gap packet, recovery result, tests, and a concise public explanation of what the coverage metric means and does not mean.

---

# Fork 14 — Maximal Sinex-backed Polylogue ADR and wire protocol

Work across the supplied Polylogue and Sinex repositories. Use Beads as roadmap authority, especially `sinex-4j2` and the corresponding Polylogue integration decisions. Explicitly supersede any doctrine that says Sinex must ultimately remain metadata-only.

## Target authority

- Sinex is the canonical durable backend for provider-native transcript material, immutable normalized transcript material, durable transcript-domain history, judgments, lifecycle, and model effects.
- Polylogue owns provider normalization semantics, AI-work ontology, lineage and compaction, physical/logical accounting, reviewed memory, context compilation, query behavior, and product UX.
- SQLite remains Polylogue’s standalone backend and local/offline edge projection.
- Beads remains work-intent authority.

## Mission

Write an implementation-grade cross-repository ADR and versioned protocol skeleton. Do not implement the full data plane, but remove ambiguity sufficient for multiple implementation agents to proceed safely.

## Required design

1. Authority matrix by data class.
2. Stable object identity, domain revision identity, Sinex interpretation event identity, and material occurrence refs.
3. Provider-native and normalized material descriptors with multiple digests and canonicalization versions.
4. Immutable segment and transcript-revision manifest format.
5. Compact event vocabulary for session/message/tool/subagent/compaction/usage/assertion/judgment/context-delivery history.
6. Domain-local ordering fields.
7. Bundle settlement, expected counts/digests, and complete-revision frontier.
8. PostgreSQL projection ownership and SQLite replica/outbox behavior.
9. Replay and stable-ref semantics.
10. Privacy classes, raw-text capability, generic MCP redaction, and model-input policy.
11. Deletion and derived-cascade behavior, carefully distinguishing Polylogue topology from Sinex derivation provenance.
12. Shared model-effect/embedding recipe identity.
13. Standalone versus Sinex-backed mode and no-dual-master rule.
14. Full `polylogue rebuild --from-sinex` parity proof.

## Owned scope

Own new ADR/schema/example packet files and narrow comments in the current thin bridge. Avoid changing production runtime behavior unless needed to make the existing contract identify itself honestly as transitional.

## Validation

Validate schemas against examples, run Markdown/link checks, and produce at least three worked traces:

- initial transcript revision;
- append/resume plus replay under new parser semantics;
- offline user assertion reconciled into Sinex and back to SQLite.

## Deliverables

Produce patches for both repositories, protocol schemas/examples, a sequence diagram, migration phases, explicit superseded decisions, validation output, and a list of Beads that should be created or rewritten.

---

# Fork 15 — Joint “Resume This Bead” Agent Work Packet demo

Work across the supplied Polylogue and Sinex repositories. Use Beads as roadmap authority, especially `sinex-a4w.3.9`, the Polylogue coordination/context programs, and `sinex-4j2`.

## Mission

Design and, as far as the current substrates permit, implement the combined flagship demonstration:

> Resume one unit of agent work from intent, transcript evidence, machine effects, verification, and reviewed lessons.

## Domain rule

There is no intrinsic `session_commit` object. Sessions remain Polylogue objects; commits remain Git objects; checks remain telemetry or source-domain objects; Beads remain task objects. The Agent Work Packet is a replayable derived relation over native refs.

## Fixture

Create or adapt one private-data-free change episode containing:

- a Bead with intent, dependencies, and status transition;
- one parent agent session and one subagent/fork;
- typed tool calls/results including one failed and one successful verification;
- repository branch, file changes, commit(s), and check result;
- optional browser or terminal evidence outside the transcript;
- one accepted lesson and one unreviewed candidate;
- a context delivery to a resumed agent;
- one deliberately missing leg to exercise honest gaps.

## Packet contract

The output should include:

- packet ID and semantics version;
- intent refs;
- session and agent topology;
- repository/worktree/branch refs;
- toil and verification observations;
- artifact and source-material refs;
- outcome class and support vector;
- reviewed lessons versus candidates;
- coverage/freshness caveats;
- context image and delivery snapshot;
- provenance for every derived relation.

## Proof

Use independent fixture manifests and direct Beads/Git/command-state oracles. The demo should support a later blinded resumption duel but must not claim uplift yet.

## Owned scope

Prefer schemas, fixture world, derivation, packet renderer, and focused tests. Avoid broad changes to either generic query language or unrelated source parsers.

## Deliverables

Produce cross-repo patches or a standalone protocol/fixture package if implementation boundaries are not ready, one human-readable packet, machine-readable packet, evidence graph, context snapshot, tests, and a list of missing product primitives discovered by the exercise.

---

# Fork 16 — Launch conductor, integration audit, and cold-reader gate

Act as the integration and release conductor for the Polylogue and Sinex external-legibility swarm. Use Beads as roadmap authority. Do not implement a large feature unless it is the only way to unblock integration; your primary job is to make independent branches cohere and to reject ungrounded public claims.

## Inputs

Expect branches or patches from narrative, semantic rendering, demo, claims, install/media, web reliability, Sinex demo, and joint architecture lanes.

## Mission

Create a release-candidate integration branch and evidence packet through aggressive patch absorption, conflict resolution, and validation. Git history elegance is secondary. Semantic correctness and explicit ownership are not.

## Merge policy

Recommended order:

1. web reliability and trust-floor blockers;
2. narrative/docs registries;
3. fixture and demo substrate;
4. semantic rendering contract;
5. terminal/web renderers;
6. claims/findings generators;
7. install/media/launch artifacts;
8. Sinex narrative and demo contracts;
9. joint architecture docs/protocols.

Use squash or direct patch application freely. Never silently resolve two competing domain decisions; record the conflict and choose under the declared authority matrix.

## Required audit

1. **Category consistency:** one category sentence per project across README, site, package metadata, media, and launch packet.
2. **Claim reachability:** every public claim resolves to a ledger entry and evidence artifact.
3. **Construct validity:** each promoted demo has question, claim, construct, fixture/intervention, product boundary, oracle, negative controls, caveats, refs, and regeneration path.
4. **Current versus planned:** no roadmap feature described as shipped.
5. **Privacy scrub:** no secrets, usernames, private paths, hostnames, private session IDs, or unlicensed source text.
6. **Roadmap truth:** no GitHub Issue navigation where Beads are authoritative.
7. **Generated-artifact drift:** docs, pages, snapshots, media, and ledgers regenerated and clean.
8. **Cold install:** supported install path works in a clean environment.
9. **Cold reader:** an uninvolved reader can answer category, payoff, evidence versus derivation, one refused claim, current status, and evidence drill-down path.
10. **Backend direction:** no surviving text incorrectly says Sinex can never store transcripts in the ultimate design.

## Single-machine scheduling

Respect resource tokens:

- one heavy Rust build at a time;
- one browser/E2E lane at a time;
- one large Polylogue verification run at a time;
- cheap docs/static lanes can run in parallel;
- avoid concurrent package-manager mutation of the same cache or virtual environment.

## Deliverables

Produce:

- integrated patch/branch;
- launch manifest with commit/patch IDs;
- validation matrix and exact commands;
- public-claim diff;
- privacy scrub result;
- cold-reader questionnaire and results;
- accepted, rejected, and deferred branch list with reasons;
- final “ship / do not ship” decision and blocking Beads.

Reject impressiveness without evidence. Prefer one strong, reproducible story over a broad but incoherent showcase.
