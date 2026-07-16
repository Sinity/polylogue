# Polylogue presentability plan

## Objective

Make a first-time technical reader understand Polylogue, run a private-data-free proof, inspect a compelling evidence chain, and distinguish shipped capability from future ambition without reading the architecture corpus.

This is deliberately a launch cut, not a plan to finish the whole product. It uses existing Beads as the authority and narrows them into a small number of mergeable vertical slices.

## Release criterion

Polylogue is presentable when a cold reader can do all of the following:

1. State the category in one sentence: “a local flight recorder and system of record for AI work.”
2. Run one supported command from a clean checkout.
3. See a structural tool failure rendered differently from assistant prose.
4. Follow a headline claim to exact evidence.
5. Understand physical versus logical session history.
6. See one capability the system refuses to infer without evidence.
7. Understand the Sinex-backed direction without believing it is already shipped.
8. Find status, security, installation, demos, and limitations from the README.

## The launch cut

### Gate 0: do not build the launch on a visibly broken surface

**Beads:** `polylogue-0hqs`, `polylogue-bby.1`, verify the already-closed `polylogue-bby.7` regression remains closed.

- Bound archive-query concurrency and return a truthful timeout/degraded result.
- Give web routes visible loading, timeout, stale, and unavailable states.
- Run a synthetic concurrent-convergence test before recording browser media.
- Until this gate passes, the canonical launch demo remains terminal-based. Do not let an unreliable web reader become the first impression.

### Gate 1: make the category and claims coherent

**Beads:** `polylogue-3tl.12`, `polylogue-3tl.15`, `polylogue-3tl.16`, `polylogue-3tl.8`.

Deliver:

- one category phrase across README, package metadata, generated site, footer, and launch copy;
- a shorter README with one evidence story above architecture detail;
- a public claims ledger with `proven`, `capability`, `field_evidence`, `experimental`, `aspirational`, and `retired` states;
- a “why not grep?” proof card;
- no dead GitHub-issue planning language where Beads are authoritative;
- an explicit status block distinguishing standalone Polylogue from the planned Sinex backend.

The static patch in this kit provides a large part of this gate.

### Gate 2: make the ontology visible

**Bead:** `polylogue-ap7`, scoped to the launch vertical slice rather than the complete renderer universe.

Implement only the semantic cards needed by the flagship story first:

1. shell/tool call with command, exit status, bounded output, and evidence ref;
2. assistant claim as ordinary authored text, visually distinct from tool outcome;
3. file edit as a diff card;
4. fork/continuation context boundary;
5. unknown tool fallback that preserves all source data.

Use one provider-neutral renderer registry shared by CLI and web. Do not create parallel terminal-only and browser-only semantics. Snapshot the semantic intermediate representation, then snapshot each backend.

Acceptance is visual and structural: a stranger must see the difference before reading explanatory prose.

### Gate 3: replace breadth-first demo narration with The Receipts

**Beads:** `polylogue-212.2`, keep infrastructure from closed `polylogue-212.7`, use closed `polylogue-212.8` as the negative-claim companion.

- Extend the deterministic corpus with an explicit claim/result discrepancy, later repair, and negative controls.
- Make `polylogue demo tour` begin with the exact incident, not archive facets.
- Keep fixture verification machine-readable but remove its full JSON dump from the human transcript.
- Generate `report.md`, `report.json`, command outputs, a tape, and an evidence manifest.
- Add a one-screen “what this proves / what it does not prove” section.
- Programmatically resolve every public evidence ref.

The patch in this kit already improves the tour ordering and report narration; the richer semantic card remains the coding task.

### Gate 4: publish findings and proof cards

**Beads:** `polylogue-3tl.4`, `polylogue-3tl.16`; use existing D4 (`polylogue-212.4`) as a technical packet rather than a hero.

Publish a small shelf:

1. The Receipts — deterministic proof.
2. Why not grep? — deterministic comparison.
3. History Has Branches — deterministic lineage proof.
4. Silent continuation after structural failure — private-archive field evidence with sampling and calibration.
5. The Honest No — unsupported inference proof.

Every page carries a status badge, corpus, date, command, oracle, caveats, and refs.

### Gate 5: installation and launch assets

**Beads:** `polylogue-3tl.7`, `.5` (closed but regenerate against the new story), `.9`, `.10`.

- Prove clean-checkout commands in isolated environments.
- Cut a version/tag only after the install matrix is recorded.
- Regenerate one deliberate, readable recording.
- Ship a release notes template, HN copy, short announcement, long technical post outline, and screenshot alt text.
- Add a standing docs/media freshness gate so the launch does not immediately drift.

## Beads explicitly deferred from the first public cut

- `polylogue-212.3` cost by outcome: valuable, but outcome attribution and provider economics need more trust-floor work.
- `polylogue-212.5` live self-capture: impressive after daemon/web stability, not a launch prerequisite.
- `polylogue-212.6` full live continuation: context provenance should ship first; efficacy requires experiment.
- `polylogue-212.9` Fable-as-Foreman: rhetorical, not foundational.
- Full web evidence cockpit and every semantic tool card: ship a narrow coherent vertical slice before breadth.
- Autonomous or multi-agent claims: present the substrate, not unproven productivity outcomes.

## Recommended merge waves

### Wave A — static story, no shared runtime files

Parallel lanes:

- README/package/site language;
- concepts/FAQ/Sinex-backend docs;
- claims ledger and findings templates;
- demo portfolio and launch copy;
- install-matrix harness design.

These can merge with minimal conflicts if ownership is strict.

### Wave B — deterministic tour and fixture

One lane owns `polylogue/demo/`, `polylogue/scenarios/`, and demo tests. Another owns the semantic renderer intermediate representation. They exchange a checked-in fixture contract, not ad hoc messages.

### Wave C — renderer backends

Terminal and web backend agents work from the frozen semantic-card contract. A contract agent owns snapshots and rejects backend-specific semantic drift.

### Wave D — web reliability and media

Run only after Gate 0. One agent operates the live daemon/web proof; another records media from a fresh deterministic archive; a third audits every claim and link.

### Wave E — integration and release decision

A single integration captain owns the release branch, Beads updates, generated files, full validation, and the final claims ledger. Other agents submit checkpoint commits or patches and do not edit the release branch directly.

## Acceptance matrix

| Surface | Required proof |
|---|---|
| README | Cold-reader review plus link/command check |
| Site | Deterministic build; no zero-valued fake live stats; category matches README |
| Tour | Fresh archive, all commands pass, first useful result bounded, refs resolve |
| Semantic cards | Provider-neutral contract tests plus terminal/web snapshots |
| Claims ledger | Every quantitative/comparative claim has status and evidence packet |
| Install | Clean isolated environments, exact versions and failures recorded |
| Web | Concurrent convergence does not create unbounded requests; degraded response is visible |
| Security | Demo corpus only; no absolute paths, tokens, private archive refs, or source text leaks |
| Sinex story | Clearly labelled target architecture; no metadata-only doctrine presented as ultimate design |

## Public surface after the cut

The README should link only a small top-level path:

- Run the demo.
- See The Receipts.
- Understand the trust model.
- Read the architecture.
- Inspect proof artifacts.
- Understand the Sinex-backed future.
- Check status and limitations.

Everything else remains available in the documentation tree but stops competing for the first ten minutes.
