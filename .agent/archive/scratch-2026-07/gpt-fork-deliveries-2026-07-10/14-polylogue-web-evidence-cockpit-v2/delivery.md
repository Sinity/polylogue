## Turn 104 (role=assistant)

# Polylogue Web Evidence Cockpit v2 delivered

**[Download the complete Web Evidence Cockpit v2 kit](sandbox:/mnt/data/Polylogue-Web-Evidence-Cockpit-v2.zip)**  
[SHA-256 checksum file](sandbox:/mnt/data/Polylogue-Web-Evidence-Cockpit-v2.zip.sha256)

Start with [00-START-HERE.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/00-START-HERE.md), then use the [artifact map](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/19-artifact-map.md).

This pass is deliberately stricter than the earlier attempt. It separates four evidence classes throughout:

- what the supplied source code establishes;
- what actually executed in the deterministic environment;
- what the static prototype demonstrates;
- what remains target architecture or an experimental hypothesis.

The package contains a source-validated defect ledger, machine contracts, an interactive prototype, executed proof records, 15 redesigned demonstrations, 35 implementation packets, 24 parallel fork prompts, maximal Sinex-backend architecture, repository patches, swarm tooling, diagrams, screenshots, an editable deck, and a full program PDF.

## The central product decision

The browser should not be developed as a conventional transcript viewer or analytics dashboard.

It should be an **evidence cockpit** whose job is to help a human or agent make consequential decisions while understanding:

1. what happened;
2. what directly supports that account;
3. which parts are interpretations or projections;
4. which evidence is missing, stale, partial, timed out, or forbidden;
5. what the current installation may honestly claim;
6. how the result can be reproduced and inspected.

Its fundamental rule is:

> **A numeric zero is valid only after complete evaluation of the declared scope.**

This produces two related but deliberately separate state systems.

Operation results use:

```text
complete
partial
unavailable
timeout
forbidden
error
```

Component and evidence condition can independently be:

```text
ready
degraded
partial
stale
missing
refused
disabled
unknown
```

This separation matters. “Stale” describes evidence or a projection. “Timeout” describes an operation. “Disabled” describes configuration. “Forbidden” describes authority. “Zero” describes a completed count. None is a synonym for the others.

The full contract is in [03-truth-contract.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/03-truth-contract.md), with the machine-readable [Evidence Envelope schema](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/schemas/evidence-envelope.schema.json).

# What the source audit found

The audit was regenerated from the supplied Polylogue snapshot rather than copied from the earlier narrative. Every recorded source reference was subsequently resolved and line-validated against the supplied Git tree.

The main defects are not cosmetic.

### Result content and status presentation can disagree

The current surface can populate result content while separate status/count presentation remains in a checking or zero-like state. The target contract makes counts part of the same result envelope as the rows they describe.

### Disabled components can appear operational

Configuration, process liveness, successful observation, evidence availability, freshness, projection readiness, and claimability are currently too easy to conflate.

The rewrite treats them as independent facts. A component started with capture disabled must say **disabled**, not running, ready, or healthy.

### Attachment state is insufficiently precise

These are now distinct:

- metadata observed;
- bytes acquired;
- digest verified;
- bytes retrievable now;
- disclosure permitted;
- bytes deleted or tombstoned.

A historical acquisition record cannot by itself support a current download or preview claim.

### Advertised routes and client-recognized routes can diverge

A route exposed by navigation is a public contract. Every advertised route must:

- render successfully;
- redirect intentionally with a reason;
- or return an explicit unsupported/refused outcome.

Blank pages and parser disagreement are not acceptable route states.

### Some pages bypass shared bounded loading

Standalone attachment and paste paths can diverge from the main shell’s fetch, timeout, cancellation, caveat, and error behavior.

The target has one bounded evidence loader for all browser reads. No page may spin indefinitely or translate failure into emptiness.

### Cost and usage need lane-level provenance

A credible usage view must expose:

- fresh input;
- cache read and write;
- reasoning;
- ordinary output;
- provider observation;
- physical versus logical scope;
- pricing recipe and version;
- unknown or unsupported lanes;
- timeout or partial status.

It must not present a catalog estimate as an invoice or silently double-count inclusive cached input.

The full ledger is [02-source-and-truth-audit.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/02-source-and-truth-audit.md). Supporting artifacts include:

- [Source excerpts](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/audit/source-excerpts.md)
- [Route inventory](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/audit/route-inventory.csv)
- [Machine-readable defect ledger](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/data/defect-ledger.csv)
- [Source-reference validation](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/audit/source-ref-validation.md)

# Executed evidence

The environment probe and all its command records are preserved rather than summarized into an overly broad success statement.

- [Executed-proof status](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/executed-proof/STATUS.md)
- [Authoritative machine manifest](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/executed-proof/manifest.json)
- [Complete executed-proof directory](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/executed-proof/)

The manifest distinguishes commands that ran, routes that responded, failures, skips, environment limitations, and source-derived proposals.

A successful deterministic route proves only that route, revision, fixture, and environment. A failed probe remains in the record. Proposed fields not supplied by the current daemon remain labeled target fields.

# Interactive prototype

**[Open the static cockpit prototype](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/prototype/index.html)**

The prototype is dependency-free HTML, CSS, and JavaScript over the public-safe Incident 14:32 proof world. It contains seven evidence-centered views.

### Archive truth

Separates:

- component configuration and process state;
- successful evidence observation;
- source coverage;
- projection and index freshness;
- disclosure authority;
- known debt;
- current claimability.

### Claim receipt

Shows:

- the assistant claim;
- the structurally failed tool result;
- later successful verification;
- evidence strength;
- exact refs;
- relevant lineage;
- authority and caveats.

### Session chronicle

Shows:

- messages and semantic tool cards;
- physical versus composed history;
- copied-prefix lineage;
- fresh subagents;
- continuation and compaction boundaries;
- assertions and context events;
- incomplete or unavailable material.

### Cost and usage

Separates token lanes, physical/logical scope, provider observations, pricing recipes, and unsupported economic interpretations.

### Context autopsy

Reconstructs:

- selected evidence;
- accepted assertions;
- generated context;
- omitted material;
- token and message budgets;
- stale items;
- lossiness;
- the actual historical delivery boundary.

### Material lifecycle

Demonstrates metadata-only, retrievable, missing, forbidden, and deleted material without collapsing them into an `acquired` Boolean.

### Agent Work Packet

Links Beads intent, Polylogue sessions, subagents, repository state, commands, verification, context, judgments, coverage, and unresolved gaps while preserving each source’s native ref and authority.

Visual artifacts:

- [Cockpit montage](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/previews/cockpit-montage.png)
- [Architecture diagram](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/diagrams/architecture.svg)
- [Truth-state diagram](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/diagrams/truth-state.svg)
- [Journey map](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/diagrams/journey-map.svg)
- [Maximal backend diagram](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/diagrams/maximal-backend.svg)

The prototype defines target behavior and information hierarchy. It is not presented as evidence that the present daemon already implements every field.

# Demonstration portfolio completely reconsidered

The revised portfolio is in [09-demo-portfolio.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/09-demo-portfolio.md). There are 15 machine-readable manifests under [demo-manifests](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/demo-manifests/).

Every demo must specify:

- a falsifiable claim;
- a consequential decision;
- independent ground truth;
- the strongest plausible simpler baseline;
- positive, negative, and refusal controls;
- a concrete falsifier;
- exact evidence refs;
- reproduction identity;
- machine and human artifacts;
- threats to validity;
- non-claims.

The redesigned portfolio is:

1. **Receipts in Browser** — claim versus structural result and later repair.
2. **Zero Is a Claim** — complete-empty versus partial, unavailable, timeout, and forbidden.
3. **Disabled Means Disabled** — configuration cannot masquerade as evidence readiness.
4. **Missing Bytes Are Not Acquired** — material lifecycle and current retrievability.
5. **Every Advertised Route Has a Truthful Outcome** — route contract parity.
6. **Count It Once, Preserve It All** — copied lineage versus fresh subagents.
7. **Cost Ledger Under Poisoned Inputs** — disjoint usage lanes and non-invoice economics.
8. **Context Autopsy** — historical delivery, omission, authority, and loss.
9. **Judgment Changes Authority, Not History** — reviewed memory lifecycle.
10. **The System Changes Its Mind Honestly** — replay and interpretation history.
11. **World Around the Claim** — the joint Agent Work Packet.
12. **Rebuild Cockpit from Sinex** — backend parity rather than metadata mirroring.
13. **Resume Under Oath** — raw-reference, generated-summary, and reviewed-context arms.
14. **Forgetting Propagates** — cross-store deletion and residual accounting.
15. **Giant Session, Bounded Truth** — load behavior without silent truncation or false zero.

The flagship fixture deliberately contains anti-grep, poisoned-accounting, stale-memory, missing-source, copied-lineage, and refusal controls. It is designed so that an attractive but semantically weak implementation fails.

Supporting contracts:

- [Demo Packet schema](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/schemas/demo-packet.schema.json)
- [Proof report template](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/templates/PROOF-REPORT.md)
- [Fault matrix](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/data/fault-matrix.csv)
- [Demo validator](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/validate-demo-packets.py)

# Rapid implementation program

The program contains **35 bounded work packages** across six waves:

```text
Wave 0 — contracts, red tests, fixture and independent oracle
Wave 1 — truth floor
Wave 2 — evidence primitives and highest-value journeys
Wave 3 — judgment and Sinex substrate
Wave 4 — accessibility, disclosure, fault, load and rebuild proof
Wave 5 — adversarial integration and launch
```

The complete plan is [10-implementation-program.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/10-implementation-program.md), with the machine graph in [work-packages.json](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/work-packages/work-packages.json).

## Packet P01 is the launch gate

`P01` implements `polylogue-bby.1`’s truthful slow and missing behavior before any major page polish.

It must establish:

- complete, partial, unavailable, timeout, forbidden, and error;
- no zero for unevaluated evidence;
- bounded compatibility for old routes;
- caveats and frontiers;
- exact refs;
- reproduction identity;
- shared CLI/web domain semantics.

Visual work is not allowed to invent substitutes while that contract is absent.

The next parallel truth-floor packets are:

- component truth;
- material lifecycle;
- route registry and capabilities;
- bounded browser loader;
- source/projection/index frontiers.

Only then do receipt, chronicle, usage, archive-truth, context, material, and judgment pages assemble over common primitives.

Existing Beads remain planning authority. The work packages are a rapid-execution grouping, not a replacement planning system. The resolved subset from the supplied snapshots is in [analysis/beads-subset.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/analysis/beads-subset.md).

# Single-machine frontier-agent swarm

The complete operating system is in [11-single-machine-swarm.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/11-single-machine-swarm.md).

Four captains centralize the things that cannot safely fork:

- **Interface captain:** result contracts, refs, routes, truth vocabulary.
- **Integration captain:** final worktree and ruthless conflict collapse.
- **Proof and claims captain:** oracle, Demo Packets, claims and non-claims.
- **Sole Beads captain:** the only planning writer during the campaign.

Implementation workers receive strict path ownership and run focused tests. Broad tests, generated surfaces, media, and final proof run only from integration.

Git history is deliberately expendable. The integration captain may:

- cherry-pick;
- extract only owned files;
- copy a known-good path;
- squash;
- discard a noisy branch;
- or reimplement a smaller compatible change.

What cannot be sacrificed is:

- interface compatibility;
- exact test evidence;
- integrated revision identity;
- proof artifacts;
- claim discipline.

The [first 72-hour cut](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/15-first-72-hours.md) is designed to produce one narrow, defensible cockpit path rapidly.

Coordination utilities include:

- [Worktree bootstrap](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/bootstrap-worktrees.sh)
- [Swarm board](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/swarm-board.py)
- [Heavy-operation lock](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/with-heavy-lock.sh)
- [Resource planner](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/resource-plan.sh)
- [tmux launcher](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/scripts/launch-tmux.sh)
- [Worker handoff](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/templates/WORKER-HANDOFF.md)
- [Interface decision](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/templates/INTERFACE-DECISION.md)
- [Cold-reader report](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/templates/COLD-READER-REPORT.md)

# Twenty-four parallel fork prompts

The prompts are indexed in [12-fork-prompts.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/12-fork-prompts.md). A single copy/paste document is available as [12a-all-fork-prompts.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/12a-all-fork-prompts.md).

They cover:

1. interface captaincy;
2. current-source regression conversion;
3. Incident 14:32 and its independent oracle;
4. result envelope;
5. component truth;
6. material lifecycle;
7. route registry;
8. bounded browser loading;
9. evidence UI primitives;
10. semantic tool cards;
11. claim-to-receipt journey;
12. session chronicle and lineage;
13. honest usage ledger;
14. archive truth;
15. context autopsy;
16. assertion and judgment;
17. accessibility;
18. browser fault/load proof;
19. disclosure security;
20. public claims and documentation;
21. transcript-complete Sinex material;
22. stable identity, settlement, and rebuild;
23. Agent Work Packet;
24. ruthless integration and release audit.

Each prompt has owning Beads, dependencies, strict paths, acceptance criteria, verification, shared constraints, and a required handoff.

# Maximal Sinex backend

The complete architecture and impedance analysis is [08-sinex-polylogue-backend-fit.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/08-sinex-polylogue-backend-fit.md).

The target authority split is:

### Sinex stores

- provider-native transcript artifacts;
- immutable Polylogue-normalized segments;
- attachments and context-image bytes;
- durable session, message, tool, usage, assertion, judgment, and delivery history;
- acquisition, coverage, replay, retention, and deletion state;
- model effects;
- ambient machine evidence;
- cross-source Agent Work Packet components.

### Polylogue owns

- provider normalization;
- AI-work ontology;
- message and block semantics;
- physical and logical session composition;
- continuation, fork, fresh-subagent, and compaction meaning;
- tool-result interpretation;
- usage and cost semantics;
- assertion and context policy;
- transcript and evidence UX;
- domain query behavior.

### SQLite remains

- standalone authority when Sinex is absent;
- edge replica in backed mode;
- local FTS/vector acceleration;
- offline cache;
- pending-write outbox;
- local UI state;
- deterministic fixture and test substrate.

The decisive rule is:

> **In Sinex-backed mode, no irreplaceable transcript evidence or durable user judgment should exist only in Polylogue SQLite.**

The analysis addresses the principal impedance mismatches:

- replay event identity versus stable citation identity;
- mutable sessions versus immutable observations;
- rewritten material and unstable byte offsets;
- session topology versus derivation provenance;
- physical evidence versus logical work;
- event granularity;
- revision settlement;
- timestamp versus causal ordering;
- differing query and schema models;
- assertion and judgment overlap;
- context artifact ownership;
- embedding recipe identity;
- SHA-256 versus BLAKE3 content descriptors;
- disclosure versus storage authority;
- deletion through shared lineage;
- offline dual-master risk;
- combined MCP tool-surface overload.

The target joint machine object is defined in the [Agent Work Packet schema](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/schemas/agent-work-packet.schema.json).

# Repository patches

## Polylogue

[Download the Polylogue cockpit starter patch](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/patches/polylogue-web-evidence-cockpit-starter.patch)

It adds:

- cockpit entry-point documentation;
- the truth contract;
- the Evidence Envelope schema;
- the interactive static prototype;
- a static web-contract audit developer tool;
- demonstration doctrine;
- explicit non-claims.

## Sinex

[Download the Sinex backend starter patch](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/patches/sinex-polylogue-cockpit-backend-starter.patch)

It adds:

- the transcript-complete backend direction;
- stable identity and settlement requirements;
- protected material versus transport distinction;
- SQLite’s backed-mode role;
- the Agent Work Packet schema;
- the decisive rebuild proof.

Both patches were checked against the exact supplied Git bases. Application instructions and base revisions are in [patches/README.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/patches/README.md).

They are intentionally additive starter patches. They do not pretend that the runtime rewrite or Sinex reconstruction has already landed.

# Presentation artifacts

- [Executive PowerPoint](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/Polylogue-Web-Evidence-Cockpit.pptx)
- [Executive deck PDF](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/Polylogue-Web-Evidence-Cockpit.pdf)
- [Full program PDF](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/Polylogue-Web-Evidence-Cockpit-Program.pdf)
- [Full program HTML](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/Polylogue-Web-Evidence-Cockpit-Program.html)
- [Deck contact sheet](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/previews/deck-contact-sheet.jpg)

# Validation

The final checker passed the distribution package. Its coverage includes:

- required artifacts;
- JSON and YAML syntax;
- JSON Schema validity;
- claims-ledger and work-package schema validation;
- Python, JavaScript, and shell syntax;
- acyclic work-package dependencies;
- 24 complete fork prompts;
- D1–D15 manifest coverage;
- package-local Markdown links;
- prototype rendering and screenshots;
- PowerPoint ZIP integrity;
- PDF integrity;
- source `file:line` resolution;
- exact-base patch application;
- credential and private-key residue heuristics;
- ZIP integrity;
- per-file SHA-256 manifest generation.

The complete record is [18-validation-report.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/18-validation-report.md), and the per-file checksums are in [PACKAGE-MANIFEST.json](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/PACKAGE-MANIFEST.json).

The most important boundary remains [17-NON-CLAIMS.md](sandbox:/mnt/data/polylogue-web-evidence-cockpit-v2/17-NON-CLAIMS.md): the package does not represent the production truth floor, semantic renderer, transcript-complete Sinex gateway, full SQLite reconstruction, selective forgetting, large-archive performance, or general agent uplift as completed merely because the design and proof machinery now exist.

---

