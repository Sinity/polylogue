# Web Evidence Cockpit v2 — GPT-Pro fork delivery summaries (recovered from capture)

Source: chatgpt conversation 6a5112f3-7798-83eb-b1ee-96ef29477c12, captured 2026-07-10.
The package FILES are unrecoverable (sandbox container dead); these are the final
delivery messages describing what was built. Treat as design input, not authority.


---

## Assistant turn 15

Web Evidence Cockpit program delivered
The complete package is ready:
Download the Polylogue Web Evidence Cockpit Kit
SHA-256 checksum
Start with
00-START-HERE.md, then use the
18-artifact-map.md.
This is substantially more than a visual redesign. The package contains:
a source-audited diagnosis of the present Polylogue web and daemon boundary;
an explicit truth contract for every visible result state;
a complete target information architecture;
a working static cockpit prototype with six evidence-centered views;
rigorous demonstration designs with independent oracles and falsifiers;
32 implementation work packages;
a single-machine swarm and worktree operating system;
20 detailed fork prompts;
apply-ready Polylogue and Sinex repository patches;
maximal Sinex-backed architecture;
JSON schemas for result envelopes and Agent Work Packets;
diagrams, screenshots, a slide deck, and a full program PDF;
an adaptive black-box probe and validation packet.
The central design decision
The cockpit should not be a conventional analytics dashboard.
Its purpose is to help a person or agent make consequential decisions from evidence while understanding the limits of that evidence.
Every important view should answer:
What happened?
What directly supports that account?
Which parts are interpretations or projections?
What is missing, stale, partial, unavailable, or forbidden?
What can this installation honestly claim right now?
How can the result be reproduced and inspected?
This leads to a product category closer to an evidence cockpit than a transcript reader or dashboard.
A session page is not “messages plus metadata.” It is an explanation of what occurred, what the assistant claimed, what tools structurally returned, how the session relates to forks and subagents, what context was delivered, and what remains uncertain.
An archive page is not a collection of green health chips. It explains whether components are configured, running, producing evidence, fully captured, materially available, projected, current, and eligible to support claims.
A joint Polylogue–Sinex page is not an event dump. It reconstructs the relationship among agent intent, transcript activity, machine effects, repository changes, task state, verification, source coverage, and outcome.
The complete argument is in the
01-executive-design.md.
The present truth defects
The source audit found several issues that cannot be solved through styling.
Disabled components can appear ready
A daemon started with watching and browser capture explicitly disabled can nevertheless report those components as running or ready. This conflates several distinct states:
configured
running
successfully observed
currently producing evidence
complete enough for a query
fresh enough for a claim
A process can be alive while a source is unavailable. A source can be configured but intentionally disabled. A projection can be current relative to incomplete evidence. None of those conditions should result in the same green chip.
“Acquired” can coexist with missing bytes
The deterministic demo verifier can recognize an attachment-acquisition construct while an attachment returned through the API has no retrievable blob.
That means at least three states have been collapsed:
attachment metadata was observed;
attachment bytes were acquired;
attachment bytes remain retrievable now.
The target cockpit treats these as separate material-lifecycle facts.
Advertised routes are not all client-resolvable
The server advertises /w/tabs and /w/timeline, while the client-side URL state parser implements stack and compare but not those advertised modes.
A route exposed through navigation is a public contract. It must either:
resolve and render successfully;
return an explicit unsupported capability result;
or be removed from navigation.
A route should not fail merely because the server and client disagree about whether it exists.
Some pages bypass bounded result handling
Standalone attachment and paste pages do not use the same bounded fetch, timeout, caveat, and error machinery as the main shell.
This permits divergent semantics:
one page may render a timeout as an error;
another may spin indefinitely;
another may show an empty state;
another may expose an unbounded payload.
The target architecture has one evidence-aware loader for every route.
The complete diagnosis is in the
02-source-and-truth-audit.md. The adaptive probe, exact environment status, and generated observations are preserved under the
README.md and its
manifest.json.
The truth contract
The most important implementation artifact is the
03-truth-contract.md.
It defines six top-level result states:
complete
partial
unavailable
timeout
forbidden
error
An empty result is valid only when the operation is complete for its declared scope and the evidence supports actual absence.
These are materially different:
complete + zero results
    No matching evidence exists in the fully evaluated scope.
partial + zero currently known results
    No matches were found in the portion that could be evaluated.
unavailable
    The required evidence or projection does not exist.
timeout
    The operation did not finish within its budget.
forbidden
    The evidence exists or may exist, but the caller lacks authority.
error
    The operation failed and no honest result can be returned.
Every result envelope can carry:
exact evidence refs;
source-material state;
projection state;
structured caveats;
truncation and continuation information;
time and row budgets;
reproduction identity;
disclosure state;
authority;
freshness;
claimability.
The machine contract is provided as an
evidence-envelope.schema.json, with a
evidence-envelope.example.json.
The aim is not to wrap every API response in decorative metadata. It is to make silent incompleteness structurally impossible at the product boundary.
Interactive cockpit prototype
The static prototype is here:
index.html
It has six deliberately different evidence views:
Session
A claim-versus-evidence receipt showing:
the assistant’s claim;
the structurally failed tool result;
the later successful verification;
evidence strength;
exact refs;
relevant lineage;
accepted and candidate assertions;
coverage caveats.
Archive truth
A system-wide view separating:
process state;
source-material state;
projection state;
freshness;
known debt;
current claimability.
It replaces undifferentiated green health chips with answers about what evidence can actually support.
Agent Work Packet
A joint Polylogue–Sinex view connecting:
Beads intent;
agent sessions and subagents;
repository and worktree;
commits and files;
commands and test outcomes;
ambient machine evidence;
context deliveries;
reviewed lessons;
unresolved gaps.
Context autopsy
A reconstruction of what an agent was actually given:
selected segments;
evidence and assertion refs;
omissions;
budget pressure;
lossiness;
stale material;
context boundary and inheritance mode.
Material lifecycle
A refusal-oriented page demonstrating that observed metadata is not proof that bytes were acquired or remain retrievable.
Overall cockpit shell
A shared navigation and result-state system designed around stable evidence refs rather than surface-specific identifiers.
A visual overview is available as the
Share
.
Individual views:
The prototype is intentionally static. It defines target behavior and information hierarchy without pretending that the current daemon already supplies every required field.
Information architecture
The proposed cockpit is organized around evidence objects and decisions rather than storage tables.
Primary objects include:
Session
Message
Content block
Tool invocation
Tool result
Material
Attachment
Assertion
Judgment
Context delivery
Lineage edge
Run
Observed event
Finding
Agent Work Packet
Primary public views include:
/w/archive
/w/session/<ref>
/w/receipt/<ref>
/w/context/<ref>
/w/material/<ref>
/w/assertion/<ref>
/w/finding/<ref>
/w/work-packet/<ref>
/w/compare
/w/timeline
The shell does not own a second object identity system. Stable Polylogue and Sinex refs remain canonical; URLs are navigational representations of those refs.
The full design is in the
04-information-architecture.md. Field-level and route-level contracts are provided separately:
05-current-to-target-contract-matrix.md
05-screen-field-contracts.csv
05-route-contracts.csv
Maximal Sinex-backed architecture
The cockpit design adopts the maximal backend direction rather than preserving the current metadata-thin boundary.
In backed mode:
Sinex stores:
provider-native transcript artifacts;
normalized transcript segments;
attachment and context-image bytes;
durable session, message, tool, usage, assertion, judgment, and delivery history;
source coverage and lifecycle state;
model effects;
cross-domain machine evidence;
Agent Work Packet components;
deletion and retention state.
Polylogue owns:
provider normalization semantics;
the AI-work ontology;
physical and logical session composition;
fork, continuation, subagent, and compaction meaning;
tool-call and tool-result interpretation;
token and cost semantics;
assertion and memory policy;
context compilation;
transcript and evidence UX;
domain-specific query behavior.
SQLite remains:
the authority in standalone mode;
an edge replica in Sinex-backed mode;
local FTS and vector acceleration;
an offline cache;
a pending-write outbox;
local UI and presentation state;
a deterministic test and demonstration substrate.
The decisive rule is:
In Sinex-backed mode, no irreplaceable transcript evidence or durable user judgment should exist only in Polylogue SQLite.
The architecture and mismatch analysis are in
07-sinex-polylogue-cockpit-fit.md. The proposed joint machine contract is an
agent-work-packet.schema.json.
The key design protections are:
domain topology is not encoded as derivation provenance;
stable Polylogue object identity is separate from replay-specific Sinex event identity;
transcript revisions have explicit completeness and settlement;
physical evidence remains inspectable while logical work is counted once;
storage authority is distinct from disclosure authority;
generic Sinex MCP access does not imply arbitrary raw-transcript access;
context delivery is immutable historical evidence, not a live query alias.
Redesigned demonstration portfolio
The proposed demonstrations are in
08-demo-portfolio.md.
Each demonstration must include:
a falsifiable claim;
a consequential user decision;
independently obtained ground truth;
the strongest plausible simpler baseline;
positive, negative, and refusal controls;
a stated falsifier;
machine-readable receipts;
exact public refs;
a non-claims section;
reproduction identity.
The ten proposed demonstrations are:
D1 — Receipts in Browser
Can the cockpit distinguish an assistant success claim from a structurally failed tool result and later verified repair?
The negative control contains prose with words such as “error” but no failed operation. This prevents the demo from being a disguised keyword search.
D2 — Disabled Means Disabled
Start the daemon with optional live components disabled. The UI and API must show disabled, not running, ready, or healthy.
D3 — Missing Bytes Are Not Acquired
Present three attachments:
metadata only;
bytes acquired and retrievable;
bytes once acquired but now unavailable.
The cockpit must not collapse them into one acquired state.
D4 — Every Advertised Route Works
Enumerate every public route and navigation target. Each must render, redirect intentionally, or return an explicit capability refusal.
D5 — Timeout Is Not No Results
Force a bounded operation to exceed its budget. The view must report timeout or partial completion rather than an empty archive.
D6 — Count It Once, Preserve It All
Show a copied-prefix fork and a fresh subagent. Every physical artifact remains resolvable, copied work is counted once logically, and the fresh subagent remains distinct.
D7 — Context Autopsy
Given a frozen agent turn, reconstruct exactly which evidence and assertions were delivered, what was omitted, and why.
D8 — Judgment Changes Authority, Not History
Accept, reject, defer, supersede, and stale several assertions. The current authority changes while the historical proposal and judgment record remains inspectable.
D9 — World Around the Claim
Start from an agent claim and recover transcript evidence, terminal activity, Git effects, Beads intent, verification outcome, and source gaps.
D10 — Rebuild Cockpit from Sinex
Discard rebuildable Polylogue state, reconstruct it from Sinex-held material and durable domain history, and compare the result against the declared parity contract.
The shared fixture is the public-safe Incident 14:32 proof world: a premature success claim, structured failure, later repair, copied lineage, fresh subagent, compaction omission, attachment variants, reviewed and stale memory, context delivery, ambient machine evidence, a source outage, and two parser-semantics versions.
Machine-readable demo contracts include:
demo-packet.schema.json
D2-disabled-means-disabled.yaml
D3-missing-bytes.yaml
Implementation program
The implementation plan contains 32 bounded work packages across six waves. See the
10-implementation-program.md and
10-work-packages.json.
Wave 0 — Freeze the contracts
Before polishing screens:
ratify the result envelope;
freeze the public route inventory;
record the current defects as failing tests;
build Incident 14:32 and independent oracles;
establish stable ref and reproduction rules.
Wave 1 — Establish the truth floor
Fix:
disabled-component reporting;
material lifecycle vocabulary;
timeout and partial-result behavior;
projection and source frontiers;
route capability reporting;
shared bounded fetch behavior.
No green chip should survive this wave unless its meaning is explicit.
Wave 2 — Build reusable evidence primitives
Implement:
evidence-ref links;
caveat and refusal panels;
completeness/freshness/authority badges;
claim-versus-result receipts;
semantic shell, edit, search, and file cards;
lineage and context-boundary cards;
shared result-state handling.
Wave 3 — Assemble the cockpit
Land:
session evidence view;
archive truth view;
context autopsy;
material inspector;
lineage explorer;
findings and receipt views;
responsive and keyboard-accessible navigation.
Wave 4 — Judgment and Sinex integration
Implement:
assertion proposal and judgment surfaces;
context-policy controls;
transcript-complete Sinex gateway;
revision settlement and projection frontiers;
Agent Work Packets;
Sinex-backed rebuild proof.
Wave 5 — Adversarial launch proof
Complete:
accessibility audit;
disclosure and raw-text capability audit;
browser and API performance budgets;
interruption and recovery tests;
all ten demonstration proof packets;
adversarial claims review;
cold-reader evaluation.
The
16-first-48-hours.md is designed to establish the truth floor and launch parallel implementation without waiting for a long architecture phase.
Single-machine frontier-agent swarm
The
11-single-machine-swarm.md trades branch elegance for rapid integration.
It uses four captains:
Interface captain — owns contracts, refs, route vocabulary, and cross-worker decisions.
Integration captain — owns the integration worktree and may cherry-pick, copy paths, extract patches, or discard noisy branch history.
Proof and claims captain — owns oracles, demos, public wording, and non-claims.
Sole Beads captain — is the only worker that updates the authoritative Beads graph during the campaign.
The machine should run:
many lightweight documentation, fixture, and static-analysis lanes;
one or two browser lanes;
one heavy Rust or broad-test lane behind a machine lock;
focused tests in worker worktrees;
broad tests and generated-surface refreshes only in integration.
The kit includes working coordination tools:
bootstrap-worktrees.sh
with-heavy-lock.sh
swarm-board.py
resource-plan.sh
launch-tmux.sh
WORKER-HANDOFF.md
INTERFACE-DECISION.md
PROOF-REPORT.md
Workers communicate through explicit interface decisions, proof reports, stable fixture contracts, and file ownership—not through assumptions about what another agent probably implemented.
Twenty parallel fork prompts
The kit contains 20 substantial prompts that can be run as parallel forks of this conversation:
12-fork-prompts.md
Individual prompt files
They cover:
interface and contract captaincy;
present-state truth audit and regression tests;
Incident 14:32 fixture construction;
daemon component truth;
attachment and material lifecycle;
shared bounded loader;
route capability and navigation parity;
timeout and partial-result semantics;
evidence UI primitives;
semantic tool cards;
session and receipt page;
archive truth page;
lineage and context autopsy;
assertion judgment surfaces;
maximal Sinex–Polylogue contract;
Sinex gateway and rebuild;
Agent Work Packet;
accessibility and disclosure security;
performance and browser E2E;
adversarial integration and claims audit.
Each prompt contains scope, constraints, relevant architecture, required controls, expected artifacts, verification commands, and handoff requirements. They are intended to produce mergeable work rather than disconnected design essays.
Apply-ready repository proposals
Polylogue
Polylogue Web Evidence Cockpit starter patch
It adds:
cockpit architecture documentation;
truth and demonstration contracts;
a machine-readable evidence-envelope schema;
the static prototype;
a web-surface contract-audit developer tool;
documentation and README entry points.
It deliberately does not claim to repair the runtime defects. Those changes require focused tests and implementation through the work packages.
Sinex
Sinex backend and cockpit architecture patch
It adds:
the transcript-complete maximal backend direction;
cockpit support contracts;
the Agent Work Packet schema;
documentation entry points.
It changes architecture contracts and documentation, not Sinex runtime behavior.
Patch bases and application notes are in
README.md and
bases.json.
Presentation and review artifacts
Polylogue-Web-Evidence-Cockpit.pptx
Polylogue-Web-Evidence-Cockpit.pdf
Polylogue-Web-Evidence-Cockpit-Program.pdf
Polylogue-Web-Evidence-Cockpit-Program.html
architecture.svg
truth-state.svg
swarm.svg
17-review-checklist.md
13-claims-ledger.yaml
14-decision-records.md
What is complete, and what is not
Complete in this package:
source-level web and daemon contract audit;
target product architecture;
truth-state and result-envelope design;
field and route contracts;
static interactive prototype;
screenshots and presentation materials;
rigorous demo redesign;
Sinex maximal-backend fit analysis;
work packages, prompts, and swarm tooling;
documentation and architecture patches;
machine-readable schemas;
adaptive probe and validation machinery;
packaged and checksummed deliverables.
Not represented as completed runtime work:
correction of the daemon’s disabled-component readiness reporting;
unified bounded loading across all current pages;
complete route-capability parity;
attachment lifecycle repair;
the production semantic renderer;
assertion judgment mutations;
a live transcript-complete Sinex gateway;
Sinex-backed Polylogue reconstruction;
selective forgetting across both systems;
general memory or agent-performance uplift.
The patches intentionally avoid disguising plans as implementation. The exact package validation record is in
15-validation-report.md, with the concise
15-validation-status.txt.
The strongest immediate execution path is the
16-first-48-hours.md: freeze the envelope and routes, turn the discovered truth defects into regressions, build Incident 14:32, and launch the truth-floor, loader, material, route, and evidence-primitives lanes in parallel.
index.html
Polylogue-Web-Evidence-Cockpit-Program.html

---

## Assistant turn 16

Polylogue Web Evidence Cockpit v2 delivered
Download the complete Web Evidence Cockpit v2 kit
SHA-256 checksum file
Start with
00-START-HERE.md, then use the
19-artifact-map.md.
This pass is deliberately stricter than the earlier attempt. It separates four evidence classes throughout:
what the supplied source code establishes;
what actually executed in the deterministic environment;
what the static prototype demonstrates;
what remains target architecture or an experimental hypothesis.
The package contains a source-validated defect ledger, machine contracts, an interactive prototype, executed proof records, 15 redesigned demonstrations, 35 implementation packets, 24 parallel fork prompts, maximal Sinex-backend architecture, repository patches, swarm tooling, diagrams, screenshots, an editable deck, and a full program PDF.
The central product decision
The browser should not be developed as a conventional transcript viewer or analytics dashboard.
It should be an evidence cockpit whose job is to help a human or agent make consequential decisions while understanding:
what happened;
what directly supports that account;
which parts are interpretations or projections;
which evidence is missing, stale, partial, timed out, or forbidden;
what the current installation may honestly claim;
how the result can be reproduced and inspected.
Its fundamental rule is:
A numeric zero is valid only after complete evaluation of the declared scope.
This produces two related but deliberately separate state systems.
Operation results use:
complete
partial
unavailable
timeout
forbidden
error
Component and evidence condition can independently be:
ready
degraded
partial
stale
missing
refused
disabled
unknown
This separation matters. “Stale” describes evidence or a projection. “Timeout” describes an operation. “Disabled” describes configuration. “Forbidden” describes authority. “Zero” describes a completed count. None is a synonym for the others.
The full contract is in
03-truth-contract.md, with the machine-readable
evidence-envelope.schema.json.
What the source audit found
The audit was regenerated from the supplied Polylogue snapshot rather than copied from the earlier narrative. Every recorded source reference was subsequently resolved and line-validated against the supplied Git tree.
The main defects are not cosmetic.
Result content and status presentation can disagree
The current surface can populate result content while separate status/count presentation remains in a checking or zero-like state. The target contract makes counts part of the same result envelope as the rows they describe.
Disabled components can appear operational
Configuration, process liveness, successful observation, evidence availability, freshness, projection readiness, and claimability are currently too easy to conflate.
The rewrite treats them as independent facts. A component started with capture disabled must say disabled, not running, ready, or healthy.
Attachment state is insufficiently precise
These are now distinct:
metadata observed;
bytes acquired;
digest verified;
bytes retrievable now;
disclosure permitted;
bytes deleted or tombstoned.
A historical acquisition record cannot by itself support a current download or preview claim.
Advertised routes and client-recognized routes can diverge
A route exposed by navigation is a public contract. Every advertised route must:
render successfully;
redirect intentionally with a reason;
or return an explicit unsupported/refused outcome.
Blank pages and parser disagreement are not acceptable route states.
Some pages bypass shared bounded loading
Standalone attachment and paste paths can diverge from the main shell’s fetch, timeout, cancellation, caveat, and error behavior.
The target has one bounded evidence loader for all browser reads. No page may spin indefinitely or translate failure into emptiness.
Cost and usage need lane-level provenance
A credible usage view must expose:
fresh input;
cache read and write;
reasoning;
ordinary output;
provider observation;
physical versus logical scope;
pricing recipe and version;
unknown or unsupported lanes;
timeout or partial status.
It must not present a catalog estimate as an invoice or silently double-count inclusive cached input.
The full ledger is
02-source-and-truth-audit.md. Supporting artifacts include:
source-excerpts.md
route-inventory.csv
defect-ledger.csv
source-ref-validation.md
Executed evidence
The environment probe and all its command records are preserved rather than summarized into an overly broad success statement.
STATUS.md
manifest.json
Complete executed-proof directory
The manifest distinguishes commands that ran, routes that responded, failures, skips, environment limitations, and source-derived proposals.
A successful deterministic route proves only that route, revision, fixture, and environment. A failed probe remains in the record. Proposed fields not supplied by the current daemon remain labeled target fields.
Interactive prototype
index.html
The prototype is dependency-free HTML, CSS, and JavaScript over the public-safe Incident 14:32 proof world. It contains seven evidence-centered views.
Archive truth
Separates:
component configuration and process state;
successful evidence observation;
source coverage;
projection and index freshness;
disclosure authority;
known debt;
current claimability.
Claim receipt
Shows:
the assistant claim;
the structurally failed tool result;
later successful verification;
evidence strength;
exact refs;
relevant lineage;
authority and caveats.
Session chronicle
Shows:
messages and semantic tool cards;
physical versus composed history;
copied-prefix lineage;
fresh subagents;
continuation and compaction boundaries;
assertions and context events;
incomplete or unavailable material.
Cost and usage
Separates token lanes, physical/logical scope, provider observations, pricing recipes, and unsupported economic interpretations.
Context autopsy
Reconstructs:
selected evidence;
accepted assertions;
generated context;
omitted material;
token and message budgets;
stale items;
lossiness;
the actual historical delivery boundary.
Material lifecycle
Demonstrates metadata-only, retrievable, missing, forbidden, and deleted material without collapsing them into an acquired Boolean.
Agent Work Packet
Links Beads intent, Polylogue sessions, subagents, repository state, commands, verification, context, judgments, coverage, and unresolved gaps while preserving each source’s native ref and authority.
Visual artifacts:
architecture.svg
truth-state.svg
journey-map.svg
maximal-backend.svg
The prototype defines target behavior and information hierarchy. It is not presented as evidence that the present daemon already implements every field.
Demonstration portfolio completely reconsidered
The revised portfolio is in
09-demo-portfolio.md. There are 15 machine-readable manifests under demo-manifests.
Every demo must specify:
a falsifiable claim;
a consequential decision;
independent ground truth;
the strongest plausible simpler baseline;
positive, negative, and refusal controls;
a concrete falsifier;
exact evidence refs;
reproduction identity;
machine and human artifacts;
threats to validity;
non-claims.
The redesigned portfolio is:
Receipts in Browser — claim versus structural result and later repair.
Zero Is a Claim — complete-empty versus partial, unavailable, timeout, and forbidden.
Disabled Means Disabled — configuration cannot masquerade as evidence readiness.
Missing Bytes Are Not Acquired — material lifecycle and current retrievability.
Every Advertised Route Has a Truthful Outcome — route contract parity.
Count It Once, Preserve It All — copied lineage versus fresh subagents.
Cost Ledger Under Poisoned Inputs — disjoint usage lanes and non-invoice economics.
Context Autopsy — historical delivery, omission, authority, and loss.
Judgment Changes Authority, Not History — reviewed memory lifecycle.
The System Changes Its Mind Honestly — replay and interpretation history.
World Around the Claim — the joint Agent Work Packet.
Rebuild Cockpit from Sinex — backend parity rather than metadata mirroring.
Resume Under Oath — raw-reference, generated-summary, and reviewed-context arms.
Forgetting Propagates — cross-store deletion and residual accounting.
Giant Session, Bounded Truth — load behavior without silent truncation or false zero.
The flagship fixture deliberately contains anti-grep, poisoned-accounting, stale-memory, missing-source, copied-lineage, and refusal controls. It is designed so that an attractive but semantically weak implementation fails.
Supporting contracts:
demo-packet.schema.json
PROOF-REPORT.md
fault-matrix.csv
validate-demo-packets.py
Rapid implementation program
The program contains 35 bounded work packages across six waves:
Wave 0 — contracts, red tests, fixture and independent oracle
Wave 1 — truth floor
Wave 2 — evidence primitives and highest-value journeys
Wave 3 — judgment and Sinex substrate
Wave 4 — accessibility, disclosure, fault, load and rebuild proof
Wave 5 — adversarial integration and launch
The complete plan is
10-implementation-program.md, with the machine graph in
work-packages.json.
Packet P01 is the launch gate
P01 implements polylogue-bby.1’s truthful slow and missing behavior before any major page polish.
It must establish:
complete, partial, unavailable, timeout, forbidden, and error;
no zero for unevaluated evidence;
bounded compatibility for old routes;
caveats and frontiers;
exact refs;
reproduction identity;
shared CLI/web domain semantics.
Visual work is not allowed to invent substitutes while that contract is absent.
The next parallel truth-floor packets are:
component truth;
material lifecycle;
route registry and capabilities;
bounded browser loader;
source/projection/index frontiers.
Only then do receipt, chronicle, usage, archive-truth, context, material, and judgment pages assemble over common primitives.
Existing Beads remain planning authority. The work packages are a rapid-execution grouping, not a replacement planning system. The resolved subset from the supplied snapshots is in
beads-subset.md.
Single-machine frontier-agent swarm
The complete operating system is in
11-single-machine-swarm.md.
Four captains centralize the things that cannot safely fork:
Interface captain: result contracts, refs, routes, truth vocabulary.
Integration captain: final worktree and ruthless conflict collapse.
Proof and claims captain: oracle, Demo Packets, claims and non-claims.
Sole Beads captain: the only planning writer during the campaign.
Implementation workers receive strict path ownership and run focused tests. Broad tests, generated surfaces, media, and final proof run only from integration.
Git history is deliberately expendable. The integration captain may:
cherry-pick;
extract only owned files;
copy a known-good path;
squash;
discard a noisy branch;
or reimplement a smaller compatible change.
What cannot be sacrificed is:
interface compatibility;
exact test evidence;
integrated revision identity;
proof artifacts;
claim discipline.
The
15-first-72-hours.md is designed to produce one narrow, defensible cockpit path rapidly.
Coordination utilities include:
bootstrap-worktrees.sh
swarm-board.py
with-heavy-lock.sh
resource-plan.sh
launch-tmux.sh
WORKER-HANDOFF.md
INTERFACE-DECISION.md
COLD-READER-REPORT.md
Twenty-four parallel fork prompts
The prompts are indexed in
12-fork-prompts.md. A single copy/paste document is available as
12a-all-fork-prompts.md.
They cover:
interface captaincy;
current-source regression conversion;
Incident 14:32 and its independent oracle;
result envelope;
component truth;
material lifecycle;
route registry;
bounded browser loading;
evidence UI primitives;
semantic tool cards;
claim-to-receipt journey;
session chronicle and lineage;
honest usage ledger;
archive truth;
context autopsy;
assertion and judgment;
accessibility;
browser fault/load proof;
disclosure security;
public claims and documentation;
transcript-complete Sinex material;
stable identity, settlement, and rebuild;
Agent Work Packet;
ruthless integration and release audit.
Each prompt has owning Beads, dependencies, strict paths, acceptance criteria, verification, shared constraints, and a required handoff.
Maximal Sinex backend
The complete architecture and impedance analysis is
08-sinex-polylogue-backend-fit.md.
The target authority split is:
Sinex stores
provider-native transcript artifacts;
immutable Polylogue-normalized segments;
attachments and context-image bytes;
durable session, message, tool, usage, assertion, judgment, and delivery history;
acquisition, coverage, replay, retention, and deletion state;
model effects;
ambient machine evidence;
cross-source Agent Work Packet components.
Polylogue owns
provider normalization;
AI-work ontology;
message and block semantics;
physical and logical session composition;
continuation, fork, fresh-subagent, and compaction meaning;
tool-result interpretation;
usage and cost semantics;
assertion and context policy;
transcript and evidence UX;
domain query behavior.
SQLite remains
standalone authority when Sinex is absent;
edge replica in backed mode;
local FTS/vector acceleration;
offline cache;
pending-write outbox;
local UI state;
deterministic fixture and test substrate.
The decisive rule is:
In Sinex-backed mode, no irreplaceable transcript evidence or durable user judgment should exist only in Polylogue SQLite.
The analysis addresses the principal impedance mismatches:
replay event identity versus stable citation identity;
mutable sessions versus immutable observations;
rewritten material and unstable byte offsets;
session topology versus derivation provenance;
physical evidence versus logical work;
event granularity;
revision settlement;
timestamp versus causal ordering;
differing query and schema models;
assertion and judgment overlap;
context artifact ownership;
embedding recipe identity;
SHA-256 versus BLAKE3 content descriptors;
disclosure versus storage authority;
deletion through shared lineage;
offline dual-master risk;
combined MCP tool-surface overload.
The target joint machine object is defined in the
agent-work-packet.schema.json.
Repository patches
Polylogue
Download the Polylogue cockpit starter patch
It adds:
cockpit entry-point documentation;
the truth contract;
the Evidence Envelope schema;
the interactive static prototype;
a static web-contract audit developer tool;
demonstration doctrine;
explicit non-claims.
Sinex
Download the Sinex backend starter patch
It adds:
the transcript-complete backend direction;
stable identity and settlement requirements;
protected material versus transport distinction;
SQLite’s backed-mode role;
the Agent Work Packet schema;
the decisive rebuild proof.
Both patches were checked against the exact supplied Git bases. Application instructions and base revisions are in
README.md.
They are intentionally additive starter patches. They do not pretend that the runtime rewrite or Sinex reconstruction has already landed.
Presentation artifacts
Polylogue-Web-Evidence-Cockpit.pptx
Polylogue-Web-Evidence-Cockpit.pdf
Polylogue-Web-Evidence-Cockpit-Program.pdf
Polylogue-Web-Evidence-Cockpit-Program.html
Deck contact sheet
Validation
The final checker passed the distribution package. Its coverage includes:
required artifacts;
JSON and YAML syntax;
JSON Schema validity;
claims-ledger and work-package schema validation;
Python, JavaScript, and shell syntax;
acyclic work-package dependencies;
24 complete fork prompts;
D1–D15 manifest coverage;
package-local Markdown links;
prototype rendering and screenshots;
PowerPoint ZIP integrity;
PDF integrity;
source file:line resolution;
exact-base patch application;
credential and private-key residue heuristics;
ZIP integrity;
per-file SHA-256 manifest generation.
The complete record is
18-validation-report.md, and the per-file checksums are in
PACKAGE-MANIFEST.json.
The most important boundary remains
17-NON-CLAIMS.md: the package does not represent the production truth floor, semantic renderer, transcript-complete Sinex gateway, full SQLite reconstruction, selective forgetting, large-archive performance, or general agent uplift as completed merely because the design and proof machinery now exist.
Polylogue-Web-Evidence-Cockpit-Program.html
